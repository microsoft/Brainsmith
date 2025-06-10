# Brainsmith-2: Developer Integration Guide

## Getting Started

### Prerequisites

**System Requirements**
- Python 3.8+ with pip package manager
- Docker (for containerized development)
- Git for version control
- 8GB+ RAM recommended for model compilation

**External Dependencies**
- FINN framework (automatically configured)
- QONNX for ONNX model preprocessing
- Tree-sitter for RTL parsing
- Brevitas for quantization (optional, for training)

### Installation and Setup

**Quick Setup (Recommended)**
```bash
# Clone the repository
git clone <repository-url> brainsmith-2
cd brainsmith-2

# Install in development mode
pip install -e .

# Verify installation
python -c "from brainsmith.core.hw_compiler import forge; print('Installation successful')"
```

**Docker-Based Setup**
```bash
# Build and run development container
./run-docker.sh

# Inside container, all dependencies are pre-configured
cd /workspace
python demos/bert/end2end_bert.py --help
```

**Development Environment Setup**
```bash
# Install additional development dependencies
pip install -r requirements.txt

# Run test suite to verify setup
python -m pytest tests/ -v

# Optional: Setup pre-commit hooks
pip install pre-commit
pre-commit install
```

### First Project Walkthrough

**BERT Model Deployment (Zero Configuration)**
```bash
# Navigate to BERT demo
cd demos/bert

# Deploy BERT-base model to FPGA (single command)
python end2end_bert.py \
    --output_dir ./output \
    --num_layers 12 \
    --num_attention_heads 12 \
    --hidden_size 384 \
    --intermediate_size 1536

# Expected output:
# ✅ Model compilation successful
# ✅ FPGA bitstream generated: ./output/bert_accelerator.bit
# ✅ Driver files generated: ./output/driver/
```

**Custom Hardware Kernel Generation (HKG)**
```bash
# Generate FINN components from RTL specification (Updated syntax)
cd examples/thresholding
python -m brainsmith.tools.hw_kernel_gen.hkg \
    thresholding_axi.sv \
    dummy_compiler_data.py \
    -o ./generated

# Expected output:
# ✅ RTL parsing successful: 4 interfaces detected
# ✅ Dataflow model built: 0 interfaces converted  
# ✅ Generated files:
#   - autothresholdingaxi.py (HWCustomOp)
#   - autothresholdingaxi_rtlbackend.py (RTL Backend)
#   - test_autothresholdingaxi.py (Test Suite)
#   - thresholding_axi_wrapper.v (Verilog Wrapper)
#   - autothresholdingaxi_README.md (Documentation)

```

## Core Workflows

### 1. Model Compilation Pipeline

**Basic Model Compilation**
```python
from brainsmith.core.hw_compiler import forge

# Compile ONNX model using BERT blueprint
result = forge(
    blueprint='bert',
    model='path/to/model.onnx',
    args={
        'target_fps': 1000,
        'resource_budget': 'moderate',
        'precision': 'mixed'
    }
)

print(f"Compilation successful: {result.success}")
print(f"Output directory: {result.output_path}")
print(f"Performance estimate: {result.performance_metrics}")
```

**Custom Blueprint Usage**
```python
from brainsmith.blueprints import register_blueprint

@register_blueprint("custom_transformer")
def custom_transformer_pipeline(model, args):
    """Custom build process for specialized transformer variants."""
    return [
        custom_preprocessing_step,
        enhanced_attention_optimization,
        custom_hardware_mapping,
        performance_validation_step
    ]

# Use custom blueprint
result = forge('custom_transformer', model, args)
```

### 2. Hardware Kernel Development

**Automated Kernel Generation**
```python
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator

# Create generator instance
hkg = HardwareKernelGenerator(
    rtl_file_path='custom_operation.sv',
    compiler_data_path='operation_metadata.py',
    output_dir='./generated'
)

# Generate complete FINN integration package
artifacts = hkg.run()

print("Generated components:")
for component_type, file_path in artifacts.items():
    print(f"  {component_type}: {file_path}")
```

**Using Interface Metadata (Recommended)**
```python
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType
from brainsmith.dataflow.core.chunking_strategy import index_chunking

# Define interfaces using enhanced metadata system
input_metadata = InterfaceMetadata(
    name="data_input",
    interface_type=DataflowInterfaceType.INPUT,
    allowed_datatypes=[
        DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True)
    ],
    chunking_strategy=index_chunking(-1, "[96]")  # Runtime-parameterized
)

# Use with AutoHWCustomOp for automatic dimension extraction
class CustomOperation(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        self._interface_metadata = [input_metadata, output_metadata]
        # Dimensions automatically extracted from ModelWrapper at runtime
        super().__init__(onnx_node, interface_metadata=self._interface_metadata, **kwargs)
        
    # All FINN methods automatically implemented by base class
    # - get_input_datatype() ✓ 
    # - get_output_datatype() ✓
    # - bram_estimation() ✓
    # - lut_estimation() ✓  
    # - dsp_estimation() ✓
    # - get_exp_cycles() ✓
```

### 3. Custom Operation Development

**Advanced AutoHWCustomOp Usage**
```python
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType
from brainsmith.dataflow.core.chunking_strategy import index_chunking

class CustomLayerNorm(AutoHWCustomOp):
    """
    Enhanced layer normalization using AutoHWCustomOp.
    Demonstrates runtime dimension extraction and automatic optimization.
    """
    
    def __init__(self, onnx_node, **kwargs):
        # Define interface metadata with chunking strategies
        self._interface_metadata = [
            InterfaceMetadata(
                name="input",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True)
                ],
                chunking_strategy=index_chunking(-1, "[hidden_size // parallelism]")
            ),
            InterfaceMetadata(
                name="output", 
                interface_type=DataflowInterfaceType.OUTPUT,
                allowed_datatypes=[
                    DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True)
                ],
                chunking_strategy=index_chunking(-1, "[hidden_size // parallelism]")
            )
        ]
        
        # Base class handles all FINN method implementations automatically
        super().__init__(onnx_node, interface_metadata=self._interface_metadata, **kwargs)
    
    # Optional: Custom methods for operation-specific functionality
    def get_performance_estimate(self, batch_size: int = 1) -> dict:
        """Get performance estimate for given batch size."""
        # Dimensions automatically extracted from ModelWrapper
        hidden_size = self.extract_runtime_dimension("input", "tDim")
        parallelism = self.extract_runtime_dimension("input", "stream_dims")
        
        cycles_per_sample = hidden_size // parallelism
        return {
            'latency_cycles': cycles_per_sample,
            'throughput_samples_per_sec': self.get_frequency() / cycles_per_sample,
            'batch_latency': cycles_per_sample * batch_size
        }
    
    # All standard FINN methods automatically implemented:
    # ✓ get_input_datatype() - uses runtime extraction
    # ✓ get_output_datatype() - uses runtime extraction  
    # ✓ bram_estimation() - uses runtime dimensions
    # ✓ lut_estimation() - uses runtime dimensions
    # ✓ dsp_estimation() - uses runtime dimensions
    # ✓ get_exp_cycles() - uses runtime dimensions
```

**Custom RTL Backend Implementation**
```python
from brainsmith.dataflow.core.auto_rtl_backend import AutoRTLBackend

class CustomLayerNormRTLBackend(AutoRTLBackend):
    """RTL backend for custom layer normalization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate_params(self, model, path):
        """Generate hardware parameters from dataflow model."""
        # Automatically extract parallelism from dataflow model
        parallelism = self.dataflow_model.get_optimal_parallelism()
        
        return {
            'HIDDEN_SIZE': model.get_nodeattr("hidden_size"),
            'PARALLELISM': parallelism['data_parallelism'],
            'DATA_WIDTH': self.get_instream_width(),
            'PRECISION': self.get_precision_config()
        }
    
    def code_generation_dict(self):
        """Generate RTL instantiation dictionary."""
        code_gen_dict = super().code_generation_dict()
        
        # Add custom RTL generation
        code_gen_dict["$CUSTOM_LOGIC$"] = self._generate_custom_logic()
        code_gen_dict["$RESOURCE_PRAGMAS$"] = self._generate_resource_pragmas()
        
        return code_gen_dict
```

## API Reference

### Primary Entry Points

**Core Compilation API**
```python
from brainsmith.core.hw_compiler import forge

def forge(blueprint: str, model: str, args: Dict[str, Any]) -> CompilationResult:
    """
    Primary entry point for hardware compilation.
    
    Args:
        blueprint: Build process identifier ('bert', 'custom_transformer', etc.)
        model: Path to ONNX model file
        args: Compilation configuration and optimization parameters
    
    Returns:
        CompilationResult with success status, output paths, and metrics
    """
```

**Hardware Kernel Generator API**
```python
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator

class HardwareKernelGenerator:
    def __init__(self, rtl_file_path: str, compiler_data_path: str, 
                 output_dir: str = "./generated"):
        """Initialize HKG with RTL specification and metadata."""
    
    def run(self) -> Dict[str, Path]:
        """Execute complete generation pipeline."""
    
    def generate_hwcustomop(self) -> Path:
        """Generate only HWCustomOp component."""
    
    def generate_rtlbackend(self) -> Path:
        """Generate only RTL backend component."""
```

**Dataflow Framework API**
```python
from brainsmith.dataflow import DataflowInterface, DataflowModel

class DataflowInterface:
    def __init__(self, name: str, interface_type: str, 
                 num_tensors: List[int], tDim: List[int], stream_dims: List[int], dtype: str):
        """Create interface with three-tier dimension system."""
    
    def validate_constraints(self) -> ValidationResult:
        """Validate dimensional relationships and constraints."""

class DataflowModel:
    def calculate_initiation_intervals(self, iPar: int, wPar: int) -> Dict[str, int]:
        """Calculate performance characteristics."""
    
    def optimize_parallelism(self, constraints: Dict) -> Dict[str, int]:
        """Find optimal parallelism configuration."""
```

### Configuration Systems

**Multi-Level Configuration**
```python
from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig

# Complete pipeline configuration
config = PipelineConfig(
    # Template configuration
    template=TemplateConfig(
        template_dirs=['custom_templates/', 'dataflow_templates/'],
        selection_strategy='auto',
        cache_templates=True
    ),
    
    # Generation configuration  
    generation=GenerationConfig(
        enabled_generators={'hwcustomop', 'rtlbackend', 'test_suite'},
        output_organization='flat',
        overwrite_existing=True
    ),
    
    # Analysis configuration
    analysis=AnalysisConfig(
        interface_detection='enhanced',
        pragma_processing=True,
        caching_enabled=True
    ),
    
    # Dataflow configuration
    dataflow=DataflowConfig(
        mode='DATAFLOW_ONLY',
        optimization_level='aggressive',
        resource_constraints={'max_luts': 100000, 'max_dsps': 500}
    )
)

# Use configuration with HKG
hkg = HardwareKernelGenerator(rtl_file, metadata_file, config=config)
```

**Environment-Specific Configuration**
```python
# Development configuration
dev_config = PipelineConfig.development_defaults()

# Production configuration
prod_config = PipelineConfig.production_defaults()

# Custom configuration from file
config = PipelineConfig.from_file('deployment_config.yaml')
```

### Extension Points

**Custom Blueprint Registration**
```python
from brainsmith.blueprints import register_blueprint, BuildStep

@register_blueprint("vision_transformer")
def vision_transformer_pipeline(model, args):
    """Custom pipeline for Vision Transformer models."""
    return [
        BuildStep("preprocess_patches", preprocess_patch_embeddings),
        BuildStep("optimize_attention", optimize_multi_head_attention),
        BuildStep("hardware_mapping", map_to_hardware_resources),
        BuildStep("validate_performance", validate_throughput_requirements)
    ]
```

**Custom Generator Development**
```python
from brainsmith.tools.hw_kernel_gen.enhanced_generator_base import EnhancedGeneratorBase

class CustomDocumentationGenerator(EnhancedGeneratorBase):
    """Generate specialized documentation for custom operations."""
    
    def generate(self, context: Dict[str, Any]) -> GeneratedArtifact:
        """Generate custom documentation format."""
        template = self.template_manager.get_template('custom_docs.j2')
        content = template.render(context)
        
        return GeneratedArtifact(
            file_name="custom_operation_guide.md",
            content=content,
            artifact_type="documentation"
        )
```

## Best Practices

### Performance Optimization

**Parallelism Configuration**
```python
# Let dataflow framework optimize automatically
model = DataflowModel(interfaces, operation_type="custom")
optimal_config = model.optimize_parallelism({
    'target_throughput': 1000,  # samples/second
    'max_resources': {'luts': 50000, 'dsps': 200},
    'power_budget': 'moderate'
})

# Apply optimized configuration
model.apply_parallelism_config(optimal_config)
```

**Resource Management**
```python
# Configure caching for development workflows
config = PipelineConfig(
    analysis=AnalysisConfig(
        caching_enabled=True,
        cache_size_limit=1000,  # MB
        cache_persistence=True
    ),
    dataflow=DataflowConfig(
        model_cache_size=500,   # MB
        interface_cache_size=100  # MB
    )
)
```

**Template Optimization**
```python
# Use base class inheritance for minimal code generation
template_config = TemplateConfig(
    selection_strategy='base_class_preferred',
    inheritance_threshold=0.8,  # Use inheritance if >80% code reuse
    optimization_level='aggressive'
)
```

### Error Handling and Debugging

**Comprehensive Error Handling**
```python
from brainsmith.core.hw_compiler import forge, CompilationError

try:
    result = forge('bert', model_path, args)
    
    if not result.success:
        print("Compilation warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
            
except CompilationError as e:
    print(f"Compilation failed: {e}")
    print(f"Error context: {e.context}")
    print(f"Suggested fixes: {e.suggestions}")
```

**Debug Mode Configuration**
```python
# Enable detailed logging and intermediate artifact preservation
debug_config = PipelineConfig(
    generation=GenerationConfig(
        preserve_intermediates=True,
        debug_output=True,
        verbose_logging=True
    ),
    validation=ValidationConfig(
        strict_validation=True,
        detailed_reports=True
    )
)
```

**Performance Profiling**
```python
from brainsmith.tools.profiling import ModelProfiler

# Profile model performance characteristics
profiler = ModelProfiler(model_path)
profile_results = profiler.analyze_computational_requirements()

print(f"Estimated cycles: {profile_results.cycle_estimate}")
print(f"Resource requirements: {profile_results.resource_estimate}")
print(f"Memory bandwidth: {profile_results.memory_bandwidth}")
```

### Testing and Validation

**Automated Testing Integration**
```python
# Generate comprehensive test suite with HKG
hkg = HardwareKernelGenerator(rtl_file, metadata, 
                             generate_tests=True)
artifacts = hkg.run()

# Run generated tests
test_file = artifacts['test_suite']
subprocess.run(['python', '-m', 'pytest', str(test_file), '-v'])
```

**Custom Validation**
```python
from brainsmith.dataflow.core.validation import ValidationSeverity

# Add custom validation rules
def validate_memory_requirements(dataflow_model):
    """Custom validation for memory usage."""
    memory_estimate = dataflow_model.estimate_memory_usage()
    
    if memory_estimate > MAX_MEMORY_BUDGET:
        return ValidationResult(
            severity=ValidationSeverity.ERROR,
            message=f"Memory usage {memory_estimate} exceeds budget",
            suggestions=["Reduce parallelism", "Use smaller data types"]
        )
    
    return ValidationResult.success()

# Register custom validator
dataflow_model.add_validator(validate_memory_requirements)
```

## Integration Examples

### CI/CD Pipeline Integration
```yaml
# .github/workflows/fpga_build.yml
name: FPGA Build Pipeline

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Brainsmith-2
      run: |
        pip install -e .
        
    - name: Validate Model
      run: |
        python -c "from brainsmith.core.hw_compiler import forge; forge('bert', 'models/bert.onnx', {'validate_only': True})"
        
    - name: Generate Hardware Components
      run: |
        python -m brainsmith.tools.hw_kernel_gen.hkg custom_ops/*.sv metadata/*.py
        
    - name: Run Tests
      run: |
        python -m pytest tests/ generated/test_*.py -v
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

COPY . /app
WORKDIR /app

RUN pip install -e .

# Pre-compile common models for faster deployment
RUN python demos/bert/end2end_bert.py --precompile_only

ENTRYPOINT ["python", "-m", "brainsmith.core.hw_compiler"]
```

This developer integration guide provides comprehensive coverage of Brainsmith-2's APIs, workflows, and best practices, enabling teams to effectively integrate the platform into their development processes and build custom FPGA AI acceleration solutions.