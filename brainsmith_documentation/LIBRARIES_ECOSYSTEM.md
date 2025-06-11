# BrainSmith Libraries Ecosystem

The BrainSmith libraries layer provides specialized tools for FPGA accelerator development, organized into five focused domains. Each library follows the same "Functions Over Frameworks" philosophy with zero-barrier contribution workflows.

## Architecture Overview

```mermaid
graph TB
    subgraph "Libraries Layer"
        A[Automation] --> A1[Parameter Sweeps]
        A --> A2[Batch Processing]
        
        B[Kernels] --> B1[Hardware Implementations]
        B --> B2[Performance Models]
        
        C[Transforms] --> C1[Model Pipeline Steps]
        C --> C2[FINN Operations]
        
        D[Analysis] --> D1[Profiling Tools]
        D --> D2[Performance Analysis]
        
        E[Blueprints] --> E1[Hardware Templates]
        E --> E2[Configuration Libraries]
    end
    
    F[Core forge Function] --> A
    F --> B
    F --> C
    F --> D
    F --> E
    
    style F fill:#4CAF50,color:#fff
    style A fill:#FF9800,color:#fff
    style B fill:#FF9800,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#FF9800,color:#fff
    style E fill:#FF9800,color:#fff
```

## Automation Library

**Location**: `brainsmith/libraries/automation/`
**Purpose**: Batch processing and parameter exploration automation

### Core Functions

#### Parameter Sweeps
Explore parameter combinations by running forge() multiple times:

```python
from brainsmith.libraries.automation import parameter_sweep

# Explore different PE counts and SIMD widths
results = parameter_sweep(
    "model.onnx", "blueprint.yaml",
    {
        'pe_count': [4, 8, 16, 32],
        'simd_width': [2, 4, 8, 16]
    },
    max_workers=4
)

# Find optimal configuration
best = find_best(results, metric='throughput')
print(f"Optimal: PE={best['sweep_info']['parameters']['pe_count']}")
```

#### Batch Processing
Process multiple model/blueprint pairs:

```python
from brainsmith.libraries.automation import batch_process

# Process multiple models
model_blueprint_pairs = [
    ("model1.onnx", "blueprint1.yaml"),
    ("model2.onnx", "blueprint2.yaml"),
    ("model3.onnx", "blueprint3.yaml")
]

batch_results = batch_process(
    model_blueprint_pairs,
    common_config={'max_luts': 0.8},
    max_workers=4
)
```

#### Statistical Analysis
Generate comprehensive statistics from results:

```python
from brainsmith.libraries.automation import aggregate_stats

stats = aggregate_stats(results)
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Mean throughput: {stats['aggregated_metrics']['throughput_ops_sec']['mean']:.0f}")
```

### Registry System
Automatic discovery of automation components:

```python
from brainsmith.libraries.automation import discover_all_automation_components

components = discover_all_automation_components()
# Automatically finds custom automation tools in contrib/ directories
```

## Kernels Library

**Location**: `brainsmith/libraries/kernels/`
**Purpose**: Hardware kernel implementation management with 93% code reduction

### Revolutionary Design

The kernels framework represents a complete transformation from enterprise complexity:

**Before**: 6,415 lines of enterprise framework
**After**: 558 lines of pure functions (93% reduction)

### Core Functions

#### Kernel Discovery
Convention-based discovery without complex registration:

```python
from brainsmith.libraries.kernels import discover_all_kernels

# Discovers all kernels in standard + additional paths
kernels = discover_all_kernels()
print(f"Found {len(kernels)} kernel packages")

# Load specific kernel
kernel = load_kernel_package("./my_custom_kernel/")
```

#### Kernel Selection
Smart selection based on requirements:

```python
from brainsmith.libraries.kernels import (
    KernelRequirements, select_optimal_kernel
)

# Define requirements
requirements = KernelRequirements(
    operator_type="Convolution",
    datatype="int8",
    max_pe=32,
    performance_requirements={'min_throughput': 500000}
)

# Select optimal kernel
selection = select_optimal_kernel(requirements, strategy='throughput')
print(f"Selected: {selection.kernel.name}")
print(f"Config: PE={selection.pe_parallelism}, SIMD={selection.simd_width}")
```

#### FINN Configuration Generation
Direct integration with FINN:

```python
from brainsmith.libraries.kernels import generate_finn_config

# Create selections for multiple layers
selections = {
    'conv1': conv_selection,
    'matmul1': matmul_selection,
    'thresh1': threshold_selection
}

# Generate complete FINN config
finn_config = generate_finn_config(selections)
```

### Zero-Barrier Contribution

Adding a custom kernel requires no registration or complex setup:

**1. Create Directory Structure:**
```bash
mkdir my_awesome_kernel/
```

**2. Add Manifest (kernel.yaml):**
```yaml
name: "my_awesome_kernel"
operator_type: "Convolution"
backend: "HLS"
version: "1.0.0"
parameters:
  pe_range: [1, 32]
  simd_range: [2, 16]
  supported_datatypes: ["int8", "int16"]
files:
  hls_source: "kernel.cpp"
  header: "kernel.hpp"
  testbench: "test_kernel.cpp"
performance:
  optimization_target: "throughput"
  estimated_throughput: 2000000
```

**3. Add Implementation Files:**
```cpp
// kernel.cpp - HLS implementation
void my_kernel(/* parameters */) {
    // Implementation here
}
```

**4. Automatic Discovery:**
```python
# No registration needed - automatic discovery!
kernels = discover_all_kernels()
assert 'my_awesome_kernel' in kernels
```

### Package Structure
Every kernel follows a standard directory structure:

```
my_kernel/
├── kernel.yaml          # Package manifest (required)
├── source_RTL.sv        # RTL implementation
├── kernel_hw_custom_op.py # Python backend
├── kernel_rtl_backend.py  # RTL backend
├── rtl_wrapper.v        # Verilog wrapper
├── testbench.py         # Tests (optional)
└── README.md            # Documentation (optional)
```

## Transforms Library

**Location**: `brainsmith/libraries/transforms/`
**Purpose**: Model transformation pipeline management

### Transform Steps
Direct access to transformation functions:

```python
from brainsmith.libraries.transforms.steps import (
    cleanup_step, streamlining_step, infer_hardware_step
)

# Process model through pipeline
model = cleanup_step(model, config)
model = streamlining_step(model, config)
model = infer_hardware_step(model, config)
```

### FINN Compatibility
Seamless integration with existing FINN workflows:

```python
from brainsmith.libraries.transforms.steps import get_step

# Get step function with FINN fallback
step_fn = get_step("cleanup")
model = step_fn(model, config)

# Validate step sequences
from brainsmith.libraries.transforms.steps import validate_step_sequence
errors = validate_step_sequence(["cleanup", "streamlining", "hardware"])
```

### Transform Operations
Direct model manipulation functions:

```python
from brainsmith.libraries.transforms.operations import (
    convert_to_hw_layers, expand_norms, shuffle_helpers
)

# Apply specific operations
model = convert_to_hw_layers(model)
model = expand_norms(model)
```

### Registry System
Automatic discovery of available transforms:

```python
from brainsmith.libraries.transforms import discover_all_transforms

transforms = discover_all_transforms()
# Finds both built-in and custom transforms
```

## Analysis Library

**Location**: `brainsmith/libraries/analysis/`
**Purpose**: Performance analysis and profiling tools

### Roofline Analysis
Comprehensive performance modeling:

```python
from brainsmith.libraries.analysis import roofline_analysis, RooflineProfiler

# Quick analysis
result = roofline_analysis(model_config, hw_config, dtypes=[4, 8])

# Detailed profiling with reports
profiler = RooflineProfiler()
profile_results = profiler.profile_model(
    model_config={'arch': 'bert', 'num_layers': 12},
    hardware_config={'dsps': 2520, 'luts': 274080}
)

# Generate HTML report
html_report = profiler.generate_report(profile_results, 'roofline_report.html')
```

### Hardware Kernel Generation
Advanced code generation tools:

```python
from brainsmith.libraries.analysis.tools import generate_hw_kernel

# Generate custom hardware kernels
kernel_code = generate_hw_kernel(
    operator_type="Convolution",
    datatype="int8",
    optimization_target="throughput"
)
```

### Registry System
Extensible analysis tool discovery:

```python
from brainsmith.libraries.analysis import discover_all_analysis_tools

tools = discover_all_analysis_tools()
# Automatically finds profiling and analysis tools
```

## Blueprints Library

**Location**: `brainsmith/libraries/blueprints/`
**Purpose**: Hardware accelerator template management

### Blueprint Discovery
Automatic discovery of blueprint templates:

```python
from brainsmith.libraries.blueprints import (
    discover_all_blueprints, BlueprintCategory
)

# Discover all available blueprints
blueprints = discover_all_blueprints()

# Find blueprints by category
basic_blueprints = find_blueprints_by_category(BlueprintCategory.BASIC)
advanced_blueprints = find_blueprints_by_category(BlueprintCategory.ADVANCED)
```

### Blueprint Loading
Access blueprint configurations:

```python
from brainsmith.libraries.blueprints import get_blueprint_by_name

# Load specific blueprint
blueprint = get_blueprint_by_name("cnn_accelerator")
print(f"Description: {blueprint.description}")
print(f"Model type: {blueprint.model_type}")
print(f"Parameters: {list(blueprint.parameters.keys())}")
```

### Blueprint Structure
Standard blueprint organization:

```
libraries/blueprints/
├── basic/
│   ├── cnn_accelerator.yaml
│   └── simple_mlp.yaml
├── advanced/
│   ├── bert_optimized.yaml
│   └── mobilenet_accelerator.yaml
└── experimental/
    └── attention_engine.yaml
```

**Sample Blueprint YAML:**
```yaml
name: "Efficient CNN Accelerator"
version: "1.2.0"
description: "Optimized CNN accelerator for edge inference"
model_type: "CNN"
target_platform: "Zynq UltraScale+"

parameters:
  pe_count:
    type: "integer"
    range: [4, 64]
    default: 16
    description: "Processing elements for parallelization"
  
  simd_width:
    type: "integer"
    range: [2, 32]
    default: 8
    description: "SIMD width for vector operations"

targets:
  throughput:
    direction: "maximize"
    priority: "high"
    target_value: 1000000
  
  power:
    direction: "minimize"
    priority: "medium"
    target_value: 10.0

constraints:
  max_lut_util: 0.85
  max_dsp_util: 0.90
  max_bram_util: 0.80
```

## Library Extension Patterns

### Common Registry Interface
All libraries follow the same registry pattern:

```python
# Standardized discovery pattern
from brainsmith.libraries.{library} import discover_all_{components}

components = discover_all_{components}()

# Standardized access pattern  
component = get_{component}_by_name(name)

# Standardized search pattern
matches = find_{components}_by_type(component_type)
```

### Contribution Workflow
Adding components to any library follows the same pattern:

1. **Create Component Directory**: Follow naming conventions
2. **Add Manifest File**: YAML configuration with metadata
3. **Implement Functionality**: Following library-specific interfaces
4. **Automatic Discovery**: No registration code required

### Health Monitoring
All registries provide health checking:

```python
from brainsmith.libraries.kernels import get_kernel_registry

registry = get_kernel_registry()
health = registry.health_check()
print(f"Status: {health['status']}")
print(f"Components: {health['total_components']}")
print(f"Success rate: {health['success_rate']:.1%}")
```

## Integration Examples

### Cross-Library Workflows
Libraries integrate seamlessly:

```python
# 1. Discover available kernels
kernels = brainsmith.libraries.kernels.discover_all_kernels()

# 2. Load blueprint for design space
blueprint = brainsmith.libraries.blueprints.get_blueprint_by_name("cnn_optimized")

# 3. Run parameter sweep with automation
results = brainsmith.libraries.automation.parameter_sweep(
    "model.onnx", "blueprint.yaml",
    {'pe_count': [8, 16, 32]}
)

# 4. Analyze results
analysis = brainsmith.libraries.analysis.roofline_analysis(
    model_config, hardware_config, [8]
)

# 5. Apply transforms in pipeline
model = brainsmith.libraries.transforms.cleanup_step(model, config)
```

### Custom Library Development
Create new library following the same patterns:

```python
# my_custom_library/__init__.py
from .registry import (
    MyCustomRegistry,
    discover_all_my_components,
    get_my_component_by_name
)

# Automatic integration with BrainSmith ecosystem
```

## Performance Characteristics

### Library Discovery Performance
- **1000+ components** discovered in <50ms
- **Parallel scanning** of directories
- **Cached results** for repeated operations
- **Smart invalidation** on filesystem changes

### Memory Efficiency
- **Lazy loading** of component implementations
- **~1KB per component** in memory for metadata
- **On-demand instantiation** of complex objects
- **Efficient caching** strategies

### Scalability
- **Linear scaling** with component count
- **Parallel processing** where applicable
- **Configurable resource limits**
- **Graceful degradation** under load

The libraries ecosystem demonstrates that following North Star principles creates systems that are both simpler and more powerful than traditional enterprise approaches, enabling rapid innovation through community contributions while maintaining professional-grade functionality.