# HWKG Modular Refactoring Plan

## Current Architecture Problems

### 1. Generator Bypass Issue
The current HWKG implementation completely bypasses the new `HWCustomOpGenerator`:
- HWKG has inline methods like `_generate_auto_hwcustomop_with_dataflow()`
- New `HWCustomOpGenerator` is never called by HWKG
- Duplication of template handling logic
- No integration with Phase 3 enhanced features

### 2. Non-Modular Design
- All generation logic embedded directly in HWKG methods
- Hardcoded template names (`hw_custom_op.py.j2` vs `hw_custom_op_slim.py.j2`)
- No way to switch between different generator types
- Tight coupling between orchestration and generation logic

### 3. Missing Extensibility
- No plugin system for new generators
- Cannot easily add new template types
- No configuration system for generator preferences
- Future generators require HWKG modifications

## Proposed Modular Architecture

### 1. Generator Registry System

```python
class GeneratorRegistry:
    """Registry for all available generators with discovery and dispatch capabilities."""
    
    def __init__(self):
        self.generators = {
            "rtl_template": {},
            "hw_custom_op": {},
            "rtl_backend": {},
            "test_suite": {},
            "documentation": {}
        }
    
    def register_generator(self, category: str, name: str, generator_class: type):
        """Register a generator for a specific category."""
        
    def get_generator(self, category: str, name: str) -> BaseGenerator:
        """Get a generator instance by category and name."""
        
    def list_available_generators(self, category: str) -> List[str]:
        """List all available generators for a category."""
```

### 2. Base Generator Interface

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

class BaseGenerator(ABC):
    """Abstract base class for all generators."""
    
    @abstractmethod
    def generate(self, hw_kernel: HWKernel, output_path: Path, **kwargs) -> Path:
        """Generate output files from HWKernel data."""
        pass
    
    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """Return list of supported features (e.g., 'enhanced_tdim', 'slim_templates')."""
        pass
    
    @abstractmethod
    def validate_requirements(self, hw_kernel: HWKernel) -> bool:
        """Validate that this generator can handle the given HWKernel."""
        pass
```

### 3. Enhanced Generator Implementations

#### RTL Template Generator (Existing)
```python
class RTLTemplateGenerator(BaseGenerator):
    """Generator for RTL wrapper templates."""
    
    def generate(self, hw_kernel: HWKernel, output_path: Path, **kwargs) -> Path:
        # Use existing generate_rtl_template function
        return generate_rtl_template(hw_kernel, output_path.parent)
    
    def get_supported_features(self) -> List[str]:
        return ["interface_sorting", "parameter_injection"]
```

#### Slim HWCustomOp Generator (Phase 3)
```python
class HWCustomOpGenerator(BaseGenerator):
    """Phase 3 slim HWCustomOp generator with enhanced TDIM pragma support."""
    
    def generate(self, hw_kernel: HWKernel, output_path: Path, **kwargs) -> Path:
        class_name = kwargs.get('class_name')
        source_file = kwargs.get('source_file', 'unknown.sv')
        
        generator = HWCustomOpGenerator()
        generator.generate_hwcustomop(hw_kernel, output_path, class_name, source_file)
        return output_path
    
    def get_supported_features(self) -> List[str]:
        return ["enhanced_tdim", "slim_templates", "automatic_chunking", "parameter_validation"]
```

#### Standard HWCustomOp Generator (Future)
```python
class StandardHWCustomOpGenerator(BaseGenerator):
    """Traditional full-featured HWCustomOp generator."""
    
    def generate(self, hw_kernel: HWKernel, output_path: Path, **kwargs) -> Path:
        # Use traditional template with full feature set
        pass
    
    def get_supported_features(self) -> List[str]:
        return ["dataflow_integration", "full_templates", "legacy_compatibility"]
```

### 4. Configuration System

```python
@dataclass
class HWKGConfig:
    """Configuration for HWKG generation preferences."""
    
    # Generator preferences
    rtl_template_generator: str = "standard"
    hw_custom_op_generator: str = "slim"  # "slim", "standard", "legacy"
    rtl_backend_generator: str = "standard"
    test_suite_generator: str = "standard"
    documentation_generator: str = "standard"
    
    # Feature flags
    enable_enhanced_tdim: bool = True
    enable_slim_templates: bool = True
    enable_legacy_support: bool = False
    
    # Template preferences
    template_style: str = "compact"  # "compact", "verbose", "legacy"
    include_documentation: bool = True
    include_test_suite: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HWKGConfig':
        """Create config from dictionary (e.g., from JSON/YAML file)."""
        pass
```

### 5. Refactored HWKG Class

```python
class HardwareKernelGenerator:
    """
    Modular orchestrator for hardware kernel generation.
    
    Dispatches generation tasks to specialized generators based on configuration.
    """
    
    def __init__(
        self,
        rtl_file_path: str,
        compiler_data_path: str,
        output_dir: str,
        config: Optional[HWKGConfig] = None,
        custom_doc_path: Optional[str] = None,
    ):
        # ... existing initialization ...
        
        # Initialize modular components
        self.config = config or HWKGConfig()
        self.generator_registry = GeneratorRegistry()
        self._register_default_generators()
        
        # Validate generator availability
        self._validate_generator_configuration()
    
    def _register_default_generators(self):
        """Register all default generators."""
        # RTL Template generators
        self.generator_registry.register_generator(
            "rtl_template", "standard", RTLTemplateGenerator
        )
        
        # HWCustomOp generators
        self.generator_registry.register_generator(
            "hw_custom_op", "slim", HWCustomOpGenerator
        )
        self.generator_registry.register_generator(
            "hw_custom_op", "standard", StandardHWCustomOpGenerator
        )
        
        # Future generators...
    
    def _generate_hw_custom_op(self):
        """
        Generate HWCustomOp using configured generator.
        """
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate HWCustomOp: RTL data not parsed.")
        
        print("--- Generating HWCustomOp Instance ---")
        
        # Get configured generator
        generator = self.generator_registry.get_generator(
            "hw_custom_op", 
            self.config.hw_custom_op_generator
        )
        
        # Validate requirements
        if not generator.validate_requirements(self.hw_kernel_data):
            raise HardwareKernelGeneratorError(
                f"Generator '{self.config.hw_custom_op_generator}' cannot handle this kernel"
            )
        
        # Prepare generation context
        class_name = generate_class_name(self.hw_kernel_data.name)
        output_file = self.output_dir / f"{class_name.lower()}.py"
        
        generation_context = {
            'class_name': class_name,
            'source_file': str(self.rtl_file_path.name),
            'dataflow_model': self.dataflow_model,
            'dataflow_interfaces': self.dataflow_interfaces,
            'config': self.config
        }
        
        # Generate using selected generator
        output_path = generator.generate(
            self.hw_kernel_data, 
            output_file, 
            **generation_context
        )
        
        self.generated_files["hw_custom_op"] = output_path
        print(f"HWCustomOp generation complete. Output: {output_path}")
        
        return output_path
```

## Implementation Steps

### Phase 1: Core Infrastructure
1. **Create Base Generator Interface**
   - Define `BaseGenerator` abstract class
   - Specify required methods and contracts
   - Add validation and feature discovery

2. **Implement Generator Registry**
   - Create registration system
   - Add generator discovery and dispatch
   - Implement configuration-based selection

3. **Create Configuration System**
   - Define `HWKGConfig` dataclass
   - Add JSON/YAML configuration loading
   - Implement validation and defaults

### Phase 2: Generator Integration
1. **Wrap Existing RTL Template Generator**
   - Create `RTLTemplateGenerator` wrapper
   - Maintain existing functionality
   - Add new interface compliance

2. **Integrate HWCustomOpGenerator**
   - Create wrapper for existing `HWCustomOpGenerator`
   - Add HWKG integration layer
   - Ensure configuration compatibility

3. **Update HWKG Orchestrator**
   - Replace inline generation methods
   - Add generator dispatch logic
   - Implement configuration-based selection

### Phase 3: Advanced Features
1. **Add Standard HWCustomOp Generator**
   - Create traditional template generator
   - Support existing templates
   - Maintain backward compatibility

2. **Implement Additional Generators**
   - RTLBackend generator
   - Test suite generator
   - Documentation generator

3. **Add Plugin System**
   - External generator loading
   - Runtime generator registration
   - Custom template support

## Configuration Examples

### Slim Generation (Phase 3 Default)
```yaml
# hwkg_config.yaml
rtl_template_generator: "standard"
hw_custom_op_generator: "slim"
enable_enhanced_tdim: true
enable_slim_templates: true
template_style: "compact"
```

### Legacy Generation
```yaml
# hwkg_legacy_config.yaml
rtl_template_generator: "standard"
hw_custom_op_generator: "standard"
enable_enhanced_tdim: false
enable_slim_templates: false
enable_legacy_support: true
template_style: "verbose"
```

### Custom Generation
```yaml
# hwkg_custom_config.yaml
rtl_template_generator: "enhanced"
hw_custom_op_generator: "custom_slim"
rtl_backend_generator: "optimized"
template_style: "minimal"
```

## Benefits

### 1. Modularity
- Each generator is self-contained
- Clear separation of concerns
- Independent testing and development

### 2. Extensibility  
- Easy addition of new generators
- Plugin system for custom generators
- No core HWKG modifications needed

### 3. Configurability
- User-selectable generator types
- Feature flag control
- Template style preferences

### 4. Maintainability
- Reduced code duplication
- Clear generator contracts
- Simplified debugging

### 5. Future-Proofing
- Ready for new template types
- Supports experimental generators
- Backward compatibility maintained

## Migration Strategy

### Step 1: Implement Infrastructure (Week 1)
- Create base classes and registry
- Add configuration system
- Implement basic dispatch logic

### Step 2: Integrate Existing Generators (Week 2) 
- Wrap RTL template generator
- Integrate HWCustomOpGenerator
- Update HWKG to use registry

### Step 3: Add Standard Generators (Week 3)
- Create remaining generator wrappers
- Implement full pipeline dispatch
- Add comprehensive testing

### Step 4: Advanced Features (Week 4)
- Add plugin system
- Implement configuration loading
- Create documentation and examples

This refactoring will transform HWKG from a monolithic generator into a modular, extensible orchestrator that can leverage the Phase 3 enhanced generators while maintaining backward compatibility and enabling future innovations.