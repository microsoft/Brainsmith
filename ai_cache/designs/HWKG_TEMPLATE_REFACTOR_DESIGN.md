# HWKG Generator System Refactoring Design

## Executive Summary

This document proposes a lightweight refactoring of the HWKG template system to improve modularity and extensibility without introducing unnecessary complexity. The core principle is **"NO BLOAT"** - every change must provide clear value without overengineering.

## Current System Pain Points

1. **Hardcoded Generator Discovery** - Generators are hardcoded in `_generate_selected_templates()`
2. **No Extension Points** - Adding new generators requires modifying core code
3. **Monolithic Context** - All generators receive the entire context dictionary
4. **Tight Coupling** - Generator management is embedded in `UnifiedGenerator`

## Proposed Architecture

### Core Design Principles

1. **Simple Generator Registry** - Dynamic discovery without complex plugin systems
2. **Full Context Always** - Pass complete context, let generators extract what they need
3. **Single Responsibility** - Separate generator management from orchestration logic
4. **Zero Breaking Changes** - Existing generators continue working

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    KernelIntegrator                         │
│  - Orchestrates generation workflow                         │
│  - Handles file I/O                                        │
│  - Delegates to GeneratorManager                           │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    GeneratorManager                         │
│  - Generator discovery and registration                     │
│  - Template rendering with full context                     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ Generators       │
                  │ - hw_custom_op   │
                  │ - rtl_wrapper    │
                  │ - test_suite     │
                  │ - [extensible]   │
                  └──────────────────┘
```

## Implementation Details

### 1. Generator Definition (Python-based with Optional Processing)

Each generator is defined as a simple Python class with minimal boilerplate:

```python
# generators/hw_custom_op_generator.py
from .base import GeneratorBase

class HWCustomOpGenerator(GeneratorBase):
    """Generates AutoHWCustomOp subclass."""
    
    name = "hw_custom_op"
    template_file = "hw_custom_op_phase2.py.j2"
    output_pattern = "{kernel_name}_hw_custom_op.py"
    
    def process_context(self, context: TemplateContext) -> Dict:
        """Transform context for this specific generator."""
        # Default: pass through full context as dict
        return context.__dict__


# generators/sv_testbench_generator.py  
class SVTestbenchGenerator(GeneratorBase):
    """SystemVerilog UVM testbench with custom processing."""
    
    name = "sv_testbench"
    template_file = "sv_testbench.sv.j2"
    output_pattern = "{kernel_name}_tb.sv"
    
    def process_context(self, context: TemplateContext) -> Dict:
        """Custom processing for testbench generation."""
        # Start with full context
        result = context.__dict__.copy()
        
        # Add testbench-specific transformations
        result['test_vectors'] = self._generate_test_vectors(context.interfaces)
        result['clock_period'] = self._calculate_clock_period(context.parameters)
        result['interface_monitors'] = self._create_monitors(context.interfaces)
        
        return result
    
    def _generate_test_vectors(self, interfaces):
        """Generate test vectors based on interface types."""
        # Custom logic for test vector generation
        ...
```

### 2. Minimal Base Class

```python
# generators/base.py
class GeneratorBase:
    """Minimal base for generator definitions."""
    
    name: str = None
    template_file: str = None
    output_pattern: str = None
    
    def process_context(self, context: TemplateContext) -> Dict:
        """Override to customize context processing."""
        return context.__dict__
    
    def get_output_filename(self, kernel_name: str) -> str:
        """Generate output filename."""
        return self.output_pattern.format(kernel_name=kernel_name)
```

### 3. GeneratorManager with Auto-Discovery

```python
class GeneratorManager:
    """Manages generator discovery, loading, and template rendering."""
    
    def __init__(self, generator_dir: Path, template_dir: Path):
        self.generator_dir = generator_dir
        self.template_dir = template_dir
        self.jinja_env = self._init_jinja_env()
        self.generators = self._discover_generators()
    
    def _discover_generators(self) -> Dict[str, GeneratorBase]:
        """Auto-discover generator definitions."""
        generators = {}
        
        # Import all *_generator.py files
        generator_pattern = self.generator_dir / "*_generator.py"
        for generator_file in glob.glob(str(generator_pattern)):
            module = self._import_generator_module(generator_file)
            
            # Find GeneratorBase subclasses
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, GeneratorBase) and
                    obj != GeneratorBase and
                    obj.name):
                    generators[obj.name] = obj()
        
        return generators
    
    def render_generator(
        self, 
        generator_name: str, 
        full_context: TemplateContext
    ) -> str:
        """Render template via generator with processed context."""
        generator = self.generators[generator_name]
        
        # Let generator process its own context
        processed_context = generator.process_context(full_context)
        
        # Load and render Jinja2 template
        template = self.jinja_env.get_template(generator.template_file)
        return template.render(**processed_context)
```

### 4. Simplified KernelIntegrator

```python
class KernelIntegrator:
    """Simplified integrator delegating to GeneratorManager."""
    
    def __init__(self, generator_dir: Path, template_dir: Path, output_dir: Path):
        self.generator_manager = GeneratorManager(generator_dir, template_dir)
        self.context_generator = TemplateContextGenerator()
        self.output_dir = output_dir
    
    def generate_and_write(
        self,
        kernel_metadata: KernelMetadata,
        include_generators: Optional[List[str]] = None
    ) -> GenerationResult:
        """Generate artifacts using generator manager."""
        # Generate context once
        context = self.context_generator.generate_template_context(kernel_metadata)
        
        # Let GeneratorManager handle generator selection and rendering
        generators = include_generators or self.generator_manager.list_generators()
        
        for generator_name in generators:
            content = self.generator_manager.render_generator(generator_name, context)
            # ... write files
```

### 5. Adding New Generators (Minimal Code)

To add a new generator:

1. Create template file: `templates/my_template.v.j2`
2. Create generator file: `generators/my_generator.py`
3. Done! Generator is automatically discovered

Example: Adding a Makefile generator with custom logic:

```python
# generators/makefile_generator.py
class MakefileGenerator(GeneratorBase):
    """Generates Makefile with build rules."""
    
    name = "makefile"
    template_file = "makefile.j2"
    output_pattern = "Makefile"
    
    def process_context(self, context: TemplateContext) -> Dict:
        """Generate Makefile-specific context."""
        base = self.context_to_dict(context)
        
        # Add custom processing
        base['source_files'] = self._collect_source_files(context)
        base['vivado_version'] = self._detect_vivado_version()
        base['synthesis_flags'] = self._determine_synth_flags(context.parameters)
        
        # Generate dependency graph
        base['dependencies'] = {
            'rtl': [context.source_file],
            'wrapper': [f"{context.module_name}_wrapper.v"],
            'xdc': self._find_constraint_files(context)
        }
        
        return base
```

## Migration Strategy

### Phase 1: Add GeneratorManager (Non-Breaking)
1. Implement `GeneratorManager` alongside existing code
2. Create simple wrapper classes for existing generators:
   ```python
   class HWCustomOpGenerator(GeneratorBase):
       name = "hw_custom_op"
       template_file = "hw_custom_op_phase2.py.j2"
       output_pattern = "{kernel_name}_hw_custom_op.py"
       # No process_context override needed - uses default
   ```
3. Update `UnifiedGenerator` to optionally use `GeneratorManager`

### Phase 2: Gradual Migration
1. Migrate generator rendering to use `GeneratorManager`
2. Remove hardcoded generator logic from `UnifiedGenerator`
3. Rename `UnifiedGenerator` to `KernelIntegrator`
4. Keep backward compatibility via adapter pattern

### Phase 3: Cleanup
1. Remove legacy generator handling
2. Simplify `KernelIntegrator` to orchestration only
3. Document extension points

## Benefits

1. **Easy Extension** - Add generators by creating one Python file
2. **Better Testing** - Test generators in isolation with mocked contexts
3. **Full Context Access** - Generators get all data, extract what they need
4. **Clear Contracts** - Python classes document generator requirements
5. **Maintainable** - Separation of concerns makes debugging easier

## Non-Goals (Avoiding Bloat)

We explicitly DO NOT:
- Build a complex plugin architecture
- Add dependency injection frameworks
- Create abstract template hierarchies
- Implement complex context transformation pipelines
- Add unnecessary configuration layers
- Support remote template loading
- Build a generator versioning system (beyond simple fallbacks)

## Example Use Cases

### 1. Xilinx IP Package Generator (Complex Processing)
```python
class XilinxIPGenerator(GeneratorBase):
    """Generate Xilinx IP packaging files."""
    
    name = "xilinx_ip"
    template_file = "component.xml.j2"
    output_pattern = "{kernel_name}_ip/component.xml"
    
    def process_context(self, context: TemplateContext) -> Dict:
        base = context.__dict__.copy()
        
        # Transform interfaces to Xilinx IP-XACT format
        base['bus_interfaces'] = []
        for intf in context.interfaces:
            if intf.protocol == "AXI-Stream":
                base['bus_interfaces'].append({
                    'name': intf.name,
                    'type': 'xilinx.com:interface:axis:1.0',
                    'mode': 'master' if intf.direction == 'output' else 'slave',
                    'port_maps': self._generate_axis_port_maps(intf)
                })
        
        # Calculate IP metadata
        base['vendor'] = 'user.org'
        base['library'] = 'hls'
        base['version'] = '1.0'
        base['supported_families'] = ['zynquplus', 'virtexuplus']
        
        return base
```

### 2. CocoTB Testbench with Test Vector Generation
```python
class CocotbTestbenchGenerator(GeneratorBase):
    """Generate CocoTB testbench with randomized testing."""
    
    name = "cocotb_test"
    template_file = "cocotb_testbench.py.j2"
    output_pattern = "test_{kernel_name}_cocotb.py"
    
    def process_context(self, context: TemplateContext) -> Dict:
        base = context.__dict__.copy()
        
        # Generate test configurations based on interfaces
        base['test_cases'] = []
        
        # Analyze dataflow patterns
        for intf in context.input_interfaces:
            if 'SIMD' in context.parameters:
                # Generate SIMD-aware test patterns
                base['test_cases'].append({
                    'name': f'test_{intf.name}_simd_patterns',
                    'data_generator': f'RandomSIMDGenerator({context.parameters["SIMD"]})',
                    'expected_behavior': self._infer_behavior(context.kernel_analysis)
                })
        
        # Add performance test if PE parameter exists
        if 'PE' in context.parameters:
            base['test_cases'].append({
                'name': 'test_throughput_pe_scaling',
                'pe_values': [1, 2, 4, context.parameters['PE']],
                'measure_cycles': True
            })
        
        return base
```

### 3. C++ Driver Generation with HLS Integration
```python
class CppDriverGenerator(GeneratorBase):
    """Generate C++ driver code for HLS kernels."""
    
    name = "cpp_driver"
    template_file = "kernel_driver.cpp.j2"
    output_pattern = "{kernel_name}_driver.cpp"
    
    def process_context(self, context: TemplateContext) -> Dict:
        base = context.__dict__.copy()
        
        # Generate type-safe buffer allocations
        base['buffer_allocations'] = []
        for intf in context.interfaces:
            if intf.interface_type in ['INPUT', 'OUTPUT', 'WEIGHT']:
                buffer_info = {
                    'name': f"{intf.name}_buffer",
                    'type': self._get_cpp_type(intf.datatype),
                    'size': self._calculate_buffer_size(intf, context.parameters),
                    'alignment': 4096  # Page alignment for DMA
                }
                base['buffer_allocations'].append(buffer_info)
        
        # Generate kernel invocation code
        base['kernel_args'] = self._order_kernel_arguments(context.interfaces)
        base['includes'] = self._determine_required_includes(context)
        
        return base
```

## Key Advantages of Python-Based Approach

1. **Custom Context Processing** - Generators can transform data as needed
2. **Type Safety** - Python classes provide better IDE support than YAML
3. **Testability** - Generator definitions can be unit tested independently  
4. **Minimal Boilerplate** - Simple class with 3 attributes and 1 optional method
5. **Gradual Complexity** - Simple generators just set attributes, complex ones override `process_context()`

## Summary

This refactoring provides a clean, extensible generator system that balances simplicity with flexibility. By using lightweight Python classes, we enable custom context processing while avoiding overengineering. Generators that need simple pass-through can use the default behavior, while complex generators can implement sophisticated transformations.

The key insight is that we need **discoverable generators** with **optional processing hooks**, not a complex framework. A generator is just a class with a name, template file, and optional context processor.