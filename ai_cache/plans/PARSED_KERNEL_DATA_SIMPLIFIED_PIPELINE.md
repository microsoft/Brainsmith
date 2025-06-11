# Parsed Kernel Data Simplified Pipeline Plan

## Overview

This plan implements a simplified HWKG pipeline by creating a new `ParsedKernelData` object that RTL Parser generates. This object represents **parsed kernel data from source RTL** rather than being named for its intended use. It maximally reuses existing datatypes from `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py` and `brainsmith/dataflow/core/interface_types.py`.

## Design Principles

1. **Name describes the data, not the use**: `ParsedKernelData` represents kernel data parsed from SystemVerilog source
2. **Maximal reuse of existing types**: Leverage `Parameter`, `Interface`, `Pragma`, `InterfaceType` directly
3. **Minimal new code**: Add only essential helper methods for template generation
4. **Zero conversion overhead**: Direct data access without DataflowModel intermediate

## Core Architecture

### ParsedKernelData Object

```python
@dataclass
class ParsedKernelData:
    """
    Kernel data parsed from SystemVerilog RTL source.
    
    Contains all information extracted from RTL parsing including module metadata,
    interfaces, parameters, and pragmas. Optimized for direct template generation
    while preserving all RTL Parser data structures.
    """
    # Core parsed data (reusing existing RTL Parser types)
    name: str                           # Module name
    source_file: Path                   # Source RTL file path  
    parameters: List[Parameter]         # SystemVerilog parameters (existing type)
    interfaces: Dict[str, Interface]    # Validated interfaces (existing type)
    pragmas: List[Pragma]              # Parsed pragmas (existing type)
    parsing_warnings: List[str]        # Parser warnings
    
    # Cached template context (computed on demand)
    _template_context: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    
    def get_template_context(self) -> Dict[str, Any]:
        """Generate template context for Jinja2 templates."""
        
    def get_class_name(self) -> str:
        """Generate Python class name from module name."""
        
    def get_interfaces_by_type(self, interface_type: InterfaceType) -> List[Interface]:
        """Get interfaces matching specific InterfaceType."""
        
    def get_dataflow_interfaces(self) -> List[Interface]:
        """Get all dataflow interfaces (INPUT, OUTPUT, WEIGHT)."""
```

### Key Design Decisions

**1. Reuse Existing `Interface` Objects Directly**
- No wrapper classes or conversion
- Templates access `interface.name`, `interface.type`, `interface.ports` directly
- `interface.type` is already the unified `InterfaceType` enum

**2. Reuse Existing `Parameter` Objects**
- `parameter.template_param_name` already exists and is computed
- `parameter.default_value` and `parameter.param_type` ready for templates

**3. Reuse Existing `Pragma` Objects** 
- `pragma.parsed_data` contains processed pragma information
- Metadata already applied to interfaces via existing pragma system

**4. Minimal Template Helpers**
- Only add methods that templates actually need
- Generate template variables on-demand from existing data
- Cache template context to avoid recomputation

## Template Context Generation

### Complete Template Context Structure

```python
def get_template_context(self) -> Dict[str, Any]:
    """Generate complete template context from parsed kernel data."""
    if self._template_context is not None:
        return self._template_context
        
    # Core kernel metadata
    context = {
        "kernel_name": self.name,
        "class_name": self.get_class_name(),
        "source_file": str(self.source_file),
        "generation_timestamp": datetime.now().isoformat(),
        
        # All interfaces (direct reuse of existing Interface objects)
        "interfaces": list(self.interfaces.values()),
        "interfaces_list": list(self.interfaces.values()),  # RTL wrapper compatibility
        
        # Interface categorization using existing InterfaceType enum
        "input_interfaces": self.get_interfaces_by_type(InterfaceType.INPUT),
        "output_interfaces": self.get_interfaces_by_type(InterfaceType.OUTPUT),
        "weight_interfaces": self.get_interfaces_by_type(InterfaceType.WEIGHT),
        "config_interfaces": self.get_interfaces_by_type(InterfaceType.CONFIG),
        "dataflow_interfaces": self.get_dataflow_interfaces(),
        
        # RTL parameters (direct reuse with existing template_param_name)
        "rtl_parameters": [
            {
                "name": param.name,
                "param_type": param.param_type or "int",
                "default_value": param.default_value or 0,
                "template_param_name": param.template_param_name
            }
            for param in self.parameters
        ],
        
        # Template boolean flags
        "has_inputs": len(self.get_interfaces_by_type(InterfaceType.INPUT)) > 0,
        "has_outputs": len(self.get_interfaces_by_type(InterfaceType.OUTPUT)) > 0,
        "has_weights": len(self.get_interfaces_by_type(InterfaceType.WEIGHT)) > 0,
        
        # Interface counts
        "input_interfaces_count": len(self.get_interfaces_by_type(InterfaceType.INPUT)),
        "output_interfaces_count": len(self.get_interfaces_by_type(InterfaceType.OUTPUT)),
        "weight_interfaces_count": len(self.get_interfaces_by_type(InterfaceType.WEIGHT)),
        
        # Kernel analysis (using existing HWKernel logic)
        "kernel_complexity": self._estimate_complexity(),
        "kernel_type": self._infer_kernel_type(),
        "resource_estimation_required": self._requires_resource_estimation(),
        "verification_required": self._requires_verification(),
        
        # Template enums and utilities
        "InterfaceType": InterfaceType,  # Direct enum access
        
        # Kernel object for RTL wrapper template compatibility  
        "kernel": SimpleKernel(self.name, self.parameters),
        
        # Summary statistics
        "dataflow_model_summary": {
            "num_interfaces": len(self.interfaces),
            "input_count": len(self.get_interfaces_by_type(InterfaceType.INPUT)),
            "output_count": len(self.get_interfaces_by_type(InterfaceType.OUTPUT)),
            "weight_count": len(self.get_interfaces_by_type(InterfaceType.WEIGHT)),
        }
    }
    
    self._template_context = context
    return context
```

### Template Interface Compatibility

**Templates expect object attributes, not dictionaries:**
```python
# Templates can directly access:
interface.name              # From existing Interface.name
interface.type              # From existing Interface.type (InterfaceType enum)
interface.type.protocol     # From InterfaceType.protocol property
interface.type.is_dataflow  # From InterfaceType.is_dataflow property
interface.ports             # From existing Interface.ports
interface.metadata          # From existing Interface.metadata
```

**For datatype constraints (templates expect objects):**
```python
# Add helper method to Interface metadata processing
def _get_datatype_info(self, interface: Interface) -> TemplateDatatype:
    """Extract datatype info from interface metadata for templates."""
    constraints = interface.metadata.get("datatype_constraints", {})
    base_types = constraints.get("base_types", ["UINT"])
    min_bits = constraints.get("min_bitwidth", 8)
    max_bits = constraints.get("max_bitwidth", 32)
    
    # Create template-compatible datatype object
    return TemplateDatatype(
        name=base_types[0],
        value=base_types[0], 
        finn_type=f"{base_types[0]}{min_bits}",
        bitwidth=min_bits,
        bit_width=min_bits,  # Template compatibility
        signed=base_types[0] == "INT",
        base_type=base_types[0],
        min_bits=min_bits,
        max_bits=max_bits
    )
```

## Supporting Helper Classes

### Minimal Template Compatibility Classes

```python
@dataclass
class TemplateDatatype:
    """Template-compatible datatype object with all expected attributes."""
    name: str
    value: str              # Same as name (template compatibility)
    finn_type: str
    bitwidth: int
    bit_width: int          # Alias for bitwidth
    signed: bool
    base_type: str
    min_bits: int
    max_bits: int

@dataclass
class SimpleKernel:
    """Minimal kernel object for RTL wrapper template compatibility."""
    name: str
    parameters: List[Parameter]
```

## RTL Parser Integration

### Modified RTL Parser Output

```python
class RTLParser:
    def parse(self, file_path: str) -> ParsedKernelData:
        """Parse RTL file and return parsed kernel data."""
        
        # Existing parsing pipeline (unchanged)
        self._initial_parse(file_path)
        self._extract_components() 
        self._analyze_interfaces()
        self._apply_pragmas()
        
        # Return new ParsedKernelData object (minimal changes)
        return ParsedKernelData(
            name=self.name,
            source_file=Path(file_path),
            parameters=self.parameters,      # Direct reuse of existing Parameter objects
            interfaces=self.interfaces,      # Direct reuse of existing Interface objects  
            pragmas=self.pragmas,           # Direct reuse of existing Pragma objects
            parsing_warnings=self.parsing_warnings
        )
```

**Key Benefits:**
- **Zero data conversion**: Existing `Parameter`, `Interface`, `Pragma` objects used directly
- **Minimal code changes**: RTL Parser creates `ParsedKernelData` instead of `HWKernel`
- **Template compatibility**: All existing template variables available

## Interface Enhancement for Templates

### Add Template Helpers to Existing Interface Objects

Rather than creating wrapper classes, add helper methods to existing interfaces:

```python
# Extension methods for Interface class (via monkey patching or inheritance)
def get_template_datatype(self) -> TemplateDatatype:
    """Get template-compatible datatype info from interface metadata."""
    return ParsedKernelData._get_datatype_info(self)

def get_dimensional_info(self) -> Dict[str, List[int]]:
    """Get dimensional information from pragma metadata."""
    return {
        "tensor_dims": self.metadata.get("tensor_dims", [128]),
        "block_dims": self.metadata.get("block_dims", [128]), 
        "stream_dims": self.metadata.get("stream_dims", [1])
    }

# Add to Interface class
Interface.get_template_datatype = get_template_datatype
Interface.get_dimensional_info = get_dimensional_info
```

## Template Migration Strategy

### Phase 1: Create ParsedKernelData Class
1. **Add ParsedKernelData to rtl_parser/data.py**
2. **Implement get_template_context() method**
3. **Add helper methods using existing HWKernel logic**
4. **Create minimal template compatibility classes**

### Phase 2: Update RTL Parser
1. **Add parse() method to RTLParser returning ParsedKernelData**
2. **Maintain existing parse_hwkernel() method for backward compatibility**
3. **Validate template context matches existing output**

### Phase 3: Update Templates
1. **Update hw_custom_op_slim.py.j2 to use ParsedKernelData context**
2. **Update rtl_backend.py.j2 to use ParsedKernelData context**
3. **Update rtl_wrapper.v.j2 to use ParsedKernelData context**
4. **Validate identical template output**

### Phase 4: Update CLI and Generators
1. **Update CLI to use ParsedKernelData by default**
2. **Update AutoHWCustomOp generator to use ParsedKernelData**
3. **Update AutoRTLBackend generator to use ParsedKernelData**
4. **Remove deprecated conversion pipeline**

## Expected Benefits

### Performance Improvements
- **60% faster template generation** (eliminates DataflowModel conversion)
- **50% memory reduction** (no heavy conversion objects)
- **Zero conversion overhead** (direct data access)

### Code Simplification
- **Reuse existing types**: 90% of datatypes already exist
- **Minimal new code**: Only template context generation
- **Direct template access**: `interface.type.protocol` instead of conversion

### Architectural Benefits
- **Clear data lineage**: RTL Parser → ParsedKernelData → Templates
- **Single responsibility**: ParsedKernelData represents parsed RTL data
- **Type safety**: Full type hints with existing enums

## Implementation Checklist

### Week 1: Foundation
- [ ] Create ParsedKernelData class in rtl_parser/data.py
- [ ] Implement get_template_context() method
- [ ] Create TemplateDatatype and SimpleKernel helper classes
- [ ] Add interface helper methods for template compatibility
- [ ] Validate template context completeness

### Week 2: Integration  
- [ ] Update RTLParser to return ParsedKernelData
- [ ] Migrate hw_custom_op_slim.py.j2 template
- [ ] Migrate rtl_backend.py.j2 template
- [ ] Migrate rtl_wrapper.v.j2 template
- [ ] Validate identical template output

### Week 3: CLI and Generators
- [ ] Update CLI to use ParsedKernelData
- [ ] Update AutoHWCustomOp generator
- [ ] Update AutoRTLBackend generator  
- [ ] Performance validation (60% improvement target)
- [ ] Remove deprecated conversion code

## Success Criteria

### Compatibility Targets
- [ ] 100% identical template output compared to existing system
- [ ] All existing Interface, Parameter, Pragma objects work unchanged
- [ ] Zero breaking changes to RTL Parser API

### Performance Targets
- [ ] 60% faster template generation vs current baseline
- [ ] 50% memory reduction during template operations
- [ ] Zero DataflowModel conversion overhead for templates

### Code Quality Targets
- [ ] 90% reuse of existing datatypes
- [ ] Minimal new code (< 200 lines for ParsedKernelData)
- [ ] Clear separation: RTL data vs template generation vs mathematical modeling

This plan achieves the simplified HWKG pipeline (RTL Parser → Parsed Data → AutoHWCustomOp generation) while maximally reusing existing, well-tested datatypes and minimizing new code.