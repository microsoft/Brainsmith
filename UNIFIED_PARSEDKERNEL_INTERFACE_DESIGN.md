# Unified ParsedKernel and InterfaceMetadata Design

## Problem Statement

Currently we have unnecessary duplication:
- RTL Parser creates `ParsedKernelData` with `Interface` objects
- Templates convert these to `InterfaceMetadata` objects
- This transformation adds complexity without value

## Proposed Solution: Direct InterfaceMetadata Creation

### Option 1: ParsedKernelData Contains InterfaceMetadata

```python
@dataclass
class ParsedKernelData:
    """Kernel data parsed from SystemVerilog RTL source."""
    name: str                              # Module name
    source_file: Path                      # Source RTL file path
    parameters: List[Parameter]            # SystemVerilog parameters
    interfaces: List[InterfaceMetadata]    # Direct InterfaceMetadata objects!
    pragmas: List[Pragma]                  # Parsed pragmas
    parsing_warnings: List[str]            # Parser warnings
```

The RTL Parser would directly create `InterfaceMetadata` objects:

```python
# In RTL Parser
interface_metadata = InterfaceMetadata(
    name=port_name,
    interface_type=self._classify_interface_type(port_name, direction),
    allowed_datatypes=self._extract_datatype_constraints(port_info, pragmas),
    chunking_strategy=self._pragma_to_chunking_strategy(pragmas)
)
```

### Option 2: Rename ParsedKernelData to KernelMetadata

Since ParsedKernelData would now contain high-level metadata objects, we could rename it:

```python
@dataclass
class KernelMetadata:
    """Complete metadata for AutoHWCustomOp generation."""
    name: str
    source_file: Path
    interfaces: List[InterfaceMetadata]
    parameters: List[Parameter]
    pragmas: List[Pragma]
    
    def to_template_context(self) -> Dict[str, Any]:
        """Convert to template context directly."""
        return {
            "class_name": self._generate_class_name(),
            "kernel_name": self.name,
            "interface_metadata": self.interfaces,
            "rtl_parameters": self.parameters,
            # ... other context
        }
```

## Benefits of Unification

### 1. **Simpler Pipeline**
```
Before: RTL → ParsedKernelData → Convert → InterfaceMetadata → AutoHWCustomOp
After:  RTL → KernelMetadata(with InterfaceMetadata) → AutoHWCustomOp
```

### 2. **Single Source of Truth**
- No data transformation between parser and template
- Interface information defined once in RTL parser
- No risk of information loss during conversion

### 3. **Direct Template Usage**
Templates can directly use the metadata without conversion:
```python
class {{ kernel_metadata.class_name }}(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        # Direct usage - no conversion needed!
        super().__init__(
            onnx_node, 
            interface_metadata={{ kernel_metadata.interfaces }},
            **kwargs
        )
```

### 4. **Better Type Safety**
- RTL Parser ensures InterfaceMetadata is properly constructed
- No intermediate representations that could have inconsistencies
- Type checking works end-to-end

## Implementation Changes Required

### 1. **Update RTL Parser**
```python
# In rtl_parser/parser.py
def parse(self, sv_file: Union[str, Path]) -> KernelMetadata:
    # ... parsing logic ...
    
    # Build InterfaceMetadata directly
    interfaces = []
    for port_name, port_info in axi_interfaces.items():
        metadata = InterfaceMetadata(
            name=port_name,
            interface_type=port_info['interface_type'],
            allowed_datatypes=self._build_datatype_constraints(port_info),
            chunking_strategy=self._extract_chunking_strategy(port_name, pragmas)
        )
        interfaces.append(metadata)
    
    return KernelMetadata(
        name=module_name,
        source_file=Path(sv_file),
        interfaces=interfaces,
        parameters=parameters,
        pragmas=pragmas
    )
```

### 2. **Simplify Template Context Generation**
```python
# No more conversion needed!
def generate_context(kernel_metadata: KernelMetadata) -> Dict[str, Any]:
    return {
        "class_name": generate_class_name(kernel_metadata.name),
        "kernel_name": kernel_metadata.name,
        "interface_metadata": kernel_metadata.interfaces,  # Direct use!
        "rtl_parameters": kernel_metadata.parameters,
    }
```

### 3. **Update Templates**
Templates become simpler since they receive ready-to-use InterfaceMetadata:
```jinja2
def __init__(self, onnx_node, **kwargs):
    interface_metadata = [
        {% for iface in interface_metadata %}
        InterfaceMetadata(
            name="{{ iface.name }}",
            interface_type=InterfaceType.{{ iface.interface_type.name }},
            allowed_datatypes={{ iface.allowed_datatypes }},
            chunking_strategy={{ iface.chunking_strategy }}
        ),
        {% endfor %}
    ]
    super().__init__(onnx_node, interface_metadata=interface_metadata, **kwargs)
```

## Conclusion

Merging InterfaceMetadata creation into the RTL Parser is not only possible but highly beneficial:

1. **Eliminates redundant transformation**
2. **Simplifies the entire pipeline**
3. **Provides better type safety**
4. **Makes templates cleaner**
5. **Reduces potential for errors**

The RTL Parser has all the information needed to create InterfaceMetadata directly, so we should do exactly that!