# Direct Template Generation Object Design Specification

## Overview

This specification defines a new `DirectTemplateResult` object that RTL Parser will generate to provide all data needed for template and AutoHWCustomOp generation without requiring DataflowModel conversion.

## Design Goals

1. **Template-First Architecture**: Object structure optimized for Jinja2 template consumption
2. **Zero Conversion Overhead**: Direct extraction from RTL parsing results
3. **Complete Data Provision**: All template variables and AutoHWCustomOp data included
4. **Performance Optimization**: Cached template context with lazy evaluation
5. **Backward Compatibility**: Maintains all existing functionality

## Current State Analysis

### Problems with Existing Objects

**HWKernel (Legacy):**
- ❌ 29 properties, only 6 used by templates (22% utilization)
- ❌ ~800 lines of template-specific code in wrong architectural location
- ❌ Heavy object creation overhead

**RTLParsingResult (Lightweight):**
- ✅ 7 properties, optimized for DataflowModel conversion
- ❌ Still requires conversion pipeline for template generation
- ❌ Missing template-specific metadata

**EnhancedRTLParsingResult (Current Best):**
- ✅ 40% performance improvement for templates
- ✅ Direct template context generation
- ❌ Still contains 500+ lines of business logic in data container
- ❌ Template interface objects created with dynamic classes

## New Object Design: DirectTemplateResult

### Core Architecture

```python
@dataclass
class DirectTemplateResult:
    """
    Template-optimized result from RTL parsing.
    
    Provides all data needed for template generation and AutoHWCustomOp creation
    without requiring DataflowModel conversion.
    """
    # Core RTL data (from RTL Parser)
    name: str
    source_file: Path
    parameters: List[RTLParameter]
    interfaces: List[TemplateInterface] 
    pragmas: List[Pragma]
    parsing_warnings: List[str]
    
    # Template context (cached)
    _template_context: Optional[Dict[str, Any]] = field(default=None, init=False)
    
    def get_template_context(self) -> Dict[str, Any]:
        """Get complete template context for Jinja2 templates."""
        
    def get_auto_hwcustomop_data(self) -> AutoHWCustomOpData:
        """Get data needed for AutoHWCustomOp instantiation."""
        
    def get_auto_rtlbackend_data(self) -> AutoRTLBackendData:
        """Get data needed for AutoRTLBackend instantiation."""
```

### Template-Optimized Interface Object

```python
@dataclass
class TemplateInterface:
    """
    Interface object optimized for template consumption.
    
    Provides all template variables as direct attributes,
    eliminating the need for dynamic class creation.
    """
    # Core interface identity
    name: str                           # "in0", "out0", "config"
    rtl_type: InterfaceType            # Unified interface type
    ports: Dict[str, RTLPort]          # Port dictionary from RTL parser
    
    # Template-required categorization
    dataflow_category: str             # "INPUT", "OUTPUT", "WEIGHT", "CONFIG"
    protocol: str                      # "axi_stream", "axi_lite", "global_control"
    direction: str                     # "input", "output", "bidirectional"
    
    # Datatype information
    dtype: TemplateDatatype           # Template-ready datatype object
    datatype_constraints: List[DataTypeConstraint]  # FINN constraint objects
    
    # AXI metadata
    axi_metadata: Dict[str, Any]      # Protocol-specific metadata
    data_width_expr: str              # Width expression for templates
    
    # Dimensional information (from pragmas or defaults)
    tensor_dims: List[int]            # Original tensor dimensions
    block_dims: List[int]             # Processing block dimensions  
    stream_dims: List[int]            # Hardware parallelism
    
    # Template utilities
    wrapper_name: str                 # Template-friendly identifier
    
    @property
    def interface_type(self) -> TemplateInterfaceType:
        """Template-compatible interface type object."""
        return TemplateInterfaceType(self.dataflow_category)
    
    @property  
    def type(self) -> TemplateInterfaceType:
        """Alias for interface_type (template compatibility)."""
        return self.interface_type
```

### Supporting Data Structures

```python
@dataclass
class TemplateDatatype:
    """Template-ready datatype information."""
    name: str                # "UINT", "INT", "FIXED"
    value: str               # Same as name (template compatibility)
    finn_type: str           # "UINT8", "INT16", etc.
    bitwidth: int            # Bit width
    bit_width: int           # Alias for bitwidth (template compatibility)
    signed: bool             # Signedness
    base_type: str           # Base type name
    min_bits: int            # Minimum allowed bits
    max_bits: int            # Maximum allowed bits

@dataclass  
class TemplateInterfaceType:
    """Template-compatible interface type object."""
    value: str              # "INPUT", "OUTPUT", "WEIGHT", "CONFIG"
    name: str               # Same as value (template compatibility)

@dataclass
class RTLParameter:
    """RTL parameter optimized for template generation."""
    name: str
    param_type: str
    default_value: str
    template_param_name: str    # e.g., "$WIDTH$"
    description: Optional[str]

@dataclass
class AutoHWCustomOpData:
    """Data package for AutoHWCustomOp instantiation."""
    interface_metadata: List[InterfaceMetadata]
    chunking_strategies: Dict[str, ChunkingStrategy]
    pragma_metadata: Dict[str, Any]

@dataclass
class AutoRTLBackendData:
    """Data package for AutoRTLBackend instantiation."""
    dataflow_interfaces: Dict[str, Dict[str, Any]]
    interface_categories: Dict[str, List[str]]
    parameter_mapping: Dict[str, Any]
```

## Template Context Generation

### Complete Template Context Structure

```python
def get_template_context(self) -> Dict[str, Any]:
    """Generate complete template context optimized for all templates."""
    if self._template_context is not None:
        return self._template_context
        
    self._template_context = {
        # Core kernel metadata
        "kernel_name": self.name,
        "class_name": self._generate_class_name(),
        "source_file": str(self.source_file),
        "generation_timestamp": datetime.now().isoformat(),
        
        # Interface organization  
        "interfaces": self.interfaces,                    # All interfaces
        "interfaces_list": self.interfaces,               # RTL wrapper compatibility
        "input_interfaces": self._filter_interfaces("INPUT"),
        "output_interfaces": self._filter_interfaces("OUTPUT"), 
        "weight_interfaces": self._filter_interfaces("WEIGHT"),
        "config_interfaces": self._filter_interfaces("CONFIG"),
        "dataflow_interfaces": self._get_dataflow_interfaces(),
        
        # RTL parameters
        "rtl_parameters": [
            {
                "name": p.name,
                "param_type": p.param_type or "int",
                "default_value": p.default_value or 0,
                "template_param_name": p.template_param_name
            }
            for p in self.parameters
        ],
        
        # Boolean flags for template conditionals
        "has_inputs": len(self._filter_interfaces("INPUT")) > 0,
        "has_outputs": len(self._filter_interfaces("OUTPUT")) > 0,
        "has_weights": len(self._filter_interfaces("WEIGHT")) > 0,
        
        # Counts for resource estimation
        "input_interfaces_count": len(self._filter_interfaces("INPUT")),
        "output_interfaces_count": len(self._filter_interfaces("OUTPUT")),
        "weight_interfaces_count": len(self._filter_interfaces("WEIGHT")),
        
        # Kernel analysis
        "kernel_complexity": self._estimate_complexity(),
        "kernel_type": self._infer_kernel_type(),
        "resource_estimation_required": self._requires_resource_estimation(),
        "verification_required": self._requires_verification(),
        
        # Metadata collections
        "interface_metadata": self._extract_interface_metadata(),
        "dimensional_metadata": self._extract_dimensional_metadata(),
        "datatype_constraints": self._extract_datatype_constraints(),
        
        # Template utilities
        "DataType": self._get_datatype_enum(),
        "InterfaceType": InterfaceType,  # Unified enum
        
        # Kernel object for RTL wrapper template compatibility
        "kernel": TemplateKernel(self.name, self.parameters),
        
        # Summary for template conditionals
        "dataflow_model_summary": {
            "num_interfaces": len(self.interfaces),
            "input_count": len(self._filter_interfaces("INPUT")),
            "output_count": len(self._filter_interfaces("OUTPUT")),
            "weight_count": len(self._filter_interfaces("WEIGHT")),
        }
    }
    
    return self._template_context
```

## RTL Parser Integration

### Modified RTL Parser Output

```python
class RTLParser:
    def parse(self, file_path: str) -> DirectTemplateResult:
        """Parse RTL file and return template-optimized result."""
        
        # Existing parsing pipeline (unchanged)
        self._initial_parse(file_path)
        self._extract_components()
        self._analyze_interfaces()
        self._apply_pragmas()
        
        # New: Create template-optimized interfaces
        template_interfaces = []
        for name, rtl_interface in self.interfaces.items():
            template_interface = self._create_template_interface(rtl_interface)
            template_interfaces.append(template_interface)
        
        # New: Create template-optimized parameters
        template_parameters = []
        for param in self.parameters:
            template_param = RTLParameter(
                name=param.name,
                param_type=param.param_type or "int",
                default_value=param.default_value or "0",
                template_param_name=f"${param.name.upper()}$",
                description=param.description
            )
            template_parameters.append(template_param)
        
        return DirectTemplateResult(
            name=self.name,
            source_file=Path(file_path),
            parameters=template_parameters,
            interfaces=template_interfaces,
            pragmas=self.pragmas,
            parsing_warnings=self.parsing_warnings
        )
    
    def _create_template_interface(self, rtl_interface: Interface) -> TemplateInterface:
        """Convert RTL Interface to template-optimized TemplateInterface."""
        
        # Determine dataflow category
        dataflow_category = self._map_to_dataflow_category(rtl_interface.type)
        
        # Create template datatype
        dtype = self._create_template_datatype(rtl_interface)
        
        # Extract dimensional information from pragmas
        tensor_dims, block_dims, stream_dims = self._extract_dimensions(rtl_interface)
        
        return TemplateInterface(
            name=rtl_interface.name,
            rtl_type=rtl_interface.type,
            ports=rtl_interface.ports,
            dataflow_category=dataflow_category,
            protocol=rtl_interface.type.protocol,
            direction=self._infer_direction(rtl_interface),
            dtype=dtype,
            datatype_constraints=self._create_datatype_constraints(rtl_interface),
            axi_metadata=self._extract_axi_metadata(rtl_interface),
            data_width_expr=self._extract_data_width_expr(rtl_interface),
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            stream_dims=stream_dims,
            wrapper_name=rtl_interface.wrapper_name or rtl_interface.name
        )
```

## AutoHWCustomOp/AutoRTLBackend Integration

### Direct Instantiation Support

```python
def get_auto_hwcustomop_data(self) -> AutoHWCustomOpData:
    """Extract data needed for AutoHWCustomOp instantiation."""
    
    interface_metadata = []
    chunking_strategies = {}
    
    for interface in self.interfaces:
        if interface.dataflow_category in ["INPUT", "OUTPUT", "WEIGHT"]:
            # Create InterfaceMetadata for AutoHWCustomOp
            metadata = InterfaceMetadata(
                name=interface.name,
                interface_type=getattr(DataflowInterfaceType, interface.dataflow_category),
                allowed_datatypes=interface.datatype_constraints,
                chunking_strategy=self._get_chunking_strategy(interface)
            )
            interface_metadata.append(metadata)
            
            # Store chunking strategy
            chunking_strategies[interface.name] = self._get_chunking_strategy(interface)
    
    return AutoHWCustomOpData(
        interface_metadata=interface_metadata,
        chunking_strategies=chunking_strategies,
        pragma_metadata=self._extract_pragma_metadata()
    )

def get_auto_rtlbackend_data(self) -> AutoRTLBackendData:
    """Extract data needed for AutoRTLBackend instantiation."""
    
    dataflow_interfaces = {}
    interface_categories = {
        "INPUT": [],
        "OUTPUT": [], 
        "WEIGHT": [],
        "CONFIG": []
    }
    
    for interface in self.interfaces:
        category = interface.dataflow_category
        interface_categories[category].append(interface.name)
        
        if category in ["INPUT", "OUTPUT", "WEIGHT"]:
            dataflow_interfaces[interface.name] = {
                "interface_type": category,
                "dtype": {"finn_type": interface.dtype.finn_type},
                "tensor_dims": interface.tensor_dims,
                "block_dims": interface.block_dims,  
                "stream_dims": interface.stream_dims
            }
    
    return AutoRTLBackendData(
        dataflow_interfaces=dataflow_interfaces,
        interface_categories=interface_categories,
        parameter_mapping=self._create_parameter_mapping()
    )
```

## Performance Optimizations

### 1. Lazy Template Context Generation
- Template context created only when requested
- Cached after first generation
- Avoids overhead for non-template use cases

### 2. Efficient Interface Categorization
- Categorization done once during object creation
- No runtime conversion or mapping needed
- Direct attribute access for templates

### 3. Minimal Object Creation
- Reuse RTL parser objects where possible
- Template-specific objects only when needed
- No dynamic class creation

## Migration Strategy

### Phase 1: Parallel Implementation
- Implement DirectTemplateResult alongside existing objects
- Add RTL Parser method: `parse_for_templates()`
- Update one template at a time to use new object

### Phase 2: Template Migration
- Migrate HWCustomOp template to use DirectTemplateResult
- Migrate RTLBackend template  
- Migrate RTL wrapper template
- Validate identical output with existing system

### Phase 3: Integration
- Update CLI to use DirectTemplateResult by default
- Update AutoHWCustomOp/AutoRTLBackend to use new data methods
- Remove deprecated conversion pipeline

## Success Criteria

### Performance Targets
- [ ] 60% faster template generation (vs current baseline)
- [ ] 50% memory reduction during template operations
- [ ] Zero template context generation overhead for non-template operations

### Compatibility Targets  
- [ ] 100% identical template output compared to existing system
- [ ] All Jinja2 templates render without modification
- [ ] AutoHWCustomOp/AutoRTLBackend instantiation preserves functionality

### Code Quality Targets
- [ ] Template interface objects use attributes, not dictionaries
- [ ] Zero dynamic class creation
- [ ] All template variables available as direct attributes
- [ ] Comprehensive unit test coverage

## Benefits

### 1. **Architectural Clarity**
- Clear separation: RTL Parser produces template-ready data
- No business logic in data containers
- Single responsibility for each component

### 2. **Performance Optimization**
- Elimination of DataflowModel conversion overhead
- Direct template context generation
- Lazy evaluation with caching

### 3. **Developer Experience**
- Template variables are actual object attributes
- Type hints and IDE support
- Clear data lineage from RTL to templates

### 4. **Maintainability**  
- Smaller, focused objects with clear purposes
- Easier testing with direct data access
- Reduced coupling between RTL parsing and dataflow modeling

This design provides a clean foundation for the simplified HWKG pipeline while maintaining all existing functionality and achieving significant performance improvements.