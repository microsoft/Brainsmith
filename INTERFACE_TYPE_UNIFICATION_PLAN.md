# Interface Type Unification Plan

## Current State Analysis

### RTL Parser Types (`brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`)
```python
class Direction(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

class InterfaceType(Enum):
    GLOBAL_CONTROL = "global"
    AXI_STREAM = "axistream"
    AXI_LITE = "axilite"
    UNKNOWN = "unknown"
```

### Dataflow Types (`brainsmith/dataflow/core/dataflow_interface.py`)
```python
class DataflowInterfaceType(Enum):
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data  
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    CONFIG = "config"    # AXI-Lite for runtime configuration
    CONTROL = "control"  # Global control signals (clk, rst, etc.)
```

## Problems with Current Design

### 1. **Semantic Mismatch**
- RTL Parser: Protocol-focused (`AXI_STREAM`, `AXI_LITE`, `GLOBAL_CONTROL`)
- Dataflow: Role-focused (`INPUT`, `OUTPUT`, `WEIGHT`, `CONFIG`, `CONTROL`)

### 2. **Missing Mappings**
- No direct conversion between `InterfaceType.AXI_STREAM` → `INPUT`/`OUTPUT`/`WEIGHT`
- RTL parser has to guess interface roles from names/directions
- Dataflow can't determine protocols from roles

### 3. **Duplication**
- Two separate type systems for the same concepts
- Maintenance burden and potential inconsistencies
- Complex conversion logic scattered throughout codebase

### 4. **Template Confusion**
- Templates need both protocol info (for RTL generation) AND role info (for dataflow)
- Current `EnhancedRTLParsingResult` creates dynamic classes to bridge gap

## Unified Type System Design

### Core Principle: **Role-Based Interface Types with Inherent Protocols**
Instead of separate type systems, create a unified enum where each interface role has an inherent protocol relationship.

### New Unified Enum Structure
```python
# File: brainsmith/dataflow/core/interface_types.py
from enum import Enum

class InterfaceType(Enum):
    """Unified interface types with inherent protocol-role relationships"""
    
    # AXI-Stream interfaces (dataflow)
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data  
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    
    # AXI-Lite interfaces (configuration)
    CONFIG = "config"    # AXI-Lite for runtime configuration
    
    # Global control signals
    CONTROL = "control"  # Global control signals (clk, rst, etc.)
    
    # Unknown/fallback
    UNKNOWN = "unknown"  # Unrecognized interfaces
    
    @property
    def protocol(self) -> str:
        """Get the hardware protocol for this interface type"""
        protocol_map = {
            InterfaceType.INPUT: "axi_stream",
            InterfaceType.OUTPUT: "axi_stream", 
            InterfaceType.WEIGHT: "axi_stream",
            InterfaceType.CONFIG: "axi_lite",
            InterfaceType.CONTROL: "global_control",
            InterfaceType.UNKNOWN: "unknown"
        }
        return protocol_map[self]
    
    @property
    def is_dataflow(self) -> bool:
        """Check if this interface participates in dataflow"""
        return self in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]
    
    @property
    def is_axi_stream(self) -> bool:
        """Check if this interface uses AXI-Stream protocol"""
        return self.protocol == "axi_stream"
    
    @property
    def is_axi_lite(self) -> bool:
        """Check if this interface uses AXI-Lite protocol"""
        return self.protocol == "axi_lite"
    
    @property
    def is_configuration(self) -> bool:
        """Check if this interface is for configuration"""
        return self in [InterfaceType.CONFIG, InterfaceType.CONTROL]
    
    @property
    def direction(self) -> str:
        """Get the expected direction for this interface type"""
        direction_map = {
            InterfaceType.INPUT: "input",
            InterfaceType.WEIGHT: "input", 
            InterfaceType.OUTPUT: "output",
            InterfaceType.CONFIG: "bidirectional",
            InterfaceType.CONTROL: "input",
            InterfaceType.UNKNOWN: "unknown"
        }
        return direction_map[self]
```

## Migration Plan

### Phase 1: Create Unified Types (Week 1)

#### 1.1 Create Core Interface Types Module in Dataflow
```python
# File: brainsmith/dataflow/core/interface_types.py
# (Implementation above)
```

#### 1.2 Update RTL Parser to Use Dataflow Types Directly
```python
# In brainsmith/tools/hw_kernel_gen/rtl_parser/data.py

# Import dataflow types directly
from brainsmith.dataflow.core.interface_types import InterfaceType

@dataclass
class Interface:
    name: str
    type: InterfaceType  # Now uses dataflow InterfaceType directly!
    ports: Dict[str, Port]
    validation_result: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict)
    wrapper_name: Optional[str] = None
    
    # No conversion needed - type IS the unified type!
```

#### 1.3 Update Protocol Validator to Determine Dataflow Types Directly
```python
# In brainsmith/tools/hw_kernel_gen/rtl_parser/protocol_validator.py

from brainsmith.dataflow.core.interface_types import InterfaceType

def validate_axi_stream(self, group: PortGroup) -> ValidationResult:
    """AXI-Stream validation with dataflow type determination"""
    # Existing validation...
    
    if result.valid:
        # Determine direction
        direction = self._determine_direction(group.ports)
        group.metadata['direction'] = direction
        
        # Determine dataflow interface type directly
        interface_type = self._determine_dataflow_type(group.name, direction)
        
        # Set the dataflow type directly on the group
        group.interface_type = interface_type
    
    return result

def validate_axi_lite(self, group: PortGroup) -> ValidationResult:
    """AXI-Lite validation - always CONFIG type"""
    # Existing validation...
    
    if result.valid:
        group.interface_type = InterfaceType.CONFIG
    
    return result

def validate_global_control(self, group: PortGroup) -> ValidationResult:
    """Global control validation - always CONTROL type"""
    # Existing validation...
    
    if result.valid:
        group.interface_type = InterfaceType.CONTROL
    
    return result

def _determine_dataflow_type(self, interface_name: str, direction: Direction) -> InterfaceType:
    """Determine dataflow interface type from name patterns and direction"""
    name_lower = interface_name.lower()
    
    # Weight interface patterns
    if any(pattern in name_lower for pattern in ['weight', 'weights', 'param', 'coeff']):
        return InterfaceType.WEIGHT
    
    # Input/output based on direction
    if direction == Direction.INPUT:
        return InterfaceType.INPUT
    elif direction == Direction.OUTPUT:
        return InterfaceType.OUTPUT
    else:
        return InterfaceType.INPUT  # Default
```

### Phase 2: Migrate Dataflow System to Use New Types (Week 2)

#### 2.1 Replace DataflowInterface Type System
```python
# In brainsmith/dataflow/core/dataflow_interface.py

# Remove old enum entirely - use the new InterfaceType directly
from .interface_types import InterfaceType

@dataclass
class DataflowInterface:
    name: str
    interface_type: InterfaceType  # Now uses the unified type directly!
    
    # Rest of fields unchanged...
    tensor_dims: List[int]
    block_dims: List[int]
    stream_dims: List[int]
    dtype: DataflowDataType
    # etc...
    
    @property
    def role(self) -> str:
        """Get interface role"""
        return self.interface_type.value
    
    @property
    def protocol(self) -> str:
        """Get interface protocol"""
        return self.interface_type.protocol
    
    @property
    def is_dataflow(self) -> bool:
        """Check if this participates in dataflow"""
        return self.interface_type.is_dataflow
```

#### 2.2 Create Conversion Utilities
```python
# File: brainsmith/dataflow/core/interface_conversion.py

def rtl_interface_to_dataflow(rtl_interface: 'RTLInterface') -> DataflowInterface:
    """Convert RTL Interface to DataflowInterface using unified types"""
    
    # Extract dimensions from metadata/pragmas
    tensor_dims = _extract_tensor_dims(rtl_interface)
    block_dims = _extract_block_dims(rtl_interface)
    stream_dims = _extract_stream_dims(rtl_interface)
    
    # Extract datatype
    dtype = _extract_datatype(rtl_interface)
    
    return DataflowInterface(
        name=rtl_interface.name,
        interface_type=rtl_interface.type,  # Direct use - same type system!
        tensor_dims=tensor_dims,
        block_dims=block_dims,
        stream_dims=stream_dims,
        dtype=dtype,
        axi_metadata=rtl_interface.metadata
    )
```

### Phase 3: Update Template System (Week 3)

#### 3.1 Simplify Template Context Generation
```python
# In templates/direct_renderer.py or similar

def build_template_context(interfaces: List[Interface]) -> Dict[str, Any]:
    """Build template context using unified types"""
    
    # Categorize by role - using single type system!
    input_interfaces = [iface for iface in interfaces if iface.type == InterfaceType.INPUT]
    output_interfaces = [iface for iface in interfaces if iface.type == InterfaceType.OUTPUT]
    weight_interfaces = [iface for iface in interfaces if iface.type == InterfaceType.WEIGHT]
    config_interfaces = [iface for iface in interfaces if iface.type == InterfaceType.CONFIG]
    
    # Filter by protocol  
    dataflow_interfaces = [iface for iface in interfaces if iface.type.is_dataflow]
    
    return {
        'input_interfaces': input_interfaces,
        'output_interfaces': output_interfaces,
        'weight_interfaces': weight_interfaces,
        'config_interfaces': config_interfaces,
        'dataflow_interfaces': dataflow_interfaces,
        
        # Counts for template conditionals
        'has_inputs': len(input_interfaces) > 0,
        'has_outputs': len(output_interfaces) > 0,
        'has_weights': len(weight_interfaces) > 0,
        'has_configs': len(config_interfaces) > 0,
        
        # Protocol information for RTL templates
        'axi_stream_interfaces': [iface for iface in interfaces if iface.type.is_axi_stream],
        'axi_lite_interfaces': [iface for iface in interfaces if iface.type.is_axi_lite],
    }
```

#### 3.2 Update Templates
```jinja2
{# In hw_custom_op_slim.py.j2 #}
{% for iface in input_interfaces %}
    # Input interface: {{ iface.name }} ({{ iface.unified_type }})
{% endfor %}

{% for iface in weight_interfaces %}
    # Weight interface: {{ iface.name }} ({{ iface.unified_type }})
{% endfor %}

{# Protocol-specific code generation #}
{% for iface in axi_stream_interfaces %}
    # AXI-Stream interface: {{ iface.name }}
    # Role: {{ iface.type.value }}
{% endfor %}
```

### Phase 4: Refactor Enhanced RTL Parsing (Week 4)

#### 4.1 Eliminate Complex Interface Mapping
```python
# BEFORE: Complex mapping in EnhancedRTLParsingResult
def _map_rtl_interface_to_category(self, interface) -> str:
    # 50+ lines of complex logic

# AFTER: Simple property access
@property
def input_interfaces(self) -> List[Interface]:
    return [iface for iface in self.interfaces.values() 
            if iface.type == InterfaceType.INPUT]

@property  
def output_interfaces(self) -> List[Interface]:
    return [iface for iface in self.interfaces.values()
            if iface.type == InterfaceType.OUTPUT]

@property
def weight_interfaces(self) -> List[Interface]:
    return [iface for iface in self.interfaces.values()
            if iface.type == InterfaceType.WEIGHT]

@property
def config_interfaces(self) -> List[Interface]:
    return [iface for iface in self.interfaces.values()
            if iface.type == InterfaceType.CONFIG]
```

#### 4.2 Simplified Template-Ready Result
```python
@dataclass
class TemplateReadyResult:
    """Simple data container using unified types"""
    name: str
    class_name: str
    source_file: Optional[Path]
    interfaces: Dict[str, Interface]  # All use dataflow InterfaceType!
    parameters: List[Parameter]
    pragmas: List[Pragma]
    
    # Pre-computed categorizations (no methods needed!)
    @property
    def input_interfaces(self) -> List[Interface]:
        return [iface for iface in self.interfaces.values() 
                if iface.type == InterfaceType.INPUT]
    
    # etc. for other roles...
```

## Benefits of Unified System

### 1. **Eliminates Type System Duplication**
- Single source of truth for interface types in dataflow module
- RTL parser simply identifies and references dataflow types
- No conversion logic between RTL and Dataflow
- Consistent terminology across entire system

### 2. **Proper Separation of Concerns**
- Dataflow module owns interface type definitions (conceptual authority)
- RTL parser identifies and tags interfaces with dataflow types (detection)
- Templates consume dataflow types directly (usage)

### 3. **Simplifies Template Generation**
- Direct property access: `interface.unified_type == InterfaceType.INPUT`
- No complex categorization methods
- Protocol information available via properties when needed

### 4. **Improves Maintainability**
- Changes to interface types in one logical place (dataflow)
- Clear semantic meaning with inherent protocol relationships
- Easy to extend (add new interface types to dataflow)

### 5. **Enables Better Validation**
- Can validate protocol-role relationships in dataflow module
- Clear rules: "INPUT/OUTPUT/WEIGHT are AXI-Stream, CONFIG is AXI-Lite"
- Type safety at compile time

### 6. **Reduces EnhancedRTLParsingResult Complexity**
- Eliminates 200+ lines of complex categorization logic
- No dynamic class creation
- Simple property-based access patterns
- RTL parser just identifies, doesn't need to understand dataflow semantics

## Migration Strategy with Detailed Checklists

### Phase 1 (Week 1): Foundation in Dataflow

#### 1.1 Create Unified Interface Type System
- [x] Create `brainsmith/dataflow/core/interface_types.py`
- [x] Implement `InterfaceType` enum with INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL, UNKNOWN
- [x] Add `protocol` property mapping interface types to protocols
- [x] Add `is_dataflow`, `is_axi_stream`, `is_axi_lite`, `is_configuration` properties
- [x] Add `direction` property for expected port directions
- [x] Write unit tests for new interface type system
- [x] Update `brainsmith/dataflow/core/__init__.py` to export `InterfaceType`

#### 1.2 Update RTL Parser Data Structures  
- [ ] Update `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
- [ ] Add import: `from brainsmith.dataflow.core.interface_types import InterfaceType`
- [ ] Update `Interface` class to use dataflow `InterfaceType` directly
- [ ] Remove old RTL parser `InterfaceType` enum (mark for deletion)
- [ ] Update `PortGroup` class to use dataflow `InterfaceType`
- [ ] Test that RTL parser data structures work with new types

#### 1.3 Update Protocol Validator
- [ ] Update `brainsmith/tools/hw_kernel_gen/rtl_parser/protocol_validator.py`
- [ ] Add import: `from brainsmith.dataflow.core.interface_types import InterfaceType`
- [ ] Update `validate_axi_stream()` to set dataflow types directly
- [ ] Update `validate_axi_lite()` to set `InterfaceType.CONFIG`
- [ ] Update `validate_global_control()` to set `InterfaceType.CONTROL`
- [ ] Implement `_determine_dataflow_type()` for weight/input/output detection
- [ ] Test protocol validator produces correct dataflow types

#### 1.4 Update Interface Builder
- [ ] Update `brainsmith/tools/hw_kernel_gen/rtl_parser/interface_builder.py`
- [ ] Ensure interface creation uses dataflow types from validator
- [ ] Test that built interfaces have correct dataflow types
- [ ] Verify interface naming and metadata still work correctly

### Phase 2 (Week 2): Dataflow Integration

#### 2.1 Update DataflowInterface
- [ ] Update `brainsmith/dataflow/core/dataflow_interface.py`
- [ ] Remove old `DataflowInterfaceType` enum entirely
- [ ] Update `DataflowInterface` class to use new `InterfaceType` directly
- [ ] Update `interface_type` field to use dataflow `InterfaceType`
- [ ] Add/update convenience properties: `role`, `protocol`, `is_dataflow`
- [ ] Test DataflowInterface creation with new types

#### 2.2 Update Interface Conversion
- [ ] Update `brainsmith/dataflow/rtl_integration/rtl_converter.py`
- [ ] Remove complex type conversion logic (no longer needed!)
- [ ] Update RTL → DataflowInterface conversion to use same type directly
- [ ] Test RTL parser → DataflowModel conversion pipeline
- [ ] Verify no type conversion errors

#### 2.3 Update Dataflow Model  
- [ ] Update `brainsmith/dataflow/core/dataflow_model.py`
- [ ] Update any interface type references to use new unified system
- [ ] Test dataflow model creation and operations
- [ ] Verify mathematical operations still work correctly

### Phase 3 (Week 3): Template System

#### 3.1 Update Template Context Generation
- [ ] Update `brainsmith/tools/hw_kernel_gen/templates/direct_renderer.py`
- [ ] Simplify interface categorization using direct type comparison
- [ ] Update `build_template_context()` to use `iface.type == InterfaceType.INPUT`
- [ ] Remove complex interface mapping functions
- [ ] Test template context generation produces correct categorizations

#### 3.2 Refactor EnhancedRTLParsingResult
- [ ] Update `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
- [ ] **DELETE** `_map_rtl_interface_to_category()` method (~50 lines)
- [ ] **DELETE** `_categorize_interfaces()` method (~45 lines)  
- [ ] **DELETE** dynamic class creation in methods (~50 lines)
- [ ] **DELETE** `_get_dtype_from_interface_metadata()` and similar (~100 lines)
- [ ] Replace with simple property access: `@property def input_interfaces()`
- [ ] Reduce EnhancedRTLParsingResult from ~500 lines to ~100 lines
- [ ] Test simplified interface access

#### 3.3 Update Templates
- [ ] Update `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_slim.py.j2`
- [ ] Update `brainsmith/tools/hw_kernel_gen/templates/rtl_backend.py.j2`
- [ ] Update `brainsmith/tools/hw_kernel_gen/templates/rtl_wrapper.v.j2`
- [ ] Change all interface references to use `iface.type.value`
- [ ] Test template rendering with unified types

#### 3.4 Update CLI and Generators
- [ ] Update `brainsmith/tools/hw_kernel_gen/cli.py`
- [ ] Update `brainsmith/tools/unified_hwkg/generator.py`
- [ ] Ensure enhanced generation pipeline uses unified types
- [ ] Test end-to-end generation with new type system

### Phase 4 (Week 4): Complete Cleanup - Zero Legacy Types

#### 4.1 Delete RTL Parser Legacy Types
- [ ] **DELETE** old `InterfaceType` enum from `rtl_parser/data.py`
- [ ] **DELETE** old `Direction` enum if no longer needed elsewhere
- [ ] **DELETE** any RTL parser specific type conversion functions
- [ ] Search codebase for old RTL `InterfaceType` references and remove
- [ ] Verify RTL parser has zero local interface type definitions

#### 4.2 Delete Dataflow Legacy Types  
- [ ] **DELETE** old `DataflowInterfaceType` enum from `dataflow_interface.py`
- [ ] **DELETE** any dataflow type conversion utilities
- [ ] Search codebase for old `DataflowInterfaceType` references and remove
- [ ] Verify dataflow module has single unified interface type system

#### 4.3 Complete EnhancedRTLParsingResult Cleanup
- [ ] **DELETE** remaining complex categorization methods
- [ ] **DELETE** template-specific type conversion functions
- [ ] **DELETE** any remaining dynamic class creation
- [ ] Verify EnhancedRTLParsingResult is truly a simple data container
- [ ] Confirm line count reduction from ~500 to ~100 lines

#### 4.4 Final Verification
- [ ] Search entire codebase for "InterfaceType" - ensure only dataflow version exists
- [ ] Search entire codebase for "DataflowInterfaceType" - ensure zero references
- [ ] Run full test suite and ensure all tests pass
- [ ] Test thresholding example end-to-end with unified types
- [ ] Verify template generation speed improvement
- [ ] Update all documentation to reflect unified type system

#### 4.5 Import Cleanup
- [ ] Verify all files import InterfaceType from `brainsmith.dataflow.core.interface_types`
- [ ] Remove any imports of old type enums
- [ ] Clean up unused imports throughout codebase
- [ ] Verify clean dependency graph: RTL parser → dataflow types

### Completion Verification Checklist

#### Architecture Verification
- [ ] Single `InterfaceType` enum exists in `brainsmith/dataflow/core/interface_types.py`
- [ ] RTL parser imports and uses dataflow types exclusively
- [ ] DataflowInterface uses same InterfaceType as RTL parser
- [ ] Templates access interface types via `iface.type` directly
- [ ] Zero type conversion logic exists anywhere

#### Code Quality Verification  
- [ ] EnhancedRTLParsingResult reduced by 200+ lines
- [ ] No dynamic class creation in interface handling
- [ ] No complex categorization methods remain
- [ ] All interface access uses simple property patterns
- [ ] Clean separation: dataflow owns types, RTL identifies them

#### Functional Verification
- [ ] RTL parsing produces correct interface types
- [ ] Dataflow model creation works with unified types
- [ ] Template generation produces correct output
- [ ] End-to-end pipeline (RTL → Dataflow → Templates) functions correctly
- [ ] All existing functionality preserved with simplified implementation

## Success Metrics

- [ ] Single interface type system used throughout (dataflow InterfaceType only)
- [ ] **Zero** legacy interface enums remaining in codebase
- [ ] EnhancedRTLParsingResult reduced by 200+ lines  
- [ ] **Zero** type conversion logic needed
- [ ] All templates use dataflow InterfaceType directly
- [ ] RTL parser imports from dataflow module
- [ ] DataflowInterface uses same InterfaceType as RTL parser
- [ ] 100% test coverage maintained
- [ ] Clean architectural separation: dataflow owns types, RTL identifies them

## Risk Mitigation

### Migration Strategy
- Create new dataflow types first (Week 1)
- Migrate RTL parser to use them directly (Week 1-2)  
- Update dataflow module to use new types (Week 2)
- Update templates and EnhancedRTLParsingResult (Week 3)
- **Delete all legacy types completely (Week 4)**
- No backward compatibility maintained - clean break

### Testing
- Comprehensive test suite for unified types
- Validation of RTL → Dataflow → Template pipeline
- Performance regression testing

### Documentation
- Clear migration guide
- Updated API documentation
- Examples using unified types

This unified approach eliminates the architectural issues with EnhancedRTLParsingResult by removing the need for complex categorization logic entirely. The RTL parser will directly populate semantic interface types that work throughout the entire system.