# Multi-Interface Datatype Mapping Implementation Plan

## Overview

This plan implements the multi-interface datatype parameter mapping strategy defined in `MULTI_INTERFACE_DATATYPE_MAPPING.md`. The goal is to automatically map RTL module parameters to interface datatype properties for kernels with multiple interfaces of the same type, using a simplified approach with per-interface parameter assignment.

## Current State Analysis

### ✅ Already Implemented
- Basic interface detection and validation (`interface_scanner.py`, `protocol_validator.py`)
- Single interface datatype constraints via `DATATYPE` pragma (`data.py:371-446`)
- Parameter extraction and whitelisting (`parser.py`)
- Interface metadata creation (`interface_builder.py`)

### ❌ Missing Functionality
- Multi-interface parameter pattern matching
- Indexed parameter detection (`INPUT0_WIDTH`, `INPUT1_WIDTH`)
- Enhanced pragma system for per-interface parameter assignment
- Optional datatype parameter mapping on InterfaceMetadata
- Automatic default parameter name generation

## Implementation Plan

### Phase 1: Core Infrastructure (High Priority)

#### 1.1 Extend InterfaceMetadata with Datatype Parameters

**File**: `brainsmith/dataflow/core/interface_metadata.py` (MODIFY)

```python
@dataclass
class InterfaceMetadata:
    name: str
    interface_type: InterfaceType
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    chunking_strategy: BlockChunkingStrategy = field(default_factory=DefaultChunkingStrategy)
    description: str = ""
    
    # NEW: Optional datatype parameter mappings
    datatype_params: Optional[Dict[str, str]] = None
    """
    Optional mapping of datatype properties to RTL parameters.
    If None, defaults to {clean_interface_name}_WIDTH, {clean_interface_name}_SIGNED pattern.
    
    Example: {"width": "INPUT0_WIDTH", "signed": "SIGNED_INPUT0"}
    """
    
    def get_datatype_parameter_name(self, property_type: str) -> str:
        """
        Get RTL parameter name for a datatype property.
        
        Args:
            property_type: 'width', 'signed', 'format', etc.
            
        Returns:
            RTL parameter name (e.g., 'INPUT0_WIDTH', 'SIGNED_INPUT0')
        """
        if self.datatype_params and property_type in self.datatype_params:
            return self.datatype_params[property_type]
        
        # Default naming convention
        clean_name = self._get_clean_interface_name()
        if property_type == 'width':
            return f"{clean_name}_WIDTH"
        elif property_type == 'signed':
            return f"SIGNED_{clean_name}"
        elif property_type == 'format':
            return f"{clean_name}_FORMAT"
        else:
            return f"{clean_name}_{property_type.upper()}"
    
    def _get_clean_interface_name(self) -> str:
        """Extract clean name from interface for parameter generation."""
        # Remove common prefixes/suffixes: s_axis_input0 -> INPUT0
        clean = self.name
        for prefix in ['s_axis_', 'm_axis_', 'axis_']:
            clean = clean.replace(prefix, '', 1)
        for suffix in ['_tdata', '_tvalid', '_tready']:
            clean = clean.replace(suffix, '')
        return clean.upper()
```

**Estimated Effort**: 1 day

#### 1.2 Create Simple Datatype Parameter Pragma

**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py` (MODIFY)

```python
class PragmaType(Enum):
    # ... existing types ...
    DATATYPE_PARAM = "datatype_param"  # Per-interface parameter assignment

@dataclass 
class DatatypeParamPragma(Pragma, InterfaceNameMatcher):
    """
    Maps specific RTL parameters to interface datatype properties.
    
    Syntax: @brainsmith DATATYPE_PARAM <interface_name> <property_type> <parameter_name>
    
    Examples:
    // @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
    // @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0
    // @brainsmith DATATYPE_PARAM s_axis_query width QUERY_WIDTH
    // @brainsmith DATATYPE_PARAM s_axis_wq width WQ_WIDTH
    """
    
    def _parse_inputs(self) -> Dict:
        if len(self.inputs) != 3:
            raise PragmaError("DATATYPE_PARAM pragma requires interface_name, property_type, parameter_name")
        
        interface_name = self.inputs[0]
        property_type = self.inputs[1].lower()
        parameter_name = self.inputs[2]
        
        # Validate property type
        valid_properties = ['width', 'signed', 'format', 'bias', 'fractional_width']
        if property_type not in valid_properties:
            raise PragmaError(f"Invalid property_type '{property_type}'. Must be one of: {valid_properties}")
        
        return {
            "interface_name": interface_name,
            "property_type": property_type, 
            "parameter_name": parameter_name
        }
    
    def applies_to_interface_metadata(self, metadata: InterfaceMetadata) -> bool:
        """Check if this pragma applies to the given interface."""
        if not self.parsed_data:
            return False
        
        pragma_interface_name = self.parsed_data.get('interface_name')
        if not pragma_interface_name:
            return False
        
        return self._interface_names_match(pragma_interface_name, metadata.name)
    
    def apply_to_metadata(self, metadata: InterfaceMetadata) -> InterfaceMetadata:
        """Apply DATATYPE_PARAM pragma to set datatype parameter mapping."""
        if not self.applies_to_interface_metadata(metadata):
            return metadata
        
        property_type = self.parsed_data['property_type']
        parameter_name = self.parsed_data['parameter_name']
        
        # Initialize datatype_params if not set
        current_params = metadata.datatype_params or {}
        current_params[property_type] = parameter_name
        
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=metadata.interface_type,
            datatype_constraints=metadata.datatype_constraints,
            chunking_strategy=metadata.chunking_strategy,
            description=metadata.description,
            datatype_params=current_params
        )
```

**Estimated Effort**: 1 day

#### 1.3 Update Pragma Handler

**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/pragma.py` (MODIFY)

```python
class PragmaHandler:
    def __init__(self, debug: bool = False):
        # ... existing code ...
        self.pragma_constructors[PragmaType.DATATYPE_PARAM] = DatatypeParamPragma
```

**Estimated Effort**: 0.5 days

### Phase 2: Template Integration (Medium Priority)

#### 2.1 Update Template Context Generation

**File**: `brainsmith/tools/hw_kernel_gen/generator.py` (MODIFY)

```python
def _generate_interface_context(self, interface: InterfaceMetadata) -> Dict[str, Any]:
    """Generate template context for a single interface."""
    context = {
        'name': interface.name,
        'type': interface.interface_type.value,
        'description': interface.description,
        # ... existing fields ...
        
        # NEW: Add datatype parameter mappings for template use
        'datatype_params': {},
    }
    
    # Generate datatype parameter names for template
    for property_type in ['width', 'signed', 'format', 'bias']:
        param_name = interface.get_datatype_parameter_name(property_type)
        context['datatype_params'][property_type] = param_name
    
    return context
```

**Estimated Effort**: 1 day

#### 2.2 Update HWCustomOp Template

**File**: `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op_phase2.py.j2` (MODIFY)

```jinja2
def get_interface_metadata(self) -> List[InterfaceMetadata]:
    """Return interface metadata for dataflow modeling."""
    interfaces = []
    
    {% for iface in interfaces_by_type.input %}
    # {{ iface.name }} interface
    interfaces.append(InterfaceMetadata(
        name="{{ iface.name }}",
        interface_type=InterfaceType.INPUT,
        datatype_constraints=[
            # Use interface-specific parameter names
            DatatypeConstraintGroup("INT", 1, self.get_nodeattr("{{ iface.datatype_params.width }}") or 8),
            DatatypeConstraintGroup("UINT", 1, self.get_nodeattr("{{ iface.datatype_params.width }}") or 8),
        ],
        # ... existing fields ...
    ))
    {% endfor %}
    
    return interfaces
```

**Estimated Effort**: 1 day

### Phase 3: Testing & Validation (High Priority)

#### 3.1 Unit Tests

**File**: `tests/tools/hw_kernel_gen/test_datatype_param_pragma.py` (NEW)

```python
class TestDatatypeParamPragma:
    def test_pragma_parsing(self):
        """Test DATATYPE_PARAM pragma parsing and validation."""
        
    def test_interface_name_matching(self):
        """Test pragma application to correct interfaces."""
        
    def test_parameter_mapping(self):
        """Test datatype_params setting on InterfaceMetadata."""
        
    def test_invalid_property_types(self):
        """Test error handling for invalid property types."""

class TestInterfaceMetadataExtensions:
    def test_default_parameter_naming(self):
        """Test default parameter name generation."""
        # s_axis_input0 -> INPUT0_WIDTH, SIGNED_INPUT0
        
    def test_custom_parameter_mapping(self):
        """Test custom datatype_params override."""
        # Custom: {"width": "QUERY_WIDTH", "signed": "QUERY_SIGNED"}
        
    def test_clean_interface_name_extraction(self):
        """Test _get_clean_interface_name method."""
        # s_axis_weights_tdata -> WEIGHTS
        # m_axis_output -> OUTPUT
```

**Estimated Effort**: 2 days

#### 3.2 Integration Tests

**File**: `tests/tools/hw_kernel_gen/test_multi_interface_integration.py` (NEW)

```python
class TestMultiInterfaceIntegration:
    def test_elementwise_add_scenario(self):
        """
        Test RTL with multiple indexed inputs:
        s_axis_input0, s_axis_input1 -> INPUT0_WIDTH, INPUT1_WIDTH
        """
        
    def test_multihead_attention_scenario(self):
        """
        Test RTL with pragma-mapped parameters:
        // @brainsmith DATATYPE_PARAM s_axis_query width QUERY_WIDTH
        // @brainsmith DATATYPE_PARAM s_axis_key width KEY_WIDTH
        """
        
    def test_default_parameter_generation(self):
        """Test automatic parameter name generation without pragmas."""
        
    def test_template_context_integration(self):
        """Test that templates receive correct datatype parameter names."""
        
    def test_generated_hwcustomop_functionality(self):
        """Test that generated HWCustomOp uses correct parameter names."""
```

**Estimated Effort**: 2 days

### Phase 4: Documentation & Examples (Low Priority)

#### 4.1 Update Documentation

**Files to Update**:
- `brainsmith/tools/hw_kernel_gen/rtl_parser/README.md`
- `tests/tools/hw_kernel_gen/HKG_DESIGN_DOCUMENT.md`
- Add examples to `MULTI_INTERFACE_DATATYPE_MAPPING.md`

**Estimated Effort**: 1 day

#### 4.2 Create Example Files

**File**: `examples/multi_interface_examples/` (NEW DIRECTORY)

```
elementwise_add.sv          # Example 1: Indexed parameters with pragmas
multihead_attention.sv      # Example 2: Named parameters with pragmas  
simple_dual_input.sv        # Example 3: Default parameter generation
test_generation.py          # Automated test of all examples
```

**Estimated Effort**: 1 day

## Implementation Schedule

### Week 1: Core Infrastructure (3 days)
- [ ] Day 1: Extend InterfaceMetadata with datatype_params attribute
- [ ] Day 2: Create DatatypeParamPragma in data.py
- [ ] Day 3: Update pragma handler and basic integration testing

### Week 2: Template Integration (2 days)
- [ ] Day 1: Update template context generation
- [ ] Day 2: Update HWCustomOp template with interface-specific parameters

### Week 3: Testing & Validation (4 days)
- [ ] Day 1-2: Write comprehensive unit tests
- [ ] Day 3-4: Create integration tests with real examples

### Week 4: Documentation & Examples (2 days)
- [ ] Day 1: Update documentation and design documents
- [ ] Day 2: Create example files and validation

## Success Criteria

### ✅ Functional Requirements
- [ ] Default parameter name generation for all interface types
- [ ] Support for indexed parameters via pragmas (`INPUT0_WIDTH`, `INPUT1_WIDTH`)
- [ ] DATATYPE_PARAM pragma for custom parameter mapping
- [ ] Template integration with interface-specific parameter names
- [ ] Backward compatibility with existing DATATYPE pragma

### ✅ Quality Requirements  
- [ ] 100% test coverage for new functionality
- [ ] Integration with existing HKG pipeline
- [ ] Backward compatibility with current pragma system
- [ ] Performance: <10ms additional overhead for parameter mapping

### ✅ Documentation Requirements
- [ ] Updated design documents with new functionality
- [ ] Code examples for pragma usage
- [ ] Clear migration path from current system

## Risk Assessment

### 🟡 Medium Risk  
- **Template Complexity**: Templates need to handle dynamic parameter names
  - *Mitigation*: Use simple Jinja2 variable substitution
- **Backward Compatibility**: New InterfaceMetadata fields may break existing code
  - *Mitigation*: Make datatype_params optional with sensible defaults

### 🟢 Low Risk
- **Pragma Integration**: New pragma follows existing patterns
  - *Mitigation*: Reuse existing pragma infrastructure
- **FINN Compatibility**: Changes are purely additive
  - *Mitigation*: Default behavior maintains current functionality

## Dependencies

### Internal Dependencies
- `brainsmith.dataflow.core.interface_metadata` - InterfaceMetadata class extension
- `brainsmith.tools.hw_kernel_gen.rtl_parser.data` - Pragma system 
- `brainsmith.tools.hw_kernel_gen.generator` - Template context generation
- Existing RTL parser infrastructure

### External Dependencies  
- None (all functionality uses existing libraries)

## Validation Plan

### Phase 1 Validation: Core Infrastructure
```bash
# Test InterfaceMetadata extensions
python -m pytest tests/tools/hw_kernel_gen/test_datatype_param_pragma.py::TestInterfaceMetadataExtensions -v

# Test DATATYPE_PARAM pragma
python -m pytest tests/tools/hw_kernel_gen/test_datatype_param_pragma.py::TestDatatypeParamPragma -v
```

### Phase 2 Validation: Template Integration
```bash
# Test template context generation
python -m pytest tests/tools/hw_kernel_gen/test_multi_interface_integration.py::test_template_context_integration -v

# Test generated HWCustomOp functionality
python -m pytest tests/tools/hw_kernel_gen/test_multi_interface_integration.py::test_generated_hwcustomop_functionality -v
```

### Phase 3 Validation: End-to-End
```bash
# Test real scenarios
python -m pytest tests/tools/hw_kernel_gen/test_multi_interface_integration.py -v

# Test with brainsmith MVU module (should work with default naming)
python -m brainsmith.tools.hw_kernel_gen.cli generate \
  brainsmith/hw_kernels/mvu/mvu_vvu_axi.sv output_test/

# Test with multi-interface example (with pragmas)
python -m brainsmith.tools.hw_kernel_gen.cli generate \
  examples/multi_interface_examples/elementwise_add.sv output_test/
```

## Example Usage

### Default Parameter Generation (No Pragmas)
```systemverilog
module simple_add #(
    int unsigned ACTIVATION_WIDTH = 8,  // Will map to: input -> ACTIVATION_WIDTH
    int unsigned OUTPUT_WIDTH = 9       // Will map to: output -> OUTPUT_WIDTH  
)(
    input logic [ACTIVATION_WIDTH-1:0] s_axis_input_tdata,
    input logic s_axis_input_tvalid,
    output logic s_axis_input_tready,
    
    output logic [OUTPUT_WIDTH-1:0] m_axis_output_tdata,
    output logic m_axis_output_tvalid,
    input logic m_axis_output_tready
);
```

### Custom Parameter Mapping (With Pragmas)
```systemverilog
// @brainsmith DATATYPE_PARAM s_axis_input0 width INPUT0_WIDTH
// @brainsmith DATATYPE_PARAM s_axis_input0 signed SIGNED_INPUT0
// @brainsmith DATATYPE_PARAM s_axis_input1 width INPUT1_WIDTH  
// @brainsmith DATATYPE_PARAM s_axis_input1 signed SIGNED_INPUT1
module elementwise_add #(
    int unsigned INPUT0_WIDTH = 8,    // Custom mapping: input0 -> INPUT0_WIDTH
    int unsigned INPUT1_WIDTH = 16,   // Custom mapping: input1 -> INPUT1_WIDTH
    bit SIGNED_INPUT0 = 1,            // Custom mapping: input0 -> SIGNED_INPUT0
    bit SIGNED_INPUT1 = 1             // Custom mapping: input1 -> SIGNED_INPUT1
)(
    // Two separate input interfaces
    input logic [INPUT0_WIDTH-1:0] s_axis_input0_tdata,
    input logic [INPUT1_WIDTH-1:0] s_axis_input1_tdata,
    // ...
);
```

This simplified implementation plan provides a clean, maintainable approach to multi-interface datatype parameter mapping with minimal complexity and maximum compatibility.