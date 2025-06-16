# Proposal: Replace Operation Type Guessing with Explicit Tensor Formatting Configuration

## Executive Summary

This proposal eliminates the problematic `EnhancedInterfaceAnalyzer` and operation type guessing by replacing it with an explicit tensor formatting configuration system. This approach is more robust, maintainable, and user-controllable while solving the same problems without assumptions.

## Problem Statement

### Current Issues with Operation Type Approach

1. **Fragile Assumptions**: The `EnhancedInterfaceAnalyzer` makes assumptions about operation types based on:
   - Class name patterns (`"mvau" in class_name`)
   - Interface dimensions (`len(interface.tensor_dims) == 2`)
   - RTL pattern detection (unreliable)

2. **Limited Scope**: Operation type is only used for:
   - **3 lines of preprocessing**: Matrix transpose, spatial flattening
   - **1 boolean flag**: SIMD flip optimization

3. **Maintenance Burden**: Complex inference logic that's hard to maintain and extend

4. **User Helplessness**: Users can't easily override incorrect inferences

### Analysis of Current Usage

```python
# Current operation type usage - MINIMAL!

# Usage 1: Tensor preprocessing (3 lines)
if op_type == "matrix_multiplication":
    return tensor.T  # Transpose for MVAU
elif op_type == "convolution":
    return tensor.reshape(channels, spatial_flat)  # Flatten for VVAU

# Usage 2: SIMD flip flag (1 boolean)
needs_simd_flip = op_type in ["matrix_multiplication", "convolution"]
```

**Observation**: We're building complex inference machinery for ~4 lines of conditional logic!

## Proposed Solution: Explicit Tensor Formatting Configuration

### Core Concept

Replace operation type guessing with explicit configuration that directly specifies the needed tensor transformations:

```python
@dataclass
class TensorFormattingConfig:
    """Explicit configuration for tensor formatting behavior."""
    needs_transpose: bool = False
    needs_spatial_flattening: bool = False  
    needs_simd_flip: bool = True
    pe_distribution: bool = True
    custom_preprocessing: Optional[Callable] = None
```

### Three-Tier Configuration System

#### Tier 1: RTL Pragma Specification (Primary)
Allow RTL authors to explicitly specify formatting requirements:

```systemverilog
// @brainsmith TENSOR_FORMAT weights transpose=true simd_flip=true
// @brainsmith TENSOR_FORMAT thresholds transpose=false simd_flip=false
module mvau_kernel(
    input  logic [SIMD*8-1:0] weights_V_TDATA,
    output logic [PE*8-1:0]   out_V_TDATA
);
```

**Benefits:**
- ✅ **Explicit**: No guessing, RTL author specifies exactly what's needed
- ✅ **Documentable**: Formatting requirements are clearly documented in RTL
- ✅ **Maintainable**: Changes to formatting are made where the RTL lives

#### Tier 2: Interface Characteristic Defaults (Fallback)
Provide intelligent defaults based on interface metadata:

```python
def get_default_formatting_config(interface: DataflowInterface) -> TensorFormattingConfig:
    """Provide intelligent defaults based on interface characteristics."""
    
    if interface.interface_type == InterfaceType.WEIGHT:
        if len(interface.tensor_dims) == 2:
            # 2D weights typically need transpose (matrix ops)
            return TensorFormattingConfig(needs_transpose=True, needs_simd_flip=True)
        elif len(interface.tensor_dims) >= 3:
            # 3D+ weights typically need spatial flattening (convolution ops)  
            return TensorFormattingConfig(needs_spatial_flattening=True, needs_simd_flip=True)
    
    elif interface.interface_type == InterfaceType.OUTPUT:
        # Output interfaces (thresholds) typically don't need SIMD flip
        return TensorFormattingConfig(needs_simd_flip=False)
    
    # Safe default for unknown cases
    return TensorFormattingConfig()
```

**Benefits:**
- ✅ **Reasonable**: Based on actual interface characteristics, not guesses
- ✅ **Safe**: Conservative defaults that work for most cases
- ✅ **Overridable**: Users can provide explicit configuration

#### Tier 3: User Override (Ultimate Control)
Allow users to provide explicit configuration:

```python
# Generated template allows user override
class MyCustomOp(AutoHWCustomOp):
    def get_tensor_formatting_config(self, interface_name: str) -> TensorFormattingConfig:
        """Override default tensor formatting configuration."""
        if interface_name == "weights":
            return TensorFormattingConfig(
                needs_transpose=False,  # Custom: no transpose needed
                needs_simd_flip=True,
                custom_preprocessing=my_custom_transform
            )
        return super().get_tensor_formatting_config(interface_name)
```

**Benefits:**
- ✅ **Ultimate Control**: Users can specify exactly what they need
- ✅ **Custom Operations**: Supports novel operations not covered by defaults
- ✅ **Debugging**: Easy to disable specific transformations for testing

## Implementation Design

### 1. TensorFormattingConfig Class

```python
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

@dataclass
class TensorFormattingConfig:
    """Configuration for tensor formatting behavior."""
    
    # Basic transformations
    needs_transpose: bool = False
    needs_spatial_flattening: bool = False
    
    # Hardware optimizations  
    needs_simd_flip: bool = True
    pe_distribution: bool = True
    
    # Advanced customization
    custom_preprocessing: Optional[Callable[[np.ndarray], np.ndarray]] = None
    custom_postprocessing: Optional[Callable[[np.ndarray], np.ndarray]] = None
    
    # Memory layout preferences
    memory_layout_hint: str = "default"  # "row_major", "column_major", "blocked"
    
    def validate(self) -> None:
        """Validate configuration for conflicts."""
        # Could add validation logic for conflicting options
        pass
```

### 2. Enhanced InterfaceMetadata

```python
@dataclass  
class InterfaceMetadata:
    """Enhanced interface metadata with formatting configuration."""
    
    name: str
    interface_type: InterfaceType
    chunking_strategy: ChunkingStrategy
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    
    # NEW: Explicit tensor formatting configuration
    tensor_formatting_config: Optional[TensorFormattingConfig] = None
    
    def get_formatting_config(self) -> TensorFormattingConfig:
        """Get tensor formatting configuration with fallback to defaults."""
        if self.tensor_formatting_config:
            return self.tensor_formatting_config
        
        # Fallback to interface-based defaults
        return get_default_formatting_config(self)
```

### 3. Simplified DataflowTensorFormatter

```python
class DataflowTensorFormatter:
    """Simplified tensor formatter using explicit configuration."""
    
    def format_tensor_for_hardware(self, 
                                 tensor: np.ndarray,
                                 interface: DataflowInterface,
                                 formatting_config: TensorFormattingConfig = None) -> np.ndarray:
        """Format tensor using explicit configuration."""
        
        # Get configuration (provided > interface metadata > defaults)
        if formatting_config is None:
            formatting_config = interface.get_formatting_config()
        
        formatting_config.validate()
        
        # Apply formatting steps based on explicit configuration
        tensor = self._validate_and_prepare(tensor, interface)
        tensor = self._apply_datatype_conversion(tensor, interface.dtype)
        tensor = self._apply_preprocessing(tensor, formatting_config)
        tensor = self._distribute_across_pes(tensor, interface.stream_dims, formatting_config)
        tensor = self._generate_hardware_layout(tensor, interface)
        tensor = self._apply_optimizations(tensor, formatting_config)
        
        return tensor
    
    def _apply_preprocessing(self, tensor: np.ndarray, config: TensorFormattingConfig) -> np.ndarray:
        """Apply preprocessing based on explicit configuration."""
        
        # Custom preprocessing takes precedence
        if config.custom_preprocessing:
            tensor = config.custom_preprocessing(tensor)
            return tensor
        
        # Standard preprocessing
        if config.needs_transpose:
            tensor = tensor.T
        
        if config.needs_spatial_flattening and len(tensor.shape) >= 3:
            # Flatten spatial dimensions
            channels = tensor.shape[-3]
            spatial_dims = tensor.shape[-2:]
            spatial_flat = np.prod(spatial_dims)
            new_shape = tensor.shape[:-3] + (channels, spatial_flat)
            tensor = tensor.reshape(new_shape)
        
        return tensor
    
    def _apply_optimizations(self, tensor: np.ndarray, config: TensorFormattingConfig) -> np.ndarray:
        """Apply hardware optimizations based on explicit configuration."""
        
        if config.needs_simd_flip:
            tensor = np.flip(tensor, axis=-1)
        
        if config.custom_postprocessing:
            tensor = config.custom_postprocessing(tensor)
        
        return tensor
```

### 4. RTL Pragma Parser

```python
class RTLTensorFormatParser:
    """Parse tensor formatting configuration from RTL pragmas."""
    
    def parse_tensor_format_pragmas(self, rtl_content: str) -> Dict[str, TensorFormattingConfig]:
        """Extract tensor formatting configurations from RTL pragmas."""
        configs = {}
        
        # Pattern: @brainsmith TENSOR_FORMAT interface_name param=value param=value
        pattern = r'@brainsmith\s+TENSOR_FORMAT\s+(\w+)\s+(.+)'
        
        for match in re.finditer(pattern, rtl_content, re.MULTILINE):
            interface_name = match.group(1)
            params_str = match.group(2)
            
            config = self._parse_config_params(params_str)
            configs[interface_name] = config
        
        return configs
    
    def _parse_config_params(self, params_str: str) -> TensorFormattingConfig:
        """Parse parameter string into configuration object."""
        params = {}
        
        for param in params_str.split():
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = self._parse_value(value.strip())
        
        return TensorFormattingConfig(
            needs_transpose=params.get('transpose', False),
            needs_spatial_flattening=params.get('spatial_flatten', False),
            needs_simd_flip=params.get('simd_flip', True),
            pe_distribution=params.get('pe_distribute', True),
            memory_layout_hint=params.get('memory_layout', 'default')
        )
    
    def _parse_value(self, value: str) -> bool:
        """Parse string value to appropriate type."""
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        else:
            return value  # Return as string for non-boolean values
```

### 5. Enhanced Template Integration

```jinja2
{# Enhanced template with explicit configuration support #}
class {{ class_name }}(AutoHWCustomOp):
    """Auto-generated with explicit tensor formatting configuration"""
    
    @staticmethod
    def get_interface_metadata() -> List[InterfaceMetadata]:
        """Interface metadata with explicit tensor formatting configuration."""
        return [
            {% for interface in interfaces %}
            InterfaceMetadata(
                name="{{ interface.name }}",
                interface_type=InterfaceType.{{ interface.interface_type.name }},
                chunking_strategy={{ interface.chunking_strategy | repr }},
                datatype_constraints={{ interface.datatype_constraints | repr }},
                {% if interface.tensor_formatting_config %}
                tensor_formatting_config=TensorFormattingConfig(
                    needs_transpose={{ interface.tensor_formatting_config.needs_transpose }},
                    needs_spatial_flattening={{ interface.tensor_formatting_config.needs_spatial_flattening }},
                    needs_simd_flip={{ interface.tensor_formatting_config.needs_simd_flip }},
                    pe_distribution={{ interface.tensor_formatting_config.pe_distribution }}
                )
                {% endif %}
            ),
            {% endfor %}
        ]
    
    # Optional: Allow user override of tensor formatting configuration
    def get_tensor_formatting_config(self, interface_name: str) -> TensorFormattingConfig:
        """Override tensor formatting configuration if needed."""
        # Default implementation uses interface metadata
        interface_metadata = next((im for im in self.get_interface_metadata() 
                                 if im.name == interface_name), None)
        if interface_metadata:
            return interface_metadata.get_formatting_config()
        
        # Fallback to safe defaults
        return TensorFormattingConfig()
```

## Benefits Analysis

### 1. Technical Benefits

**Robustness**
- ✅ **No Assumptions**: RTL authors explicitly specify requirements
- ✅ **No Guessing**: Eliminates brittle inference logic  
- ✅ **Extensible**: Easy to add new formatting options
- ✅ **Debuggable**: Clear configuration makes debugging easier

**Maintainability** 
- ✅ **Simpler Code**: Eliminates complex operation type inference
- ✅ **Clear Logic**: Explicit boolean flags instead of string matching
- ✅ **Fewer Edge Cases**: No special cases for different operation types
- ✅ **Local Changes**: Formatting changes are made where RTL lives

**Performance**
- ✅ **Faster**: No complex pattern matching or inference
- ✅ **Deterministic**: Same configuration always produces same result
- ✅ **Cacheable**: Configuration can be cached and reused

### 2. User Experience Benefits

**Control**
- ✅ **Explicit Specification**: RTL authors control their formatting requirements
- ✅ **Easy Override**: Users can override defaults when needed
- ✅ **Custom Operations**: Supports novel operations not covered by defaults

**Documentation**
- ✅ **Self-Documenting**: RTL pragmas document formatting requirements
- ✅ **Clear Dependencies**: Easy to see what formatting a kernel needs
- ✅ **Version Control**: Changes to formatting are tracked with RTL changes

**Debugging**
- ✅ **Easy Testing**: Can disable individual transformations
- ✅ **Clear Errors**: Configuration validation provides clear error messages
- ✅ **Reproducible**: Same configuration always produces same result

### 3. Development Benefits

**Faster Development**
- ✅ **No Guessing Logic**: Eliminates time spent on inference algorithms
- ✅ **Explicit Requirements**: RTL authors specify exactly what they need
- ✅ **Fewer Bugs**: Explicit configuration reduces edge cases

**Better Testing**
- ✅ **Testable**: Can test specific configurations in isolation
- ✅ **Predictable**: Known configuration produces known results
- ✅ **Comprehensive**: Can test all configuration combinations

## Migration Strategy

### Phase 1: Implement Configuration System
1. Create `TensorFormattingConfig` class
2. Enhance `InterfaceMetadata` with configuration support
3. Update `DataflowTensorFormatter` to use explicit configuration
4. Create RTL pragma parser

### Phase 2: Update Templates
1. Enhance templates to include configuration in interface metadata
2. Add user override capability to generated classes
3. Update generation pipeline to parse RTL pragmas

### Phase 3: Migrate Existing Operations
1. Add RTL pragmas to existing kernels
2. Update interface metadata to include explicit configuration
3. Remove operation type inference logic
4. Remove `EnhancedInterfaceAnalyzer`

### Phase 4: Validation and Documentation
1. Comprehensive testing of all configuration combinations
2. Update documentation with new pragma syntax
3. Create migration guide for existing users

## Risk Analysis

### Technical Risks

**Risk**: RTL authors might forget to specify pragmas
**Mitigation**: Provide intelligent defaults based on interface characteristics

**Risk**: Configuration conflicts or invalid combinations  
**Mitigation**: Add validation logic to `TensorFormattingConfig.validate()`

**Risk**: Complex custom preprocessing functions
**Mitigation**: Provide common preprocessing functions as utilities

### Adoption Risks

**Risk**: Learning curve for RTL pragma syntax
**Mitigation**: Simple, clear syntax with good documentation and examples

**Risk**: Existing code needs migration
**Mitigation**: Backwards compatibility through intelligent defaults

## Comparison with Current Approach

| Aspect | Current (Operation Type) | Proposed (Explicit Config) |
|--------|-------------------------|----------------------------|
| **Robustness** | Brittle assumptions | Explicit specification |
| **Maintainability** | Complex inference logic | Simple boolean logic |
| **User Control** | Limited (can't override) | Full (RTL pragmas + overrides) |
| **Extensibility** | Hard (new operation types) | Easy (new config options) |
| **Debugging** | Difficult (hidden inference) | Easy (explicit configuration) |
| **Performance** | Slower (inference overhead) | Faster (direct configuration) |
| **Documentation** | Hidden in code | Self-documenting pragmas |

## Conclusion

The proposed explicit tensor formatting configuration system is superior to operation type guessing in every measurable way:

1. **More Robust**: No brittle assumptions or guessing
2. **More Maintainable**: Simpler logic, fewer edge cases
3. **More User-Friendly**: Explicit control with reasonable defaults
4. **More Extensible**: Easy to add new formatting options
5. **Better Performance**: No inference overhead

The current operation type approach is solving a 4-line problem with 200+ lines of complex inference machinery. The proposed approach solves the same problem with explicit, maintainable, user-controllable configuration.

**Recommendation**: Proceed with implementation of explicit tensor formatting configuration system and remove operation type guessing entirely.