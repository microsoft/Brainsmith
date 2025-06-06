# Enhanced Tensor Chunking Specification

## Overview

This document specifies the enhanced tensor chunking methods required to support the new TDIM pragma syntax: `// @brainsmith TDIM intf_name shape index`.

The enhanced approach moves from explicit dimension specification to a flexible shape + index strategy that works with both runtime tensor shapes and parameter-defined shapes.

## Enhanced TDIM Pragma Syntax

### Current Syntax (To Be Deprecated)
```systemverilog
// @brainsmith TDIM in0_V_data_V PE*CHANNELS 1
// @brainsmith TDIM weights BATCH_SIZE*FEATURES HIDDEN_DIM
```

### Enhanced Syntax (New Approach)
```systemverilog
// @brainsmith TDIM intf_name index

// Examples:
// @brainsmith TDIM in0_V_data_V -1        // Chunk at last dimension (shape auto-extracted)
// @brainsmith TDIM weights -2             // Chunk at 2nd-to-last dimension
// @brainsmith TDIM config 0               // Chunk at first dimension
```

### Syntax Components

1. **`intf_name`**: Interface name (unchanged)
2. **`index`**: Dimension index to chunk (Python-style indexing)
   - Positive: `0` = first dimension, `1` = second dimension, etc.
   - Negative: `-1` = last dimension, `-2` = second-to-last dimension, etc.

### Shape Extraction Strategy

The enhanced system **automatically extracts tensor shapes** from the input tensors in the ONNX graph, eliminating the need for manual shape specification:

- **Input interfaces**: Shape extracted from corresponding input tensor
- **Output interfaces**: Shape typically matches input tensor or can be computed from input
- **Weight interfaces**: Shape extracted from weight tensor (if available) or parameter-defined
- **Layout inference**: Smart defaults based on tensor dimensions (4D→NCHW, 3D→CHW, 2D→NC, 1D→C)

## Enhanced Tensor Chunking Methods

The following methods need to be added to `brainsmith/dataflow/core/tensor_chunking.py`:

### 1. Index-Based Chunking Strategy

```python
def apply_index_chunking_strategy(self, tensor_shape: List[int], tensor_layout: str,
                                chunk_index: int) -> Tuple[List[int], List[int]]:
    """
    Apply index-based chunking strategy from enhanced TDIM pragma.
    
    Args:
        tensor_shape: Tensor shape extracted from input tensor
        tensor_layout: Tensor layout (e.g., "NCHW", "NHWC") - inferred or specified
        chunk_index: Dimension index to chunk (Python-style indexing)
        
    Returns:
        Tuple of (qDim, tDim) lists
        
    Examples:
        tensor_shape=[1, 8, 32, 32], chunk_index=-1 → qDim=[1, 8, 32, 1], tDim=[1, 1, 1, 32]
        tensor_shape=[128, 64], chunk_index=0 → qDim=[1, 64], tDim=[128, 1]
    """
    if not tensor_shape:
        return [1], [1]
        
    # Normalize negative indices
    normalized_index = chunk_index
    if chunk_index < 0:
        normalized_index = len(tensor_shape) + chunk_index
        
    # Validate index bounds
    if not (0 <= normalized_index < len(tensor_shape)):
        raise ValueError(f"Chunk index {chunk_index} out of bounds for shape {tensor_shape}")
    
    # Create qDim and tDim based on chunking at the specified index
    qDim = [1] * len(tensor_shape)
    tDim = [1] * len(tensor_shape)
    
    # Set the chunked dimension
    qDim[normalized_index] = tensor_shape[normalized_index]
    tDim[normalized_index] = 1
    
    # Set all other dimensions in tDim to their full size
    for i, dim_size in enumerate(tensor_shape):
        if i != normalized_index:
            tDim[i] = dim_size
            qDim[i] = 1
    
    return qDim, tDim

def extract_tensor_shape_from_input(self, interface_name: str, onnx_node) -> List[int]:
    """
    Extract tensor shape from the corresponding input tensor in the ONNX graph.
    
    Args:
        interface_name: Interface name (e.g., "in0_V_data_V")
        onnx_node: ONNX node containing input tensor references
        
    Returns:
        List of tensor dimensions
        
    Examples:
        interface_name="in0_V_data_V" → extracts shape from input_tensor[0]
        interface_name="weights" → extracts shape from weight tensor
    """
    # Map interface name to input tensor index
    input_index = self._map_interface_to_input_index(interface_name)
    
    if input_index is None:
        raise ValueError(f"Cannot map interface {interface_name} to input tensor")
    
    try:
        # Get input tensor name
        input_tensor_name = onnx_node.input[input_index]
        
        # Extract shape from tensor (implementation depends on FINN's model wrapper)
        if hasattr(self, '_model_wrapper') and self._model_wrapper:
            tensor_shape = self._model_wrapper.get_tensor_shape(input_tensor_name)
            if tensor_shape:
                return [int(dim) for dim in tensor_shape]
        
        # Fallback: try to get shape from node attributes
        shape_attr = f"input_{input_index}_shape"
        fallback_shape = getattr(onnx_node, shape_attr, None)
        if fallback_shape:
            return fallback_shape
            
        raise ValueError(f"Cannot extract shape for tensor {input_tensor_name}")
        
    except (IndexError, AttributeError) as e:
        raise ValueError(f"Failed to extract tensor shape for interface {interface_name}: {e}")

def _map_interface_to_input_index(self, interface_name: str) -> Optional[int]:
    """Map interface name to input tensor index using naming conventions."""
    # Standard mapping based on interface naming patterns
    if "in0" in interface_name or interface_name.startswith("input"):
        return 0
    elif "in1" in interface_name:
        return 1
    elif "weights" in interface_name or "weight" in interface_name:
        return 1  # Typically the second input
    else:
        return 0  # Default to first input

def infer_layout_from_shape(self, tensor_shape: List[int]) -> str:
    """Infer tensor layout from shape dimensions with smart defaults."""
    if len(tensor_shape) == 4:
        return "NCHW"  # Default for 4D tensors (batch, channel, height, width)
    elif len(tensor_shape) == 3:
        return "CHW"   # Default for 3D tensors (channel, height, width)
    elif len(tensor_shape) == 2:
        return "NC"    # Default for 2D tensors (batch, features)
    elif len(tensor_shape) == 1:
        return "C"     # Default for 1D tensors (features)
    else:
        return "UNKNOWN"
```

### 2. Simplified Pragma Data Processing

```python
def process_enhanced_tdim_pragma(self, pragma_data: Dict[str, Any],
                               interface_name: str,
                               onnx_node) -> Tuple[List[int], List[int]]:
    """
    Process enhanced TDIM pragma data to compute qDim and tDim.
    
    Args:
        pragma_data: Parsed TDIM pragma data from RTL analysis
        interface_name: Interface name for tensor shape extraction
        onnx_node: ONNX node for tensor access
        
    Returns:
        Tuple of (qDim, tDim) lists
        
    Example pragma_data:
        {
            "chunk_index": -1
        }
    """
    chunk_index = pragma_data.get("chunk_index")
    
    if chunk_index is None:
        # Fallback to default chunking
        tensor_shape = self.extract_tensor_shape_from_input(interface_name, onnx_node)
        tensor_layout = self.infer_layout_from_shape(tensor_shape)
        return self.infer_dimensions_with_layout(tensor_layout, tensor_shape)
    
    # Extract tensor shape automatically
    tensor_shape = self.extract_tensor_shape_from_input(interface_name, onnx_node)
    tensor_layout = self.infer_layout_from_shape(tensor_shape)
    
    # Apply index-based chunking
    return self.apply_index_chunking_strategy(tensor_shape, tensor_layout, chunk_index)
```

### 3. Layout-Aware Index Optimization

```python
def optimize_chunk_index_for_layout(self, tensor_layout: str, chunk_index: int, 
                                  shape_length: int) -> int:
    """
    Optimize chunk index based on tensor layout for better performance.
    
    Args:
        tensor_layout: Tensor layout (e.g., "NCHW", "NHWC")
        chunk_index: Requested chunk index
        shape_length: Length of tensor shape
        
    Returns:
        Optimized chunk index
        
    Note: This method can suggest better chunking strategies based on layout,
    but respects explicit user choices from TDIM pragmas.
    """
    # Normalize negative index
    normalized_index = chunk_index
    if chunk_index < 0:
        normalized_index = shape_length + chunk_index
    
    # Layout-specific optimizations (optional - for suggestions only)
    if tensor_layout == "NCHW" and shape_length >= 4:
        # For NCHW, chunking the width (last dim) is often optimal for streaming
        if normalized_index == shape_length - 1:  # Width dimension
            return chunk_index  # Already optimal
        # Could log suggestions for better performance
        
    elif tensor_layout == "NHWC" and shape_length >= 4:
        # For NHWC, chunking channels (last dim) enables channel parallelism
        if normalized_index == shape_length - 1:  # Channel dimension
            return chunk_index  # Already optimal
    
    # Return original index - respect user choice
    return chunk_index
```

### 4. Validation and Error Handling

```python
def validate_enhanced_tdim_pragma(self, pragma_data: Dict[str, Any],
                                interface_name: str) -> List[str]:
    """
    Validate enhanced TDIM pragma data for correctness.
    
    Args:
        pragma_data: Parsed TDIM pragma data
        interface_name: Interface name for error reporting
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    chunk_index = pragma_data.get("chunk_index")
    
    # Validate chunk_index (the only required parameter now)
    if chunk_index is None:
        errors.append(f"Interface {interface_name}: TDIM pragma missing chunk_index")
    elif not isinstance(chunk_index, int):
        errors.append(f"Interface {interface_name}: TDIM pragma chunk_index must be integer")
    
    return errors
```

## RTL Parser Integration

The enhanced TDIM pragma parsing needs to be integrated into the RTL parser pipeline:

### 1. Enhanced TDimPragma Class Updates

Update `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`:

```python
@dataclass
class TDimPragma(Pragma):
    """Enhanced TDIM pragma for shape + index chunking specification."""
    
    def _parse_inputs(self) -> Dict:
        """
        Parse enhanced TDIM pragma: @brainsmith TDIM <interface_name> <shape> <index>
        
        Examples:
            @brainsmith TDIM in0_V_data_V [:] -1
            @brainsmith TDIM weights [C, H, W] -2
        """
        if len(self.inputs) != 3:
            raise PragmaError("Enhanced TDIM pragma requires interface_name, shape, and index")
        
        interface_name = self.inputs[0]
        shape_str = self.inputs[1]
        index_str = self.inputs[2]
        
        # Parse shape specification
        if shape_str == "[:]":
            shape_spec = "runtime"
        elif shape_str.startswith("[") and shape_str.endswith("]"):
            # Parameter list: [param1, param2, ...]
            param_names = [p.strip() for p in shape_str[1:-1].split(",")]
            shape_spec = param_names  # Will be resolved later with parameter values
        else:
            raise PragmaError(f"Invalid shape specification: {shape_str}")
        
        # Parse chunk index
        try:
            chunk_index = int(index_str)
        except ValueError:
            raise PragmaError(f"Invalid chunk index: {index_str}")
        
        return {
            "interface_name": interface_name,
            "shape_spec": shape_spec,
            "chunk_index": chunk_index
        }
    
    def apply(self, **kwargs) -> Any:
        """Apply enhanced TDIM pragma with parameter resolution."""
        interfaces = kwargs.get('interfaces')
        parameters = kwargs.get('parameters', {})
        
        if not interfaces:
            return
            
        interface_name = self.parsed_data.get("interface_name")
        shape_spec = self.parsed_data.get("shape_spec")
        chunk_index = self.parsed_data.get("chunk_index")
        
        # Find target interface
        target_interface = None
        for iface in interfaces.values():
            if iface.name == interface_name:
                target_interface = iface
                break
                
        if not target_interface:
            logger.warning(f"Enhanced TDIM pragma: interface '{interface_name}' not found")
            return
        
        # Resolve parameter-based shape if needed
        resolved_shape_spec = shape_spec
        if isinstance(shape_spec, list):
            # Resolve parameter names to values
            resolved_shape = []
            for param_name in shape_spec:
                if param_name in parameters:
                    param_value = parameters[param_name]
                    try:
                        resolved_shape.append(int(param_value))
                    except (ValueError, TypeError):
                        logger.error(f"Enhanced TDIM pragma: parameter '{param_name}' has invalid value: {param_value}")
                        return
                else:
                    logger.error(f"Enhanced TDIM pragma: parameter '{param_name}' not found")
                    return
            resolved_shape_spec = resolved_shape
        
        # Store enhanced pragma data
        target_interface.metadata["enhanced_tdim"] = {
            "shape_spec": resolved_shape_spec,
            "chunk_index": chunk_index
        }
        
        logger.info(f"Applied enhanced TDIM pragma: {interface_name} shape={resolved_shape_spec} index={chunk_index}")
```

## Integration with AutoHWCustomOp

The enhanced tensor chunking integrates with the refactored AutoHWCustomOp:

```python
def _compute_dimensions_from_tensor_info(self, interface_name: str, tensor_shape: List[int],
                                       tensor_layout: str, metadata: 'InterfaceMetadata') -> Tuple[List[int], List[int]]:
    """Compute qDim and tDim using enhanced chunking logic."""
    from brainsmith.dataflow.core.tensor_chunking import TensorChunking
    
    # Check for enhanced TDIM pragma
    enhanced_tdim = metadata.pragma_metadata.get("enhanced_tdim")
    
    if enhanced_tdim:
        # Use enhanced TDIM pragma
        chunker = TensorChunking()
        qDim, tDim = chunker.process_enhanced_tdim_pragma(
            enhanced_tdim, tensor_shape, tensor_layout
        )
    else:
        # Use default layout-based chunking
        chunker = TensorChunking()
        qDim, tDim = chunker.infer_dimensions_with_layout(tensor_layout, tensor_shape)
    
    return qDim, tDim
```

## Migration Strategy

### Phase 1: Backward Compatibility
- Keep existing TDIM pragma parsing for compatibility
- Add enhanced TDIM pragma parsing alongside
- Both syntaxes work during transition period

### Phase 2: Enhanced Feature Rollout
- Update HKG templates to use enhanced pragma data
- Update documentation with new syntax examples
- Provide migration tools for existing RTL kernels

### Phase 3: Deprecation
- Mark old TDIM syntax as deprecated
- Provide warnings for old syntax usage
- Eventually remove old syntax support

## Testing Requirements

### Unit Tests
- Test `apply_index_chunking_strategy()` with various shapes and indices
- Test `process_enhanced_tdim_pragma()` with different pragma configurations
- Test validation methods with invalid inputs

### Integration Tests
- Test RTL parsing with enhanced TDIM pragmas
- Test end-to-end chunking from RTL to AutoHWCustomOp
- Test parameter resolution for shape specifications

### Example Test Cases

```python
def test_index_chunking_strategy():
    chunker = TensorChunking()
    
    # Test last dimension chunking
    qDim, tDim = chunker.apply_index_chunking_strategy([1, 8, 32, 32], "NCHW", -1)
    assert qDim == [1, 8, 32, 1]
    assert tDim == [1, 1, 1, 32]
    
    # Test first dimension chunking
    qDim, tDim = chunker.apply_index_chunking_strategy([128, 64], "NC", 0)
    assert qDim == [1, 64]
    assert tDim == [128, 1]

def test_enhanced_pragma_processing():
    chunker = TensorChunking()
    
    pragma_data = {
        "shape_spec": "runtime",
        "chunk_index": -1
    }
    
    qDim, tDim = chunker.process_enhanced_tdim_pragma(
        pragma_data, [1, 8, 32, 32], "NCHW"
    )
    
    assert qDim == [1, 8, 32, 1]
    assert tDim == [1, 1, 1, 32]
```

This enhanced tensor chunking specification provides a flexible, powerful approach to tensor dimension management that integrates seamlessly with FINN's workflow while providing RTL developers with intuitive control over chunking strategies.