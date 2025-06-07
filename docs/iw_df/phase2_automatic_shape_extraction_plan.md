# Phase 2: Automatic Tensor Shape Extraction and HWKG Integration

## Overview

With the per-interface chunking strategy architecture complete, Phase 2 focuses on implementing automatic tensor shape extraction and enabling HWKG pragma parsing to create chunking strategies. This phase transforms the system from requiring manual tensor configuration to zero-configuration operation.

## Phase 2 Objectives

1. **Automatic Shape Extraction**: Extract tensor shapes from ONNX model inputs automatically
2. **Smart Layout Inference**: Infer NCHW/CHW/NC layouts from tensor dimensions  
3. **ModelWrapper Integration**: Connect to FINN's model representation for accurate shapes
4. **HWKG Pragma Integration**: Parse RTL pragmas and create chunking strategies
5. **Zero-Configuration Operation**: Enable users to create nodes without manual qDim/tDim

## Current Architecture Status

✅ **Completed in Phase 1:**
- Per-interface chunking strategy pattern
- [`ChunkingStrategy`](brainsmith/dataflow/core/chunking_strategy.py:1) base class with concrete implementations
- [`InterfaceMetadata`](brainsmith/dataflow/core/interface_metadata.py:48) with strategy ownership
- Simplified [`TensorChunking`](brainsmith/dataflow/core/tensor_chunking.py:1) delegation system
- 47/47 tests passing with comprehensive coverage

🎯 **Phase 2 Tasks:**
- Enhance tensor shape extraction from ONNX inputs
- Add ModelWrapper integration for accurate shapes  
- Implement smart layout inference (4D→NCHW, 3D→CHW, etc.)
- Create HWKG pragma parser for strategy generation
- Enable zero-configuration FINN workflow integration

## Implementation Tasks

### Task 2.1: Enhanced Tensor Shape Extraction

**Files to Modify:**
- [`brainsmith/dataflow/core/tensor_chunking.py`](brainsmith/dataflow/core/tensor_chunking.py:1)

**Current State:**
```python
def get_default_shape_for_interface(self, interface_name: str) -> List[int]:
    """Get default tensor shape for interface (placeholder)."""
    return [1, 8, 32, 32]  # Fixed default
```

**Target Implementation:**
```python
def extract_tensor_shape_from_input(self, interface_name: str) -> List[int]:
    """Extract actual tensor shape from ONNX input or ModelWrapper."""
    if self._model_wrapper:
        # Get shape from ModelWrapper (FINN integration)
        input_index = self._map_interface_to_input_index(interface_name)
        if input_index is not None:
            tensor_name = self._model_wrapper.graph.input[input_index].name
            return self._model_wrapper.get_tensor_shape(tensor_name)
    
    # Fallback to ONNX node analysis
    return self._extract_shape_from_onnx_node(interface_name)

def _map_interface_to_input_index(self, interface_name: str) -> Optional[int]:
    """Map interface name to ONNX input index using naming patterns."""
    # Common patterns: in0_V_data_V → input[0], in1_V_data_V → input[1]
    # Implementation here

def _extract_shape_from_onnx_node(self, interface_name: str) -> List[int]:
    """Extract shape from ONNX node attributes or value_info."""
    # Implementation here
```

**Acceptance Criteria:**
- [ ] Extract shapes from ModelWrapper when available
- [ ] Fallback to ONNX node attribute parsing
- [ ] Handle interface name → input index mapping
- [ ] Support common RTL interface naming patterns
- [ ] Return sensible defaults when extraction fails

### Task 2.2: Smart Layout Inference

**Files to Modify:**
- [`brainsmith/dataflow/core/tensor_chunking.py`](brainsmith/dataflow/core/tensor_chunking.py:1)

**Target Implementation:**
```python
def infer_layout_from_shape(self, tensor_shape: List[int]) -> str:
    """Infer tensor layout with smart defaults."""
    layout_map = {
        4: "NCHW",  # 4D tensors → batch, channels, height, width
        3: "CHW",   # 3D tensors → channels, height, width  
        2: "NC",    # 2D tensors → batch, channels
        1: "C"      # 1D tensors → channels
    }
    return layout_map.get(len(tensor_shape), f"DIM{len(tensor_shape)}")

def get_default_chunking_for_layout(self, layout: str, tensor_shape: List[int]) -> Tuple[List[int], List[int]]:
    """Provide layout-aware default chunking strategies."""
    if layout == "NCHW":
        # Default: no chunking on batch/channels, stream on width
        return ([1, 1, 1, tensor_shape[3]], [1, tensor_shape[1], tensor_shape[2], 1])
    elif layout == "CHW": 
        # Default: stream on width
        return ([1, 1, tensor_shape[2]], [tensor_shape[0], tensor_shape[1], 1])
    elif layout == "NC":
        # Default: stream on channels
        return ([1, tensor_shape[1]], [tensor_shape[0], 1])
    else:
        # Conservative default: full tensor
        return ([1] * len(tensor_shape), tensor_shape)
```

**Acceptance Criteria:**
- [ ] Correct layout inference for common tensor dimensions
- [ ] Layout-aware chunking defaults that make sense
- [ ] Support for unusual tensor dimensions
- [ ] Comprehensive test coverage for all layout types

### Task 2.3: ModelWrapper Integration

**Files to Modify:**
- [`brainsmith/dataflow/core/auto_hw_custom_op.py`](brainsmith/dataflow/core/auto_hw_custom_op.py:204)

**Current State:**
```python
def _build_dataflow_model(self):
    """Build DataflowModel from interface metadata."""
    # Basic implementation without ModelWrapper
```

**Target Implementation:**
```python
def _build_dataflow_model(self):
    """Build DataflowModel with automatic shape extraction."""
    interfaces = {}
    
    for metadata in self.interface_metadata.interfaces:
        # Extract actual tensor shape
        tensor_shape = self.tensor_chunking.extract_tensor_shape_from_input(metadata.name)
        
        # Compute chunking using interface strategy
        qDim, tDim = self.tensor_chunking.compute_chunking_for_interface(metadata, tensor_shape)
        
        # Create DataflowInterface with extracted information
        interfaces[metadata.name] = DataflowInterface(
            interface_name=metadata.name,
            interface_type=metadata.interface_type,
            qDim=qDim,
            tDim=tDim,
            # ... other properties
        )
    
    self._dataflow_model = DataflowModel(self.onnx_node, interfaces)

def set_model_wrapper(self, model_wrapper):
    """Set ModelWrapper for accurate tensor shape extraction."""
    self.tensor_chunking._model_wrapper = model_wrapper
    self._invalidate_dataflow_model()  # Rebuild with new information
```

**Acceptance Criteria:**
- [ ] Integrate with FINN's ModelWrapper for shape extraction
- [ ] Automatic shape extraction during DataflowModel building
- [ ] Model invalidation when ModelWrapper changes
- [ ] Fallback behavior when ModelWrapper unavailable

### Task 2.4: HWKG Pragma Integration

**Files to Create:**
- `brainsmith/tools/hw_kernel_gen/pragma_to_strategy.py`

**Target Implementation:**
```python
from brainsmith.dataflow.core.chunking_strategy import (
    default_chunking, index_chunking, last_dim_chunking, 
    spatial_chunking, FullTensorChunkingStrategy
)

class PragmaToStrategyConverter:
    """Convert parsed RTL pragmas to chunking strategies."""
    
    def convert_tdim_pragma(self, pragma_data: Dict[str, Any]) -> ChunkingStrategy:
        """Convert TDIM pragma to appropriate chunking strategy."""
        pragma_type = pragma_data.get('type', 'default')
        
        if pragma_type == 'index':
            return index_chunking(
                pragma_data['start_index'], 
                pragma_data['shape']
            )
        elif pragma_type == 'spatial':
            return spatial_chunking(
                pragma_data['height'],
                pragma_data['width'] 
            )
        elif pragma_type == 'last_dim':
            return last_dim_chunking(pragma_data['chunk_size'])
        elif pragma_type == 'none':
            return FullTensorChunkingStrategy()
        else:
            return default_chunking()
    
    def parse_enhanced_tdim_pragma(self, pragma_string: str) -> Dict[str, Any]:
        """Parse enhanced TDIM pragma string."""
        # Parse: "@brainsmith TDIM in0_V_data_V -1 [16]"
        # Parse: "@brainsmith TDIM weights spatial 8x8" 
        # Parse: "@brainsmith TDIM bias none"
        # Implementation here
```

**Files to Modify:**
- `brainsmith/tools/hw_kernel_gen/templates/hw_custom_op.py.j2`

**Target Template:**
```jinja2
class {{ class_name }}(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        # Import chunking strategies
        from brainsmith.dataflow.core.chunking_strategy import (
            default_chunking, index_chunking, spatial_chunking, FullTensorChunkingStrategy
        )
        
        self._interface_metadata = [
            {% for interface in interfaces %}
            InterfaceMetadata(
                name="{{ interface.name }}",
                interface_type=DataflowInterfaceType.{{ interface.type }},
                allowed_datatypes=[
                    {% for dtype in interface.datatypes %}
                    DataTypeConstraint(
                        finn_type="{{ dtype.finn_type }}", 
                        bit_width={{ dtype.bit_width }}
                    ),
                    {% endfor %}
                ],
                chunking_strategy={{ interface.chunking_strategy_code }}
            ),
            {% endfor %}
        ]
        super().__init__(onnx_node, **kwargs)
```

**Acceptance Criteria:**
- [ ] Parse enhanced TDIM pragma syntax correctly
- [ ] Convert pragmas to appropriate chunking strategies
- [ ] Generate clean template code with strategy objects
- [ ] Support all common pragma patterns (index, spatial, none, default)

### Task 2.5: Zero-Configuration FINN Integration

**Files to Create:**
- `examples/zero_config_finn_demo.py`

**Target Usage:**
```python
# Zero-configuration usage - shapes extracted automatically
node = onnx.helper.make_node(
    "ThresholdingAxi",
    inputs=["input_tensor"],      # Shape extracted from input
    outputs=["output_tensor"], 
    in0_V_data_V_dtype="UINT8"    # Only datatype required
    # No qDim, tDim, or shape needed!
)

# Traditional usage still supported
node = onnx.helper.make_node(
    "ThresholdingAxi", 
    inputs=["input_tensor"],
    outputs=["output_tensor"],
    qDim=[1, 1, 1, 32],          # Manual override if needed
    tDim=[1, 8, 32, 1],
    in0_V_data_V_dtype="UINT8"
)
```

**Integration Test:**
```python
def test_zero_config_finn_workflow():
    """Test complete zero-configuration FINN workflow."""
    # Create model with input tensor
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 8, 32, 32]
    )
    
    # Create node without manual configuration
    node = onnx.helper.make_node(
        "ThresholdingAxi",
        inputs=["input"],
        outputs=["output"], 
        in0_V_data_V_dtype="UINT8"
    )
    
    # Create ModelWrapper and AutoHWCustomOp
    model = onnx.helper.make_model(
        onnx.helper.make_graph([node], "test", [input_tensor], [])
    )
    model_wrapper = ModelWrapper(model)
    op = ThresholdingAxi(node)
    op.set_model_wrapper(model_wrapper)
    
    # Verify automatic shape extraction
    dataflow_model = op.get_dataflow_model()
    assert dataflow_model.get_tensor_shape("input") == [1, 8, 32, 32]
    
    # Verify automatic chunking
    interface = dataflow_model.get_interface("in0_V_data_V")
    assert interface.qDim is not None
    assert interface.tDim is not None
```

**Acceptance Criteria:**
- [ ] Zero-configuration node creation works
- [ ] Automatic shape extraction from input tensors
- [ ] Backward compatibility with manual configuration
- [ ] Integration with FINN transformation passes
- [ ] Comprehensive end-to-end testing

## Testing Strategy

### Unit Tests
- [ ] Tensor shape extraction with various ONNX structures
- [ ] Layout inference for all supported dimensions
- [ ] Pragma parsing and strategy conversion
- [ ] ModelWrapper integration edge cases

### Integration Tests  
- [ ] Complete HWKG pipeline with pragma parsing
- [ ] FINN workflow compatibility testing
- [ ] Zero-configuration vs manual configuration comparison
- [ ] Performance benchmarking

### Validation Tests
- [ ] Real RTL parsing with enhanced TDIM pragmas
- [ ] Generated class functionality verification
- [ ] Backward compatibility with existing code
- [ ] Error handling and graceful fallbacks

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Zero-Config Usage | 90%+ scenarios | Test coverage |
| Shape Extraction Accuracy | 95%+ | Validation tests |
| FINN Compatibility | 100% | Integration tests |
| Pragma Parsing Success | 100% | RTL parser tests |
| Performance | No regression | Benchmark comparison |

## Risk Assessment

### High Risk
- **Complex ONNX Shape Extraction**: Some models may have unusual tensor configurations
  - *Mitigation*: Comprehensive fallback strategies and extensive test coverage

### Medium Risk  
- **RTL Pragma Complexity**: Enhanced pragma syntax may be difficult to parse
  - *Mitigation*: Start with simple patterns, expand gradually

### Low Risk
- **Backward Compatibility**: Manual configuration should continue working
  - *Mitigation*: Extensive regression testing

## Timeline

**Estimated Duration:** 3-4 weeks

- **Week 1:** Enhanced shape extraction and layout inference
- **Week 2:** ModelWrapper integration and chunking improvements
- **Week 3:** HWKG pragma integration and template updates
- **Week 4:** Testing, validation, and documentation

## Next Phase Preview

**Phase 3:** HWKG Template Optimization and Code Generation
- Slim template generation (50-80 lines vs 300+)
- Enhanced pragma syntax finalization
- Automated class migration tools
- Performance optimization and benchmarking

This phase will deliver the zero-configuration experience that transforms AutoHWCustomOp from a manual, verbose system into an elegant, automatic solution.