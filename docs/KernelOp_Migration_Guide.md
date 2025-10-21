# KernelOp Migration Guide for FINN Developers

## Executive Summary

**KernelOp** replaces manual shape calculations, datatype inference, and validation with a declarative **KernelSchema** that:
- Automatically computes folded shapes via template resolution
- Validates constraints during model building (not at runtime)
- Caches tensor context and kernel models for performance
- Eliminates ~60% of boilerplate code per kernel

**Migration effort**: 2-4 hours per kernel, depending on complexity.

---

## Overview: What Changes

### Before (Manual HWCustomOp)
```python
class MVAU(HWCustomOp):
    def get_nodeattr_types(self):
        return {
            "PE": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "MW": ("i", True, 0),
            "MH": ("i", True, 0),
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # ... 15 more attributes
        }

    def get_folded_input_shape(self, ind=0):
        mw = self.get_nodeattr("MW")
        simd = self.get_nodeattr("SIMD")
        vecs = self.get_nodeattr("numInputVectors")
        sf = mw // simd
        assert mw % simd == 0, "SIMD must divide MW"
        return tuple(vecs + [sf, simd])

    def get_folded_output_shape(self, ind=0):
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        vecs = self.get_nodeattr("numInputVectors")
        nf = mh // pe
        assert mh % pe == 0, "PE must divide MH"
        return tuple(vecs + [nf, pe])

    def verify_node(self):
        # Manual checking of 20+ conditions...
        pass
```

### After (KernelOp)
```python
MVAU_SCHEMA = KernelSchema(
    name="MVAU",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=[":"],               # Copy tensor shape
            stream_tiling=["SIMD"],           # Fold by SIMD
            datatype_attr="inputDataType",
            constraints=[
                DimensionDivisible("input", -1, "SIMD", ShapeHierarchy.STREAM)
            ]
        ),
        InputSchema(
            name="weights",
            is_weight=True,
            # ... weight-specific config
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=[":"],
            stream_tiling=["PE"],
            datatype_attr="outputDataType",
            constraints=[
                DimensionDivisible("output", -1, "PE", ShapeHierarchy.STREAM)
            ]
        )
    ]
)

@kernel(description="Matrix-Vector-Activation Unit")
class MVAU(KernelOp):
    kernel_schema = MVAU_SCHEMA

    # get_folded_input_shape() inherited, automatic
    # get_folded_output_shape() inherited, automatic
    # verify_node() inherited, automatic from constraints
```

**Key benefits**:
- **Declarative**: Schema is data, not imperative code
- **Automatic**: Shape calculations, validation, caching built-in
- **Consistent**: Same API across all kernels
- **Debuggable**: Validation errors show location and suggestions

---

## Migration Taxonomy: 8 Operation Categories

### Category 1: Shape-Preserving, Single Input/Output
**Examples**: LayerNorm, Thresholding, StreamingEltwise

**Characteristics**:
- Input shape == output shape
- Single tensor in, single tensor out
- PE-folded streaming

**Migration complexity**: ⭐ Easy (1-2 hours)

**Key challenges**:
- Translate PE/SIMD folding to `stream_tiling`
- Express divisibility constraints

**Schema pattern**:
```python
KernelSchema(
    name="LayerNorm",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=[":"],               # Preserve tensor shape
            stream_tiling=["SIMD"],           # PE-fold last dimension
            constraints=[
                DimensionDivisible("input", -1, "SIMD", ShapeHierarchy.STREAM)
            ]
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=[":"],               # Same as input
            stream_tiling=[DerivedDim("input", -1)],  # Match input streaming
        )
    ]
)
```

---

### Category 2: Multi-Input Operations
**Examples**: AddStreams, ElementwiseBinary

**Characteristics**:
- Multiple data inputs (not weights)
- Output datatype may be derived from inputs
- All inputs must have compatible shapes

**Migration complexity**: ⭐⭐ Moderate (2-3 hours)

**Key challenges**:
- Handle `inputDataTypes` list (multiple datatypes)
- Express output datatype derivation
- Ensure input shape compatibility

**Schema pattern**:
```python
KernelSchema(
    name="AddStreams",
    inputs=[
        InputSchema(
            name="input0",
            block_tiling=[":"],
            stream_tiling=["PE"],
            datatype_attr="inputDataType0"
        ),
        InputSchema(
            name="input1",
            block_tiling=[":"],
            stream_tiling=["PE"],
            datatype_attr="inputDataType1"
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=[":"],
            stream_tiling=[DerivedDim("input0", -1)],
            datatype_attr="outputDataType"
        )
    ]
)
```

**Special handling**: Output datatype derivation
```python
def get_output_datatype(self, ind=0):
    """Override to compute output type from input types."""
    # Get input datatypes from kernel_model
    idt0 = self.kernel_model.inputs[0].datatype
    idt1 = self.kernel_model.inputs[1].datatype

    # Compute output type based on input range
    min_input = min(idt0.min(), idt1.min())
    max_input = max(idt0.max(), idt1.max())

    if min_input >= 0:
        odt = DataType[f"UINT{compute_bitwidth(max_input)}"]
    else:
        odt = DataType[f"INT{compute_bitwidth(min_input, max_input)}"]

    # Cache it
    self.set_nodeattr("outputDataType", odt.name)
    return odt
```

---

### Category 3: Shape-Transforming Operations
**Examples**: Pool, ConvolutionInputGenerator, Upsampler

**Characteristics**:
- Input shape != output shape
- Requires custom `make_shape_compatible_op()`
- Often have kernel size, stride, padding parameters

**Migration complexity**: ⭐⭐⭐ Complex (3-4 hours)

**Key challenges**:
- Express shape transformation in `block_tiling`
- Handle multi-dimensional transformations
- Preserve FINN's im2col semantics

**Schema pattern for Pool**:
```python
KernelSchema(
    name="Pool",
    inputs=[
        InputSchema(
            name="input",
            # Input: (Batch, OutH, OutW, KernelH*KernelW*Channels)
            # This is post-im2col format from ConvInputGen
            block_tiling=[1, 1, 1, ":"],      # Process full window
            stream_tiling=[1, 1, 1, "PE"],    # Fold channels
            constraints=[
                DimensionDivisible("input", -1, "PE", ShapeHierarchy.STREAM)
            ]
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            # Output: (Batch, OutH, OutW, Channels)
            block_tiling=[1, 1, 1, ":"],
            stream_tiling=[1, 1, 1, "PE"],
        )
    ]
)
```

**Special handling**: Custom shape inference
```python
def make_shape_compatible_op(self, model):
    """Pool requires im2col preprocessing - shape not inferrable from ONNX."""
    # Get expected output shape from nodeattrs
    batch_size = self.get_nodeattr("BatchSize")
    odims = self.get_nodeattr("OutImgDims")
    channels = self.get_nodeattr("Channels")
    oshape = (batch_size, *odims, channels)

    # Return shape-compatible op for ONNX
    return super().make_const_shape_op(oshape)
```

---

### Category 4: Multi-Output Operations
**Examples**: StreamingSplit, DuplicateStreams

**Characteristics**:
- Single input, multiple outputs
- Outputs may have different shapes
- Must handle `get_n_outputs()`

**Migration complexity**: ⭐⭐ Moderate (2-3 hours)

**Key challenges**:
- Define multiple `OutputSchema` with different shapes
- Handle indexing in inherited methods
- Support variable number of outputs

**Schema pattern**:
```python
# Note: Number of outputs determined at runtime from nodeattr
# This is a limitation - need to build schema dynamically

def build_split_schema(channels_per_stream):
    """Factory function to create schema with variable outputs."""
    outputs = []
    for i, channels in enumerate(channels_per_stream):
        outputs.append(
            OutputSchema(
                name=f"output{i}",
                # Each output gets its share of channels
                block_tiling=[1],  # Will be resolved based on tensor_context
                stream_tiling=["SIMD"],
            )
        )

    return KernelSchema(
        name="Split",
        inputs=[
            InputSchema(
                name="input",
                block_tiling=[":"],
                stream_tiling=["SIMD"],
            )
        ],
        outputs=outputs
    )
```

**Migration note**: Multi-output operations may require **schema factories** or **dynamic schema construction** in `__init__()`. This is a current limitation of the declarative approach.

---

### Category 5: Compute Kernels with Weights
**Examples**: MatrixVectorActivation (MVAU), Lookup, Thresholding

**Characteristics**:
- Have weight/parameter inputs (ONNX initializers)
- Weight memory modes (embedded, decoupled, external)
- Complex weight tensor transformations
- Accumulator datatype management

**Migration complexity**: ⭐⭐⭐⭐ Very Complex (4-6 hours)

**Key challenges**:
- Handle optional weight inputs (may be ONNX initializer or streaming)
- Express weight-specific constraints
- Preserve weight transformation logic (`get_hw_compatible_weight_tensor`)
- Manage accumulator bitwidth minimization

**Schema pattern**:
```python
KernelSchema(
    name="MVAU",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=[":"],
            stream_tiling=["SIMD"],
            datatype_attr="inputDataType",
            constraints=[
                DimensionDivisible("input", -1, "SIMD", ShapeHierarchy.STREAM)
            ]
        ),
        InputSchema(
            name="weights",
            is_weight=True,  # Mark as weight
            optional=True,   # May be initializer
            # Weight shape: (MW, MH) in ONNX
            # Folded: (1, PE, WMEM, SIMD)
            block_tiling=[1],  # Custom handling
            datatype_attr="weightDataType",
        ),
        InputSchema(
            name="thresholds",
            is_weight=True,
            optional=True,
            datatype_attr="accDataType"  # Note: threshold dt == acc dt
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=[":"],
            stream_tiling=["PE"],
            datatype_attr="outputDataType",
            constraints=[
                DimensionDivisible("output", -1, "PE", ShapeHierarchy.STREAM)
            ]
        )
    ]
)
```

**Special handling**: Weight tensor transformations
```python
def _create_input_model(self, index: int, interfaces: dict):
    """Override for weight inputs with custom shapes."""
    schema = self.kernel_schema.inputs[index]

    if schema.is_weight and index == 1:  # Weight input
        # Weights have special layout
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")

        # Original ONNX shape
        tensor_shape = (mw, mh)

        # Hardware layout shape
        wmem = (mw * mh) // (pe * simd)
        block_shape = (1, pe, wmem, simd)
        stream_shape = (1, pe, 1, simd)

        return InputModel(
            name=schema.name,
            tensor_shape=tensor_shape,
            block_shape=block_shape,
            stream_shape=stream_shape,
            datatype=DataType[self.get_nodeattr("weightDataType")],
            is_weight=True
        )

    # For other inputs, use default logic
    return super()._create_input_model(index, interfaces)
```

**Migration recommendation**: Start with **simple compute kernels** (no weights) to learn the pattern, then tackle weight-based kernels.

---

### Category 6: Infrastructure Operations
**Examples**: StreamingFIFO, StreamingDataWidthConverter

**Characteristics**:
- Minimal or no computation
- Pass-through semantics
- Focus on buffering, format conversion
- Folded shape stored as nodeattr (not inferred)

**Migration complexity**: ⭐⭐ Moderate (2 hours)

**Key challenges**:
- Shape is **externally determined** (from neighboring ops)
- Can't use template resolution
- Requires direct nodeattr access

**Schema pattern**:
```python
KernelSchema(
    name="StreamingFIFO",
    inputs=[
        InputSchema(
            name="input",
            # Shape comes from nodeattr, not tensor context
            block_tiling=None,  # Not applicable
            stream_tiling=None,
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=None,
            stream_tiling=None,
        )
    ]
)
```

**Special handling**: Direct nodeattr access
```python
def get_folded_input_shape(self, ind=0):
    """FIFO shape is stored directly, not computed."""
    return tuple(self.get_nodeattr("folded_shape"))

def get_normal_input_shape(self, ind=0):
    return tuple(self.get_nodeattr("normal_shape"))

# No template resolution needed - shapes are explicit
```

**Migration note**: Infrastructure ops may **not benefit** from KernelOp's template resolution. Consider keeping them as manual `HWCustomOp` unless you need the caching/validation features.

---

### Category 7: Table-Based Operations
**Examples**: Lookup, Thresholding (with embedded tables)

**Characteristics**:
- Embedded parameter tables
- Memory mode affects interface
- Table data requires layout transformations

**Migration complexity**: ⭐⭐⭐ Complex (3-4 hours)

**Key challenges**:
- Handle memory mode variations (embedded vs external)
- Express table as weight input or initializer
- Preserve table layout transformations

**Schema pattern for Lookup**:
```python
KernelSchema(
    name="Lookup",
    inputs=[
        InputSchema(
            name="indices",
            block_tiling=[":"],
            stream_tiling=[1],  # One index per cycle
            datatype_attr="InputType"
        ),
        InputSchema(
            name="embeddings",
            is_weight=True,
            optional=True,  # May be external memory
            # Table shape: (NumEmbeddings, EmbeddingDim)
            datatype_attr="EmbeddingType"
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            # Output shape: input_shape + [EmbeddingDim]
            # This is shape-extending - requires custom logic
            block_tiling=[":"],
            stream_tiling=[1],
        )
    ]
)
```

**Special handling**: Shape extension
```python
def make_shape_compatible_op(self, model):
    """Lookup extends shape by appending embedding dimension."""
    input_shape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
    emb_dim = self.get_nodeattr("EmbeddingDim")

    # Output shape = input_shape + [emb_dim]
    output_shape = input_shape + (emb_dim,)

    return super().make_const_shape_op(output_shape)
```

---

### Category 8: 1D vs 2D Spatial Operations
**Examples**: ConvolutionInputGenerator with `is1D` flag

**Characteristics**:
- Support both 1D and 2D spatial data
- Dimension normalization logic
- Different kernel semantics for [H,W] vs [1,D]

**Migration complexity**: ⭐⭐⭐ Complex (3-4 hours)

**Key challenges**:
- Handle dimension reshaping transparently
- Express 1D/2D variants in same schema
- Preserve normalization semantics

**Schema pattern**:
```python
def build_conv_input_gen_schema(is1d: bool):
    """Factory for 1D/2D variants."""
    if is1d:
        # 1D: (Batch, SeqLen, Channels)
        input_dims = [1, ":", ":"]
        kernel_dims = [1, "K"]
    else:
        # 2D: (Batch, H, W, Channels)
        input_dims = [1, ":", ":", ":"]
        kernel_dims = ["KH", "KW"]

    return KernelSchema(
        name="ConvolutionInputGenerator",
        inputs=[
            InputSchema(
                name="input",
                block_tiling=input_dims + [":"],
                stream_tiling=input_dims[:-1] + ["SIMD"],
            )
        ],
        outputs=[
            OutputSchema(
                name="output",
                # Shape transformation happens here
                # Output: sliding windows
                block_tiling=[1] + kernel_dims + [":"],
                stream_tiling=[1] + kernel_dims + ["SIMD"],
            )
        ]
    )
```

**Migration note**: Dimension-varying operations may need **schema factories** instantiated in `__init__()` based on nodeattrs.

---

## Step-by-Step Migration Procedure

### Phase 1: Analysis (15-30 min)

1. **Categorize your operation** using taxonomy above
2. **Identify shape semantics**:
   - Does input shape == output shape?
   - How many inputs/outputs?
   - What are tensor/block/stream dimensions?
3. **List nodeattrs**:
   - Which are parallelization (PE, SIMD)?
   - Which are datatypes?
   - Which are shape-related (MW, MH, NumChannels)?
4. **Extract constraints**:
   - What assertions exist in verify_node()?
   - What % checks exist in folded shape methods?
   - Are there min/max value requirements?

### Phase 2: Schema Definition (30-60 min)

1. **Create KernelSchema module-level constant**:
```python
KERNEL_SCHEMA = KernelSchema(
    name="YourKernel",
    inputs=[...],
    outputs=[...],
    metadata={"description": "..."}
)
```

2. **Define each InputSchema**:
   - `block_tiling`: How does tensor_shape map to block_shape?
     - `[":"]` → copy entire tensor
     - `[":", "PE"]` → fold last dim by PE
     - `[1, ":", ":"]` → prepend singleton
   - `stream_tiling`: How does block_shape map to stream_shape?
     - `["SIMD"]` → fold by SIMD parameter
     - `[DerivedDim("input", -1)]` → copy from another interface
   - `constraints`: List of `InterfaceConstraint` objects
   - `datatype_attr`: Name of nodeattr holding datatype

3. **Define each OutputSchema**:
   - Use `DerivedDim` to reference input dimensions when possible
   - Leave `stream_tiling=None` if it should match input automatically

4. **Validate schema**:
```python
# Test schema creation
schema = KERNEL_SCHEMA
print(f"Inputs: {[inp.name for inp in schema.inputs]}")
print(f"Outputs: {[out.name for out in schema.outputs]}")

# Schema validation happens in __post_init__
# Check for error messages
```

### Phase 3: Class Migration (30-60 min)

1. **Update class declaration**:
```python
# Before
class MyKernel(HWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

# After
from brainsmith.core.finn import KernelOp
from brainsmith.core.plugins import kernel

@kernel(description="My kernel")
class MyKernel(KernelOp):
    kernel_schema = KERNEL_SCHEMA

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
```

2. **Update get_nodeattr_types()**:
   - Keep parallelization params (PE, SIMD)
   - Keep functional params (epsilon, ActVal)
   - Keep exec_mode, mem_mode
   - **Remove** datatypes (now in schema)
   - **Remove** shape params (MW, MH, NumChannels) if they can be inferred

3. **Delete methods made redundant**:
   - ❌ `get_folded_input_shape()` → inherited
   - ❌ `get_folded_output_shape()` → inherited
   - ❌ `get_normal_input_shape()` → inherited
   - ❌ `get_normal_output_shape()` → inherited
   - ❌ `get_instream_width()` → inherited
   - ❌ `get_outstream_width()` → inherited
   - ❌ `verify_node()` → automatic from constraints
   - ❌ `infer_node_datatype()` → automatic via `refresh_tensor_context()`

4. **Update remaining methods**:
   - Replace `self.get_nodeattr("inputDataType")` → `self.kernel_model.inputs[0].datatype`
   - Replace manual shape math → `self.kernel_model.inputs[0].stream_shape`
   - Keep `execute_node()` unchanged
   - Keep `generate_params()` unchanged

### Phase 4: Backend Migration (HLS/RTL) (30-60 min)

If your kernel has HLS/RTL backends:

1. **Update backends to use kernel_model**:
```python
# Before
def defines(self, var):
    simd = self.get_nodeattr("SIMD")
    width = self.get_nodeattr("MW")
    idt = DataType[self.get_nodeattr("inputDataType")]

# After
def defines(self, var):
    input_model = self.kernel_model.inputs[0]
    simd = input_model.stream_shape[-1]
    width = input_model.tensor_shape[-1]
    idt = input_model.datatype
```

2. **Update template generation** to use `kernel_model` properties throughout

### Phase 5: Testing (30-60 min)

1. **Unit tests for schema**:
```python
def test_kernel_schema():
    from your_kernel import KERNEL_SCHEMA

    # Check basic structure
    assert len(KERNEL_SCHEMA.inputs) == 1
    assert len(KERNEL_SCHEMA.outputs) == 1
    assert KERNEL_SCHEMA.inputs[0].name == "input"

    # Check constraints exist
    assert len(KERNEL_SCHEMA.inputs[0].constraints) > 0
```

2. **Integration tests**:
```python
def test_kernel_model_building():
    # Create node
    node = make_test_node()
    op = MyKernel(node)

    # Simulate tensor context refresh
    op.refresh_tensor_context(model)

    # Access kernel_model (triggers build)
    km = op.kernel_model

    # Verify shapes
    assert km.inputs[0].stream_shape[-1] == expected_simd
    assert km.outputs[0].tensor_shape == expected_output_shape
```

3. **End-to-end tests**:
   - Run through FINN build flow
   - Verify cppsim execution
   - Check rtlsim parity

---

## Common Migration Pitfalls

### 1. Protected Attributes
**Problem**: Trying to modify datatype nodeattrs directly
```python
# ❌ This will fail
self.set_nodeattr("inputDataType", "INT8")
```

**Solution**: Datatypes are set by `refresh_tensor_context()`, not manually
```python
# ✅ Let tensor context handle it
op.refresh_tensor_context(model)
# Datatypes are now synchronized
```

### 2. Shape Mismatch on Optional Inputs
**Problem**: Optional weight inputs (empty string in ONNX) cause index mismatches

**Solution**: Use `optional=True` in schema, handle None in `_create_input_model()`:
```python
InputSchema(name="weights", optional=True, is_weight=True)

def _create_input_model(self, index, interfaces):
    schema = self.kernel_schema.inputs[index]
    tensor = self.tensor_context.inputs[index]

    if tensor is None and schema.optional:
        return None  # Skip optional inputs

    return super()._create_input_model(index, interfaces)
```

### 3. Circular DerivedDim References
**Problem**: Output references another output
```python
# ❌ This fails schema validation
OutputSchema(
    name="output1",
    stream_tiling=[DerivedDim("output0", -1)]  # Can't reference output
)
```

**Solution**: Outputs can only reference inputs
```python
# ✅ Reference input instead
OutputSchema(
    name="output1",
    stream_tiling=[DerivedDim("input", -1)]  # OK
)
```

### 4. Template Rank Mismatch
**Problem**: Template has fewer dimensions than reference shape
```python
# Given tensor_shape = (1, 128, 768)
block_tiling = ["SIMD"]  # Only 1 dimension
```

**Solution**: Template auto-pads with singletons
```python
# Becomes [1, 1, "SIMD"] automatically
# No action needed - just be aware
```

### 5. Constraint on Wrong Hierarchy
**Problem**: Checking SIMD divisibility on tensor shape
```python
DimensionDivisible("input", -1, "SIMD")  # Default: ShapeHierarchy.STREAM
```

**Solution**: Constraints default to STREAM, which is usually correct
```python
# Explicit specification (rarely needed)
DimensionDivisible("input", -1, "SIMD", ShapeHierarchy.STREAM)

# For tensor-level checks
DimensionDivisible("input", -1, "BlockSize", ShapeHierarchy.TENSOR)
```

---

## When NOT to Migrate

Consider **keeping manual HWCustomOp** if:

1. **Infrastructure operations** with externally-determined shapes (FIFO, DWC)
2. **Operations with extreme dynamic behavior** (variable number of outputs determined at runtime)
3. **Legacy operations** scheduled for deprecation
4. **Operations with complex state** not captured by schema (e.g., stateful operations)

KernelOp is optimized for **dataflow compute kernels** with:
- Clear input → output dataflow
- Template-expressible shapes
- Constraint-based validation

---

## Migration Checklist

Use this checklist for each kernel:

- [ ] **Analysis**
  - [ ] Categorized operation (1-8 above)
  - [ ] Identified all shape transformations
  - [ ] Listed all nodeattrs
  - [ ] Extracted all constraints from verify_node()

- [ ] **Schema Definition**
  - [ ] Created KernelSchema module constant
  - [ ] Defined all InputSchema with block/stream tiling
  - [ ] Defined all OutputSchema
  - [ ] Added all constraints
  - [ ] Specified datatype_attr for each interface
  - [ ] Schema validates without errors

- [ ] **Class Migration**
  - [ ] Changed base class to KernelOp
  - [ ] Added @kernel decorator
  - [ ] Set kernel_schema class attribute
  - [ ] Updated get_nodeattr_types()
  - [ ] Deleted redundant methods
  - [ ] Updated remaining methods to use kernel_model

- [ ] **Backend Migration** (if applicable)
  - [ ] Updated HLS backend to use kernel_model
  - [ ] Updated RTL backend to use kernel_model
  - [ ] Updated code generation templates

- [ ] **Testing**
  - [ ] Unit test for schema
  - [ ] Unit test for model building
  - [ ] Integration test with FINN flow
  - [ ] Cppsim execution test
  - [ ] Rtlsim parity test

- [ ] **Documentation**
  - [ ] Updated docstrings
  - [ ] Added schema explanation comments
  - [ ] Updated CLAUDE.md if needed

---

## Example: Complete Migration (Thresholding)

See `brainsmith/kernels/layernorm/auto_layernorm.py` for a complete reference implementation.

**Before** (FINN Thresholding): 270 lines, manual shape math, manual validation

**After** (AutoLayerNorm): 202 lines, declarative schema, automatic validation

**Reduction**: 25% fewer lines, ~60% less boilerplate

---

## Getting Help

1. **Review reference implementations**:
   - `brainsmith/kernels/layernorm/auto_layernorm.py` (shape-preserving)
   - `brainsmith/kernels/rotaryembedding/rope_axi.py` (shape-transforming)

2. **Check test cases**:
   - `brainsmith/dataflow/tests/test_schema_validation.py`
   - `brainsmith/dataflow/tests/test_template_resolution.py`

3. **Read the system documentation**:
   - `docs/dataflow_system_analysis.md` (this document)
   - `docs/dimension_lifecycle.md`

4. **Common patterns**:
   - Shape-preserving: Use `block_tiling=[":"]`
   - Dimension folding: Use `stream_tiling=["PE"]`
   - Match input streaming: Use `DerivedDim("input", -1)`
   - Validate divisibility: Use `DimensionDivisible`

---

## Summary

**KernelOp** transforms kernel development from **imperative shape calculations** to **declarative schemas**.

**Migration is a one-time investment** that pays dividends in:
- **Correctness**: Constraints validated automatically
- **Maintainability**: Schema is self-documenting
- **Consistency**: Same API across all kernels
- **Performance**: Intelligent caching reduces redundant computation

**Start with Category 1 (shape-preserving)** kernels to learn the patterns, then progress to more complex categories.

**Arete**: Every line of schema serves a purpose. Every deleted method is a victory.
