# Best Practices and Troubleshooting

This chapter provides practical guidance for writing robust, maintainable kernel implementations.

## Schema Design Patterns

### Pattern: Pass-Through Operations

Element-wise operations that preserve structure:

```python
ACTIVATION_SCHEMA = KernelSchema(
    inputs=[InputSchema(block_tiling=FULL_SHAPE, stream_tiling=["PE"])],
    outputs=[OutputSchema(block_tiling=FULL_SHAPE, stream_tiling=[("input", -1)],
                          datatype="input", preserves_input_layout=True)],
    constraints=[ShapesEqual(("input", "output"))]
)
```

### Pattern: Reduction Operations

```python
REDUCE_SCHEMA = KernelSchema(
    inputs=[InputSchema(block_tiling=FULL_SHAPE, stream_tiling=[1, 1, 1, "SIMD"])],
    outputs=[OutputSchema(block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM, 1],
                          stream_tiling=[1, 1, 1, 1])],
    constraints=[DimensionEquals("output", -1, 1)]
)
```

### Pattern: Binary Operations with Broadcasting

```python
BINARY_OP_SCHEMA = KernelSchema(
    inputs=[InputSchema(block_tiling=FULL_SHAPE, stream_tiling=["PE"]),
            InputSchema(block_tiling=FULL_SHAPE, stream_tiling=[("input0", -1)])],
    outputs=[OutputSchema(block_tiling=FULL_SHAPE, stream_tiling=[("input0", -1)],
                          datatype=add_datatype("input0", "input1"))]
    # No shape equality constraint - allows broadcasting
)
```

### Pattern: Matrix Operations

```python
MATMUL_SCHEMA = KernelSchema(
    inputs=[InputSchema(stream_tiling=["SIMD"]),
            InputSchema(stream_tiling=["SIMD", "PE"])],  # 2D parallelization
    outputs=[OutputSchema(stream_tiling=["PE"])],
    internal_datatypes={"accumulator": accumulator_datatype},
    constraints=[CustomConstraint(check_matmul_dims)]
)
```

## Common Pitfalls and Solutions

### Pitfall: Forgetting to Build Design Space

**Problem:**
```python
op = MyKernel(node)
valid_ranges = op.get_valid_ranges(model)  # May fail!
```

**Solution:**
```python
# Explicit initialization
op = MyKernel(node)
op.build_design_space(model)
valid_ranges = op.design_space.dimensions

# Or use method that auto-initializes
valid_ranges = op.get_valid_ranges(model)  # Safe - auto-builds
```

**Why it happens:** Design space is lazily initialized. Direct attribute access may occur before initialization.

### Pitfall: Modifying Nodeattrs During DSE

**Problem:** `op.set_nodeattr("SIMD", simd)` invalidates design point, triggering expensive rebuild.

**Solution:** Use `design_space.configure({"SIMD": simd})` for fast exploration, then `op.apply_design_point(best_point)` when done.

### Pitfall: Hardcoded Dimensions

**Problem:** `block_tiling=[1, 784]` breaks when tensor size changes.

**Solution:** Use `block_tiling=FULL_SHAPE` and `stream_tiling=["SIMD"]` for adaptability.

### Pitfall: Wrong Hierarchy in Constraints

**Problem:** STREAM constraints fail at build time (stream shapes not available yet).

**Solution:** Use TENSOR hierarchy for structural constraints. STREAM checks happen during `configure()`.

### Pitfall: Assuming Interface Order

**Problem:** `input_list[0]` is fragile.

**Solution:** Use `inputs["input0"]` for robust name-based access.

### Pitfall: Ignoring Broadcast Constraints

**Problem:** `ShapesEqual` constraint breaks broadcasting.

**Solution:** Omit shape equality constraints for ops that support broadcasting.

## Testing Best Practices

### Pattern: Parameterized Tests

Test multiple configurations systematically:

```python
import pytest

@pytest.mark.parametrize("simd", [1, 2, 4, 8, 16, 32, 64])
def test_channelwise_add_simd_sweep(simd):
    """Test ChannelwiseAdd with different SIMD values."""
    # Build model
    model = make_test_model(input_shape=(1, 64))

    # Create kernel
    op = ChannelwiseAdd(node)
    op.build_design_space(model)

    # Configure
    point = op.design_space.configure({"SIMD": simd})

    # Validate
    assert point.initiation_interval == expected_cycles(simd)
    assert point.input_list[0].stream_shape[-1] == simd
```

### Pattern: Golden Reference Testing

Compare against known-good implementation:

```python
def test_kernel_correctness():
    """Validate kernel produces correct results."""
    # Create test data
    input_data = np.random.randint(-128, 127, size=(1, 224, 224, 64), dtype=np.int8)
    bias_data = np.random.randint(-10, 10, size=(64,), dtype=np.int8)

    # Expected output (NumPy golden reference)
    expected = input_data + bias_data

    # Kernel output (cppsim or rtlsim)
    model = build_test_model(input_data, bias_data)
    result = execute_kernel(model)

    # Compare
    np.testing.assert_array_equal(result, expected)
```

### Pattern: Constraint Validation Testing

Test that constraints catch errors:

```python
def test_constraint_validation():
    """Test that invalid configs are rejected."""
    # Valid config - should work
    point = design_space.configure({"SIMD": 32})  # Divides 64
    assert point is not None

    # Invalid config - should fail
    with pytest.raises(ValueError, match="SIMD=5"):
        design_space.configure({"SIMD": 5})  # Doesn't divide 64
```

### Pattern: Edge Case Testing

Test boundary conditions:

```python
def test_edge_cases():
    """Test minimum and maximum configurations."""
    design_space = build_design_space()

    # Minimum resources
    min_point = design_space.configure({
        "SIMD": design_space.dim_min("SIMD")
    })
    assert min_point.config["SIMD"] == 1

    # Maximum resources
    max_point = design_space.configure({
        "SIMD": design_space.dim_max("SIMD")
    })
    assert max_point.initiation_interval <= min_point.initiation_interval
```

## Performance Optimization

### Cache Design Space

```python
class MyKernel(KernelOp):
    """Kernel with cached design space."""

    def __init__(self, node):
        super().__init__(node)
        self._design_space_cache = None

    def build_design_space(self, model):
        """Build with caching."""
        if self._design_space_cache is None:
            self._design_space_cache = super().build_design_space(model)
        return self._design_space_cache
```

**Benefit:** Avoid redundant builds across multiple operations.

### Use Interface-Based Navigation

When writing generic DSE code:

```python
# Generic - works for any kernel
def sweep_first_input_parallelism(design_space):
    """Sweep first input's parallelism parameter."""
    base = design_space.default_point()

    # Get parameter name (could be "SIMD", "PE", "MW", etc.)
    param_name = base.get_input_stream_param(0)

    # Sweep using interface methods
    for point in base.sweep_input_stream(0):
        yield point

# Specific - only works for kernels with "SIMD"
def sweep_simd(design_space):
    """Sweep SIMD parameter."""
    base = design_space.default_point()
    for point in base.sweep_dimension("SIMD"):
        yield point
```

### Batch Configuration

Configure multiple points at once:

```python
# Efficient - reuses validation
configs = [
    {"SIMD": 1, "PE": 1},
    {"SIMD": 2, "PE": 2},
    {"SIMD": 4, "PE": 4},
]

points = [design_space.configure(cfg) for cfg in configs]

# Less efficient - validates constraints each time
for cfg in configs:
    point = design_space.configure(cfg)
    # Process immediately
```

## Debugging Techniques

### Inspect Design Space

```python
def debug_design_space(design_space):
    """Print design space details."""
    print(f"Kernel: {design_space.name}")

    print("\nInputs:")
    for name, inp in design_space.inputs.items():
        print(f"  {name}:")
        print(f"    tensor_shape: {inp.tensor_shape}")
        print(f"    block_shape: {inp.block_shape}")
        print(f"    stream_tiling: {inp.stream_tiling}")
        print(f"    datatype: {inp.datatype}")

    print("\nOutputs:")
    for name, out in design_space.outputs.items():
        print(f"  {name}:")
        print(f"    tensor_shape: {out.tensor_shape}")
        print(f"    block_shape: {out.block_shape}")
        print(f"    stream_tiling: {out.stream_tiling}")
        print(f"    datatype: {out.datatype}")

    print("\nDimensions:")
    for name, dim in design_space.dimensions.items():
        if hasattr(dim, 'values'):
            print(f"  {name}: {dim.values[:5]}... ({len(dim.values)} values)")
        else:
            print(f"  {name}: {dim}")

debug_design_space(op.design_space)
```

### Trace Configuration

```python
def debug_configure(design_space, config):
    """Debug configuration process."""
    print(f"Configuring: {config}")

    try:
        point = design_space.configure(config)
        print("✓ Configuration successful")

        print("\nResolved stream shapes:")
        for name, inp in point.inputs.items():
            print(f"  {name}: {inp.stream_shape}")
        for name, out in point.outputs.items():
            print(f"  {name}: {out.stream_shape}")

        return point

    except ValueError as e:
        print(f"✗ Configuration failed: {e}")
        return None

debug_configure(design_space, {"SIMD": 32})
```

### Validate Constraints Manually

```python
def check_constraints(design_space):
    """Manually validate all constraints."""
    from brainsmith.dataflow.validation import DesignSpaceValidationContext

    # Create validation context
    ctx = DesignSpaceValidationContext(
        inputs=design_space.inputs,
        outputs=design_space.outputs,
        internal_datatypes=design_space.internal_datatypes,
        param_getter=lambda k: None
    )

    # Check each constraint
    for i, constraint in enumerate(design_space.constraints):
        error = constraint.check(ctx)
        if error:
            print(f"✗ Constraint {i} failed: {constraint.describe()}")
            print(f"  Error: {error}")
        else:
            print(f"✓ Constraint {i} passed: {constraint.describe()}")

check_constraints(op.design_space)
```

## Documentation Guidelines

### Document Schema Intent

```python
CHANNELWISE_SCHEMA = KernelSchema(
    name="ChannelwiseAdd",

    # Purpose: Add per-channel bias to activation tensor
    # Hardware: Input streams from DRAM, bias buffered in BRAM
    # Parallelization: PE channels processed per cycle

    inputs=[
        InputSchema(
            name="input",
            # Activation data (dynamic, streaming)
            block_tiling=FULL_SHAPE,
            stream_tiling=["PE"],
        ),
        InputSchema(
            name="bias",
            # Per-channel bias (static, buffered)
            block_tiling=FULL_SHAPE,
            stream_tiling=[("input", -1)],
        ),
    ],
    # ...
)
```

### Document Complex Constraints

```python
def matmul_compatibility(ctx):
    """Validate matrix multiply dimension compatibility.

    For A × B → C:
    - A shape: (M, K)
    - B shape: (K, N)
    - C shape: (M, N)

    This constraint checks K dimensions match.
    """
    # Implementation...
```

### Document DSE Trade-offs

```python
dse_dimensions={
    "ram_style": DSEDimension(
        "ram_style",
        {"distributed", "block"},
        default="block"
    ),
    # Trade-off:
    # - "distributed": Lower latency, higher resource usage
    # - "block": Higher latency, more efficient packing
}
```

## Migration and Refactoring

### From Legacy FINN to Kernel Op

```python
# Before (FINN HWCustomOp):
class MyKernel_old(HWCustomOp):
    def get_nodeattr(self, name):
        return self.onnx_node.get_attribute(name)

    def make_shape_compatible_op(self, model):
        # Manual shape logic...

    def minimize_accumulator_width(self, model):
        # Manual datatype logic...

# After (KernelOp with schema):
MYKERNEL_SCHEMA = KernelSchema(
    name="MyKernel",
    inputs=[
        InputSchema(
            block_tiling=FULL_SHAPE,  # Automatic shape handling
            stream_tiling=["SIMD"],
            datatype=VALUE_OPTIMIZED   # Automatic optimization
        )
    ],
    # ...
)

@kernel
class MyKernel(KernelOp):
    @classmethod
    def build_schema(cls, node, model):
        return MYKERNEL_SCHEMA
```

**Benefits:**
- Declarative shape/datatype handling
- Automatic DSE support
- Better validation

### Incremental Adoption

```python
class HybridKernel(KernelOp):
    """Kernel that supports both old and new APIs."""

    def get_folded_input_shape(self):
        """Legacy FINN API."""
        if self._design_point:
            # New path - use design point
            return self._design_point.input_list[0].folded_shape
        else:
            # Old path - manual computation
            return self._legacy_get_folded_shape()
```

## Troubleshooting Checklist

When things go wrong:

### Build Failures

- [ ] Did you define a valid schema?
- [ ] Are all required constraints specified?
- [ ] Do tensor shapes exist in ONNX graph?
- [ ] Are datatypes valid in graph?
- [ ] Did you handle variable rank correctly?

### Configuration Failures

- [ ] Is the configuration valid (divisibility)?
- [ ] Did you check dimension ranges first?
- [ ] Are optimization constraints satisfied?
- [ ] Did you try both boundaries (min/max)?

### Unexpected Results

- [ ] Is design space cached correctly?
- [ ] Did nodeattrs change unexpectedly?
- [ ] Are you comparing correct hierarchies (TENSOR vs STREAM)?
- [ ] Did broadcasting affect shapes?

### Performance Issues

- [ ] Are you rebuilding design space unnecessarily?
- [ ] Are you modifying nodeattrs during DSE?
- [ ] Could you use interface-based navigation?
- [ ] Is caching enabled?

## Summary

Key takeaways:

**Schema Design:**
- Use templates (FULL_SHAPE) for flexibility
- Derive datatypes when possible
- Validate with constraints

**DSE:**
- Build once, configure many
- Use immutable navigation
- Cache design space

**Testing:**
- Parameterized tests for coverage
- Golden reference for correctness
- Edge case validation

**Debugging:**
- Inspect design space structure
- Trace configuration steps
- Validate constraints manually

**Performance:**
- Avoid unnecessary rebuilds
- Batch configurations
- Use interface-based APIs for generic code

With these practices, you can build robust, efficient, and maintainable kernel implementations!
