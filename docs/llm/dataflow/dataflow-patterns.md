# Dataflow Patterns and Best Practices

This guide covers common patterns, best practices, and real-world examples for using the Brainsmith dataflow module effectively.

## Common Kernel Patterns

### 1. Element-wise Operations

Element-wise operations process tensors element-by-element with no spatial dependencies.

```python
def create_elementwise_kernel(op_name: str, constraint_type: str = "INT"):
    """Generic element-wise operation pattern."""
    kernel_def = KernelDefinition(name=op_name)
    
    # Input and output have same shape
    kernel_def.add_input(InputDefinition(
        name="x",
        datatype_constraints=[DatatypeConstraintGroup(constraint_type, 8, 32)],
        block_tiling=[1, "CH_TILES", ":", ":"],  # Tile channels
        stream_tiling=[1, "SIMD", ":", ":"]      # Parallel channels
    ))
    
    kernel_def.add_output(OutputDefinition(
        name="y",
        datatype_constraints=[DatatypeConstraintGroup(constraint_type, 8, 32)],
        block_tiling=[1, "CH_TILES", ":", ":"]
    ))
    
    # Enforce shape equality
    kernel_def.add_relationship("x", "y", RelationType.EQUAL)
    
    return kernel_def

# Examples
relu_kernel = create_elementwise_kernel("relu")
sigmoid_kernel = create_elementwise_kernel("sigmoid", "FIXED")
add_kernel = create_elementwise_kernel("add")
```

### 2. Binary Operations

Operations with two inputs of compatible shapes.

```python
def create_binary_op_kernel(op_name: str, output_scaling: float = 1.0):
    """Binary operation with broadcasting support."""
    kernel_def = KernelDefinition(name=op_name)
    
    # First input
    kernel_def.add_input(InputDefinition(
        name="a",
        datatype_constraints=[DatatypeConstraintGroup("INT", 8, 16)],
        block_tiling=[1, "CH", ":", ":"],
        stream_tiling=[1, "SIMD", 1, 1]
    ))
    
    # Second input (may broadcast)
    kernel_def.add_input(InputDefinition(
        name="b", 
        datatype_constraints=[DatatypeConstraintGroup("INT", 8, 16)],
        block_tiling=[1, "CH", ":", ":"],
        stream_tiling=[1, "SIMD", 1, 1]
    ))
    
    # Output (may have different bitwidth)
    output_width = 32 if op_name in ["multiply", "mac"] else 16
    kernel_def.add_output(OutputDefinition(
        name="y",
        datatype_constraints=[DatatypeConstraintGroup("INT", output_width, output_width)],
        block_tiling=[1, "CH", ":", ":"]
    ))
    
    # Relationships
    kernel_def.add_relationship("a", "b", RelationType.EQUAL)
    
    if output_scaling != 1.0:
        kernel_def.add_relationship(
            "a", "y", RelationType.MULTIPLE,
            factor=output_scaling
        )
    else:
        kernel_def.add_relationship("a", "y", RelationType.EQUAL)
    
    return kernel_def
```

### 3. Reduction Operations

Patterns for operations that reduce dimensions.

```python
def create_reduction_kernel(reduce_dim: int, keep_dim: bool = False):
    """Reduction along specified dimension."""
    kernel_def = KernelDefinition(name=f"reduce_dim{reduce_dim}")
    
    # Input with all dimensions
    kernel_def.add_input(InputDefinition(
        name="input",
        datatype_constraints=[DatatypeConstraintGroup("INT", 8, 32)],
        block_tiling=["BATCH", "CH", "H", "W"],
        stream_tiling=[1, "SIMD", 1, 1]
    ))
    
    # Output with reduced dimension
    if keep_dim:
        output_tiling = ["BATCH", "CH", "H", "W"]
        output_tiling[reduce_dim] = 1
    else:
        output_tiling = ["BATCH", "CH", "H", "W"]
        output_tiling.pop(reduce_dim)
    
    kernel_def.add_output(OutputDefinition(
        name="output",
        datatype_constraints=[DatatypeConstraintGroup("INT", 32, 32)],
        block_tiling=output_tiling
    ))
    
    return kernel_def

# Global average pooling
gap_kernel = create_reduction_kernel(reduce_dim=2)  # Reduce H
gap_kernel = create_reduction_kernel(reduce_dim=3)  # Then W
```

### 4. Convolution Family

Different convolution patterns for various use cases.

```python
# Standard Convolution
def create_conv2d_kernel(kernel_size: int = 3, groups: int = 1):
    """Standard 2D convolution."""
    kernel_def = KernelDefinition(name=f"conv2d_k{kernel_size}_g{groups}")
    
    # Input activations
    kernel_def.add_input(InputDefinition(
        name="input",
        datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
        block_tiling=[1, "CH_IN_TILES", ":", ":"],
        stream_tiling=[1, "SIMD", 1, 1]
    ))
    
    # Convolution weights
    kernel_def.add_input(InputDefinition(
        name="weights",
        datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
        block_tiling=["CH_OUT_TILES", "CH_IN_TILES", kernel_size, kernel_size],
        stream_tiling=["PE", "SIMD", 1, 1],
        is_weight=True
    ))
    
    # Bias (optional)
    kernel_def.add_input(InputDefinition(
        name="bias",
        datatype_constraints=[DatatypeConstraintGroup("INT", 32, 32)],
        block_tiling=["CH_OUT_TILES"],
        stream_tiling=["PE"],
        optional=True,
        is_weight=True
    ))
    
    # Output feature maps
    kernel_def.add_output(OutputDefinition(
        name="output",
        datatype_constraints=[DatatypeConstraintGroup("INT", 32, 32)],
        block_tiling=[1, "CH_OUT_TILES", ":", ":"]
    ))
    
    # Channel relationships
    if groups == 1:
        # Standard conv: input channels match weight input channels
        kernel_def.add_relationship(
            "input", "weights", RelationType.DEPENDENT,
            source_dim=1, target_dim=1
        )
    
    return kernel_def

# Depthwise Convolution
def create_depthwise_conv_kernel(kernel_size: int = 3):
    """Depthwise separable convolution."""
    kernel_def = KernelDefinition(name=f"depthwise_k{kernel_size}")
    
    kernel_def.add_input(InputDefinition(
        name="input",
        block_tiling=[1, "CHANNELS", ":", ":"],
        stream_tiling=[1, "CH_PAR", 1, 1]
    ))
    
    kernel_def.add_input(InputDefinition(
        name="weights",
        block_tiling=["CHANNELS", 1, kernel_size, kernel_size],
        stream_tiling=["CH_PAR", 1, 1, 1],
        is_weight=True
    ))
    
    kernel_def.add_output(OutputDefinition(
        name="output",
        block_tiling=[1, "CHANNELS", ":", ":"]
    ))
    
    # Channels must match exactly
    kernel_def.add_relationship(
        "input", "weights", RelationType.DEPENDENT,
        source_dim=1, target_dim=0
    )
    
    return kernel_def
```

### 5. Matrix Operations

Matrix multiplication and related operations.

```python
def create_matmul_kernel(transA: bool = False, transB: bool = False):
    """Matrix multiplication with optional transposes."""
    kernel_def = KernelDefinition(name="matmul")
    
    # Matrix A
    a_tiling = ["TILE_M", "TILE_K"] if not transA else ["TILE_K", "TILE_M"]
    kernel_def.add_input(InputDefinition(
        name="A",
        datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
        block_tiling=a_tiling,
        stream_tiling=["PE_M", "PE_K"]
    ))
    
    # Matrix B
    b_tiling = ["TILE_K", "TILE_N"] if not transB else ["TILE_N", "TILE_K"]
    kernel_def.add_input(InputDefinition(
        name="B",
        datatype_constraints=[DatatypeConstraintGroup("INT", 8, 8)],
        block_tiling=b_tiling,
        stream_tiling=["PE_K", "PE_N"]
    ))
    
    # Output matrix C
    kernel_def.add_output(OutputDefinition(
        name="C",
        datatype_constraints=[DatatypeConstraintGroup("INT", 32, 32)],
        block_tiling=["TILE_M", "TILE_N"]
    ))
    
    # Inner dimension must match
    a_inner = 1 if not transA else 0
    b_inner = 0 if not transB else 1
    kernel_def.add_relationship(
        "A", "B", RelationType.DEPENDENT,
        source_dim=a_inner, target_dim=b_inner
    )
    
    return kernel_def

# Batched Matrix Multiplication
def create_bmm_kernel():
    """Batched matrix multiply for transformers."""
    kernel_def = KernelDefinition(name="batched_matmul")
    
    # 3D inputs with batch dimension
    kernel_def.add_input(InputDefinition(
        name="A",
        block_tiling=["BATCH", "TILE_M", "TILE_K"],
        stream_tiling=[1, "PE_M", "PE_K"]
    ))
    
    kernel_def.add_input(InputDefinition(
        name="B",
        block_tiling=["BATCH", "TILE_K", "TILE_N"],
        stream_tiling=[1, "PE_K", "PE_N"]
    ))
    
    kernel_def.add_output(OutputDefinition(
        name="C",
        block_tiling=["BATCH", "TILE_M", "TILE_N"]
    ))
    
    # Batch dimensions must match
    kernel_def.add_relationship(
        "A", "B", RelationType.DEPENDENT,
        source_dim=0, target_dim=0
    )
    
    # Inner dimensions must match
    kernel_def.add_relationship(
        "A", "B", RelationType.DEPENDENT,
        source_dim=2, target_dim=1
    )
    
    return kernel_def
```

## SDIM Configuration Patterns

### Uniform vs Non-uniform Streaming

```python
# Uniform: Same streaming for all dimensions
def configure_uniform_streaming(model, interface_name, sdim_value):
    """Apply uniform SDIM to all free dimensions."""
    model.configure_sdim({interface_name: sdim_value})

# Non-uniform: Different streaming per dimension
def configure_optimized_streaming(model, interface_name):
    """Optimize SDIM based on dimension characteristics."""
    params = model.get_sdim_parameters()
    
    if interface_name in params:
        info = params[interface_name]
        sdim_config = {}
        
        for dim in info.free_dimensions:
            if dim == 0:  # Batch dimension
                sdim_config[dim] = 1  # No batch parallelism
            elif dim == 1:  # Channel dimension
                sdim_config[dim] = min(16, info.block_dims[dim])
            else:  # Spatial dimensions
                sdim_config[dim] = min(4, info.block_dims[dim])
        
        model.configure_sdim({interface_name: sdim_config})
```

### Relationship-Aware Configuration

```python
def configure_with_relationships(kernel_model):
    """Configure SDIM respecting relationships."""
    # Start with primary input
    primary_sdim = {"input": [1, 16, 1, 1]}
    kernel_model.configure_sdim(primary_sdim)
    
    # Relationships will propagate to dependent inputs
    # No need to configure them explicitly
    
    # Verify propagation
    sdim_state = kernel_model.get_sdim_state()
    for name, sdim in sdim_state.items():
        print(f"{name}: {sdim}")
```

## Performance Optimization Patterns

### Memory-Compute Balance

```python
def optimize_for_memory_bandwidth(kernel_def, memory_bandwidth_gbps, clock_freq_mhz):
    """Balance compute and memory bandwidth."""
    # Calculate available bandwidth per cycle
    bytes_per_cycle = (memory_bandwidth_gbps * 1e9) / (clock_freq_mhz * 1e6)
    
    # Determine optimal SDIM based on datatype
    datatype_bytes = 1  # INT8
    max_elements_per_cycle = int(bytes_per_cycle / datatype_bytes)
    
    # Find factorization for SDIM
    sdim_configs = []
    for pe in range(1, min(17, max_elements_per_cycle + 1)):
        simd = max_elements_per_cycle // pe
        if pe * simd <= max_elements_per_cycle:
            sdim_configs.append({"PE": pe, "SIMD": simd})
    
    return sdim_configs
```

### Pipeline Depth Optimization

```python
def optimize_for_pipeline_depth(kernel_def, pipeline_depth):
    """Match tiling to pipeline characteristics."""
    # Ensure enough work to fill pipeline
    min_iterations = pipeline_depth * 2
    
    # Configure block size for efficient pipelining
    return {
        "MIN_BLOCK_SIZE": min_iterations,
        "PREFERRED_BLOCK_SIZE": min_iterations * 4
    }
```

## Error Handling Patterns

### Graceful Degradation

```python
def create_model_with_fallback(kernel_def, ideal_config, fallback_configs):
    """Try configurations in order until one succeeds."""
    errors = []
    
    # Try ideal configuration first
    try:
        model = kernel_def.create_model(**ideal_config)
        return model, "ideal"
    except ValueError as e:
        errors.append(f"Ideal config failed: {e}")
    
    # Try fallbacks
    for i, config in enumerate(fallback_configs):
        try:
            model = kernel_def.create_model(**config)
            return model, f"fallback_{i}"
        except ValueError as e:
            errors.append(f"Fallback {i} failed: {e}")
    
    # All failed
    raise ValueError(f"All configurations failed:\n" + "\n".join(errors))
```

### Validation Helpers

```python
def validate_kernel_compatibility(kernel_def, input_shapes, param_ranges):
    """Pre-validate kernel configuration."""
    issues = []
    
    # Check basic validity
    val_errors = kernel_def.validate()
    if val_errors:
        issues.extend(val_errors)
    
    # Check parameter ranges
    required_params = kernel_def.get_required_parameters()
    for param, context in required_params.items():
        if param not in param_ranges:
            issues.append(f"Missing parameter range for {param}")
        else:
            min_val, max_val = param_ranges[param]
            if min_val <= 0:
                issues.append(f"Parameter {param} must be positive")
    
    # Check shape compatibility
    for inp_def in kernel_def.input_definitions:
        if inp_def.name in input_shapes:
            shape = input_shapes[inp_def.name]
            if inp_def._block_tiling_spec:
                errors = inp_def._block_tiling_spec.validate_against_shape(shape)
                issues.extend(errors)
    
    return issues
```

## Integration Patterns

### FINN HWCustomOp Generation

```python
def generate_finn_node_attrs(kernel_def):
    """Generate FINN nodeattr_types from kernel definition."""
    nodeattrs = {}
    
    # Add datatype attributes
    for inp in kernel_def.input_definitions:
        nodeattrs[f"{inp.name}DataType"] = ('s', False, 'INT8')
    
    for out in kernel_def.output_definitions:
        nodeattrs[f"{out.name}DataType"] = ('s', False, 'INT32')
    
    # Add tiling parameters
    params = kernel_def.get_required_parameters()
    for param_name in params:
        nodeattrs[param_name] = ('i', False, 1)
    
    # Add optimization hints
    nodeattrs["ram_style"] = ('s', False, 'auto')
    nodeattrs["runtime_writeable_weights"] = ('i', False, 0)
    
    return nodeattrs
```

### Multi-Kernel Pipelines

```python
def create_pipeline(kernel_defs, connections):
    """Create a processing pipeline from kernels."""
    pipeline = {
        "kernels": {},
        "connections": connections
    }
    
    # Validate connections
    for src_kernel, src_port, dst_kernel, dst_port in connections:
        # Get output from source
        src_def = kernel_defs[src_kernel]
        src_output = src_def.get_output(src_port)
        
        # Get input from destination
        dst_def = kernel_defs[dst_kernel]
        dst_input = dst_def.get_input(dst_port)
        
        # Validate type compatibility
        # ... validation logic ...
    
    return pipeline
```

## Testing Patterns

### Property-Based Testing

```python
def test_kernel_properties(kernel_def, property_tests):
    """Test kernel properties with random inputs."""
    import hypothesis.strategies as st
    from hypothesis import given
    
    # Generate valid configurations
    @given(
        batch=st.integers(1, 32),
        channels=st.integers(16, 256),
        height=st.integers(14, 224),
        width=st.integers(14, 224)
    )
    def test_with_shape(batch, channels, height, width):
        shape = (batch, channels, height, width)
        
        # Create model
        model = kernel_def.create_model(
            input_specs={"input": (shape, DataType["INT8"])},
            output_specs={"output": (shape, DataType["INT8"])},
            parameter_binding={"SIMD": 8, "PE": 4}
        )
        
        # Test properties
        for prop_test in property_tests:
            prop_test(model)
    
    test_with_shape()
```

### Performance Regression Testing

```python
def benchmark_kernel_configurations(kernel_def, configs):
    """Benchmark different kernel configurations."""
    results = []
    
    for config_name, config in configs.items():
        model = kernel_def.create_model(**config["model_params"])
        model.configure_sdim(config["sdim_config"])
        
        metrics = model.calculate_performance_metrics(
            frequency_mhz=config.get("frequency", 200)
        )
        
        results.append({
            "config": config_name,
            "throughput_fps": metrics["aggregate"]["throughput_fps"],
            "bandwidth_mbps": metrics["aggregate"]["total_bandwidth_mbps"],
            "initiation_interval": metrics["aggregate"]["initiation_interval"]
        })
    
    return results
```

## Anti-Patterns to Avoid

### 1. Over-Parameterization

```python
# Bad: Too many parameters
InputDefinition(
    block_tiling=["P1", "P2", "P3", "P4"],
    stream_tiling=["P5", "P6", "P7", "P8"]
)  # 8 parameters to tune!

# Good: Strategic parameters
InputDefinition(
    block_tiling=[1, "CH_TILES", ":", ":"],
    stream_tiling=[1, "SIMD", 1, 1]
)  # 2 key parameters
```

### 2. Ignoring Relationships

```python
# Bad: No relationships defined
kernel_def.add_input(input_a)
kernel_def.add_input(input_b)
kernel_def.add_output(output)
# Missing relationships!

# Good: Express all constraints
kernel_def.add_relationship("input_a", "input_b", RelationType.EQUAL)
kernel_def.add_relationship("input_a", "output", RelationType.EQUAL)
```

### 3. Fixed Assumptions

```python
# Bad: Hard-coded dimensions
InputDefinition(
    block_tiling=[1, 64, 224, 224]  # Only works for one size!
)

# Good: Flexible design
InputDefinition(
    block_tiling=[1, "CHANNELS", ":", ":"]  # Works for any size
)
```

## Summary

These patterns provide a foundation for effectively using the dataflow module:
- Use appropriate patterns for your operation type
- Configure SDIM thoughtfully based on hardware constraints
- Validate early and handle errors gracefully
- Integrate cleanly with downstream tools
- Test thoroughly with realistic configurations

The key is balancing flexibility with performance, using the type system and validation to catch errors early while maintaining hardware efficiency.