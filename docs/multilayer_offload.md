# Multilayer Offload (MLO)

Multilayer Offload (MLO) is a powerful feature in Brainsmith that enables the implementation of much larger neural networks by implementing a repeating slice of the model (such as a single transformer encoder layer) in hardware and cycling model weights through external memory (DRAM/HBM). This technique allows the acceleration of models that would otherwise be too large to fit on the FPGA.

## Overview

Traditional FPGA accelerators store all model weights on-chip in BRAM or UltraRAM, which severely limits the size of models that can be implemented. MLO overcomes this limitation by:

1. **Implementing a single repeating layer** (e.g., one transformer encoder) in hardware
2. **Storing weights off-chip** in high-bandwidth memory (HBM/DRAM)
3. **Streaming weights** into the accelerator as needed for each layer
4. **Reusing the same hardware** to process multiple layers sequentially

This approach trades some throughput for the ability to handle much larger models, making it ideal for large language models, vision transformers, and other deep architectures.

## How It Works

### Loop Body Hierarchy

MLO works by identifying a repeating structure in the neural network and implementing only that structure in hardware. This is configured using the `loop_body_hierarchy` parameter:

```yaml
finn_config:
  loop_body_hierarchy: [['encoder', 'encoder.layer.0']]
```

This configuration tells Brainsmith:
- Look for a repeating pattern called 'encoder'
- The repeating unit is 'encoder.layer.0' (the first encoder layer)
- All encoder layers (layer.0, layer.1, layer.2, etc.) will be processed using the same hardware

### Weight Streaming

Instead of storing all weights on-chip, MLO:
1. **Streams weights from HBM/DRAM** for each layer as needed
2. **Prefetches weights** for the next layer while processing the current one
3. **Manages weight buffers** to overlap computation and memory access
4. **Reuses computation hardware** across all layers

### Loop Rolling Process

The loop rolling transformation in Brainsmith:
1. **Identifies repeating patterns** in the ONNX graph based on `loop_body_hierarchy`
2. **Extracts the loop body** (single repeating layer)
3. **Creates a rolled implementation** that processes all layers sequentially
4. **Generates weight streaming logic** to load parameters from external memory
5. **Adds loop control logic** to iterate through all layers

## Configuration

### Basic MLO Setup

To enable MLO in your blueprint, add the `loop_body_hierarchy` configuration:

```yaml
name: "BERT with MLO"
description: "BERT model with Multilayer Offload"

finn_config:
  loop_body_hierarchy: [['encoder', 'encoder.layer.0']]
  split_large_fifos: true
  fifosim_n_inferences: 2  # Speed up FIFO simulation

design_space:
  steps:
    - "qonnx_to_finn"
    - "bert_streamlining"
    - "infer_kernels"
    - "create_dataflow_partition"
    - "specialize_layers"
    - "loop_rolling"        # This step implements MLO
    - "target_fps_parallelization"
    - "apply_folding_config"
    # ... rest of pipeline
```

### BERT MLO Example

For BERT models, a typical MLO configuration looks like:

```yaml
# bert_mlo_demo.yaml
name: "BERT Demo"
description: "Hugging face BERT model with MLO"

extends: "../../brainsmith/blueprints/bert.yaml"

finn_config:
  loop_body_hierarchy: [['encoder', 'encoder.layer.0']]
  split_large_fifos: true
  fifosim_n_inferences: 2
  verify_steps: ['folded_hls_cppsim', 'stitched_ip_rtlsim']

design_space:
  steps:
    - at_start:
        insert:
          - "bert_cleanup"
          - "remove_head"
          - "remove_tail"
          - "generate_reference_io"
    - at_end:
        insert: "shell_metadata_handover"
```

## Performance Characteristics

### Memory Bandwidth Requirements

MLO places high demands on memory bandwidth since weights must be streamed continuously:

- **Weight streaming bandwidth**: Model size × layers × clock frequency / execution cycles
- **Activation memory**: Only need to store activations for current layer
- **Memory efficiency**: Much lower on-chip memory usage

### Throughput vs. Latency Trade-offs

**Advantages:**
- **Much larger models** can be implemented
- **Lower on-chip memory usage** (BRAM/UltraRAM)
- **Better memory utilization** across layers

**Trade-offs:**
- **Reduced throughput** due to sequential layer processing
- **Higher memory bandwidth requirements**
- **Increased latency** for single inference
- **More complex control logic**

### When to Use MLO

**Use MLO when:**
- Model is too large to fit on-chip (>24 layers typical threshold)
- High-bandwidth memory is available (HBM preferred)
- Batch processing can amortize sequential layer costs
- Model has clear repeating structure (transformers, CNNs with residual blocks)

**Avoid MLO when:**
- Model easily fits on-chip with traditional approach
- Ultra-low latency is critical
- Limited memory bandwidth available
- Model lacks clear repeating structure

## Implementation Details

### Folding Configuration

MLO requires special consideration for folding (parallelization) parameters:

```python
# Generate folding config for MLO
python gen_folding_config.py \
    --simd 4 \
    --pe 4 \
    --num_layers 2 \  # Number of layers to implement
    -t 1 \
    -o ./configs/bert_mlo_demo.json
```

The folding configuration affects both:
- **Compute parallelism** within each layer
- **Memory bandwidth requirements** for weight streaming

### Weight Management

MLO generates additional logic for:
- **Weight buffer management**: Double/triple buffering for overlap
- **DMA controllers**: Efficient weight streaming from external memory
- **Address generation**: Calculating weight addresses for each layer
- **Synchronization**: Coordinating weight loads with computation

### Loop Control

The generated accelerator includes:
- **Layer counters**: Track current layer being processed
- **State machines**: Control weight loading and computation phases
- **Flow control**: Manage data flow between layers
- **Completion detection**: Signal when all layers are processed

## Roofline Analysis

Brainsmith includes built-in roofline analysis for MLO configurations:

```python
# MLO models in roofline analysis
bert_large_mlo = {
    'offload': True,           # Enable MLO mode
    'arch': 'bert',
    'num_layers': 24,          # Total layers (only 1 implemented)
    'seq_len': 512,
    'num_heads': 16,
    'head_size': 64,
    'intermediate': 4*16*64,
}
```

The `'offload': True` flag tells the roofline model to:
- Calculate sequential execution cycles (`num_layers` iterations)
- Account for weight streaming bandwidth requirements
- Model memory access patterns for large models

## Example: BERT MLO Demo

The `examples/bert/bert_mlo_demo.sh` demonstrates a complete MLO workflow:

```bash
#!/bin/bash
# BERT MLO Demo

# Generate folding configuration
python gen_folding_config.py \
    --simd 4 \
    --pe 4 \
    --num_layers 2 \
    -t 1 \
    -o ./configs/bert_mlo_demo.json

# Run BERT demo with MLO
python bert_demo.py \
    -o bert_mlo_demo \
    -n 4 \                    # 4 attention heads
    -l 2 \                    # 2 layers total
    -z 64 \                   # Hidden size 64
    -i 256 \                  # Intermediate size 256
    -b 8 \                    # 8-bit quantization
    -q 32 \                   # Sequence length 32
    --blueprint ./bert_mlo_demo.yaml
```

This creates a BERT model with 2 encoder layers where only the first layer is implemented in hardware, and the second layer reuses the same hardware with different weights.

## Best Practices

### Memory System Design

1. **Use HBM when available** - Higher bandwidth than DDR for weight streaming
2. **Optimize memory access patterns** - Sequential access is more efficient
3. **Size buffers appropriately** - Balance memory usage vs. bandwidth utilization

### Model Architecture

1. **Ensure clear layer boundaries** in your model structure
2. **Consistent layer shapes** across the repeated structure
3. **Minimize cross-layer dependencies** that complicate weight streaming

### Performance Tuning

1. **Profile memory bandwidth utilization** - Should be >80% for efficiency
2. **Balance compute and memory** - Don't over-parallelize if memory-bound
3. **Consider mixed precision** - Lower precision reduces bandwidth requirements
4. **Optimize FIFO depths** - Critical for maintaining pipeline efficiency

### Verification

1. **Use smaller models first** - Debug with 2-3 layers before scaling up
2. **Compare against non-MLO** - Verify functional correctness
3. **Test weight loading** - Ensure correct weights loaded for each layer
4. **Monitor memory bandwidth** - Verify streaming performance

## Debugging MLO Issues

### Common Problems

**Incorrect loop body identification:**
- Check `loop_body_hierarchy` matches your model structure
- Verify layer naming conventions in ONNX graph

**Memory bandwidth bottlenecks:**
- Profile actual vs. theoretical bandwidth usage
- Consider reducing parallelism or increasing memory frequency

**Weight loading errors:**
- Check weight buffer sizes and addressing logic
- Verify DMA controller configuration

**Pipeline stalls:**
- Analyze FIFO depths and utilization
- Look for producer/consumer mismatches

### Debug Tools

1. **Save intermediate models** - Use `save_intermediate_models: true`
2. **Enable verification** - Use RTL simulation to check correctness
3. **Memory tracing** - Monitor weight loading patterns
4. **Performance counters** - Track cycles, bandwidth utilization

## Future Enhancements

### Planned Features

- **Multi-level loop rolling** - Support for nested repeating structures
- **Dynamic weight caching** - Intelligent caching of frequently accessed weights
- **Mixed-precision streaming** - Different precision for different layers
- **Async weight prefetching** - More sophisticated memory scheduling

### Research Directions

- **Sparse weight streaming** - Skip zero weights to reduce bandwidth
- **Compressed weight formats** - On-the-fly decompression
- **Multi-model support** - Switch between different models dynamically
- **Cross-layer optimization** - Optimize across layer boundaries

## See Also

- [Design Space Exploration](design_space_exploration.md) - Understanding execution trees
- [Blueprint Schema](blueprint_schema.md) - Configuration syntax
- [Hardware Kernels](hardware_kernels.md) - Building custom accelerators
- [BERT Examples](../examples/bert/) - Complete MLO implementations
