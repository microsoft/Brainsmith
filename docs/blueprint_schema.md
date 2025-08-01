# Blueprint Schema Reference

Blueprints are YAML files that declaratively define the design space for FPGA accelerator generation, including hardware kernels, build steps, and exploration parameters.

## Schema Structure

```yaml
# Required: Blueprint metadata
name: "string"                          # Blueprint name
description: "string"                   # Optional: Blueprint description

# Optional: Inherit from parent blueprint
extends: "relative/path/to/parent.yaml" # Path relative to child blueprint

# Required: Core configuration
clock_ns: 5.0                          # Target clock period in nanoseconds (required)
output: "estimates"                    # Output type: estimates | rtl | bitfile
                                      # Default: "estimates"
board: "Pynq-Z1"                      # Target FPGA board (required for rtl/bitfile)

# Optional: Additional configuration
verify: false                         # Enable verification (default: false)
verify_data: "path/to/verify_data/"   # Directory with input.npy and expected_output.npy
save_intermediate_models: false       # Save intermediate models (default: false)

# Optional: Direct FINN parameter overrides
finn_config:
  minimize_bit_width: false
  rtlsim_batch_size: 100
  shell_flow_type: "vivado_zynq"
  # Any other FINN DataflowBuildConfig parameter...

# Required: Design space definition
design_space:
  # Kernel definitions (for hardware mapping)
  kernels:
    - KernelName                      # Use all available backends
    - KernelName: BackendName         # Specific backend only
    - KernelName: [Backend1, Backend2] # Multiple specific backends
    
  # Build pipeline steps
  steps:
    - "step_name"                     # Single step
    - ["optionA", "optionB"]          # Branch: mutually exclusive options
    - ["step", ~]                     # Optional step (~ or null = skip)
    
    # Step operations (for inheritance or organization)
    - after: "target_step"
      insert: "new_step"              # Insert single step
    - before: "target_step"  
      insert: 
        - "step1"                     # Insert multiple steps 
        - "step2"
        - ["step3a", "step3b"]        # Insert a branching point
    - replace: "old_step"
      with: [["new1", "new2"]]        # Replace with a branching point
    - remove: "unwanted_step"         # Remove a step
    - at_start:
        insert: "first_step"
    - at_end:
        insert: ["last1", "last2"]   
```

## Field Definitions

### Core Configuration

#### clock_ns (required)
The target clock period in nanoseconds. This is the only required configuration field.
```yaml
clock_ns: 5.0    # 5ns = 200MHz clock frequency
```

#### output
Determines how far to proceed in the build pipeline:
- `"estimates"` (default) - Generate resource estimates only
- `"rtl"` - Generate RTL code and IP blocks
- `"bitfile"` - Full synthesis to FPGA bitstream

#### board
Target FPGA board. Required when `output` is `"rtl"` or `"bitfile"`.
Common boards include:
- `"Pynq-Z1"`, `"Pynq-Z2"` - Xilinx PYNQ boards
- `"Ultra96"` - Avnet Ultra96
- `"ZCU104"`, `"ZCU102"` - Xilinx ZCU boards
- `"VCK190"` - Xilinx Versal board
- `"V80"` - AMD Versal V80

### Optional Configuration

#### verify & verify_data
Enable verification with test data:
```yaml
verify: true
verify_data: "path/to/verify_data/"  # Directory containing input.npy and expected_output.npy
```

#### debug
Enable debug logging and additional diagnostics (not currently implemented).

#### save_intermediate_models
Save model state after each transformation step (useful for debugging).

### FINN Configuration Overrides

The `finn_config` section allows direct access to FINN's DataflowBuildConfig parameters:
```yaml
finn_config:
  minimize_bit_width: false      # Skip bit-width optimization
  rtlsim_batch_size: 100        # Batch size for RTL simulation
  shell_flow_type: "vivado_zynq" # Shell type for Zynq devices
  generate_outputs: ["estimate_only", "rtlsim_performance"]
```

## Design Space Definition

### Kernels

Kernels define the hardware implementations available for neural network layers. When the `infer_kernels` step is executed, Brainsmith automatically maps layers to these kernels.

**Pre-Release Note**: Backend registration is to support future features in the FINN Kernel backend rework. As of now, specifying backends in the blueprint *has no impact on the build* and it will default based on the `preferred_impl_style` nodeattr in the HWCustomOp.

```yaml
kernels:
  # Use all available backends for a kernel
  - Thresholding                      # Thresholding layers
  
  # Specify particular backends
  - LayerNorm: LayerNorm_hls          # Use HLS backend only
  - MVAU: [MVAU_hls, MVAU_rtl]        # Use both HLS and RTL backends
```

Common FINN kernels:
- `MVAU` - Matrix-Vector-Activation Unit (dense/linear layers)
- `Thresholding` - Quantized activation functions
- `ElementwiseBinaryOperation` - Element-wise operations
- `Pool` - Pooling layers
- `LayerNorm` - Layer normalization
- `HWSoftmax` - Hardware softmax
- `DuplicateStreams` - Stream duplication
- `Shuffle` - Tensor shuffling/reshaping

### Steps

Steps define the transformation pipeline. They can be:
1. **Linear** - Applied unconditionally
2. **Branching** - Create alternative paths
3. **Optional** - Can be skipped

```yaml
steps:
  # Linear steps - always executed
  - "qonnx_to_finn"                   # Convert QONNX to FINN format
  - "tidy_up"                         # Clean up the model
  
  # Branching - creates multiple execution paths
  - ["streamline", "streamline_aggressive"]  # Try both approaches
  
  # Optional steps - creates paths with and without
  - ["minimize_bit_width", ~]         # ~ means skip this step
  
  # Special step for kernel inference
  - "infer_kernels"                   # Maps layers to hardware kernels
  
  # Common FINN pipeline steps
  - "create_dataflow_partition"       # Partition into dataflow regions
  - "specialize_layers"               # Specialize to hardware
  - "apply_folding_config"            # Apply parallelization
  - "generate_estimate_reports"       # Generate resource estimates
```

### Step Operations

Step operations allow you to modify the step list when inheriting from parent blueprints or organizing complex pipelines.

#### Operation Types

**after** - Insert steps after a target step:
```yaml
- after: "tidy_up"
  insert: "custom_cleanup"          # Insert single step
  
- after: "streamline"
  insert: ["verify", "log_stats"]   # Insert multiple steps
```

**before** - Insert steps before a target step:
```yaml
- before: "generate_reports"
  insert: ["save_checkpoint", "validate_results"]
```

**replace** - Replace a step with alternatives:
```yaml
- replace: "minimize_bit_width"
  with: ["quantize_weights", "quantize_activations"]
```

**remove** - Remove unwanted steps:
```yaml
- remove: "debug_step"              # Remove single step
```

**at_start/at_end** - Insert at list boundaries:
```yaml
- at_start:
    insert: "initialize_environment"
- at_end:
    insert: ["cleanup", "generate_summary"]
```

### Inheritance

Blueprint inheritance allows reusing and extending existing configurations.

```yaml
# parent.yaml
name: "Base FINN Pipeline"
clock_ns: 5.0
output: "estimates"

design_space:
  steps:
    - "qonnx_to_finn"
    - "tidy_up"
    - "streamline"
    
  kernels:
    - MVAU

# child.yaml - extends parent
extends: "parent.yaml"
name: "Extended Pipeline"
output: "rtl"                      # Override parent
board: "Pynq-Z1"                   # Add new field

design_space:
  steps:
    # Parent steps are inherited, then these operations applied:
    - after: "streamline"
      insert: "custom_optimization"
    - at_end:
      insert: "package_ip"
      
  # Kernels are replaced entirely (no merge)
  kernels:
    - MVAU
    - Thresholding                 # Add new kernel
```

#### Inheritance Rules

1. **Simple fields** (name, clock_ns, etc.) - Child overrides parent
2. **finn_config** - Deep merged (child fields override parent fields)
3. **steps** - Parent steps are inherited. Use step operations to modify them. If child specifies direct steps without operations, parent steps are replaced entirely.
4. **kernels** - Child replaces parent entirely (no merge)

## Execution Semantics

### Execution Tree Structure

Brainsmith builds an execution tree from the blueprint where:
- **Nodes** represent execution segments (groups of sequential steps)
- **Branches** occur at variation points (lists in steps)
- **Leaves** represent complete execution paths

### Branch Expansion

Given these steps:
```yaml
steps:
  - "tidy_up"                      # Linear step
  - ["streamline", "streamline_aggressive"]  # Branch point (2 options)
  - "convert_to_hw"                # Linear step
  - ["fold_constants", ~]          # Branch point (2 options)
```

This creates 4 execution paths:
1. `tidy_up → streamline → convert_to_hw → fold_constants`
2. `tidy_up → streamline → convert_to_hw → (skip)`
3. `tidy_up → streamline_aggressive → convert_to_hw → fold_constants`
4. `tidy_up → streamline_aggressive → convert_to_hw → (skip)`

### Segment-Based Execution

To optimize performance, Brainsmith groups sequential steps into segments:
- Steps between branch points form a single segment
- Each segment is executed as one FINN build
- Artifacts are shared at branch points to avoid redundant computation

**Pre-Release Note**: Segment-based execution is functional but requires further testing and refinement for large design spaces.
