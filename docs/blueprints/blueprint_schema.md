# Blueprint Schema Reference

Based on code analysis of `blueprint_parser.py` and related modules.

## Schema Structure

```yaml
# Optional: Blueprint metadata
name: "string"
description: "string"

# Optional: Inherit from parent blueprint
extends: "relative/path/to/parent.yaml"

# Global configuration - all fields optional, can be flat or under global_config
output_stage: "compile_and_package"    # generate_reports | compile_and_package | synthesize_bitstream
working_directory: "work"               # Default: "work"
save_intermediate_models: false         # Default: false
fail_fast: false                        # Default: false
max_combinations: 100000                # Or env BRAINSMITH_MAX_COMBINATIONS
timeout_minutes: 60                     # Or env BRAINSMITH_TIMEOUT_MINUTES

# Smithy parameters (mapped to FINN config)
platform: "Pynq-Z1"                     # Maps to finn_config.board
target_clk: "5ns"                       # Maps to finn_config.synth_clk_period_ns
                                       # Supports units: ps, ns, us, ms (default: ns)

# Optional: Overrides for FINN DataflowBuildConfig
finn_config:
  shell_flow_type: "vivado_zynq"
  folding_config_file: "path/to/folding.json"
  specialize_layers_config_file: "path/to/spec.json"
  standalone_thresholds: true
  # Any other FINN DataflowBuildConfig field...

# Required for meaningful execution: Design space
design_space:
  # Build steps with branching options
  steps:
    - "step_name"                  # Single step
    - ["optionA", "optionB"]       # Mutually exclusive options
    - ["step", ~]                  # Optional step (~ = skip)
    
  # Optional: Kernel definitions
  kernels:
    - KernelName                   # Use all available backends
    - KernelName: BackendName      # Specific backend
    - KernelName: [Backend1, Backend2]  # Multiple backends
```

## Field Definitions

### Output Stages
Valid values for `output_stage`:
- `"generate_reports"` - Stop after generating reports
- `"compile_and_package"` - Compile to RTL/IP
- `"synthesize_bitstream"` - Full synthesis to bitstream

### Flat vs Nested Configuration
Parameters can be specified either:
1. **Flat** (recommended) - directly at top level
2. **Nested** - under `global_config` for backwards compatibility

Example of equivalent configurations:
```yaml
# Flat style (recommended)
output_stage: "compile_and_package"
max_combinations: 1000

# Nested style (backwards compatible)
global_config:
  output_stage: "compile_and_package"
  max_combinations: 1000
```

### Smithy Parameters
Brainsmith provides cleaner parameter names that map to FINN:
- `platform` → `board`
- `target_clk` → `synth_clk_period_ns` with unit conversion

### Time Unit Conversion
The `target_clk` parameter supports unit suffixes:
- `"5"` or `"5ns"` → 5.0 nanoseconds
- `"5000ps"` → 5.0 nanoseconds
- `"0.005us"` → 5.0 nanoseconds
- `"0.000005ms"` → 5.0 nanoseconds

### Step Variations Syntax
```yaml
steps:
  # Single step applied unconditionally
  - "cleanup"
    
  # Branching with exclusive options
  - ["cleanup_minimal", "cleanup_aggressive", "cleanup_experimental"]  # Creates 3 branches
    
  # Optional steps
  - ["optimization", ~]               # ~ means skip
  - ["final_step", null]            # null also means skip
    
  # Complex combinations
  - "preprocessing"                   # Always applied
  - ["fast_path", "accurate_path"]   # Choose one
  - ["verification", ~]             # Optional
  - "packaging"                     # Always applied
```

### Inheritance

#### Basic Inheritance (List Replacement)
```yaml
# parent.yaml
output_stage: "compile_and_package"
max_combinations: 10000
design_space:
  steps:
    - "base_cleanup"
    - "optimization"

# child.yaml - traditional approach
extends: "parent.yaml"  # Relative to child location
max_combinations: 5000  # Override parent
design_space:
  steps:
    - "base_cleanup"      # Must repeat from parent
    - "optimization"      # Must repeat from parent
    - "extra_processing"  # Added in child
```

#### Advanced Step Inheritance
```yaml
# child.yaml - new step manipulation approach
extends: "parent.yaml"
design_space:
  steps:
    # Parent steps are automatically inherited when using 'extends'
    # These operations modify the inherited steps:
    
    # Insert after specific step
    - after: "base_cleanup"
      insert: "validation"
    
    # Replace a step
    - replace: "optimization"
      with: ["optimize_fast", "optimize_thorough"]
    
    # Insert before step
    - before: "extra_processing"
      insert: ["prepare_data", ~]  # Optional step
    
    # Remove step
    - remove: "unused_step"
    
    # Insert at boundaries
    - at_start:
      insert: "initialization"
    - at_end:
      insert: "finalization"
```

#### Step Operations Without Inheritance
```yaml
# standalone.yaml - operations within single blueprint
design_space:
  steps:
    # Define initial steps
    - "load_model"
    - "preprocessing"
    - "optimization"
    - "build"
    
    # Modify them with operations
    - after: "preprocessing"
      insert: ["validate_model", ~]
    - replace: "optimization"
      with: ["optimize_speed", "optimize_size"]
```

## Execution Semantics

### Branch Point Detection
Branches occur when:
1. Step has multiple options (list format): `["optionA", "optionB"]`
2. Kernel inference with multiple backend options (future feature)

### Segment Construction
- Steps between branch points form segments
- Linear steps accumulate into current segment
- Branch points flush pending steps and create child segments

### Combination Expansion
For steps `["cleanup", ["fast", "thorough"], ["verify", ~]]`:
1. `cleanup → fast → verify`
2. `cleanup → fast → (skip)`
3. `cleanup → thorough → verify`
4. `cleanup → thorough → (skip)`

## Environment Variables
- `BRAINSMITH_MAX_COMBINATIONS` - Override max_combinations
- `BRAINSMITH_TIMEOUT_MINUTES` - Override timeout_minutes

## Validation Rules
1. Step names must exist in plugin registry
2. Kernel backends must support specified kernel
3. Tree combinations must not exceed max_combinations
4. For hardware synthesis (output_stage != generate_reports):
   - Must have synth_clk_period_ns (or target_clk)
   - Must have board (or platform)

## Example Blueprints

### Minimal Blueprint
```yaml
name: "simple-exploration"
output_stage: "generate_reports"

design_space:
  steps:
    - "cleanup"
    - ["optimize_fast", "optimize_thorough"]
  kernels: []
```

### Hardware Synthesis Blueprint
```yaml
name: "fpga-accelerator"

# Flat configuration style
output_stage: "synthesize_bitstream"
platform: "Pynq-Z1"
target_clk: "5ns"
max_combinations: 50

design_space:
  steps:
    - "preprocessing"
    - "cleanup"
    - ["quantize_int8", "quantize_int4"]
    - ["optimize", ~]  # Optional optimization
    - "infer_kernels"
    - "create_dataflow_partition"
    
  kernels:
    - MVAU: mvau_hls
    - Conv: conv_hls

# Optional FINN overrides
finn_config:
  shell_flow_type: "vivado_zynq"
  generate_outputs: ["estimate_only"]
```