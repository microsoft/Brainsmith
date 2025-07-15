# Blueprint Configuration Structure

## Blueprint YAML Architecture (ASCII Art)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BLUEPRINT YAML STRUCTURE                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┴────────────────────────────────┐
    │                                                                 │
    ▼                                                                 ▼
┌──────────────────┐                                    ┌──────────────────┐
│   HW_COMPILER    │                                    │     SEARCH       │
├──────────────────┤                                    ├──────────────────┤
│ kernels:         │                                    │ strategy:        │
│   - MatMul       │ ◄── Auto-discovery                │   - exhaustive   │
│   - [LayerNorm,  │ ◄── Mutually exclusive            │   - random       │
│      RMSNorm]    │                                    │   - genetic      │
│   - [~, Dropout] │ ◄── Optional                      │                  │
│                  │                                    │ constraints:     │
│ transforms:      │                                    │   - lut ≤ 0.85   │
│   cleanup:       │                                    │   - fps ≥ 1000   │
│   - RemoveId     │                                    │                  │
│   optimization:  │                                    │ max_evaluations: │
│   - [SetPumped,  │                                    │   1000           │
│      SetTiled]   │                                    └──────────────────┘
│                  │                                              │
│ build_steps:     │                                              │
│   - PrepareIP    │                                              ▼
│   - HLSSynthIP   │                                    ┌──────────────────┐
└──────────────────┘                                    │     GLOBAL       │
         │                                              ├──────────────────┤
         │                                              │ output_stage:    │
         ▼                                              │   - rtl          │
┌──────────────────┐                                    │ working_dir:     │
│ Plugin Discovery │                                    │   ./builds       │
│   & Validation   │                                    │ cache_results:   │
└──────────────────┘                                    │   true           │
                                                        └──────────────────┘
```

## Complete Blueprint Example

```yaml
version: "3.0"
name: "BERT Layer Optimization"
description: "Comprehensive exploration of BERT acceleration options"

# Hardware compiler configuration
hw_compiler:
  # Kernel configurations with backend selection
  kernels:
    # Simple kernel - auto-discovers all backends
    - "MatMul"
    
    # Kernel with specific backends
    - ("LayerNorm", ["LayerNormHLS", "LayerNormRTL"])
    
    # Mutually exclusive kernels (choose one)
    - ["GELU", "ReLU", "Swish"]
    
    # Optional kernel (can be skipped)
    - ["~", "Dropout"]
    
  # Transform pipeline configuration
  transforms:
    # Stage-based organization
    cleanup:
      - "RemoveIdentityOps"
      - "FoldConstants"
      - "RemoveUnusedTensors"
      
    optimization:
      # Mutually exclusive transforms
      - ["Streamline", "StreamlineLight"]
      # Optional transform
      - ["~", "AggressiveOptimization"]
      
    hardware:
      - "ConvertToHW"
      - "InferDataLayouts"
      
  # Build steps (for legacy FINN backend)
  build_steps:
    - "step_tidy_up"
    - "step_streamline"
    - "step_convert_to_hw"
    - "step_specialize_layers"
    - "step_hw_codegen"
    - "step_create_stitched_ip"
    
  # Hardware configuration flags
  config_flags:
    target_fps: 1000
    target_device: "xczu7ev-ffvc1156-2-e"
    folding_config_file: "./folding_config.json"

# FINN-specific configuration
finn_config:
  board: "ZCU104"
  shell_flow_type: "vivado_zynq"
  vitis_platform: "xilinx_zcu104_base_202020_1"
  auto_fifo_depths: true
  auto_fifo_strategy: "characterize"

# Search strategy configuration
search:
  # Exploration strategy
  strategy: "exhaustive"  # or "random", "genetic", "bayesian"
  
  # Design constraints
  constraints:
    - metric: "lut_utilization"
      operator: "<="
      value: 0.85
      
    - metric: "bram_utilization"
      operator: "<="
      value: 0.90
      
    - metric: "throughput"
      operator: ">="
      value: 1000.0
      
    - metric: "latency"
      operator: "<="
      value: 10.0
  
  # Search limits
  max_evaluations: 5000
  timeout_minutes: 720  # 12 hours
  
  # Sampling configuration (for random/genetic)
  sample_size: 100
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 50
    min_delta: 0.01

# Global configuration
global:
  # Output stage selection
  output_stage: "stitched_ip"  # or "rtl", "hw_codegen", "synthesis"
  
  # Working directory
  working_directory: "./dse_results"
  
  # Caching configuration
  cache_results: true
  cache_directory: "./.dse_cache"
  
  # Parallelization
  parallel_builds: 8
  
  # Logging
  log_level: "INFO"
  log_file: "dse_exploration.log"
  
  # Artifact management
  save_all_artifacts: false
  save_top_n: 10

# Model-specific settings
model_config:
  input_shape: [1, 384]
  output_shape: [1, 384]
  batch_size: 1
  precision: "int8"
```

## Configuration Patterns

### 1. Kernel Selection Patterns

**Auto-discovery**:
```yaml
kernels:
  - "MatMul"  # Finds: MatMulHLS, MatMulRTL, MatMulDSP
```

**Specific backends**:
```yaml
kernels:
  - ("LayerNorm", ["LayerNormHLS", "LayerNormRTL"])
```

**Mutually exclusive**:
```yaml
kernels:
  - ["Conv2D", "DepthwiseConv2D", "GroupedConv2D"]
```

**Optional**:
```yaml
kernels:
  - ["~", "BatchNorm"]  # Can be skipped entirely
```

### 2. Transform Patterns

**Stage organization**:
```yaml
transforms:
  cleanup: [...]
  optimization: [...]
  hardware: [...]
```

**Nested options**:
```yaml
transforms:
  optimization:
    - "BaseOptimization"
    - ["Option1", "Option2"]  # Choose one
    - ["~", "AggressiveOpt"]  # Optional
```

### 3. Constraint Patterns

**Resource constraints**:
```yaml
constraints:
  - metric: "lut_utilization"
    operator: "<="
    value: 0.85
```

**Performance requirements**:
```yaml
constraints:
  - metric: "throughput"
    operator: ">="
    value: 1000.0
```

**Compound constraints**:
```yaml
constraints:
  - metric: "power_efficiency"  # fps/watt
    operator: ">="
    value: 100.0
```

## Validation Rules

1. **Schema Version**: Must specify compatible version
2. **Required Fields**: name, hw_compiler, global
3. **Plugin Existence**: All referenced plugins must exist
4. **Constraint Validity**: Metrics must be measurable
5. **Path Validity**: All file paths must be valid
6. **Type Checking**: Values must match expected types

## Best Practices

1. **Start Simple**: Begin with minimal configuration
2. **Incremental Complexity**: Add options gradually
3. **Use Comments**: Document design decisions
4. **Version Control**: Track blueprint changes
5. **Modular Design**: Split large blueprints
6. **Validate Early**: Test with small design spaces