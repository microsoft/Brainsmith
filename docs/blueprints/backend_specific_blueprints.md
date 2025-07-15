# Backend-Specific Blueprint Examples

## Legacy FINN Backend Blueprint

The legacy FINN backend uses only `build_steps` - no kernels or transforms in the blueprint. All transformations are handled within the build steps themselves.

```yaml
version: "3.0"
name: "BERT Legacy FINN Production"
description: "Production-ready BERT using legacy FINN backend with comprehensive build pipeline"

hw_compiler:
  # Legacy FINN backend ignores these - uses build_steps only
  kernels: []
  transforms: {}
  
  # Complete build pipeline for legacy FINN
  build_steps:
    # Custom preprocessing steps
    - "cleanup"                           # Remove unnecessary ops
    - "remove_head"                       # Remove classification head
    - "remove_tail"                       # Remove final layers
    - "qonnx_to_finn"                     # Convert QONNX to FINN format
    - "streamlining"                      # Apply streamlining transformations
    - "infer_hardware"                    # Infer hardware annotations
    
    # Standard FINN build steps
    - "step_create_dataflow_partition"    # Partition into dataflow regions
    - "step_specialize_layers"            # Convert to hardware layers
    - "step_target_fps_parallelization"   # Set parallelization for target FPS
    - "step_apply_folding_config"         # Apply folding configuration
    - "step_minimize_bit_width"           # Optimize bit widths
    - "step_generate_estimate_reports"    # Generate resource estimates
    - "step_hw_codegen"                   # Generate HLS code
    - "step_hw_ipgen"                     # Generate IP blocks
    - "step_set_fifo_depths"             # Configure FIFO depths
    - "step_create_stitched_ip"          # Create final IP
    - "step_measure_rtlsim_performance"   # Measure RTL simulation performance
    
    # Custom post-processing
    - "shell_metadata_handover"           # Transfer metadata to shell
    
  config_flags:
    # Hardware target
    target_device: "xczu7ev-ffvc1156-2-e"
    shell_flow_type: "vivado_zynq"
    
    # Performance configuration
    target_fps: 1000
    standalone_thresholds: true
    minimize_bit_width: true
    
    # Folding configuration
    folding_config_file: "./folding_configs/bert_balanced.json"
    
    # Verification steps
    verify_steps:
      - "FOLDED_HLS_CPPSIM"
      - "STITCHED_IP_RTLSIM"

finn_config:
  # Board configuration
  board: "ZCU104"
  fpga_part: "xczu7ev-ffvc1156-2-e"
  
  # Build configuration
  generate_outputs: 
    - "estimate_reports"
    - "stitched_ip"
    - "rtlsim_performance"
    - "synthesis_reports"
  
  # FIFO optimization
  auto_fifo_depths: true
  auto_fifo_strategy: "characterize"
  default_fifo_depth: 32
  
  # Synthesis configuration
  synth_clk_period_ns: 3.33
  board_interface:
    - "axi_lite"
    - "axis"

search:
  # Simple exhaustive search for legacy backend
  strategy: "exhaustive"
  
  constraints:
    - metric: "estimated_throughput_fps"
      operator: ">="
      value: 1000.0
      
    - metric: "estimated_lut_usage"
      operator: "<="
      value: 0.85
      
    - metric: "estimated_bram_usage"
      operator: "<="
      value: 0.90

global:
  output_stage: "stitched_ip"
  working_directory: "./legacy_finn_builds"
  save_intermediate_models: true
  
  # Legacy backend specific
  finn_build_dir: "/tmp/finn_build"
  code_gen_dir: "./generated_hw"
  
  # Build management
  log_level: "INFO"
  save_top_n: 10
  cache_results: true
```

---

## Future FINN Backend Blueprint

The future FINN backend uses kernels and transforms but no build_steps. The backend internally manages the build pipeline.

```yaml
version: "3.0"
name: "BERT Future FINN Production"
description: "Production-ready BERT using future FINN backend with kernel/transform-based approach"

hw_compiler:
  # Kernel specifications with backend options
  kernels:
    # Core transformer components
    - ["MultiHeadAttention", ["streaming_mha", "parallel_mha"]]
    - ["LayerNorm", ["layernorm_hls", "layernorm_rtl"]]
    - ["FeedForward", ["dense_hls", "sparse_rtl"]]
    - ["GELU", ["gelu_lut", "gelu_polynomial"]]
    
    # Optional optimizations
    - ["~", "AttentionCache"]             # Optional KV caching
    - ["~", "QuantizedEmbedding"]         # Optional quantized embeddings
    
    # Output layers
    - ["Pooler", ["mean_pool", "cls_pool"]]
    - ["Classifier", ["linear_hls"]]
    
  # Transform pipeline organized by stages
  transforms:
    # Graph preparation
    preparation:
      - "ValidateModel"
      - "NormalizeGraph"
      - "FuseConsecutiveOps"
      
    # Quantization transforms
    quantization:
      - ["INT8Quantization", "MixedPrecisionQuant"]
      - "FoldQuantizedWeights"
      - "PropagateQuantInfo"
      
    # Graph optimization
    optimization:
      - "RemoveIdentityOps"
      - "FoldConstants"
      - ["StreamlineLight", "StreamlineAggressive"]
      - "ConvertToChannelsLast"
      
    # Hardware mapping
    hardware_mapping:
      - "InferStreamingDataflow"
      - "AnnotateResources"
      - "InsertDWC"
      - "BalanceFIFODepths"
      
    # Backend-specific optimizations
    backend_opt:
      - ["EnableDSPPacking", "~"]
      - ["ShareBRAMs", "~"]
      - "OptimizeCriticalPath"
  
  # No build_steps for future backend
  build_steps: []
  
  config_flags:
    # Target configuration
    target_device: "xcvu9p-flga2104-2-i"
    target_clock_ns: 2.5
    
    # Kernel configuration
    kernel_defaults:
      compute_precision: "INT8"
      accumulator_precision: "INT32"
      activation_type: "INT8"
      
    # Parallelization hints
    parallelization:
      MultiHeadAttention:
        num_heads_parallel: 4
        embed_dim_parallel: 8
      FeedForward:
        hidden_parallel: 16
      LayerNorm:
        vector_parallel: 8
    
    # Memory configuration
    memory_hierarchy:
      on_chip_buffer_kb: 1024
      external_bandwidth_gbps: 19.2
      prefetch_distance: 4

# Future backend configuration
future_finn_config:
  # Execution mode
  compilation_mode: "performance"  # or "balanced", "area"
  
  # Advanced features
  enable_features:
    - "automatic_pipelining"
    - "cross_layer_optimization"
    - "dynamic_precision_scaling"
    - "hardware_software_partitioning"
  
  # Resource allocation
  resource_budget:
    lut_percent: 85
    bram_percent: 90
    dsp_percent: 95
    uram_percent: 100
  
  # Quality of results
  optimization_level: 3
  timing_closure_effort: "high"

search:
  # Smart search for kernel/transform combinations
  strategy: "bayesian"
  
  # Multi-objective optimization
  objectives:
    - metric: "throughput"
      weight: 0.4
      direction: "maximize"
    - metric: "resource_efficiency"
      weight: 0.3
      direction: "maximize"
    - metric: "power_efficiency"
      weight: 0.3
      direction: "maximize"
  
  constraints:
    - metric: "latency_us"
      operator: "<="
      value: 100.0
      
    - metric: "accuracy_drop"
      operator: "<="
      value: 0.01  # Max 1% accuracy drop
      
    - metric: "total_power_w"
      operator: "<="
      value: 75.0
  
  # Search configuration
  max_evaluations: 1000
  timeout_minutes: 480
  parallel_evaluations: 16
  
  # Bayesian optimization settings
  acquisition_function: "expected_improvement"
  n_initial_points: 50

global:
  output_stage: "optimized_design"
  working_directory: "./future_finn_builds"
  
  # Future backend outputs
  save_outputs:
    - "optimized_graph"
    - "hardware_design"
    - "performance_model"
    - "deployment_package"
  
  # Advanced caching
  cache_strategy: "incremental"
  cache_key_inputs: ["model_hash", "kernel_config", "transform_seq"]
  
  # Profiling and analysis
  enable_profiling: true
  profile_granularity: "layer"
  collect_bottleneck_analysis: true
  
  # Deployment
  deployment_target: "edge"
  generate_runtime: true
  include_calibration_data: true
```

---

## Key Differences

### Legacy FINN Backend
- **Uses only `build_steps`** - sequential pipeline execution
- **No kernels/transforms** in blueprint (empty arrays)
- **Fixed pipeline** - customization through step ordering and config flags
- **Step names** prefixed with "step_" for FINN steps
- **Direct control** over each compilation phase

### Future FINN Backend
- **Uses kernels and transforms** - declarative specification
- **No build_steps** - backend manages pipeline internally
- **Flexible composition** - mix and match kernels/backends
- **Stage-based transforms** - logical grouping of optimizations
- **Higher abstraction** - focus on what, not how

### Migration Strategy

To migrate from legacy to future backend:

1. **Identify kernel usage** in build steps → map to kernel specifications
2. **Extract transforms** from custom steps → organize into transform stages
3. **Convert config flags** → map to kernel-specific configurations
4. **Update constraints** → use new metric names
5. **Adjust outputs** → future backend has different artifacts

Both backends can coexist during transition, allowing gradual migration of models.