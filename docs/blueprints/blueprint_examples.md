# Brainsmith Blueprint Examples

This document provides example blueprints of varying complexity to demonstrate the capabilities of the Brainsmith DSE v3 system.

## Table of Contents

1. [Minimal Blueprint](#minimal-blueprint)
2. [Basic Exploration Blueprint](#basic-exploration-blueprint)
3. [Intermediate Blueprint with Constraints](#intermediate-blueprint-with-constraints)
4. [Advanced Multi-Kernel Blueprint](#advanced-multi-kernel-blueprint)
5. [Complex Production Blueprint](#complex-production-blueprint)
6. [Experimental Features Blueprint](#experimental-features-blueprint)

---

## Minimal Blueprint

The simplest possible blueprint that just runs a model through the default pipeline.

```yaml
version: "3.0"
name: "Minimal Example"
description: "Bare minimum configuration"

hw_compiler:
  kernels: []
  transforms: {}
  build_steps:
    - "cleanup"
    - "streamline"
    - "convert_to_hw"
    - "create_dataflow_partition"
    - "hw_codegen"

global:
  output_stage: "rtl"
  working_directory: "./minimal_build"
```

---

## Basic Exploration Blueprint

A simple blueprint that explores a few kernel options without constraints.

```yaml
version: "3.0"
name: "Basic BERT Layer"
description: "Simple exploration of BERT acceleration options"

hw_compiler:
  # Auto-discover backends for these kernels
  kernels:
    - "MatrixVectorUnit"
    - "LayerNorm"
    - "Softmax"
    
  # Basic transform pipeline
  transforms:
    cleanup:
      - "RemoveIdentityOps"
      - "FoldConstants"
    
  build_steps:
    - "cleanup"
    - "qonnx_to_finn"
    - "streamline"
    - "create_dataflow_partition"
    - "specialize_layers"
    - "hw_codegen"
    - "create_stitched_ip"
    
  config_flags:
    target_device: "xczu7ev-ffvc1156-2-e"
    target_clock_ns: 5.0

search:
  strategy: "exhaustive"

global:
  output_stage: "stitched_ip"
  working_directory: "./basic_exploration"
  log_level: "INFO"
```

---

## Intermediate Blueprint with Constraints

Adds performance constraints and mutually exclusive options.

```yaml
version: "3.0"
name: "Constrained BERT Exploration"
description: "BERT acceleration with resource and performance constraints"

hw_compiler:
  kernels:
    - ["MatrixVectorUnit", ["rtl"]] # Explicit backend selection
    - ["LayerNorm", "RMSNorm"]  # Mutually exclusive normalization options
    - ["~", "GELU"] # Optional activation
    - ["Softmax", ["Softmax_hls", "Softmax_rtl"]] # Multiple backend options
    
  transforms:
    cleanup:
      - "RemoveIdentityOps"
      - "RemoveUnusedTensors"
      
    kernel_opt:
      - ["~", "AggressiveFolding"] # Optional aggressive optimization
      - "ConvertToHW"
      
    graph_opt:
      - "InferDataLayouts"
      - "AnnotateCycles"

search:
  strategy: "exhaustive"
  
  constraints:
    - metric: "lut_utilization"
      operator: "<="
      value: 0.80
      
    - metric: "latency"
      operator: "<="
      value: 10.0
  
  max_evaluations: 100
  timeout_minutes: 120

global:
  output_stage: "stitched_ip"
  working_directory: "./constrained_exploration"
  cache_results: true
  save_artifacts: true
  log_level: "INFO"
```

---

## Advanced Multi-Kernel Blueprint

Demonstrates complex kernel configurations and transform stages.

```yaml
version: "3.0"
name: "Multi-Model Exploration"
description: "Advanced exploration with multiple kernel variants and optimization strategies"

hw_compiler:
  kernels:
    # Attention mechanism options
    - ["MultiHeadAttention", ["hls", "rtl", "dsp"]]
    
    # Normalization variants with optional selection
    - ["LayerNorm", "GroupNorm", "~BatchNorm"]
    
    # Activation function exploration
    - ["GELU", "SiLU", "ReLU", "Swish"]
    
    # Optional quantization-aware kernels
    - ["~", "QuantizedMatMul"]
    - ["~", "MixedPrecisionConv"]
    
    # Specialized kernels with backend preferences
    - ["FFN", ["streaming", "parallel"]]
    - ["Embedding", ["lookup_table", "hash"]]
    
  transforms:
    # Multi-stage transform pipeline
    preparation:
      - "ValidateDataTypes"
      - "CanonicalizeGraph"
      
    cleanup:
      - "RemoveIdentityOps"
      - "MergeConsecutiveTransposes"
      - ["RemoveUnusedTensors", "~"]  # Optional cleanup
      
    quantization:
      - ["QuantizeWeights", "MixedPrecisionQuantization", "~"]
      - "FoldQuantizedConstants"
      
    optimization:
      # Mutually exclusive optimization strategies
      - ["ConservativeOpt", "BalancedOpt", "AggressiveOpt"]
      - "PropagateDataTypes"
      - ["~", "ExperimentalFusion"]  # Optional experimental
      
    parallelization:
      - ["SetFIFODepths", "AutoFIFODepths"]
      - "BalanceDataflow"
      
    hardware:
      - "ConvertToHW"
      - "InferStreamingDataflow"
      - "AnnotateResources"
    
  build_steps:
    # Comprehensive build pipeline
    - "preparation"
    - "cleanup"
    - "quantization_prep"
    - "qonnx_to_finn"
    - "tidy_up"
    - "streamline"
    - "convert_to_hw"
    - "infer_data_layouts"
    - "create_dataflow_partition"
    - "specialize_layers"
    - "set_folding_config"
    - "generate_estimate_reports"
    - "dataflow_performance_estimation"
    - "minimize_bit_width"
    - "allocate_resources"
    - "hw_codegen"
    - "hw_ipgen"
    - "set_fifo_depths"
    - "insert_dwc"
    - "insert_fifo"
    - "create_stitched_ip"
    - "synthesize_bitstream"
    
  config_flags:
    # Advanced configuration
    target_device: "xcvu9p-flga2104-2-i"
    target_clock_ns: 2.5
    
    # Folding configuration
    default_folding: 
      PE: 16
      SIMD: 8
    
    # Memory configuration
    ram_style: "ultra"
    fifo_impl_style: "vivado"
    
    # Optimization flags
    fast_mode: false
    verify_every_step: true
    generate_reports: true

finn_config:
  board: "VCU118"
  shell_flow_type: "vitis_alveo"
  vitis_platform: "xilinx_vcu118_base_202110_1"
  
  # Advanced FIFO configuration
  auto_fifo_depths: true
  auto_fifo_strategy: "characterize"
  fifo_target_depth_multiplier: 2.0
  
  # Synthesis options
  synth_clk_period_ns: 2.5
  mvau_wwidth_max: 128

search:
  strategy: "genetic"  # Use genetic algorithm for large space
  
  constraints:
    # Resource constraints
    - metric: "lut_utilization"
      operator: "<="
      value: 0.85
      
    - metric: "bram_utilization"
      operator: "<="
      value: 0.90
      
    - metric: "dsp_utilization"
      operator: "<="
      value: 0.95
      
    # Performance requirements
    - metric: "throughput"
      operator: ">="
      value: 5000.0
      
    - metric: "latency"
      operator: "<="
      value: 5.0
      
    # Power constraint
    - metric: "total_power"
      operator: "<="
      value: 75.0
  
  # Genetic algorithm parameters
  population_size: 50
  generations: 100
  mutation_rate: 0.1
  crossover_rate: 0.7
  
  # Search limits
  max_evaluations: 1000
  timeout_minutes: 720  # 12 hours
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 0.001

global:
  output_stage: "synthesize_bitstream"
  working_directory: "./advanced_exploration"
  
  # Caching configuration
  cache_results: true
  cache_directory: "./.dse_cache"
  
  # Parallelization
  parallel_builds: 16
  
  # Logging and debugging
  log_level: "DEBUG"
  log_file: "advanced_exploration.log"
  profile_builds: true
  
  # Artifact management
  save_all_artifacts: false
  save_top_n: 20
  compress_artifacts: true
  
  # Partial build support
  checkpoint_interval: 10
  resume_on_failure: true
```

---

## Complex Production Blueprint

A production-ready blueprint with all features for a complex model.

```yaml
version: "3.0"
name: "Production BERT-Large Deployment"
description: "Production-optimized BERT-Large for edge deployment with comprehensive DSE"

# Model configuration
model_config:
  architecture: "bert-large"
  hidden_size: 1024
  num_layers: 24
  num_heads: 16
  max_seq_length: 512
  vocab_size: 30522

hw_compiler:
  kernels:
    # Core transformer kernels with production backends
    - ["MultiHeadSelfAttention", ["StreamingMHSA_v2", "ParallelMHSA_v2"]]
    - ["CrossAttention", ["StreamingXAttn", "~"]]  # Optional for encoder-only
    
    # Normalization with fallback options
    - ["LayerNorm", "RMSNorm", "SimplifiedLayerNorm"]
    
    # FFN variants for different layers
    - ["GatedFFN", ["GeGLU", "SwiGLU", "Standard"]]
    
    # Embeddings with optimization options  
    - ["TokenEmbedding", ["HashEmbed", "SparseEmbed", "DenseEmbed"]]
    - ["PositionalEncoding", ["Learned", "Sinusoidal", "Rotary"]]
    
    # Specialized kernels for production
    - ["AttentionMask", ["rtl"]]
    - ["~", "KVCache"]  # Optional for inference optimization
    - ["~", "FlashAttention"]  # Optional for memory efficiency
    
    # Output processing
    - ["Pooler", ["MeanPool", "CLSPool"]]
    - ["Classifier", ["LinearHLS", "LinearRTL"]]
    
  transforms:
    # Comprehensive transform pipeline with alternatives
    validation:
      - "ValidateQuantization"
      - "CheckNumericalStability"
      
    preparation:
      - "NormalizeGraphStructure"
      - "FuseConsecutiveOps"
      - ["~", "ExperimentalGraphOpt"]
      
    quantization:
      # Production quantization strategies
      - ["INT8Quantization", "MixedINT8INT4", "FP16Baseline"]
      - "QuantizationAwareTraining"
      - "CalibrationDataAnnotation"
      
    graph_optimization:
      # Layer-specific optimizations
      - "FuseLayerNormIntoMatMul"
      - ["FuseActivations", "KeepActivationsSeparate"]
      - "EliminateCommonSubexpressions"
      - ["~", "AggressiveConstantFolding"]
      
    memory_optimization:
      - ["ReuseIntermediateBuffers", "SeparateBuffers"]
      - "OptimizeMemoryLayout"
      - ["~", "EnableMemoryCompression"]
      
    parallelization_strategy:
      # Different parallelization approaches
      - ["PipelineParallelism", "DataParallelism", "TensorParallelism"]
      - "AutoTuneParallelConfig"
      
    hardware_mapping:
      - "MapToHardwarePrimitives"
      - "AssignComputeResources"
      - ["CoarseGrainReconfiguration", "FineGrainReconfiguration", "~"]
      
    backend_optimization:
      - "BackendSpecificOpt"
      - ["EnableDSPPacking", "~"]
      - ["EnableBRAMSharing", "~"]
      
  build_steps:
    # Production build pipeline with checkpointing
    - "validate_model"
    - "preparation_phase"
    - "cleanup"
    - "apply_quantization"
    - "checkpoint_1"
    
    - "qonnx_to_finn"
    - "apply_graph_optimizations"
    - "tidy_up"
    - "streamline"
    - "checkpoint_2"
    
    - "convert_to_hw"
    - "apply_memory_optimizations"
    - "create_dataflow_partition"
    - "apply_parallelization"
    - "checkpoint_3"
    
    - "specialize_layers"
    - "target_fps_parallelization"
    - "apply_folding_config"
    - "round_thresholds"
    - "minimize_bit_width"
    - "checkpoint_4"
    
    - "allocate_hw_resources"
    - "generate_estimate_reports"
    - "dataflow_performance_estimation"
    - "checkpoint_5"
    
    - "hw_codegen"
    - "apply_backend_optimizations"
    - "hw_ipgen"
    - "checkpoint_6"
    
    - "set_fifo_depths"
    - "insert_dwc"
    - "insert_fifo"
    - "floorplan_timing_optimizations"
    - "checkpoint_7"
    
    - "create_stitched_ip"
    - "package_final_product"
    - "synthesize_bitstream"
    - "generate_deployment_package"
    
  config_flags:
    # Production hardware target
    target_device: "xcvu13p-flga2577-2-e"
    target_clock_ns: 2.0
    
    # Performance targets
    target_fps: 10000
    batch_size: 1
    
    # Folding configuration per layer type
    folding_config:
      defaults:
        PE: 32
        SIMD: 16
      
      MultiHeadSelfAttention:
        PE: 64
        SIMD: 32
        
      GatedFFN:
        PE: 128
        SIMD: 64
        
      LayerNorm:
        PE: 16
        SIMD: 16
    
    # Memory configuration
    ram_style: "ultra"
    fifo_impl_style: "vivado"
    fifo_ram_style: "distributed"
    
    # Precision configuration
    activation_precision:
      default: "INT8"
      attention_scores: "INT16"
      
    weight_precision:
      embeddings: "INT8"
      attention: "INT4"
      ffn: "INT8"
      
    accumulator_precision:
      default: "INT32"
      critical_path: "INT48"
    
    # Optimization flags
    enable_op_fusion: true
    enable_hw_parallel: true
    enable_resource_sharing: true
    enable_critical_path_opt: true
    
    # Verification
    verify_transformed_model: true
    verify_hw_layers: true
    compare_hw_sw_outputs: true

# FINN-specific production configuration
finn_config:
  board: "U280"
  platform: "xilinx_u280_xdma_201920_3"
  shell_flow_type: "vitis_alveo"
  
  # Memory interfaces
  mem_mode: "external"
  num_axilite_workers: 4
  
  # Advanced FIFO configuration
  auto_fifo_depths: true
  auto_fifo_strategy: "characterize_and_optimize"
  fifo_target_depth_multiplier: 1.5
  large_fifo_mem_style: "uram"
  
  # Synthesis options
  synth_clk_period_ns: 2.0
  nworkers_synth: 8
  
  # Implementation strategies
  vivado_prop:
    "SYNTH_STRATEGY": "Flow_PerfOptimized_high"
    "SYNTH_FLOW": "AlternateRoutability"
    "OPT_DIRECTIVE": "ExploreWithRemap"
    "PLACE_DIRECTIVE": "ExtraNetDelay_high"
    "PHYS_OPT_DIRECTIVE": "AggressiveExplore"
    "ROUTE_DIRECTIVE": "AggressiveExplore"
    
  # DMA configuration
  axi_port_width: 512
  ddr_bandwidth_GBps: 19.2

# Production search configuration
search:
  strategy: "bayesian"  # Smart search for production
  
  # Critical production constraints
  constraints:
    # Hard resource limits
    - metric: "lut_utilization"
      operator: "<="
      value: 0.90
      
    - metric: "bram_utilization"  
      operator: "<="
      value: 0.95
      
    - metric: "uram_utilization"
      operator: "<="
      value: 0.95
      
    - metric: "dsp_utilization"
      operator: "<="
      value: 0.90
      
    # Performance requirements
    - metric: "throughput"
      operator: ">="
      value: 10000.0  # samples/sec
      
    - metric: "latency_p99"  # 99th percentile
      operator: "<="
      value: 1.0  # milliseconds
      
    # Quality constraints
    - metric: "accuracy"
      operator: ">="
      value: 0.985  # 98.5% of FP32 accuracy
      
    # Power and thermal
    - metric: "total_power"
      operator: "<="
      value: 100.0  # Watts
      
    - metric: "power_efficiency"  # inferences/watt
      operator: ">="
      value: 100.0
  
  # Bayesian optimization config
  acquisition_function: "expected_improvement"
  exploration_weight: 0.1
  
  # Multi-objective optimization
  objectives:
    - name: "throughput"
      weight: 0.4
      direction: "maximize"
      
    - name: "accuracy"
      weight: 0.3
      direction: "maximize"
      
    - name: "resource_efficiency"
      weight: 0.3
      direction: "maximize"
  
  # Search execution
  max_evaluations: 5000
  timeout_minutes: 2880  # 48 hours
  
  # Parallel execution
  parallel_evaluations: 32
  async_evaluations: true
  
  # Smart sampling
  initial_samples: 100
  sample_around_best: true
  
  # Early stopping with validation
  early_stopping:
    enabled: true
    patience: 50
    min_delta: 0.0001
    validation_interval: 10

# Production deployment configuration
global:
  output_stage: "deployment_package"
  working_directory: "/mnt/fast_ssd/production_builds"
  
  # Distributed build support
  distributed_build: true
  build_nodes: ["gpu-node-1", "gpu-node-2", "gpu-node-3", "gpu-node-4"]
  
  # Advanced caching
  cache_results: true
  cache_directory: "/mnt/shared/dse_cache"
  cache_eviction_policy: "lru"
  cache_size_gb: 100
  
  # Build management
  parallel_builds: 32
  priority_queue: true
  build_timeout_s: 7200
  
  # Checkpointing and recovery
  checkpoint_interval: 25
  checkpoint_directory: "/mnt/shared/checkpoints"
  resume_on_failure: true
  max_retries: 3
  
  # Monitoring and logging
  log_level: "INFO"
  log_directory: "/var/log/brainsmith"
  enable_telemetry: true
  telemetry_endpoint: "http://metrics.internal:8080"
  
  # Profiling
  profile_builds: true
  profile_directory: "/mnt/shared/profiles"
  collect_hw_counters: true
  
  # Artifact management
  save_all_artifacts: false
  save_top_n: 50
  save_pareto_frontier: true
  artifact_retention_days: 30
  compress_artifacts: true
  
  # Deployment package
  package_format: "container"
  include_runtime: true
  include_test_vectors: true
  sign_package: true
  
  # Validation and testing
  run_hardware_validation: true
  hardware_test_vectors: 1000
  compare_tolerance: 0.001

# Processing configuration
processing:
  preprocessing:
    - module: "data_preparation"
      config:
        normalize_inputs: true
        add_calibration_data: true
        
  postprocessing:
    - module: "deployment_optimization"
      config:
        strip_debug_info: true
        optimize_for_inference: true
        
    - module: "documentation_generator"
      config:
        generate_datasheet: true
        generate_integration_guide: true

# Test configuration
test_config:
  unit_tests:
    - "test_kernel_functionality"
    - "test_transform_correctness"
    
  integration_tests:
    - "test_end_to_end_accuracy"
    - "test_performance_requirements"
    
  hardware_tests:
    - "test_bitstream_functionality"
    - "test_thermal_limits"
    - "test_error_recovery"
```

---

## Experimental Features Blueprint

Showcases experimental and cutting-edge features.

```yaml
version: "3.0"
name: "Experimental Features Showcase"
description: "Blueprint demonstrating experimental DSE features and future capabilities"

# Experimental model configuration with dynamic shapes
model_config:
  dynamic_batch: true
  batch_range: [1, 32]
  dynamic_sequence: true
  sequence_range: [16, 512]

hw_compiler:
  # Experimental kernel features
  kernels:
    # Sparse kernels with density hints
    - ["SparseMatMul", ["structured_sparse", "unstructured_sparse"]]
      metadata:
        expected_sparsity: 0.9
        pattern: "block_2x2"
    
    # Multi-version kernels for A/B testing
    - kernel: "AttentionV2"
      versions:
        - "flash_attention_v2"
        - "linear_attention"
        - "cosformer_attention"
      selection_strategy: "performance_based"
    
    # Conditional kernels based on input characteristics
    - conditional: "sequence_length > 256"
      then: ["LongformerAttention", ["sliding_window"]]
      else: ["StandardAttention", ["dense"]]
    
    # Learnable kernels (neural architecture search)
    - ["LearnableBlock", ["nas_cell_1", "nas_cell_2", "nas_cell_3"]]
    
  # Experimental transform features
  transforms:
    # Graph rewriting with custom rules
    graph_rewrite:
      - transform: "CustomPatternMatcher"
        rules_file: "./custom_patterns.yaml"
        priority: "high"
      
    # ML-based optimization
    ml_optimization:
      - "NeuralGraphOptimizer"
        model_path: "./graph_opt_model.onnx"
        confidence_threshold: 0.8
      
    # Differential transforms (try multiple paths)
    differential:
      - branches:
          aggressive: ["FuseEverything", "MaximizeParallelism"]
          conservative: ["SafeFusions", "BalancedParallelism"]
          experimental: ["QuantumInspiredOpt", "StochasticRounding"]
        merge_strategy: "best_performance"
    
    # Profile-guided optimization
    profile_guided:
      - "CollectRuntimeProfile"
      - "OptimizeHotPaths"
      - "SpecializeForWorkload"
  
  # Experimental build system features
  build_steps:
    # Incremental compilation
    - "incremental_checkpoint"
    - "detect_changes"
    - "selective_rebuild"
    
    # Multi-target compilation
    - "compile_for_targets":
        targets: ["ZCU104", "U280", "VCK5000"]
        strategy: "unified_bitstream"
    
    # Just-in-time optimization
    - "jit_optimization":
        trigger: "first_run_profiling"
        optimization_budget_s: 60
    
  # Experimental configuration with meta-parameters
  config_flags:
    # Auto-tuning configuration
    auto_tune:
      enabled: true
      parameters: ["folding", "precision", "parallelism"]
      tuning_dataset: "./tuning_data.npz"
      tuning_iterations: 100
    
    # Adaptive precision
    adaptive_precision:
      enabled: true
      min_bits: 4
      max_bits: 16
      sensitivity_analysis: true
    
    # Hardware-software co-design
    hw_sw_codesign:
      enabled: true
      cpu_offload_threshold: 0.1
      dynamic_scheduling: true

# Experimental search strategies
search:
  # Multi-stage search pipeline
  strategy: "multi_stage"
  stages:
    # Stage 1: Broad exploration
    - name: "exploration"
      strategy: "sobol_sampling"
      samples: 1000
      duration_minutes: 60
      
    # Stage 2: Refinement
    - name: "refinement"
      strategy: "bayesian_optimization"
      candidates_from: "exploration.top_10_percent"
      iterations: 500
      
    # Stage 3: Fine-tuning
    - name: "fine_tuning"
      strategy: "gradient_based"
      candidates_from: "refinement.pareto_frontier"
      learning_rate: 0.01
      iterations: 100
  
  # Advanced constraints with soft penalties
  constraints:
    hard:
      - metric: "timing_closure"
        operator: "=="
        value: true
        
    soft:
      - metric: "resource_balance"
        operator: "maximize"
        weight: 0.3
        
      - metric: "thermal_headroom"
        operator: ">="
        value: 10.0  # Celsius
        penalty: "quadratic"
  
  # Experimental features
  features:
    # Predictive modeling
    performance_prediction:
      enabled: true
      model: "xgboost"
      features: ["kernel_config", "transform_seq", "model_stats"]
      
    # Transfer learning from previous explorations
    transfer_learning:
      enabled: true
      source_explorations: ["bert_base_*.yaml", "gpt2_*.yaml"]
      transfer_ratio: 0.2
      
    # Ensemble decisions
    ensemble_evaluation:
      enabled: true
      evaluators: ["performance", "power", "reliability"]
      aggregation: "weighted_vote"

# Experimental global features
global:
  # Cloud-native execution
  execution_mode: "kubernetes"
  k8s_config:
    namespace: "brainsmith-experimental"
    node_selector:
      accelerator: "xilinx-u280"
    resources:
      requests:
        memory: "32Gi"
        cpu: "16"
      limits:
        memory: "64Gi"
        cpu: "32"
  
  # Advanced artifact management
  artifact_store:
    type: "s3"
    bucket: "brainsmith-artifacts"
    versioning: true
    lifecycle_policy:
      transition_to_glacier_days: 30
      expiration_days: 365
  
  # Experimental monitoring
  observability:
    tracing:
      enabled: true
      backend: "jaeger"
      sample_rate: 0.1
      
    metrics:
      enabled: true
      backend: "prometheus"
      custom_metrics:
        - "design_space_coverage"
        - "pareto_hypervolume"
        - "exploration_efficiency"
    
    alerts:
      - condition: "build_failure_rate > 0.2"
        action: "reduce_parallelism"
        
      - condition: "exploration_stalled"
        action: "inject_random_candidates"
  
  # Experimental features flags
  experimental:
    quantum_annealing_search: false
    neuromorphic_kernels: false
    photonic_acceleration: false
    in_memory_computing: true
    approximate_computing: true
    
  # Continuous learning
  continuous_learning:
    enabled: true
    feedback_collection: true
    model_update_frequency: "weekly"
    a_b_testing: true

# Experimental extensions
extensions:
  # Custom plugins
  plugins:
    - name: "quantum_optimizer"
      path: "./plugins/quantum_opt.py"
      config:
        num_qubits: 20
        
    - name: "neural_architecture_search"
      path: "./plugins/nas.py"
      config:
        search_space: "darts"
        
  # External integrations
  integrations:
    - service: "wandb"
      config:
        project: "brainsmith-experimental"
        tags: ["experimental", "v3"]
        
    - service: "mlflow"
      config:
        tracking_uri: "http://mlflow.internal"
        experiment_name: "dse_experimental"
```

---

## Usage Guide

### Choosing the Right Blueprint

1. **Minimal Blueprint**: Start here for initial testing or when you just need a working build
2. **Basic Blueprint**: Use for simple explorations with a few kernel options
3. **Intermediate Blueprint**: Add constraints and explore optimization trade-offs
4. **Advanced Blueprint**: Full-featured exploration for production models
5. **Complex Production**: Production deployment with all optimizations
6. **Experimental**: Testing cutting-edge features and research ideas

### Blueprint Evolution Strategy

1. Start with a minimal blueprint to verify your model works
2. Gradually add kernels and transforms based on profiling results
3. Introduce constraints as you understand resource requirements
4. Move to advanced features once basic optimization is complete
5. Use production blueprints for final deployment
6. Experiment with cutting-edge features in isolated tests

### Best Practices

1. **Version Control**: Always version your blueprints with your model
2. **Incremental Complexity**: Add features gradually, testing each addition
3. **Document Decisions**: Use YAML comments to explain choices
4. **Reuse Components**: Create blueprint templates for common patterns
5. **Monitor Metrics**: Track exploration efficiency and result quality
6. **Validate Early**: Test constraints and configurations with small runs

### Common Patterns

#### Kernel Selection
```yaml
# Auto-discovery for maximum exploration
kernels:
  - "LayerNorm"

# Specific backends for controlled exploration  
kernels:
  - ["LayerNorm", ["hls", "rtl"]]

# Mutually exclusive options
kernels:
  - ["GELU", "ReLU", "SiLU"]

# Optional components
kernels:
  - ["~", "Dropout"]
```

#### Transform Organization
```yaml
# Stage-based for clarity
transforms:
  cleanup: [...]
  optimization: [...]
  hardware: [...]

# With alternatives
transforms:
  optimization:
    - ["Conservative", "Aggressive"]
```

#### Search Strategies
```yaml
# Quick exploration
search:
  strategy: "random"
  samples: 100

# Thorough exploration
search:
  strategy: "exhaustive"

# Smart exploration
search:
  strategy: "bayesian"
```

Remember: The blueprint system is designed to grow with your needs. Start simple and add complexity as you learn what works for your specific model and deployment requirements.