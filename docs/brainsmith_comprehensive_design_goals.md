# Brainsmith Comprehensive Design Goals and Architecture

## Mission Statement

**Brainsmith** is a meta search framework for creating optimized FPGA dataflow accelerators through systematic global design space exploration (DSE) that orchestrates multiple FINN flows across diverse hardware architectures, model transformations, and optimization strategies.

## Core Design Philosophy

### **Two-Tier Optimization Hierarchy**

#### **Global Design Space (Brainsmith)**
- **Design Space Definition**: YAML blueprints specify ranges of architectures, strategies, and constraints
- **DSE Orchestration**: Iterative execution of FINN flows with different configurations
- **Multi-Objective Optimization**: Balance throughput, resources, power, latency across Pareto frontiers
- **Strategy Selection**: Grid search, genetic algorithms, Bayesian optimization, custom strategies

#### **Local Search Space (FINN)** 
- **Single Architecture Optimization**: Within-architecture parameter tuning
- **DataflowBuildConfig**: Network optimizations, FIFO sizing, kernel parallelism
- **Hardware Implementation**: LUT vs DSP, RTL vs HLS kernel variations

### **Four-Hook FINN Interface Architecture**

Based on the evolving FINN builder interface, Brainsmith must configure FINN through four extensibility hooks:

#### **1. Model Ops Hook**
```python
class ModelOpsConfig:
    """Configure ONNX node handling and frontend processing."""
    
    supported_ops: List[str]           # ONNX ops FINN can handle
    custom_op_mappings: Dict[str, str] # Custom ONNX -> FINN mappings
    frontend_adjustments: List[Callable] # Model preprocessing functions
    cleanup_transforms: List[Callable]  # Post-import cleanup
    
    # Example configuration
    supported_ops = ["MatMul", "Add", "Relu", "BrainsmithLayerNorm"]
    custom_op_mappings = {
        "BrainsmithLayerNorm": "LayerNorm_hls",
        "BrainsmithSoftmax": "HwSoftmax_hls"
    }
```

#### **2. Model Transforms Hook**
```python
class ModelTransformsConfig:
    """Configure network topology optimization strategies."""
    
    transform_sequence: List[str]      # Ordered transform pipeline
    transform_configs: Dict[str, Any]  # Per-transform configuration
    architecture_specific: Dict[str, List[str]] # Architecture-specific transforms
    
    # Example for transformer architectures
    transform_sequence = [
        "InferShapes",
        "FoldConstants", 
        "StreamliningAggressive",     # Brainsmith-specific
        "OptimizeTransformerBlocks",  # Architecture-aware
        "MinimizeAccumulatorWidth"
    ]
    
    architecture_specific = {
        "transformer": ["OptimizeAttention", "FuseLayerNorm"],
        "cnn": ["OptimizeConvBlocks", "FuseBatchNorm"]
    }
```

#### **3. HW Kernels Hook**
```python
class HWKernelsConfig:
    """Configure available hardware kernels and instantiation priority."""
    
    available_kernels: Dict[str, List[str]]  # Op -> kernel implementations
    instantiation_priority: Dict[str, List[str]] # Preferred order
    kernel_constraints: Dict[str, Dict]      # Resource/performance limits
    custom_kernels: List[str]                # User-defined kernels
    
    # Example configuration
    available_kernels = {
        "MatMul": ["MVAU_rtl", "MVAU_hls", "DynMatMul_hls"],
        "Add": ["StreamingEltwise", "AddStreams"],
        "LayerNorm": ["LayerNorm_hls", "LayerNorm_rtl"]  # Brainsmith custom
    }
    
    instantiation_priority = {
        "high_throughput": ["rtl", "hls", "streaming"],
        "low_resource": ["hls", "streaming", "rtl"],
        "balanced": ["rtl", "hls", "streaming"]
    }
```

#### **4. HW Optimization Hook**
```python
class HWOptimizationConfig:
    """Configure automatic hardware parameter optimization."""
    
    folding_strategy: str              # "auto", "manual", "template"
    optimization_targets: List[str]    # ["throughput", "resource", "power"]
    constraint_functions: List[Callable] # Custom constraint evaluators
    optimization_algorithms: Dict[str, Any] # Algorithm configurations
    
    # Example optimization strategies
    optimization_algorithms = {
        "genetic": {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.1
        },
        "bayesian": {
            "acquisition_function": "ei",
            "n_initial_points": 10
        },
        "gradient_free": {
            "method": "nelder_mead",
            "max_iterations": 200
        }
    }
```

## Enhanced Blueprint Architecture

### **Multi-Dimensional Design Space Definition**

```yaml
# Enhanced Blueprint: brainsmith/blueprints/yaml/bert_dse.yaml
name: "bert_dse"
description: "BERT transformer with comprehensive DSE"
architecture: "transformer"

# Global Design Space Dimensions
design_space:
  # Platform variations
  platforms:
    - board: "V80"
      fpga_part: "xcvu80p"
      target_clocks: [200, 250, 300]
    - board: "ZCU104"  
      fpga_part: "xczu7ev"
      target_clocks: [150, 200, 250]
  
  # Model Ops Configurations
  model_ops:
    custom_ops_enabled: [true, false]
    frontend_optimization_levels: ["conservative", "aggressive"]
    supported_precision: ["int8", "int4", "mixed"]
  
  # Model Transform Strategies  
  model_transforms:
    streamlining_modes: ["conservative", "balanced", "aggressive"]
    architecture_optimization: ["standard", "attention_optimized", "memory_optimized"]
    graph_surgery: ["minimal", "standard", "aggressive"]
  
  # HW Kernel Selections
  hw_kernels:
    mvau_implementations: ["rtl", "hls", "mixed"]
    compute_priorities: ["high_throughput", "low_resource", "balanced"]
    custom_kernel_usage: [true, false]
  
  # HW Optimization Strategies
  hw_optimization:
    folding_algorithms: ["genetic", "bayesian", "gradient_free"]
    optimization_targets: 
      - ["throughput"]
      - ["resource"] 
      - ["throughput", "resource"]  # Multi-objective
    constraint_sets:
      - name: "datacenter"
        max_luts: 400000
        max_dsps: 2000
        min_throughput: 1000
      - name: "edge"
        max_luts: 100000  
        max_dsps: 500
        min_throughput: 100

# DSE Strategy Configuration
dse_config:
  global_strategy: "multi_objective_genetic"
  max_iterations: 100
  early_stopping:
    patience: 15
    min_improvement: 0.01
  
  parallelization:
    max_concurrent_finn_runs: 4
    resource_management: "queue"
  
  result_analysis:
    pareto_dimensions: ["throughput", "lut_utilization", "power"]
    constraint_violations: "reject"  # or "penalize"

# Traditional Build Steps (per FINN iteration)
build_steps:
  - "common.cleanup"
  - "transformer.remove_head"
  - "transformer.remove_tail"
  - "transformer.qonnx_to_finn"
  - "transformer.generate_reference_io"
  - "transformer.streamlining"
  - "transformer.infer_hardware"
  - "step_create_dataflow_partition"
  - "step_specialize_layers"
  - "step_target_fps_parallelization" 
  - "step_apply_folding_config"
  - "transformer.constrain_folding_and_set_pumped_compute"
  - "step_minimize_bit_width"
  - "step_generate_estimate_reports"
  - "step_hw_codegen"
  - "step_hw_ipgen"
  - "step_measure_rtlsim_performance"
  - "step_set_fifo_depths"
  - "step_create_stitched_ip"
  - "transformer.shell_metadata_handover"
```

## Four-Hook Configuration Architecture

### **Hook Configuration Classes**

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable

@dataclass
class FINNHooksConfig:
    """Complete FINN hook configuration for a design space point."""
    
    model_ops: ModelOpsConfig
    model_transforms: ModelTransformsConfig  
    hw_kernels: HWKernelsConfig
    hw_optimization: HWOptimizationConfig
    
    @classmethod
    def from_design_point(cls, design_point: Dict[str, Any]) -> 'FINNHooksConfig':
        """Create hook configuration from a design space point."""
        return cls(
            model_ops=ModelOpsConfig.from_dict(design_point['model_ops']),
            model_transforms=ModelTransformsConfig.from_dict(design_point['model_transforms']),
            hw_kernels=HWKernelsConfig.from_dict(design_point['hw_kernels']),
            hw_optimization=HWOptimizationConfig.from_dict(design_point['hw_optimization'])
        )
    
    def to_finn_config(self) -> Dict[str, Any]:
        """Convert to FINN builder configuration."""
        return {
            'model_ops_config': self.model_ops.to_dict(),
            'model_transforms_config': self.model_transforms.to_dict(),
            'hw_kernels_config': self.hw_kernels.to_dict(),
            'hw_optimization_config': self.hw_optimization.to_dict()
        }
```

### **Enhanced Design Space Explorer**

```python
class AdvancedDesignSpaceExplorer:
    """Advanced DSE engine with four-hook FINN integration."""
    
    def __init__(self, blueprint: Blueprint, config: CompilerConfig):
        self.blueprint = blueprint
        self.config = config
        self.design_space = blueprint.get_design_space()
        self.dse_strategy = self._create_dse_strategy()
        self.finn_interface = FINNBuilderInterface()
        
    def explore(self, model: onnx.ModelProto) -> DSEResult:
        """Run comprehensive design space exploration."""
        
        results = []
        for iteration in range(self.config.max_dse_iterations):
            
            # Generate next design point across all dimensions
            design_point = self.dse_strategy.next_design_point(self.design_space)
            
            # Configure FINN hooks for this design point
            finn_hooks = FINNHooksConfig.from_design_point(design_point)
            
            # Run FINN with this configuration
            finn_result = self._run_finn_with_hooks(model, finn_hooks, design_point)
            
            # Evaluate multi-objective fitness
            fitness = self._evaluate_fitness(finn_result, design_point)
            
            # Update DSE strategy with results
            self.dse_strategy.update(design_point, fitness)
            results.append((design_point, finn_result, fitness))
            
            # Check convergence criteria
            if self._has_converged(results):
                break
                
        return self._analyze_results(results)
    
    def _run_finn_with_hooks(self, model: onnx.ModelProto, 
                           hooks: FINNHooksConfig, 
                           design_point: Dict) -> FINNResult:
        """Execute FINN with specific hook configuration."""
        
        # Create enhanced DataflowBuildConfig
        df_cfg = build_cfg.DataflowBuildConfig(
            steps=self.blueprint.get_build_steps(),
            target_fps=design_point['target_fps'],
            synth_clk_period_ns=design_point['clk_period'],
            board=design_point['board'],
            
            # Four-hook configuration
            model_ops_config=hooks.model_ops.to_dict(),
            model_transforms_config=hooks.model_transforms.to_dict(),
            hw_kernels_config=hooks.hw_kernels.to_dict(),
            hw_optimization_config=hooks.hw_optimization.to_dict(),
            
            # Traditional parameters
            output_dir=self._get_iteration_dir(design_point),
            folding_config_file=design_point.get('folding_config'),
            auto_fifo_depths=design_point['auto_fifo_depths'],
            verification_atol=self.config.verification_atol
        )
        
        return self.finn_interface.build_dataflow_cfg(model, df_cfg)
```

## Enhanced Library Interface

### **DSE-First API Design**

```python
import brainsmith

# Simple DSE interface
result = brainsmith.explore_design_space(
    model=model,
    blueprint="bert_dse",
    output_dir="./build",
    max_iterations=100,
    strategy="multi_objective_genetic"
)

# Advanced four-hook configuration
hooks_config = brainsmith.FINNHooksConfig(
    model_ops=brainsmith.ModelOpsConfig(
        custom_op_mappings={"MyOp": "MyCustomKernel"},
        frontend_adjustments=["my_preprocessing"]
    ),
    hw_kernels=brainsmith.HWKernelsConfig(
        instantiation_priority={"MatMul": ["rtl", "hls"]},
        custom_kernels=["my_optimized_mvau"]
    )
)

result = brainsmith.explore_design_space(
    model=model,
    blueprint="bert_dse", 
    output_dir="./build",
    finn_hooks_override=hooks_config,
    constraints={
        "max_luts": 300000,
        "min_throughput": 1500
    }
)

# Access comprehensive results
pareto_frontier = result.get_pareto_frontier(["throughput", "lut_utilization"])
best_config = result.get_best_config(metric="throughput_per_lut")
design_insights = result.analyze_design_trends()
```

### **Configuration-Driven Workflow**

```yaml
# dse_experiment.yaml
blueprint: bert_dse
model_path: ./models/bert_quantized.onnx
output_dir: ./experiments/bert_dse_run1

dse_config:
  strategy: multi_objective_genetic
  max_iterations: 200
  population_size: 50
  
  objectives:
    - name: "throughput"
      weight: 0.4
      direction: "maximize"
    - name: "lut_efficiency" 
      weight: 0.3
      direction: "maximize"
    - name: "power"
      weight: 0.3
      direction: "minimize"

constraints:
  hard_constraints:
    max_luts: 400000
    max_dsps: 2000
    max_power_w: 50
  soft_constraints:
    preferred_throughput: 2000
    preferred_latency_ms: 5

# Four-hook overrides
finn_hooks:
  model_ops:
    custom_ops_enabled: true
    precision_mode: "mixed_int8_int4"
  
  hw_kernels:
    prefer_rtl_kernels: true
    enable_custom_kernels: true
    
  hw_optimization:
    folding_algorithm: "bayesian"
    optimization_time_budget_hours: 4

analysis:
  generate_pareto_plots: true
  sensitivity_analysis: true
  design_space_coverage_report: true
```

## Enhanced CLI Interface

### **DSE-Centric Commands**

```bash
# Design space exploration
brainsmith explore model.onnx --blueprint bert_dse --output ./build \
  --strategy multi_objective_genetic --max-iterations 100 \
  --constraint max_luts=400000 --constraint min_throughput=1000

# Configuration-driven exploration  
brainsmith explore --config dse_experiment.yaml

# Four-hook customization
brainsmith explore model.onnx --blueprint bert_dse --output ./build \
  --model-ops-config custom_ops.yaml \
  --hw-kernels-config prefer_rtl.yaml \
  --hw-optimization-config bayesian_opt.yaml

# Results analysis and visualization
brainsmith analyze ./build/dse_results.json \
  --pareto-plot throughput,lut_utilization \
  --sensitivity-analysis \
  --export-report ./report.pdf

# Design space coverage analysis
brainsmith coverage-report ./build/dse_results.json \
  --dimensions platform,transform_strategy,kernel_config

# Compare multiple experiments
brainsmith compare-experiments \
  ./exp1/dse_results.json \
  ./exp2/dse_results.json \
  --metrics throughput,efficiency,resource_utilization

# Interactive design space exploration
brainsmith interactive-dse model.onnx --blueprint bert_dse
```

## Implementation Roadmap

### **Phase 1: Core DSE Infrastructure**
1. **Four-Hook Configuration Classes**: Complete hook configuration system
2. **Enhanced Blueprint System**: Multi-dimensional design space definitions  
3. **Advanced DSE Engine**: Multi-objective optimization with constraint handling
4. **FINN Interface Layer**: Four-hook integration with FINN builder

### **Phase 2: DSE Strategies and Analysis**
1. **DSE Algorithms**: Genetic, Bayesian, gradient-free optimization
2. **Multi-Objective Optimization**: Pareto frontier analysis
3. **Result Analysis Tools**: Sensitivity analysis, design insights
4. **Parallel Execution**: Concurrent FINN runs for efficiency

### **Phase 3: User Experience and Integration**
1. **Enhanced CLI**: DSE-focused command structure
2. **Configuration Management**: YAML-driven experimental workflows
3. **Visualization Tools**: Interactive plots and reports
4. **Documentation and Examples**: Comprehensive user guides

### **Phase 4: Advanced Features**
1. **Custom DSE Strategies**: User-defined optimization algorithms
2. **Transfer Learning**: Knowledge transfer between similar models
3. **Real-time Optimization**: Online DSE during deployment
4. **Integration APIs**: REST APIs for external tool integration

## Success Metrics

### **Technical Goals**
1. **Design Space Coverage**: Explore 10x more design points than manual approaches
2. **Optimization Quality**: Find Pareto-optimal solutions within 5% of theoretical optimum
3. **FINN Integration**: Seamless four-hook configuration without FINN modifications
4. **Scalability**: Support 100+ concurrent FINN runs across cluster resources

### **User Experience Goals**
1. **Ease of Use**: Single command DSE with sensible defaults
2. **Configurability**: Full control over all optimization dimensions
3. **Insight Generation**: Actionable design insights and recommendations
4. **Reproducibility**: Deterministic results from configuration files

This comprehensive design positions Brainsmith as the definitive meta search framework for FPGA dataflow accelerator optimization, providing systematic exploration of the global design space while seamlessly integrating with FINN's evolving four-hook architecture.