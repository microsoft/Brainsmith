# Brainsmith DSE Architecture Update

## Mission-Critical Context

After reviewing the `FINN_Brainsmith_Interfacing.md` document, it's clear that Brainsmith serves a **meta search framework** role that operates at a higher level than FINN. This significantly impacts our architectural design.

## Key Insights from FINN/Brainsmith Interface

### Core Mission
**Brainsmith** = Meta search framework for creating FPGA dataflow accelerators
- **Global Design Space Exploration (DSE)** across architectures and strategies
- **Iterative FINN execution** with different configurations
- **Blueprint-driven** optimization across multiple search spaces

### Interface Hierarchy
```
Brainsmith Core (Global DSE)
├── Blueprint → defines design space parameters
├── forge() → iteratively runs FINN flows
└── Multiple Search Spaces → each optimized by FINN

FINN Builder (Local Search Space)
├── DataflowBuildConfig → defines local search space
├── build_dataflow() → single FINN flow execution
└── Single Search Space → network, FIFO, kernel optimizations
```

### Search vs Design Space Distinction
- **Search Space (FINN)**: Single architecture implementation space
- **Design Space (Brainsmith)**: Multiple architectures and DSE strategies

## Architectural Impact on Our Design

### 1. **Blueprint Role Expansion**
Our blueprints must now support **DSE configuration**, not just step sequences:

```yaml
# Enhanced BERT Blueprint (brainsmith/blueprints/yaml/bert.yaml)
name: "bert"
description: "BERT transformer model with global DSE"
architecture: "transformer"

# Design Space Parameters
design_space:
  platforms:
    - board: "V80"
      fpga_part: "xcvu80p"
    - board: "ZCU104" 
      fpga_part: "xczu7ev"
  
  kernel_implementations:
    mvau_variants: ["rtl", "hls"]
    compute_modes: ["lut", "dsp", "mixed"]
  
  hw_targets:
    target_clocks: [200, 250, 300]  # MHz
    mvau_wwidth_max: [32, 64, 128]
  
  dse_strategies:
    streamlining: ["conservative", "aggressive"]
    auto_folding: ["balanced", "throughput", "resource"]

# Build Steps (per search space iteration)
build_steps:
  - "common.cleanup"
  - "transformer.remove_head"
  # ... rest of steps

# DSE Configuration
dse_config:
  search_strategy: "grid_search"  # or "genetic", "bayesian"
  max_iterations: 50
  early_stopping:
    metric: "throughput_per_lut"
    patience: 10
  
  constraints:
    max_luts: 400000
    max_dsps: 2000
    min_throughput: 1000  # inferences/sec
```

### 2. **Compiler Architecture Updates**

#### **Enhanced CompilerConfig**
```python
@dataclass
class CompilerConfig:
    blueprint: str
    output_dir: str
    
    # DSE Configuration
    dse_enabled: bool = True
    dse_strategy: str = "grid_search"  # grid_search, genetic, bayesian
    max_dse_iterations: int = 50
    early_stopping_patience: int = 10
    
    # Design Space Constraints
    resource_constraints: Dict[str, int] = field(default_factory=dict)
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Search Space Override (for single FINN run)
    single_search_space: bool = False
    finn_config_override: Optional[Dict[str, Any]] = None
    
    # Traditional parameters (now per-iteration defaults)
    target_fps: int = 3000
    clk_period_ns: float = 3.33
    board: str = "V80"
    # ... rest unchanged
```

#### **DSE Engine Component**
```python
class DesignSpaceExplorer:
    """Orchestrates global design space exploration."""
    
    def __init__(self, config: CompilerConfig, blueprint: Blueprint):
        self.config = config
        self.blueprint = blueprint
        self.dse_strategy = self._create_strategy()
        self.results_tracker = DSEResultsTracker()
    
    def explore(self, model: onnx.ModelProto) -> DSEResult:
        """Run global DSE across the design space."""
        design_space = self.blueprint.get_design_space()
        
        for iteration in range(self.config.max_dse_iterations):
            # Generate next search space configuration
            search_config = self.dse_strategy.next_configuration(design_space)
            
            # Create FINN configuration for this search space
            finn_config = self._create_finn_config(search_config)
            
            # Run single FINN flow
            finn_result = self._run_finn_iteration(model, finn_config)
            
            # Evaluate results
            self.results_tracker.add_result(search_config, finn_result)
            
            # Check early stopping
            if self._should_stop():
                break
        
        return self.results_tracker.get_best_result()
    
    def _run_finn_iteration(self, model: onnx.ModelProto, finn_config: Dict) -> FINNResult:
        """Run a single FINN build with specific configuration."""
        # Create DataflowBuildConfig from search space parameters
        df_cfg = build_cfg.DataflowBuildConfig(
            steps=self.blueprint.get_build_steps(),
            target_fps=finn_config['target_fps'],
            synth_clk_period_ns=finn_config['clk_period'],
            board=finn_config['board'],
            # ... other FINN-specific parameters
        )
        
        # Execute FINN flow
        return build.build_dataflow_cfg(model_path, df_cfg)
```

#### **Updated HardwareCompiler**
```python
class HardwareCompiler:
    """Main hardware compiler with DSE capabilities."""
    
    def __init__(self, config: CompilerConfig):
        self.config = config
        self.preprocessor = ModelPreprocessor(config)
        self.dse_engine = DesignSpaceExplorer(config) if config.dse_enabled else None
        self.finn_builder = DataflowBuilder(config)  # For single runs
        self.postprocessor = OutputProcessor(config)
    
    def compile(self, model: onnx.ModelProto) -> CompilerResult:
        """Compile with optional global DSE."""
        result = CompilerResult(success=False, output_dir=self.config.output_dir, config=self.config)
        result.start_timing()
        
        try:
            # Preprocess model
            preprocessed_model = self.preprocessor.preprocess(model, result)
            
            if self.config.dse_enabled and self.dse_engine:
                # Global DSE mode
                result.add_log("Starting global design space exploration")
                dse_result = self.dse_engine.explore(preprocessed_model)
                result.dse_summary = dse_result
                result.add_log(f"DSE completed: {dse_result.iterations} iterations, best: {dse_result.best_config}")
            else:
                # Single FINN run mode
                result.add_log("Running single FINN build (DSE disabled)")
                finn_result = self.finn_builder.build(preprocessed_model, result)
            
            # Process outputs
            self.postprocessor.process_outputs(result)
            result.success = True
            
        except Exception as e:
            result.add_error(f"Compilation failed: {e}")
        
        result.end_timing()
        return result
```

### 3. **Enhanced Library Interface**

#### **DSE-Aware API**
```python
# Traditional single-run interface (DSE disabled)
result = brainsmith.compile_model(
    model=model,
    blueprint="bert",
    output_dir="./build",
    dse_enabled=False
)

# DSE-enabled interface
result = brainsmith.compile_model_with_dse(
    model=model,
    blueprint="bert",
    output_dir="./build",
    dse_strategy="genetic",
    max_iterations=100,
    resource_constraints={
        "max_luts": 400000,
        "max_dsps": 2000
    },
    performance_requirements={
        "min_throughput": 1000
    }
)

# Advanced DSE configuration
config = brainsmith.CompilerConfig(
    blueprint="bert",
    output_dir="./build",
    dse_enabled=True,
    dse_strategy="bayesian",
    max_dse_iterations=50,
    resource_constraints={"max_luts": 300000},
    performance_requirements={"min_throughput": 1500}
)
compiler = brainsmith.HardwareCompiler(config)
result = compiler.compile(model)

# Access DSE results
if result.dse_summary:
    print(f"Best configuration: {result.dse_summary.best_config}")
    print(f"Pareto frontier: {result.dse_summary.pareto_frontier}")
    print(f"All results: {result.dse_summary.all_results}")
```

### 4. **Enhanced CLI Interface**

#### **DSE Commands**
```bash
# Traditional single run (DSE disabled)
brainsmith compile model.onnx --blueprint bert --output ./build --no-dse

# DSE-enabled compilation
brainsmith compile model.onnx --blueprint bert --output ./build \
  --dse-strategy genetic --max-iterations 100 \
  --resource-constraint max_luts=400000 \
  --performance-requirement min_throughput=1000

# DSE configuration file
brainsmith compile model.onnx --config dse_config.yaml

# DSE analysis commands
brainsmith dse analyze ./build/dse_results.json
brainsmith dse compare run1/dse_results.json run2/dse_results.json
brainsmith dse plot-pareto ./build/dse_results.json --metric throughput_per_lut
```

#### **DSE Configuration File**
```yaml
# dse_config.yaml
blueprint: bert
output_dir: ./build
dse_enabled: true

dse_config:
  strategy: genetic
  max_iterations: 100
  early_stopping_patience: 15
  
  population_size: 20  # for genetic
  mutation_rate: 0.1
  crossover_rate: 0.8

constraints:
  resources:
    max_luts: 400000
    max_dsps: 2000
    max_bram_18k: 1000
  
  performance:
    min_throughput: 1000  # inferences/sec
    max_latency: 10       # ms
    min_efficiency: 0.7   # throughput/resources

design_space:
  platforms: ["V80", "ZCU104"]
  target_clocks: [200, 250, 300]
  mvau_variants: ["rtl", "hls"]
  streamlining_modes: ["conservative", "aggressive"]
```

## Updated Benefits

### **For DSE Users (Primary Use Case)**:
1. **Automated Optimization**: Global search across design spaces
2. **Multi-objective**: Balance throughput, resources, power
3. **Constraint-driven**: Hardware and performance limits
4. **Pareto Analysis**: Trade-off exploration
5. **Strategy Selection**: Grid, genetic, Bayesian optimization

### **For FINN Users (Secondary Use Case)**:
1. **Single-run Mode**: Traditional FINN workflow
2. **Configuration Override**: Direct FINN parameter control
3. **Blueprint Reuse**: Leverage Brainsmith steps without DSE
4. **Migration Path**: Easy upgrade to DSE when ready

### **For Researchers**:
1. **Design Space Definition**: YAML-based space specification
2. **Strategy Comparison**: Multiple DSE algorithms
3. **Result Analysis**: Rich metrics and visualization
4. **Reproducibility**: Configuration-driven experiments

## Mission Alignment

This updated architecture now properly reflects Brainsmith's role as a **meta search framework**:

1. **Global DSE**: Orchestrates multiple FINN runs across design spaces
2. **Blueprint-driven**: Design spaces defined in YAML blueprints
3. **Strategy-aware**: Supports multiple DSE algorithms
4. **FINN Integration**: Properly interfaces with FINN's local search
5. **Research-focused**: Enables systematic design space exploration

The design maintains backward compatibility while enabling the true power of Brainsmith as a global design space exploration platform for FPGA dataflow accelerators.