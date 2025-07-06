# Brainsmith Core: Design Space Exploration System

A comprehensive FPGA design space exploration system that transforms ONNX neural network models into optimal hardware configurations through systematic exploration of the design space.

## System Overview

Brainsmith Core implements an integrated two-phase architecture for FPGA design space exploration:

**Phase 1: Design Space Constructor** - Transforms ONNX models and Blueprint YAML specifications into validated, structured design spaces with automatic plugin discovery and configuration validation.

**Phase 2: Design Space Explorer** - Systematically explores design spaces through exhaustive or intelligent sampling, executes builds, and identifies optimal configurations using multi-objective optimization.

## Quick Start

### Basic Usage

```python
from brainsmith.core.phase1 import forge
from brainsmith.core.phase2 import explore, MockBuildRunner

# Phase 1: Create design space from model and blueprint
design_space = forge(
    model_path="models/bert.onnx",
    blueprint_path="blueprints/bert_exploration.yaml"
)

# Phase 2: Explore design space
results = explore(
    design_space=design_space,
    build_runner_factory=lambda: MockBuildRunner(success_rate=0.8)
)

# Analyze results
print(f"Explored {results.evaluated_count} configurations")
print(f"Best configuration: {results.best_config.id}")
print(f"Pareto optimal: {len(results.pareto_optimal)} configs")
```

### Complete Workflow Example

```python
from brainsmith.core.phase1 import forge
from brainsmith.core.phase2 import ExplorerEngine, LoggingHook, CachingHook
from brainsmith.core.phase3 import FinnBuildRunner

# Phase 1: Simple construction
design_space = forge(
    model_path="models/resnet18.onnx",
    blueprint_path="blueprints/resnet_exploration.yaml"
)

print(f"Design space created with {design_space.get_total_combinations():,} combinations")

# Phase 2: Advanced exploration with hooks
hooks = [
    LoggingHook(log_level="INFO", log_file="exploration.log"),
    CachingHook(cache_dir=".cache/resnet_exploration")
]

explorer = ExplorerEngine(
    build_runner_factory=FinnBuildRunner.create,
    hooks=hooks
)

# Explore with resume capability
results = explorer.explore(
    design_space=design_space,
    resume_from="dse_abc12345_config_00050"  # Resume from checkpoint
)

# Detailed analysis
print(f"\nExploration Results:")
print(f"Duration: {(results.end_time - results.start_time).total_seconds():.1f} seconds")
print(f"Success rate: {results.success_count/results.evaluated_count*100:.1f}%")

if results.best_config:
    best_result = next(r for r in results.evaluations if r.config_id == results.best_config.id)
    print(f"\nBest Configuration ({results.best_config.id}):")
    print(f"  Throughput: {best_result.metrics.throughput:.2f} inferences/sec")
    print(f"  Latency: {best_result.metrics.latency:.2f} μs")
    print(f"  LUT Utilization: {best_result.metrics.lut_utilization:.1%}")

# Pareto frontier analysis
print(f"\nPareto Optimal Configurations:")
for i, config in enumerate(results.pareto_optimal[:5]):
    result = next(r for r in results.evaluations if r.config_id == config.id)
    print(f"  {i+1}. {config.id}: {result.metrics.throughput:.1f} fps, "
          f"{result.metrics.lut_utilization:.1%} LUT")
```

## Blueprint Configuration

### Basic Blueprint Structure

```yaml
version: "3.0"
name: "BERT Base Exploration"
description: "Comprehensive exploration of BERT-base model configurations"

hw_compiler:
  kernels:
    - "MatMul"                          # Auto-discover all backends
    - ("Softmax", ["SoftmaxHLS", "SoftmaxRTL"])  # Specific backends only
    - ["LayerNorm", "RMSNorm"]          # Mutually exclusive options
    - ["~", "Dropout"]                  # Optional kernel (skip or include)
    
  transforms:
    cleanup:
      - "RemoveIdentity"
      - "FoldConstants"
    optimization:
      - ["~", "Streamline"]             # Optional transform
      - ["SetPumped", "SetTiled"]       # Mutually exclusive
    
  build_steps:
    - "PrepareIP"
    - "HLSSynthIP"
    - "CreateStitchedIP"
  
  config_flags:
    target_fps: 1000
    target_platform: "U250"

# Transform stages for pre/post processing
  transforms:
    pre_proc:
      - "Normalize"        # Model normalization
      - "ConvertFormat"    # Format conversion
    post_proc:
      - "VerifyOps"        # Operation verification
      - "AnalyzeLatency"   # Performance analysis

search:
  strategy: "exhaustive"
  constraints:
    - metric: "lut_utilization"
      operator: "<="
      value: 0.85
    - metric: "throughput"
      operator: ">="
      value: 100.0
  max_evaluations: 1000
  timeout_minutes: 120

global:
  output_stage: "rtl"
  working_directory: "./builds"
  cache_results: true
  save_artifacts: true
  log_level: "INFO"
```

### Advanced Configuration Patterns

```yaml
# Kernel auto-discovery with constraints
hw_compiler:
  kernels:
    - "MatMul"                          # Auto-discovers: MatMulHLS, MatMulRTL, MatMulDSP
    - ("MVAU", ["hls"])                 # Only HLS backend
    - ["Conv2D", "DepthwiseConv2D"]     # Choose one
    - ["~", "BatchNorm"]                # Optional (can be skipped)

# Stage-based transforms
  transforms:
    cleanup: ["RemoveIdentity", "FoldConstants"]
    streamline: ["~Streamline"]         # Optional stage
    optimization: [["SetPumped", "SetTiled"]]  # Mutually exclusive options

# Conditional constraints
search:
  constraints:
    - metric: "lut_utilization"
      operator: "<="
      value: 0.80                       # Conservative LUT usage
    - metric: "dsp_utilization"
      operator: "<="
      value: 0.90                       # Allow higher DSP usage
```

## API Reference

### Phase 1: Design Space Constructor

#### Core API

```python
from brainsmith.core.phase1 import forge, ForgeAPI

# Simple construction
design_space = forge("model.onnx", "blueprint.yaml")

# Advanced construction
api = ForgeAPI(verbose=True)
design_space = api.forge("model.onnx", "blueprint.yaml")

# Optimized construction (faster plugin loading)
design_space = api.forge_optimized(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml",
    optimize_plugins=True
)
```

#### Error Handling

```python
from brainsmith.core.phase1.exceptions import (
    BlueprintParseError, 
    PluginNotFoundError, 
    ValidationError
)

try:
    design_space = forge("model.onnx", "blueprint.yaml")
except BlueprintParseError as e:
    print(f"Blueprint syntax error at line {e.line}: {e.message}")
    if e.suggestions:
        print(f"Suggestions: {', '.join(e.suggestions)}")
except PluginNotFoundError as e:
    print(f"Plugin '{e.plugin_name}' not found")
    if e.alternatives:
        print(f"Available alternatives: {', '.join(e.alternatives)}")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    for error in e.errors:
        print(f"  - {error}")
```

### Phase 2: Design Space Explorer

#### Basic Exploration

```python
from brainsmith.core.phase2 import explore, MockBuildRunner

# Simple exploration with mock runner
results = explore(
    design_space=design_space,
    build_runner_factory=lambda: MockBuildRunner(success_rate=0.8)
)

# With real FINN backend
from brainsmith.core.phase3 import FinnBuildRunner
results = explore(
    design_space=design_space,
    build_runner_factory=FinnBuildRunner.create
)
```

#### Advanced Exploration

```python
from brainsmith.core.phase2 import (
    ExplorerEngine, 
    LoggingHook, 
    CachingHook,
    EarlyStoppingHook
)

# Custom hooks
class CustomNotificationHook(ExplorationHook):
    def on_exploration_complete(self, exploration_results):
        send_slack_notification(f"Exploration complete: {exploration_results.success_count} successful builds")

# Explorer with hooks
explorer = ExplorerEngine(
    build_runner_factory=FinnBuildRunner.create,
    hooks=[
        LoggingHook(log_level="INFO", log_file="exploration.log"),
        CachingHook(cache_dir=".cache"),
        EarlyStoppingHook(max_failures=50, min_successes=10),
        CustomNotificationHook()
    ]
)

results = explorer.explore(design_space)
```

#### Results Analysis

```python
# Basic metrics
print(f"Total combinations: {results.total_combinations}")
print(f"Evaluated: {results.evaluated_count}")
print(f"Success rate: {results.success_count/results.evaluated_count*100:.1f}%")

# Best configuration
if results.best_config:
    best_result = next(r for r in results.evaluations if r.config_id == results.best_config.id)
    print(f"Best: {results.best_config.id}")
    print(f"  Throughput: {best_result.metrics.throughput:.2f} fps")
    print(f"  Latency: {best_result.metrics.latency:.2f} μs")

# Top N configurations
top_configs = results.get_top_n_configs(n=10, metric="throughput")
for config, result in top_configs:
    print(f"{config.id}: {result.metrics.throughput:.2f} fps")

# Pareto frontier
print(f"Pareto optimal configurations: {len(results.pareto_optimal)}")
for config in results.pareto_optimal:
    result = next(r for r in results.evaluations if r.config_id == config.id)
    print(f"  {config.id}: {result.metrics.throughput:.1f} fps, "
          f"{result.metrics.lut_utilization:.1%} LUT")

# Failure analysis
failure_summary = results.get_failed_summary()
for error_msg, count in failure_summary.items():
    print(f"'{error_msg}': {count} failures")
```

## Advanced Features

### Resume Capability

```python
# Long-running exploration with checkpointing
explorer = ExplorerEngine(
    build_runner_factory=FinnBuildRunner.create,
    hooks=[CachingHook(cache_dir=".cache/my_exploration")]
)

try:
    results = explorer.explore(design_space)
except KeyboardInterrupt:
    print("Exploration interrupted")

# Resume from last checkpoint
results = explorer.explore(
    design_space,
    resume_from="dse_abc12345_config_00075"  # Last completed config
)
```

### Custom Build Runners

```python
from brainsmith.core.phase2.interfaces import BuildRunnerInterface
from brainsmith.core.phase2.data_structures import BuildResult, BuildStatus

class CustomBuildRunner(BuildRunnerInterface):
    def __init__(self, backend_type: str):
        self.backend_type = backend_type
    
    def run(self, config: BuildConfig) -> BuildResult:
        result = BuildResult(
            config_id=config.id,
            status=BuildStatus.RUNNING
        )
        
        try:
            # Custom build logic here
            # Model path is in config.model_path
            # All configuration in config object
            
            # Simulate build
            metrics = self.execute_build(config)
            result.metrics = metrics
            result.complete(BuildStatus.SUCCESS)
            
        except Exception as e:
            result.complete(BuildStatus.FAILED, error_message=str(e))
        
        return result
    
    def execute_build(self, config: BuildConfig) -> BuildMetrics:
        # Implementation-specific build logic
        pass

# Use custom runner
results = explore(
    design_space=design_space,
    build_runner_factory=lambda: CustomBuildRunner("custom_backend")
)
```

### Hook System Extensions

```python
class MLGuidedExplorationHook(ExplorationHook):
    def __init__(self, model_path: str):
        self.predictor = load_performance_model(model_path)
        self.promising_configs = []
    
    def on_combinations_generated(self, configs):
        # Use ML model to predict promising configurations
        predictions = self.predictor.predict([self.config_to_features(c) for c in configs])
        
        # Sort by predicted performance
        config_scores = list(zip(configs, predictions))
        config_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 20% for evaluation
        top_count = max(1, len(configs) // 5)
        self.promising_configs = [c for c, _ in config_scores[:top_count]]
        
        logger.info(f"ML guidance: selected {len(self.promising_configs)} promising configs")
    
    def config_to_features(self, config):
        # Convert BuildConfig to feature vector for ML model
        return [
            len(config.kernels),
            len(config.transforms),
            # ... other features
        ]

# Use ML-guided exploration
ml_hook = MLGuidedExplorationHook("models/performance_predictor.pkl")
explorer = ExplorerEngine(
    build_runner_factory=FinnBuildRunner.create,
    hooks=[LoggingHook(), CachingHook(), ml_hook]
)
```

## Performance Guidelines

### Small Design Spaces (< 100 configurations)

```python
# Direct exploration without caching
results = explore(
    design_space=design_space,
    build_runner_factory=MockBuildRunner
)
```

### Medium Design Spaces (100-10,000 configurations)

```python
# Use caching and logging
explorer = ExplorerEngine(
    build_runner_factory=FinnBuildRunner.create,
    hooks=[
        LoggingHook(log_level="INFO"),
        CachingHook(cache_dir=".cache")
    ]
)

results = explorer.explore(design_space)
```

### Large Design Spaces (> 10,000 configurations)

```python
# Use all optimizations: caching, early stopping, constraints
explorer = ExplorerEngine(
    build_runner_factory=FinnBuildRunner.create,
    hooks=[
        LoggingHook(log_level="INFO", log_file="large_exploration.log"),
        CachingHook(cache_dir=".cache"),
        EarlyStoppingHook(max_failures=100, target_successes=50)
    ]
)

# Use tight constraints to reduce space
design_space.search_config.constraints.extend([
    SearchConstraint("lut_utilization", "<=", 0.75),
    SearchConstraint("throughput", ">=", 500.0)
])

results = explorer.explore(design_space)
```

## Troubleshooting

### Common Issues

**Blueprint Parsing Errors**
```python
# Check YAML syntax
import yaml
with open("blueprint.yaml") as f:
    try:
        data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"YAML syntax error: {e}")

# Validate blueprint structure
design_space = forge("model.onnx", "blueprint.yaml")  # Will show specific errors
```

**Plugin Not Found Errors**
```python
# Check available plugins
from brainsmith.core.plugins import get_registry
registry = get_registry()
print("Available kernels:", list(registry.kernels.keys()))
print("Available transforms:", list(registry.transforms.keys()))

# Check plugin backends
print("MatMul backends:", registry.get_backends_by_kernel("MatMul"))
```

**Memory Issues with Large Design Spaces**
```python
# Check design space size before exploration
total_combos = design_space.get_total_combinations()
if total_combos > 50000:
    print(f"Warning: Large design space ({total_combos:,} combinations)")
    print("Consider adding constraints or using sampling")

# Add constraints to reduce space
design_space.search_config.constraints.append(
    SearchConstraint("lut_utilization", "<=", 0.8)
)
```

**Failed Builds**
```python
# Check failure patterns
results = explore(design_space, build_runner_factory)
failure_summary = results.get_failed_summary()
for error, count in failure_summary.items():
    print(f"{error}: {count} occurrences")

# Use mock runner to test exploration logic
mock_results = explore(
    design_space, 
    lambda: MockBuildRunner(success_rate=0.9)
)
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive joint architecture documentation
- **[Phase 1 Architecture](phase1/PHASE1_ARCHITECTURE.md)** - Design Space Constructor details
- **[Phase 2 Architecture](phase2/PHASE2_ARCHITECTURE.md)** - Design Space Explorer details
- **[Plugin System](plugins/ARCHITECTURE.md)** - Plugin system documentation
- **[Integration Reports](PHASE1_INTEGRATION_REPORT.md)** - Integration analysis

## System Status

✅ **Phase 1: Complete** - Design Space Constructor fully implemented
✅ **Phase 2: Complete** - Design Space Explorer fully implemented  
🚧 **Phase 3: In Development** - Build Runner implementation ongoing

The Phase 1-2 system is production-ready and provides a complete solution for design space construction and exploration with mock build execution.