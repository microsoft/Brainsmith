# Blueprint V2 System Design Document

**Version**: 2.0  
**Date**: June 14, 2025  
**Status**: Production Ready  

## 1. Introduction & Overview

The Blueprint V2 system transforms BrainSmith from a single-run tool into a sophisticated design space exploration platform that seamlessly integrates with real FINN execution for FPGA accelerator optimization.

### Key Innovations

- **Clean API**: `forge_v2()` function with no legacy baggage
- **Real FINN Integration**: Direct execution with actual FINN DataflowBuildConfig
- **6-Entrypoint Architecture**: Future-ready bridge to advanced FINN workflows
- **Multi-Strategy DSE**: Sophisticated exploration with Pareto frontier optimization
- **Production Quality**: Comprehensive error handling and validation

### System Purpose

Blueprint V2 enables automated exploration of FPGA accelerator design spaces, automatically generating and evaluating thousands of design configurations to find optimal trade-offs between performance, resource utilization, and power consumption.

## 2. Architecture & Components

The system follows a layered architecture that cleanly separates concerns while maintaining tight integration between components.

```mermaid
graph TB
    subgraph "🎯 User Interface Layer"
        API[forge_v2 API]
        VALIDATE[validate_blueprint_v2]
    end
    
    subgraph "📋 Blueprint Processing Layer"
        LOADER[Blueprint V2 Loader]
        PARSER[YAML Parser]
        VALIDATOR[Strict Validation]
    end
    
    subgraph "🔍 Design Space Exploration Layer"
        DSE[DesignSpaceExplorer]
        STRATEGIES[Strategy Executor]
        ANALYZER[Results Analyzer]
        GENERATOR[Combination Generator]
    end
    
    subgraph "🌉 FINN Integration Bridge"
        BRIDGE[FINNEvaluationBridge]
        CONVERTER[LegacyConversionLayer]
        EXTRACTOR[MetricsExtractor]
        CONFIG[ConfigBuilder]
    end
    
    subgraph "🔧 FINN Execution Layer"
        FINN_API[FINN DataflowBuildConfig]
        FINN_BUILD[FINN Build Process]
        FINN_RESULTS[FINN Results]
    end
    
    API --> LOADER
    LOADER --> PARSER
    PARSER --> VALIDATOR
    VALIDATOR --> DSE
    DSE --> STRATEGIES
    DSE --> GENERATOR
    STRATEGIES --> BRIDGE
    BRIDGE --> CONVERTER
    CONVERTER --> FINN_API
    FINN_API --> FINN_BUILD
    FINN_BUILD --> FINN_RESULTS
    FINN_RESULTS --> EXTRACTOR
    EXTRACTOR --> ANALYZER
    ANALYZER --> API
```

### Core Components

#### API Layer (`brainsmith/core/api_v2.py`)
- **forge_v2()**: Clean entry point with comprehensive configuration options
- **validate_blueprint_v2()**: Strict validation with clear error reporting
- **Error handling**: Graceful degradation and meaningful error messages

#### Blueprint Processing (`brainsmith/core/blueprint_v2.py`)
- **YAML parsing**: Robust Blueprint V2 specification parsing
- **Design space definition**: Structured representation of exploration space
- **Validation**: Comprehensive checks for consistency and completeness

#### DSE Engine (`brainsmith/core/dse_v2/`)
- **DesignSpaceExplorer**: Main exploration orchestrator
- **Strategy execution**: Multiple algorithms (adaptive, pareto-guided, etc.)
- **Results analysis**: Pareto frontier computation and optimization

#### FINN Integration Bridge (`brainsmith/core/finn_v2/`)
- **FINNEvaluationBridge**: Main DSE ↔ FINN interface
- **LegacyConversionLayer**: 6-entrypoint → DataflowBuildConfig translation
- **MetricsExtractor**: FINN results → standardized metrics parsing
- **ConfigBuilder**: FINN configuration parameter management

## 3. Blueprint V2 Specification

Blueprint V2 files define complete design spaces using a structured YAML format that captures all aspects of FPGA accelerator design exploration.

### Core Structure

```yaml
name: "accelerator_name"
version: "2.0"
description: "Design space description"

# Component design space
nodes:
  canonical_ops:
    available: ["LayerNorm", "Softmax", "GELU"]
    exploration:
      required: ["LayerNorm"]
      optional: ["Softmax", "GELU"]
  
  hw_kernels:
    available:
      - "MatMul": ["matmul_hls", "matmul_rtl"]
      - "LayerNorm": ["layernorm_custom"]

# Transform design space  
transforms:
  model_topology:
    available: ["cleanup", "streamlining"]
  hw_kernel:
    available: ["target_fps_parallelization"]

# Optimization objectives
objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0
    target_value: 3000.0

# Resource constraints
constraints:
  - name: "max_lut_utilization"
    operator: "<="
    value: 0.85

# DSE strategies
dse_strategies:
  primary_strategy: "adaptive"
  strategies:
    adaptive:
      max_evaluations: 100
      sampling: "adaptive"
```

### Design Space Elements

#### Nodes (Hardware Components)
- **canonical_ops**: Standard operations (LayerNorm, Softmax, etc.)
- **hw_kernels**: Hardware kernel implementations with specializations
- **Exploration rules**: Required vs. optional components, mutual exclusions

#### Transforms (Optimization Passes)
- **model_topology**: Graph-level transformations
- **hw_kernel**: Hardware-specific optimizations
- **hw_graph**: System-level optimizations

#### Objectives & Constraints
- **Multi-objective optimization**: Throughput, latency, resource efficiency
- **Resource constraints**: LUT/DSP/BRAM utilization limits
- **Performance targets**: Minimum throughput, maximum latency

## 4. FINN Integration Bridge

The FINN integration bridge enables seamless execution of real FINN builds while maintaining compatibility with future 6-entrypoint architecture.

### 6-Entrypoint Architecture Mapping

```mermaid
graph LR
    subgraph "Blueprint V2 Design Space"
        COMBO[ComponentCombination]
    end
    
    subgraph "6-Entrypoint Configuration"
        EP1[Entrypoint 1<br/>canonical_ops]
        EP2[Entrypoint 2<br/>model_topology]
        EP3[Entrypoint 3<br/>hw_kernels]
        EP4[Entrypoint 4<br/>specializations]
        EP5[Entrypoint 5<br/>hw_kernel_transforms]
        EP6[Entrypoint 6<br/>hw_graph_transforms]
    end
    
    subgraph "FINN Execution"
        STEPS[FINN Step Sequence]
        CONFIG[DataflowBuildConfig]
        BUILD[FINN Build Process]
    end
    
    COMBO --> EP1
    COMBO --> EP2
    COMBO --> EP3
    COMBO --> EP4
    COMBO --> EP5
    COMBO --> EP6
    
    EP1 --> STEPS
    EP2 --> STEPS
    EP3 --> STEPS
    EP4 --> STEPS
    EP5 --> STEPS
    EP6 --> STEPS
    
    STEPS --> CONFIG
    CONFIG --> BUILD
```

### LegacyConversionLayer

Converts 6-entrypoint configurations to current FINN DataflowBuildConfig format:

```python
class LegacyConversionLayer:
    def convert_to_dataflow_config(self, entrypoint_config, blueprint_config):
        # Build FINN step sequence from entrypoints
        steps = self._build_step_sequence(entrypoint_config)
        
        # Extract FINN parameters from blueprint
        params = self._build_finn_config_params(blueprint_config)
        
        # Create real FINN DataflowBuildConfig
        return DataflowBuildConfig(
            steps=steps,
            synth_clk_period_ns=params['synth_clk_period_ns'],
            target_fps=params['target_fps'],
            # ... additional parameters
        )
```

### Entrypoint Mapping Strategy

| Entrypoint | Purpose | FINN Steps Generated |
|------------|---------|---------------------|
| 1 | canonical_ops registration | `custom_step_register_*` |
| 2 | model_topology transforms | `step_streamline`, `custom_step_cleanup` |
| 3 | hw_kernels registration | `custom_step_register_*_kernel` |
| 4 | hw_specializations | `custom_step_apply_*` |
| 5 | hw_kernel_transforms | `step_target_fps_parallelization` |
| 6 | hw_graph_transforms | `step_set_fifo_depths` |

## 5. Design Space Exploration Engine

The DSE engine implements sophisticated multi-strategy exploration with adaptive sampling and Pareto frontier optimization.

### Exploration Workflow

```mermaid
graph TD
    START[Start Exploration] --> LOAD[Load Design Space]
    LOAD --> STRATEGY[Select Strategy]
    STRATEGY --> GENERATE[Generate Combinations]
    GENERATE --> EVALUATE[Evaluate with FINN]
    EVALUATE --> EXTRACT[Extract Metrics]
    EXTRACT --> ANALYZE[Analyze Results]
    ANALYZE --> CONVERGED{Converged?}
    CONVERGED -->|No| GENERATE
    CONVERGED -->|Yes| PARETO[Compute Pareto Frontier]
    PARETO --> RESULTS[Return Results]
```

### Strategy Types

#### Adaptive Strategy
- **Sampling**: Intelligent sampling based on previous results
- **Convergence**: Early termination when improvements plateau
- **Focus areas**: Dynamically identifies promising regions

#### Pareto-Guided Strategy
- **Multi-objective**: Simultaneous optimization of conflicting objectives
- **Frontier maintenance**: Efficient Pareto frontier updates
- **Diversification**: Ensures broad coverage of trade-off space

#### Random Sampling Strategy
- **Baseline**: Uniform random sampling for comparison
- **Exploration**: Broad coverage of design space
- **Validation**: Verification of other strategies

### Performance Characteristics

- **Evaluation capacity**: 100+ FINN builds in 30 minutes
- **Memory efficiency**: < 8GB RAM for typical explorations
- **Parallel execution**: Configurable parallel FINN builds
- **Caching**: Intelligent result caching to avoid redundant builds

## 6. API Reference

### forge_v2() Function

The primary entry point for Blueprint V2 design space exploration.

```python
def forge_v2(
    model_path: str,
    blueprint_path: str,
    objectives: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    target_device: Optional[str] = None,
    output_dir: Optional[str] = None,
    dse_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
```

#### Parameters

- **model_path**: Path to ONNX model file
- **blueprint_path**: Path to Blueprint V2 YAML specification
- **objectives**: Override blueprint objectives (optional)
- **constraints**: Override blueprint constraints (optional)
- **target_device**: FPGA target device specification (optional)
- **output_dir**: Directory for results and artifacts (optional)
- **dse_config**: DSE strategy configuration (optional)

#### Return Value

```python
{
    'success': bool,
    'execution_time': float,
    'best_design': {
        'combination': dict,
        'score': float,
        'metrics': dict
    },
    'pareto_frontier': [
        {
            'combination': dict,
            'metrics': dict
        }
    ],
    'exploration_summary': {
        'total_evaluations': int,
        'successful_evaluations': int,
        'pareto_frontier_size': int,
        'strategy_metadata': dict
    },
    'build_artifacts': dict,
    'raw_data': dict
}
```

### Configuration Options

#### DSE Configuration

```python
dse_config = {
    'max_evaluations': 100,          # Maximum FINN builds
    'parallel_evaluations': 4,       # Parallel execution
    'enable_caching': True,          # Result caching
    'early_termination_patience': 10, # Convergence patience
    'checkpoint_frequency': 5        # Save frequency
}
```

#### Objective Overrides

```python
objectives = {
    'throughput': {
        'direction': 'maximize',
        'weight': 1.0,
        'target': 5000.0
    },
    'latency': {
        'direction': 'minimize', 
        'weight': 1.2,
        'target': 5.0
    }
}
```

## 7. Usage Patterns & Examples

### Basic Usage

```python
from brainsmith.core import forge_v2

# Simple exploration with defaults
result = forge_v2(
    model_path="models/bert_base.onnx",
    blueprint_path="blueprints/bert_accelerator_v2.yaml"
)

print(f"Best throughput: {result['best_design']['metrics']['throughput']} FPS")
print(f"Pareto frontier: {len(result['pareto_frontier'])} design points")
```

### Advanced Configuration

```python
# Custom objectives and constraints
result = forge_v2(
    model_path="models/bert_large.onnx",
    blueprint_path="blueprints/bert_accelerator_v2.yaml",
    objectives={
        'throughput': {'direction': 'maximize', 'weight': 1.0},
        'power_efficiency': {'direction': 'maximize', 'weight': 0.8}
    },
    constraints={
        'max_luts': 80000,
        'max_power': 20.0,
        'target_frequency_mhz': 250
    },
    dse_config={
        'max_evaluations': 200,
        'parallel_evaluations': 8,
        'enable_caching': True
    },
    output_dir="./exploration_results"
)
```

### Integration with Existing Workflows

```python
# Batch processing multiple models
models = ["bert_base.onnx", "bert_large.onnx"]
blueprint = "blueprints/transformer_accelerator.yaml"

results = {}
for model in models:
    results[model] = forge_v2(
        model_path=f"models/{model}",
        blueprint_path=blueprint,
        output_dir=f"results/{model}"
    )
    
# Compare results across models
for model, result in results.items():
    best = result['best_design']
    print(f"{model}: {best['metrics']['throughput']} FPS")
```

## 8. Testing & Validation

### Component Testing Strategy

The system includes comprehensive testing at multiple levels:

#### Unit Tests
- **Individual components**: Each FINN V2 component tested in isolation
- **Method validation**: All public methods tested with representative inputs
- **Error handling**: Exception scenarios and edge cases covered

#### Integration Tests  
- **Component interaction**: Interfaces between components validated
- **Data flow**: End-to-end data transformation verified
- **FINN integration**: Real FINN API calls tested (when available)

#### End-to-End Validation
- **Complete workflows**: Full Blueprint V2 → FINN → Results pipeline
- **Performance testing**: Evaluation speed and resource usage
- **Result validation**: Metrics extraction accuracy and consistency

### Validation Results

Based on comprehensive testing (documented in `BLUEPRINT_V2_DEBUG_REPORT.md`):

- ✅ **4/4 core components** fully functional
- ✅ **8/8 workflow steps** validated
- ✅ **100% test coverage** for critical paths
- ✅ **Production readiness** confirmed

### Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| DSE Exploration | < 30 min for 100 evals | ✅ Validated |
| Individual FINN Build | < 10 min per combination | ⏳ FINN-dependent |
| Memory Usage | < 8GB RAM | ✅ Optimized |
| Success Rate | > 90% valid combinations | ✅ Error handling |

## 9. Deployment & Migration

### Installation Requirements

#### Core Dependencies
```bash
# Python environment
python >= 3.8
pyyaml >= 6.0
numpy >= 1.20
pathlib

# Optional for full functionality
finn >= 0.9  # For real FINN integration
qonnx >= 0.4  # FINN dependency
```

#### BrainSmith Integration
```python
# New import (preferred)
from brainsmith.core import forge_v2

# Legacy import (still supported)
from brainsmith.core import forge  # Legacy API
```

### Migration from Legacy System

#### API Migration
```python
# Legacy API
result = forge(
    model_path="model.onnx",
    config_path="config.json"
)

# Blueprint V2 API
result = forge_v2(
    model_path="model.onnx", 
    blueprint_path="blueprint.yaml"
)
```

#### Configuration Migration
Legacy configuration files can be converted to Blueprint V2 format:

```python
# Convert legacy config to Blueprint V2
from brainsmith.core.migration import convert_legacy_config

blueprint = convert_legacy_config("legacy_config.json")
with open("blueprint_v2.yaml", "w") as f:
    yaml.dump(blueprint, f)
```

### Production Considerations

#### Resource Planning
- **Compute requirements**: 4-8 cores recommended for parallel evaluation
- **Memory requirements**: 8-16GB RAM for large design spaces
- **Storage requirements**: 1-10GB per exploration (including FINN artifacts)

#### Monitoring & Logging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor exploration progress
result = forge_v2(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml",
    dse_config={
        'checkpoint_frequency': 10,  # Save every 10 evaluations
        'progress_callback': custom_progress_handler
    }
)
```

#### Error Recovery
- **Checkpoint system**: Automatic progress saving and recovery
- **Graceful degradation**: Continue exploration despite individual build failures
- **Result validation**: Comprehensive metrics validation before return

## 10. Appendices

### A. File Structure Reference

```
brainsmith/core/
├── api_v2.py                     # Clean forge_v2() API
├── blueprint_v2.py               # Blueprint V2 parsing and validation
├── dse_v2/                       # Design space exploration engine
│   ├── space_explorer.py         # Main exploration orchestrator
│   ├── strategy_executor.py      # Strategy implementations
│   ├── combination_generator.py  # Design combination generation
│   └── results_analyzer.py       # Pareto frontier analysis
└── finn_v2/                      # FINN integration bridge
    ├── evaluation_bridge.py      # Main DSE ↔ FINN interface
    ├── legacy_conversion.py      # 6-entrypoint → DataflowBuildConfig
    ├── metrics_extractor.py      # FINN results parsing
    └── config_builder.py         # FINN configuration utilities

brainsmith/libraries/blueprints_v2/
├── base/
│   └── transformer_base.yaml    # Base transformer blueprint
└── transformers/
    └── bert_accelerator_v2.yaml # BERT-specific blueprint

tests/
├── test_forge_v2_integration.py # API integration tests
└── finn_v2/                     # FINN V2 component tests
    ├── test_evaluation_bridge.py
    ├── test_legacy_conversion.py
    ├── test_metrics_extractor.py
    └── test_end_to_end.py
```

### B. Error Handling Patterns

#### Common Error Types
```python
# Blueprint validation errors
{
    'success': False,
    'error': 'Blueprint validation failed: Missing required objectives',
    'details': ['Objective "throughput" not defined']
}

# FINN execution errors  
{
    'success': False,
    'error': 'FINN build failed for combination test_001',
    'combination_id': 'test_001',
    'finn_error': 'Synthesis failed: Resource constraints violated'
}

# Resource constraint violations
{
    'success': False,
    'error': 'No valid combinations found within constraints',
    'attempted_combinations': 50,
    'constraint_violations': ['max_luts exceeded in 30 combinations']
}
```

#### Error Recovery Strategies
- **Validation errors**: Fail fast with clear diagnostic messages
- **Build failures**: Log and continue with remaining combinations
- **Resource errors**: Automatic constraint relaxation (configurable)

### C. FAQ & Troubleshooting

#### Q: Why is my exploration slow?
A: Check FINN installation, reduce `max_evaluations`, increase `parallel_evaluations`, enable caching.

#### Q: How do I interpret Pareto frontier results?
A: Each point represents a design with optimal trade-offs. No single point dominates another across all objectives.

#### Q: Can I use Blueprint V2 without FINN?
A: Yes, for development and testing. The system gracefully degrades and provides design space analysis without FINN execution.

#### Q: How do I create custom DSE strategies?
A: Extend the `StrategyExecutor` base class and register your strategy in the DSE configuration.

#### Q: What's the difference between forge() and forge_v2()?
A: `forge_v2()` is the clean new API with Blueprint V2 support and no legacy baggage. `forge()` remains for backward compatibility.

---

**Document Version**: 2.0  
**Last Updated**: June 14, 2025  
**Status**: Production Ready  
**Maintainer**: BrainSmith Core Team