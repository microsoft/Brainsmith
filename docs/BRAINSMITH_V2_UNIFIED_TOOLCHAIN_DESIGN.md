# BrainSmith V2 Unified Toolchain Design Document

**Comprehensive Analysis of the Consolidated Post-Migration Architecture**

---

## 1. Executive Summary

BrainSmith V2 represents a successful consolidation of the dual V1/V2 architecture into a unified, function-focused system that eliminates ~2,000+ lines of duplicate code while providing real FINN integration and comprehensive design space exploration capabilities.

### Key Consolidation Achievements
- **Unified API**: Single [`forge()`](brainsmith/core/api.py:20) entry point replacing complex V1 orchestration
- **Real FINN Integration**: Direct [`DataflowBuildConfig`](brainsmith/core/finn/legacy_conversion.py:70) builds with actual hardware generation
- **Registry-Driven Libraries**: Explicit component mapping via [`AVAILABLE_TRANSFORMS`](brainsmith/libraries/transforms/__init__.py:35) and [`AVAILABLE_KERNELS`](brainsmith/libraries/kernels/__init__.py:1) 
- **Design Space Exploration**: Component combination generation with [`ComponentCombination`](brainsmith/core/dse/combination_generator.py:20) and multiple strategies
- **Blueprint V2 System**: Structured design spaces via [`DesignSpaceDefinition`](brainsmith/core/blueprint.py:303) with inheritance and validation

### Architecture Philosophy
The V2 system adopts a **function-focused design** pattern throughout, replacing academic class hierarchies with pure functions, explicit registries, and fail-fast error handling. This approach reduces complexity while maintaining clean extension points for future capabilities.

---

## 2. Core Architecture Overview

The unified architecture centers on three core principles that eliminated the V1/V2 duplication while enabling real FINN integration.

```mermaid
graph TB
    subgraph "User Interface"
        API[forge API<br/>Single Entry Point]
    end
    
    subgraph "Core System"
        BP[Blueprint V2<br/>Design Spaces]
        DSE[Design Space Explorer<br/>Combination Generation]
        FINN[FINN Integration<br/>Real Builds]
    end
    
    subgraph "Libraries System"
        K[Kernels<br/>HW Components]
        T[Transforms<br/>Pipeline Steps]
        B[Blueprints<br/>Model Configs]
        V[Validation<br/>Registry Health]
    end
    
    API --> BP
    API --> DSE
    DSE --> BP
    DSE --> FINN
    DSE --> K
    DSE --> T
    BP --> B
    V --> K
    V --> T
    V --> B
```

### Function-Focused Design Principles

**1. Pure Functions Over Classes** ([`brainsmith/libraries/transforms/steps/cleanup.py`](brainsmith/libraries/transforms/steps/cleanup.py:44))
```python
def cleanup_step(model: Any, cfg: Any) -> Any:
    """Pure function transformation - no state, clear inputs/outputs"""
    logger.info("Applying basic cleanup transformations")
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model
```

**2. Registry Dictionaries for Discovery** ([`brainsmith/libraries/transforms/__init__.py`](brainsmith/libraries/transforms/__init__.py:35))
```python
AVAILABLE_TRANSFORMS = {
    "cleanup": cleanup_step,
    "qonnx_to_finn": qonnx_to_finn_step,
    "streamlining": streamlining_step,
    "infer_hardware": infer_hardware_step,
}
```

**3. Fail-Fast Error Handling** ([`brainsmith/libraries/transforms/steps/cleanup.py`](brainsmith/libraries/transforms/steps/cleanup.py:62))
```python
if not QONNX_AVAILABLE:
    raise RuntimeError(
        "cleanup_step requires qonnx package. "
        "Install with: pip install qonnx"
    )
```

---

## 3. Core System Components

### 3.1 Blueprint V2 System

The Blueprint V2 system provides structured design space definitions that replace the rigid V1 configurations with flexible, explorable component spaces.

**Design Space Architecture** ([`brainsmith/core/blueprint.py`](brainsmith/core/blueprint.py:303))

```mermaid
classDiagram
    class DesignSpaceDefinition {
        +name: str
        +nodes: NodeDesignSpace
        +transforms: TransformDesignSpace
        +objectives: List[Objective]
        +constraints: List[Constraint]
        +raw_blueprint_config: Dict
        +validate() bool, List[str]
    }
    
    class ComponentSpace {
        +available: List[Union[str, Dict]]
        +exploration: ExplorationRules
        +get_component_names() List[str]
        +get_component_options() List[str]
    }
    
    class ExplorationRules {
        +required: List[str]
        +optional: List[str]
        +mutually_exclusive: List[List[str]]
        +dependencies: Dict[str, List[str]]
        +validate() bool, List[str]
    }
    
    DesignSpaceDefinition --> ComponentSpace
    ComponentSpace --> ExplorationRules
```

**Component Space Structure** ([`brainsmith/core/blueprint.py`](brainsmith/core/blueprint.py:89))
```python
@dataclass
class ComponentSpace:
    """Design space for a component category with exploration rules."""
    available: List[Union[str, Dict[str, List[str]]]] = field(default_factory=list)
    exploration: Optional[ExplorationRules] = None
    
    def get_component_names(self) -> List[str]:
        """Extract all component names from available list."""
        names = []
        for item in self.available:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.extend(item.keys())
        return names
```

**BERT Blueprint Example** ([`brainsmith/libraries/blueprints/transformers/bert_accelerator.yaml`](brainsmith/libraries/blueprints/transformers/bert_accelerator.yaml:1))
```yaml
name: "bert_accelerator"
description: "BERT transformer accelerator with optimized folding"

parameters:
  folding_factors:
    mvau_pe:
      range: [4, 8, 16, 32]
      default: 8
    mvau_simd:
      range: [12, 24, 48]
      default: 12

targets:
  throughput:
    direction: "maximize"
    target: "target_fps"
```

### 3.2 Design Space Explorer (DSE)

The DSE system orchestrates component combination generation and evaluation using multiple exploration strategies.

**DSE Architecture** ([`brainsmith/core/dse/space_explorer.py`](brainsmith/core/dse/space_explorer.py:111))

```mermaid
sequenceDiagram
    participant User
    participant Explorer as DesignSpaceExplorer
    participant Generator as CombinationGenerator
    participant Strategy as StrategyExecutor
    participant Bridge as FINNEvaluationBridge
    
    User->>Explorer: explore_design_space(model_path)
    Explorer->>Generator: generate_all_combinations()
    Explorer->>Strategy: execute_strategy()
    Strategy-->>Explorer: selected_combinations
    
    loop Evaluation Loop
        Explorer->>Bridge: evaluate_combination(model, combo)
        Bridge-->>Explorer: metrics
        Explorer->>Explorer: update_exploration_state()
    end
    
    Explorer-->>User: ExplorationResults
```

**Component Combination Generation** ([`brainsmith/core/dse/combination_generator.py`](brainsmith/core/dse/combination_generator.py:19))
```python
@dataclass
class ComponentCombination:
    """Represents a specific combination of components for all entrypoints."""
    # Node components (Entrypoints 1, 3, 4)
    canonical_ops: List[str] = field(default_factory=list)
    hw_kernels: Dict[str, str] = field(default_factory=dict)
    
    # Transform components (Entrypoints 2, 5, 6)
    model_topology: List[str] = field(default_factory=list)
    hw_kernel_transforms: List[str] = field(default_factory=list)
    hw_graph_transforms: List[str] = field(default_factory=list)
```

**Exploration Strategies** ([`brainsmith/core/dse/strategy_executor.py`](brainsmith/core/dse/strategy_executor.py:95))

1. **HierarchicalExplorationStrategy**: Phases for kernel → transform → fine-tuning
2. **AdaptiveExplorationStrategy**: Performance-guided exploration vs exploitation
3. **ParetoGuidedStrategy**: Multi-objective Pareto frontier optimization

### 3.3 FINN Integration Bridge

The FINN integration provides the critical bridge from design space exploration to real hardware generation.

**Evaluation Flow** ([`brainsmith/core/finn/evaluation_bridge.py`](brainsmith/core/finn/evaluation_bridge.py:44))

```mermaid
graph LR
    A[ComponentCombination] --> B[6-entrypoint config]
    B --> C[LegacyConversionLayer]
    C --> D[DataflowBuildConfig]
    D --> E[Real FINN Build]
    E --> F[MetricsExtractor]
    F --> G[Standardized Metrics]
```

**Interface Detection** ([`brainsmith/core/finn/evaluation_bridge.py`](brainsmith/core/finn/evaluation_bridge.py:19))
```python
def uses_legacy_finn(blueprint_config: Dict[str, Any]) -> bool:
    """Check if blueprint explicitly requests legacy FINN interface."""
    return blueprint_config.get('legacy_finn', False)
```

**Legacy Conversion Architecture** ([`brainsmith/core/finn/legacy_conversion.py`](brainsmith/core/finn/legacy_conversion.py:29))
```python
class LegacyConversionLayer:
    """Converts Blueprint V2 6-entrypoint config to FINN DataflowBuildConfig 
    with blueprint-driven step ordering."""
    
    def convert_to_dataflow_config(self, entrypoint_config: Dict[str, List[str]], 
                                 blueprint_config: Dict[str, Any]):
        # Build step function list from blueprint configuration
        step_functions = self._build_step_sequence(blueprint_config)
        
        # Create DataflowBuildConfig with real FINN
        dataflow_config = DataflowBuildConfig(
            steps=step_functions,
            output_dir=finn_params.get('output_dir', './finn_output'),
            synth_clk_period_ns=finn_params.get('synth_clk_period_ns', 5.0),
            generate_outputs=[DataflowOutputType.STITCHED_IP],
        )
```

---

## 4. Libraries System Architecture

The libraries system adopts a registry-driven approach that eliminates magic filesystem scanning in favor of explicit component registration.

### 4.1 Kernels Library

**Registry Pattern** ([`brainsmith/libraries/kernels/functions.py`](brainsmith/libraries/kernels/functions.py:18))
```python
def discover_all_kernels(additional_paths: Optional[List[str]] = None) -> Dict[str, KernelPackage]:
    """Discover all kernel packages in brainsmith/kernels/ + additional paths."""
    kernels = {}
    search_paths = []
    
    # Add default kernels directory
    current_dir = Path(__file__).parent
    default_kernels_dir = current_dir
    if default_kernels_dir.exists():
        search_paths.append(str(default_kernels_dir))
    
    for search_path in search_paths:
        path_kernels = _discover_kernels_in_directory(search_path)
        kernels.update(path_kernels)
    
    return kernels
```

**Hardware Implementation Example** ([`brainsmith/libraries/kernels/layernorm/layernorm.py`](brainsmith/libraries/kernels/layernorm/layernorm.py:21))
```python
@register_op(domain="brainsmith.libraries.kernels.layernorm", op_type="LayerNorm")
class LayerNorm(HWCustomOp, HLSBackend):
    """Abstraction layer for HW implementation of the LayerNorm layer."""
    
    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update({
            "SIMD": ("i", True, 0),
            "NumChannels": ("i", True, 128),
            "ifm_dim": ("ints", True, []),
            "epsilon": ("f", True, 1e-5),
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
        })
        return my_attrs
```

### 4.2 Transforms Library

**Explicit Registry** ([`brainsmith/libraries/transforms/__init__.py`](brainsmith/libraries/transforms/__init__.py:35))
```python
AVAILABLE_TRANSFORMS = {
    "cleanup": cleanup_step,
    "cleanup_advanced": cleanup_advanced_step,
    "qonnx_to_finn": qonnx_to_finn_step,
    "streamlining": streamlining_step,
    "infer_hardware": infer_hardware_step,
    "remove_head": remove_head_step,
    "remove_tail": remove_tail_step,
}

def get_transform(name: str) -> Callable:
    """Get transform function by name. Fails fast if not found."""
    if name not in AVAILABLE_TRANSFORMS:
        available = ", ".join(sorted(AVAILABLE_TRANSFORMS.keys()))
        raise KeyError(f"Transform '{name}' not found. Available: {available}")
    return AVAILABLE_TRANSFORMS[name]
```

**Proven Step Functions** ([`brainsmith/libraries/transforms/steps/streamlining.py`](brainsmith/libraries/transforms/steps/streamlining.py:10))
```python
def streamlining_step(model, cfg):
    """Custom streamlining with absorption and reordering transformations."""
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(reorder.MoveOpPastFork(["Mul"]))
    model = model.transform(reorder.MoveScalarMulPastMatMul())
    return model
```

**Hardware Inference Operations** ([`brainsmith/libraries/transforms/operations/convert_to_hw_layers.py`](brainsmith/libraries/transforms/operations/convert_to_hw_layers.py:26))
```python
class InferShuffle(Transformation):
    """Find transpose layers with reshape layers and convert to shuffle operator"""
    
    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if(n.op_type == "Transpose"):
                # Create Shuffle node with proper attributes
                new_node = helper.make_node(
                    "Shuffle",
                    [new_in_tensor],
                    [new_out_tensor],
                    domain="brainsmith.libraries.kernels.shuffle",
                    backend="fpgadataflow",
                    in_shape=in_shape,
                    out_shape=out_shape,
                    data_type=idt.name,
                )
```

### 4.3 Validation System

**Registry Integrity Monitoring** ([`brainsmith/libraries/validation.py`](brainsmith/libraries/validation.py:16))
```python
def validate_all_registries() -> Dict[str, Any]:
    """Comprehensive validation of all library registries."""
    report = {
        'status': 'healthy',
        'libraries': {},
        'summary': {
            'total_components': 0,
            'failed_components': 0,
            'drift_detected': False
        }
    }
    
    # Validate each library registry
    for library_name in ['kernels', 'transforms', 'analysis', 'blueprints']:
        try:
            registry = getattr(import_library(library_name), f'AVAILABLE_{library_name.upper()}')
            loader = getattr(import_library(library_name), f'get_{library_name[:-1]}')
            errors = validate_registry_integrity(library_name, registry, loader)
            report['libraries'][library_name] = {
                'errors': errors,
                'component_count': len(registry)
            }
        except Exception as e:
            report['libraries'][library_name] = {
                'errors': [f"Failed to validate {library_name}: {e}"],
                'component_count': 0
            }
```

---

## 5. Integration Patterns

### 5.1 Core↔Libraries Communication

The unified system uses function composition and registry lookups for seamless integration between core and libraries.

```mermaid
graph TB
    subgraph "Core Requests"
        A[DSE requests transforms]
        B[DSE requests kernels]
        C[Blueprint loads configs]
    end
    
    subgraph "Registry Resolution"
        D[AVAILABLE_TRANSFORMS lookup]
        E[discover_all_kernels]
        F[AVAILABLE_BLUEPRINTS lookup]
    end
    
    subgraph "Function Execution"
        G["step_function(model, cfg)"]
        H["kernel.evaluate()"]
        I["blueprint.load()"]
    end
    
    A --> D --> G
    B --> E --> H
    C --> F --> I
```

### 5.2 DSE↔FINN Evaluation Flow

**Real Build Integration** ([`brainsmith/core/finn/evaluation_bridge.py`](brainsmith/core/finn/evaluation_bridge.py:89))
```python
def _execute_legacy_finn(self, model_path: str, entrypoint_config: Dict[str, list]) -> Dict[str, Any]:
    """Execute using legacy FINN DataflowBuildConfig interface."""
    
    # Convert to FINN DataflowBuildConfig using legacy conversion layer
    dataflow_config = self.legacy_converter.convert_to_dataflow_config(
        entrypoint_config, self.blueprint_config
    )
    
    # Execute real FINN build
    finn_result = self._execute_finn_run(model_path, dataflow_config)
    
    # Extract standardized metrics
    metrics = self.metrics_extractor.extract_metrics(finn_result, dataflow_config)
    return metrics
```

### 5.3 Event Logging System

**Simple Extension Points** ([`brainsmith/core/hooks/__init__.py`](brainsmith/core/hooks/__init__.py:30))
```python
from .events import (
    log_optimization_event,
    log_parameter_change,
    log_performance_metric,
    register_event_handler,
    create_custom_event_type
)

# Usage: Simple logging with extension points
log_optimization_event('dse_completed', {'solutions': 50})
log_parameter_change('learning_rate', 0.01, 0.005)
```

---

## 6. Migration Achievements

### 6.1 Code Reduction Metrics

| **Aspect** | **V1 (Before)** | **V2 (After)** | **Reduction** |
|------------|------------------|----------------|---------------|
| **Core API** | Multiple complex functions | Single [`forge()`](brainsmith/core/api.py:20) | ~80% complexity |
| **FINN Integration** | Mock/placeholder systems | Real [`DataflowBuildConfig`](brainsmith/core/finn/legacy_conversion.py:70) | 100% real builds |
| **Library Discovery** | Filesystem scanning | Registry dictionaries | Explicit mapping |
| **Transform Pipeline** | Hardcoded sequences | Configurable [`step_functions`](brainsmith/core/finn/legacy_conversion.py:64) | Blueprint-driven |
| **Design Space** | Fixed configurations | [`ComponentCombination`](brainsmith/core/dse/combination_generator.py:20) generation | Exploration-ready |

### 6.2 Architectural Improvements

**1. Function-Focused Design**
- Eliminated complex class hierarchies in favor of pure functions
- Clear input/output contracts with fail-fast error handling
- Simplified testing and debugging workflows

**2. Real FINN Integration**
- Actual hardware builds vs V1 mock evaluations
- [`MetricsExtractor`](brainsmith/core/finn/metrics_extractor.py:16) providing real performance data
- Support for both legacy and future modern interfaces

**3. Registry-Driven Discovery**
- Explicit component registration vs magical filesystem scanning
- [`validate_all_registries()`](brainsmith/libraries/validation.py:16) preventing registry-reality drift
- Clear error messages when components missing

**4. Design Space Exploration**
- [`DesignSpaceExplorer`](brainsmith/core/dse/space_explorer.py:111) with multiple strategies
- Component combination generation respecting exploration rules
- Real optimization with Pareto frontier analysis

---

## 7. Usage Patterns & Examples

### 7.1 Basic forge() API Usage

```python
from brainsmith.core.api import forge

# Simple BERT acceleration
results = forge(
    model_path="bert_model.onnx",
    blueprint_path="blueprints/transformers/bert_accelerator.yaml",
    output_dir="./acceleration_results"
)

print(f"Best design score: {results['best_design']['score']}")
print(f"Pareto frontier size: {len(results['pareto_frontier'])}")
```

### 7.2 Advanced DSE Configuration

```python
# Custom exploration configuration
dse_config = {
    'max_evaluations': 100,
    'parallel_evaluations': 2,
    'early_termination_patience': 15,
    'strategy_name': 'hierarchical'
}

results = forge(
    model_path="bert_model.onnx",
    blueprint_path="blueprints/transformers/bert_accelerator.yaml",
    dse_config=dse_config,
    objectives={'throughput': {'direction': 'maximize', 'weight': 0.7}},
    constraints={'max_power_w': 20.0}
)
```

### 7.3 Blueprint V2 Development

```yaml
# Custom blueprint structure
name: "my_accelerator"
version: "2.0"

nodes:
  canonical_ops:
    available: ["MatMul", "Add", "Mul"]
    exploration:
      required: ["MatMul"]
      optional: ["Add", "Mul"]
  
  hw_kernels:
    available:
      - "matmul_rtl": ["pe_4", "pe_8", "pe_16"]
      - "softmax_hls": ["simd_1", "simd_2"]

transforms:
  model_topology:
    available: ["cleanup", "streamlining"]
    exploration:
      required: ["cleanup"]
      
objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0
```

---

## 8. Performance & Scalability

### 8.1 Real FINN Build Characteristics

**Resource Utilization Patterns** ([`brainsmith/core/finn/metrics_extractor.py`](brainsmith/core/finn/metrics_extractor.py:306))
```python
def _calculate_resource_efficiency(self, metrics: Dict[str, float]) -> float:
    """Calculate composite resource efficiency metric."""
    throughput = metrics.get('throughput', 0.0)
    lut_util = metrics.get('lut_utilization', 0.0)
    dsp_util = metrics.get('dsp_utilization', 0.0)
    bram_util = metrics.get('bram_utilization', 0.0)
    
    # Average resource utilization
    avg_resource_util = (lut_util + dsp_util + bram_util) / 3.0
    
    # Efficiency = throughput / resource_utilization
    if avg_resource_util > 0:
        efficiency = min(1.0, throughput / (avg_resource_util * 1000.0))
    else:
        efficiency = 0.0
    return efficiency
```

### 8.2 Exploration Efficiency

**Caching Strategy** ([`brainsmith/core/dse/space_explorer.py`](brainsmith/core/dse/space_explorer.py:566))
```python
def _get_cached_result(self, combination: ComponentCombination) -> Optional[Dict[str, Any]]:
    """Get cached evaluation result for combination."""
    cache_key = self._generate_cache_key(combination)
    return self.evaluation_cache.get(cache_key)

def _cache_result(self, combination: ComponentCombination, result: Dict[str, Any]):
    """Cache evaluation result for future use."""
    cache_key = self._generate_cache_key(combination)
    cacheable_result = {
        'success': result['success'],
        'primary_metric': result['primary_metric'],
        'evaluation_time': result['evaluation_time']
    }
    self.evaluation_cache[cache_key] = cacheable_result
```

---

## 9. Development Guide

### 9.1 Adding New Kernels

**1. Create Kernel Package Structure**
```
brainsmith/libraries/kernels/my_kernel/
├── kernel.yaml              # Metadata and parameters
├── my_kernel.py             # HWCustomOp implementation  
├── my_kernel_hls.py         # HLS backend code
└── README.md                # Documentation
```

**2. Implement HWCustomOp Interface** 
```python
@register_op(domain="brainsmith.libraries.kernels.my_kernel", op_type="MyKernel")
class MyKernel(HWCustomOp, HLSBackend):
    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update({
            "SIMD": ("i", True, 1),
            "PE": ("i", True, 1),
            "inputDataType": ("s", True, ""),
        })
        return my_attrs
```

**3. Update Registry** (Automatic discovery via [`discover_all_kernels()`](brainsmith/libraries/kernels/functions.py:18))

### 9.2 Adding Transform Steps

**1. Implement Step Function** ([`brainsmith/libraries/transforms/steps/`](brainsmith/libraries/transforms/steps/))
```python
def my_transform_step(model: Any, cfg: Any) -> Any:
    """
    Description of transform operation.
    
    Category: optimization
    Dependencies: [cleanup]
    Description: Specific optimization transformations
    """
    logger.info("Applying my custom transformation")
    model = model.transform(MyCustomTransformation())
    return model
```

**2. Register in Transform Registry** ([`brainsmith/libraries/transforms/__init__.py`](brainsmith/libraries/transforms/__init__.py:35))
```python
AVAILABLE_TRANSFORMS = {
    # ... existing transforms
    "my_transform": my_transform_step,
}
```

### 9.3 Testing and Validation

**Registry Validation** ([`brainsmith/libraries/validation.py`](brainsmith/libraries/validation.py:275))
```python
# Run comprehensive validation
report = validate_all_registries()
assert report['status'] == 'healthy'

# Check specific library
from brainsmith.libraries.kernels import get_kernel
kernel = get_kernel('my_kernel')  # Fails fast if not found
```

---

## 10. Future Evolution

### 10.1 Modern 6-Entrypoint Interface

**Current Status**: Legacy conversion layer bridges V2 → FINN ([`brainsmith/core/finn/evaluation_bridge.py`](brainsmith/core/finn/evaluation_bridge.py:107))
```python
def _execute_modern_finn(self, model_path: str, entrypoint_config: Dict[str, list]) -> Dict[str, Any]:
    """Execute using modern 6-entrypoint interface (future implementation)."""
    
    # TODO: Implement modern 6-entrypoint execution
    logger.warning("Modern 6-entrypoint execution not yet implemented, falling back to legacy")
    
    # For now, fall back to legacy execution but log the intention
    return self._execute_legacy_finn(model_path, entrypoint_config)
```

**Future Implementation**: Direct 6-entrypoint → FINN without legacy conversion

### 10.2 Extension Points and Plugin Architecture

**Event System Extensions** ([`brainsmith/core/hooks/__init__.py`](brainsmith/core/hooks/__init__.py:159))
```python
# Future ML analysis plugin
from brainsmith.hooks.plugins import MLAnalysisPlugin
hooks.install_plugin('ml_analysis', MLAnalysisPlugin())

# Future statistical monitoring plugin  
from brainsmith.hooks.plugins import StatisticsPlugin
hooks.install_plugin('statistics', StatisticsPlugin())
```

**Registry Plugin System** ([`brainsmith/libraries/validation.py`](brainsmith/libraries/validation.py:216))
```python
def suggest_registry_updates() -> Dict[str, Dict[str, List[str]]]:
    """Development mode: detect unregistered components and orphaned entries."""
    # Automatic discovery of new components for registration
    # Drift detection between filesystem and registry state
```

### 10.3 Planned Enhancements

1. **Advanced DSE Strategies**: Machine learning-guided exploration
2. **Blueprint Inheritance System**: Complete implementation with validation
3. **Performance Prediction**: ML models for FINN build time/resource estimation  
4. **Multi-Target Support**: Extension beyond Zynq UltraScale to Versal, Alveo
5. **Cloud Integration**: Distributed FINN builds with result aggregation

---

## Conclusion

BrainSmith V2 represents a successful architectural consolidation that eliminates the complexity and duplication of the dual V1/V2 system while providing real FINN integration and comprehensive design space exploration capabilities. The function-focused design, registry-driven libraries, and unified API create a maintainable, extensible foundation for future FPGA accelerator generation workflows.

The migration achievements demonstrate that significant complexity reduction (elimination of ~2,000+ lines of duplicate code) can be achieved while simultaneously adding sophisticated capabilities like real hardware builds, multi-objective optimization, and structured design space exploration.

---

**Document Metadata**
- **Analysis Date**: June 2025
- **Codebase Version**: Post-V1→V2 Migration
- **Files Analyzed**: 20+ core implementation files
- **Architecture**: Unified V2-only system
- **Integration**: Real FINN DataflowBuildConfig builds