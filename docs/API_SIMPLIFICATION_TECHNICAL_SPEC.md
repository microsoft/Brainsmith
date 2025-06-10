# 🔧 **BrainSmith API Simplification - Technical Specification**
## Detailed Implementation Guide for Code Review Response

---

## 📋 **Executive Summary**

This document provides the complete technical specification for implementing the code review feedback to simplify the BrainSmith API. The primary change is replacing the complex multi-function API with a single `forge` function that serves as the core toolchain.

---

## 🎯 **Core Design: The `forge` Function**

### **Function Signature**
```python
def forge(
    model_path: str,
    blueprint_path: str,
    objectives: Dict[str, Any] = None,
    constraints: Dict[str, Any] = None,
    target_device: str = None,
    is_hw_graph: bool = False,
    build_core: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
```

### **Input Specifications**

#### **Required Inputs**
- **`model_path`**: Path to pre-quantized ONNX neural network model
- **`blueprint_path`**: Path to YAML file containing declarative design space specification

#### **Optional Configuration**
- **`objectives`**: Dictionary defining optimization targets
  ```python
  objectives = {
      'throughput': {'direction': 'maximize', 'target': 1000, 'weight': 1.0},
      'latency': {'direction': 'minimize', 'target': 10, 'weight': 0.8},
      'power': {'direction': 'minimize', 'weight': 0.6}
  }
  ```

- **`constraints`**: Hardware resource budgets and limits
  ```python
  constraints = {
      'max_luts': 0.8,           # Maximum LUT utilization (80%)
      'max_dsps': 0.7,           # Maximum DSP utilization (70%)
      'max_brams': 0.6,          # Maximum BRAM utilization (60%)
      'max_power': 20.0,         # Maximum power consumption (W)
      'target_frequency': 200e6   # Target clock frequency (Hz)
  }
  ```

- **`target_device`**: FPGA device specification (e.g., 'xcvu9p-flga2104-2-i')

#### **Control Flags**
- **`is_hw_graph`** (bool, default False): 
  - If True: Input model is already a Dataflow Graph (ONNX with HWCustomOps)
  - If False: Input is a standard quantized ONNX model
  - When True: Skips model analysis and transformation, goes directly to HW optimization

- **`build_core`** (bool, default True):
  - If True: Generate complete Dataflow Core (stitched IP design with HW code)
  - If False: Exit after Dataflow Graph generation (checkpoint mode)

- **`output_dir`** (Optional[str]): Directory for saving results and artifacts

### **Output Specification**

```python
# Return dictionary structure
{
    'dataflow_graph': {
        'onnx_model': <ONNX ModelProto>,           # HWCustomOps graph
        'metadata': {
            'kernel_mapping': {...},                # Op -> Kernel assignments
            'resource_estimates': {...},           # Resource usage estimates
            'performance_estimates': {...}         # Performance predictions
        }
    },
    'dataflow_core': {                             # Only if build_core=True
        'ip_files': [...],                         # Generated IP core files
        'synthesis_results': {...},               # Post-synthesis metrics
        'driver_code': {...},                     # Host interface code
        'bitstream': <path_to_bitstream>          # Generated bitstream
    },
    'dse_results': {
        'best_configuration': {...},              # Optimal design point
        'pareto_frontier': [...],                 # Multi-objective solutions
        'exploration_history': [...],            # All evaluated points
        'convergence_metrics': {...}             # DSE algorithm performance
    },
    'metrics': {
        'performance': {
            'throughput_ops_sec': <float>,        # Operations per second
            'latency_ms': <float>,                # End-to-end latency
            'frequency_mhz': <float>              # Achieved clock frequency
        },
        'resources': {
            'lut_utilization': <float>,           # LUT usage percentage
            'dsp_utilization': <float>,           # DSP usage percentage
            'bram_utilization': <float>,          # BRAM usage percentage
            'power_consumption_w': <float>        # Power consumption
        }
    },
    'analysis': {
        'design_space_coverage': <float>,         # Percentage explored
        'optimization_quality': <float>,         # Solution quality metric
        'recommendations': [...],                # Improvement suggestions
        'warnings': [...]                       # Potential issues
    }
}
```

---

## 🏗️ **Implementation Architecture**

### **Main Function Flow**

```python
def forge(...) -> Dict[str, Any]:
    # 1. Input Validation
    _validate_inputs(model_path, blueprint_path, objectives, constraints)
    
    # 2. Blueprint Loading (Hard Error)
    blueprint = _load_and_validate_blueprint(blueprint_path)
    
    # 3. DSE Configuration Setup
    dse_config = _setup_dse_configuration(blueprint, objectives, constraints, target_device)
    
    # 4. Execution Branch
    if is_hw_graph:
        # Hardware Graph Optimization Path
        dataflow_graph = _load_dataflow_graph(model_path)
        dse_results = _run_hw_optimization_dse(dataflow_graph, dse_config)
    else:
        # Full Model-to-Hardware Path  
        dse_results = _run_full_dse(model_path, dse_config)
        dataflow_graph = dse_results.best_result.dataflow_graph
    
    # 5. Dataflow Core Generation (Optional)
    dataflow_core = None
    if build_core:
        dataflow_core = _generate_dataflow_core(dataflow_graph, dse_config)
    
    # 6. Results Assembly
    results = _assemble_results(dataflow_graph, dataflow_core, dse_results)
    
    # 7. Output Handling
    if output_dir:
        _save_forge_results(results, output_dir)
    
    return results
```

### **Helper Function Specifications**

#### **Input Validation**
```python
def _validate_inputs(model_path: str, blueprint_path: str, objectives: Dict, constraints: Dict):
    """Validate all input parameters with descriptive error messages."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    if not model_path.lower().endswith('.onnx'):
        raise ValueError(f"Model must be ONNX format, got: {model_path}")
    
    if not blueprint_path.lower().endswith(('.yaml', '.yml')):
        raise ValueError(f"Blueprint must be YAML format, got: {blueprint_path}")
    
    # Validate objectives format
    if objectives:
        for obj_name, obj_config in objectives.items():
            if 'direction' not in obj_config:
                raise ValueError(f"Objective '{obj_name}' missing 'direction' field")
            if obj_config['direction'] not in ['maximize', 'minimize']:
                raise ValueError(f"Objective '{obj_name}' direction must be 'maximize' or 'minimize'")
    
    # Validate constraints format  
    if constraints:
        numeric_constraints = ['max_luts', 'max_dsps', 'max_brams', 'max_power', 'target_frequency']
        for key, value in constraints.items():
            if key in numeric_constraints and not isinstance(value, (int, float)):
                raise ValueError(f"Constraint '{key}' must be numeric, got {type(value)}")
```

#### **Blueprint Loading (Hard Error)**
```python
def _load_and_validate_blueprint(blueprint_path: str):
    """Load and validate blueprint - throw hard error if invalid."""
    try:
        from ..blueprints.base import Blueprint
        blueprint = Blueprint.from_yaml_file(Path(blueprint_path))
        
        # Validate blueprint configuration
        is_valid, errors = blueprint.validate_library_config()
        if not is_valid:
            raise ValueError(f"Blueprint validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        logger.info(f"Successfully loaded blueprint: {blueprint.name}")
        return blueprint
        
    except ImportError:
        raise RuntimeError("Blueprint system not available. Cannot proceed without valid blueprint.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    except Exception as e:
        raise ValueError(f"Failed to load blueprint '{blueprint_path}': {str(e)}")
```

#### **DSE Configuration Setup**
```python
def _setup_dse_configuration(blueprint, objectives, constraints, target_device):
    """Setup comprehensive DSE configuration."""
    from ..dse.interface import DSEConfiguration, DSEObjective, OptimizationObjective
    
    # Extract design space from blueprint
    design_space = blueprint.get_design_space()
    
    # Setup objectives
    dse_objectives = []
    if objectives:
        for obj_name, obj_config in objectives.items():
            direction = OptimizationObjective.MAXIMIZE if obj_config['direction'] == 'maximize' else OptimizationObjective.MINIMIZE
            weight = obj_config.get('weight', 1.0)
            target = obj_config.get('target', None)
            
            dse_objectives.append(DSEObjective(
                name=obj_name,
                direction=direction,
                weight=weight,
                target_value=target
            ))
    else:
        # Default objectives
        dse_objectives = [
            DSEObjective('throughput', OptimizationObjective.MAXIMIZE, weight=1.0),
            DSEObjective('latency', OptimizationObjective.MINIMIZE, weight=0.8)
        ]
    
    # Setup constraints  
    dse_constraints = constraints.copy() if constraints else {}
    if target_device:
        dse_constraints['target_device'] = target_device
        
    # Add default constraints if not specified
    if 'max_luts' not in dse_constraints:
        dse_constraints['max_luts'] = 0.8
    if 'max_dsps' not in dse_constraints:
        dse_constraints['max_dsps'] = 0.8
    
    return DSEConfiguration(
        design_space=design_space,
        objectives=dse_objectives,
        constraints=dse_constraints,
        blueprint=blueprint
    )
```

#### **DSE Execution Paths**
```python
def _run_full_dse(model_path: str, dse_config):
    """Execute full model-to-hardware DSE pipeline."""
    from ..dse.interface import DSEInterface
    
    logger.info("Starting full DSE: Model analysis -> Transformation -> Kernel mapping -> HW optimization")
    
    dse_engine = DSEInterface(dse_config)
    
    # Execute complete pipeline
    results = dse_engine.explore_design_space(
        model_path=model_path,
        stages=['analysis', 'transformation', 'kernel_mapping', 'hw_optimization']
    )
    
    logger.info(f"DSE completed: {len(results.results)} design points evaluated")
    return results

def _run_hw_optimization_dse(dataflow_graph, dse_config):
    """Execute hardware optimization DSE on existing Dataflow Graph."""
    from ..dse.interface import DSEInterface
    
    logger.info("Starting HW optimization DSE on existing Dataflow Graph")
    
    dse_engine = DSEInterface(dse_config)
    
    # Execute only HW optimization stage
    results = dse_engine.optimize_dataflow_graph(
        dataflow_graph=dataflow_graph,
        stages=['hw_optimization']
    )
    
    logger.info(f"HW optimization completed: {len(results.results)} configurations evaluated")
    return results
```

#### **Dataflow Core Generation**
```python
def _generate_dataflow_core(dataflow_graph, dse_config):
    """Generate complete stitched IP design from Dataflow Graph."""
    from ..finn.orchestration import FINNBuildOrchestrator
    
    logger.info("Generating Dataflow Core (stitched IP design)")
    
    orchestrator = FINNBuildOrchestrator(dse_config)
    
    # Generate IP core with synthesis
    core_results = orchestrator.generate_ip_core(
        dataflow_graph=dataflow_graph,
        generate_bitstream=True,
        run_synthesis=True,
        generate_drivers=True
    )
    
    logger.info("Dataflow Core generation completed")
    return core_results
```

---

## 🛠️ **Tools Interface Migration**

### **New Tools Module Structure**
```
brainsmith/tools/
├── __init__.py                 # Public tools interface
├── profiling/
│   ├── __init__.py            # Profiling tools interface  
│   ├── roofline.py            # Existing roofline analysis
│   ├── roofline_runner.py     # Existing roofline runner
│   └── model_profiling.py     # Existing model profiling
└── hw_kernel_gen/             # Existing kernel generation tools
    └── ...
```

### **Tools Public Interface**
```python
# brainsmith/tools/__init__.py
"""
BrainSmith Supplementary Tools

Tools that are not part of the core forge toolflow but provide 
additional analysis and profiling capabilities.
"""

from .profiling import roofline_analysis, RooflineProfiler

__all__ = [
    'roofline_analysis',
    'RooflineProfiler'
]
```

### **Roofline Tools Interface**
```python
# brainsmith/tools/profiling/__init__.py
"""Model profiling and roofline analysis tools."""

from .roofline import roofline_analysis as _roofline_analysis
from .model_profiling import RooflineModel

class RooflineProfiler:
    """High-level interface for roofline analysis."""
    
    def __init__(self):
        self.model = RooflineModel()
    
    def profile_model(self, model_config: Dict[str, Any], hardware_config: Dict[str, Any]) -> Dict[str, Any]:
        """Profile model and generate roofline analysis."""
        # Set up model configuration
        if model_config['arch'] == 'bert':
            self.model.profile_bert(model=model_config)
        elif model_config['arch'] == 'slm_pp':
            self.model.profile_slm_pp(model=model_config)
        elif model_config['arch'] == 'slm_tg':
            self.model.profile_slm_tg(model=model_config)
        else:
            raise ValueError(f"Unsupported model architecture: {model_config['arch']}")
        
        profile = self.model.get_profile()
        
        # Generate analysis for different data types
        results = {}
        for dtype in [4, 8]:
            results[f'{dtype}bit'] = self._analyze_dtype(profile, hardware_config, dtype)
        
        return results
    
    def _analyze_dtype(self, profile, hw_config, dtype):
        """Analyze profile for specific data type."""
        # Implementation using existing roofline analysis logic
        pass

def roofline_analysis(model_config: Dict, hw_config: Dict, dtypes: List[int]) -> Dict[str, Any]:
    """
    Wrapper for existing roofline analysis functionality.
    
    Args:
        model_config: Model configuration dictionary
        hw_config: Hardware configuration dictionary  
        dtypes: List of data type bit widths to analyze
        
    Returns:
        Dictionary with roofline analysis results for each data type
    """
    return _roofline_analysis(model_config, hw_config, dtypes)
```

---

## 📚 **Updated Public API**

### **Simplified `__init__.py`**
```python
# brainsmith/__init__.py

# Core toolchain (primary interface)
from .core.api import forge, validate_blueprint

# Core data structures
from .core.design_space import DesignSpace, DesignPoint, ParameterDefinition

# Blueprint system
try:
    from .blueprints import Blueprint, load_blueprint, list_blueprints
except ImportError:
    Blueprint = None
    load_blueprint = None
    list_blueprints = None

# DSE system  
try:
    from .dse import DSEInterface, DSEAnalyzer, ParetoAnalyzer
except ImportError:
    DSEInterface = None
    DSEAnalyzer = None
    ParetoAnalyzer = None

# Supplementary tools (not part of core toolflow)
from .tools import roofline_analysis, RooflineProfiler

__all__ = [
    # Core toolchain
    'forge',
    'validate_blueprint',
    
    # Core data structures
    'DesignSpace',
    'DesignPoint', 
    'ParameterDefinition',
    
    # Blueprint system
    'Blueprint',
    'load_blueprint',
    'list_blueprints',
    
    # DSE system
    'DSEInterface',
    'DSEAnalyzer',
    'ParetoAnalyzer',
    
    # Supplementary tools
    'roofline_analysis',
    'RooflineProfiler'
]

__version__ = "0.5.0"  # Updated for API simplification
__author__ = "Microsoft Research"
__description__ = "Simplified FPGA accelerator design space exploration platform"
```

---

## 📖 **Usage Examples**

### **Basic Usage**
```python
import brainsmith

# Core toolchain usage
results = brainsmith.forge(
    model_path="bert_model.onnx",
    blueprint_path="bert_blueprint.yaml"
)

# Access results
dataflow_graph = results['dataflow_graph']
metrics = results['metrics']
print(f"Throughput: {metrics['performance']['throughput_ops_sec']:.2f} ops/sec")
print(f"LUT Utilization: {metrics['resources']['lut_utilization']:.1%}")
```

### **Advanced Configuration**
```python
# Advanced usage with objectives and constraints
results = brainsmith.forge(
    model_path="bert_model.onnx",
    blueprint_path="bert_blueprint.yaml",
    objectives={
        'throughput': {'direction': 'maximize', 'target': 1000, 'weight': 1.0},
        'latency': {'direction': 'minimize', 'target': 10, 'weight': 0.8},
        'power': {'direction': 'minimize', 'weight': 0.6}
    },
    constraints={
        'max_luts': 0.8,
        'max_dsps': 0.7,
        'target_device': 'xcvu9p-flga2104-2-i'
    },
    output_dir="./results"
)
```

### **Hardware Graph Optimization**
```python
# Skip model analysis for pre-transformed graphs
results = brainsmith.forge(
    model_path="dataflow_graph.onnx",  # Already contains HWCustomOps
    blueprint_path="optimization_blueprint.yaml",
    is_hw_graph=True,  # Skip to HW optimization
    build_core=True
)
```

### **Checkpoint Mode**
```python
# Generate only Dataflow Graph (no IP synthesis)
results = brainsmith.forge(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml", 
    build_core=False  # Exit after Dataflow Graph
)

dataflow_graph = results['dataflow_graph']
# Save checkpoint for later use
dataflow_graph['onnx_model'].save("checkpoint.onnx")
```

### **Supplementary Tools Usage**
```python
# Use roofline analysis (not part of core toolflow)
from brainsmith.tools import roofline_analysis

roofline_results = roofline_analysis(
    model_config={
        'arch': 'bert',
        'num_layers': 12,
        'seq_len': 512,
        'num_heads': 12,
        'head_size': 64
    },
    hw_config={
        'luts': 1728000,
        'dsps': 12288, 
        'lut_hz': 250e6,
        'dsp_hz': 500e6
    },
    dtypes=[4, 8]
)
```

---

## ⚠️ **Migration Guide**

### **Old API → New API Mapping**

| Old Function | New Equivalent | Notes |
|--------------|----------------|-------|
| `brainsmith_explore()` | `forge()` | Single unified function |
| `brainsmith_roofline()` | `roofline_analysis()` | Moved to tools module |
| `brainsmith_dataflow_analysis()` | `forge(build_core=False)` | Use checkpoint mode |
| `brainsmith_generate()` | `forge(build_core=True)` | Default behavior |
| `explore_design_space()` | `forge()` | Legacy function removed |

### **Breaking Changes**
1. **Roofline Analysis**: Moved from core API to `brainsmith.tools`
2. **Blueprint Validation**: No longer falls back to mock blueprints (hard error)
3. **Legacy Functions**: All removed, no backward compatibility
4. **Return Format**: Standardized dictionary structure
5. **Import Paths**: Tools moved to separate module

### **Migration Steps**
1. Replace all old function calls with `forge()`
2. Update roofline usage to import from `brainsmith.tools`
3. Add proper error handling for blueprint validation
4. Update result parsing to use new dictionary structure
5. Test thoroughly with new API

This technical specification provides complete implementation guidance for addressing all code review feedback while maintaining functionality and improving API clarity.