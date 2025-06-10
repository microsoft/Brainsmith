# 🔥 **BrainSmith API Simplification Implementation Plan**
## Response to Code Review Feedback

---

## 📋 **Code Review Summary**

### **Primary Feedback:**
1. **api.py is vastly overly complicated** - Simplify to core toolchain design
2. **Introduce `forge` function** - Core BrainSmith toolchain with DSE and flags
3. **Move roofline analysis** - Not part of core toolflow, move to tools/
4. **Hard error on blueprint validation** - No defaulting to mock blueprints
5. **Remove legacy interfaces** - Deprecate and remove unused legacy code

### **New Core Design: `forge` Function**
- **Input**: Model (ONNX), Blueprint (YAML), Objectives & Constraints
- **Flags**: `is_hw_graph`, `build_core`
- **Output**: Dataflow Graph, Dataflow Core (optional)

---

## 🎯 **Implementation Plan**

### **Phase 1: Core API Simplification (Priority: CRITICAL)**

#### **1.1 Create New `forge` Function**
**File**: `brainsmith/core/api.py`

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
    """
    Core BrainSmith toolchain: DSE on input model to produce Dataflow Core.
    
    Args:
        model_path: Path to pre-quantized ONNX model
        blueprint_path: Path to blueprint YAML (design space specification)
        objectives: Target objectives (latency/throughput requirements)
        constraints: Hardware resource budgets, optimization priorities
        target_device: Target FPGA device specification
        is_hw_graph: If True, input is already a Dataflow Graph, skip to HW optimization
        build_core: If False, exit after Dataflow Graph generation
        output_dir: Optional output directory for results
        
    Returns:
        Dict containing:
        - dataflow_graph: ONNX graph of HWCustomOps describing Dataflow Core
        - dataflow_core: Stitched IP design (if build_core=True)
        - metrics: Performance and resource utilization metrics
        - analysis: DSE analysis and recommendations
    """
```

#### **1.2 Remove Legacy Functions**
**Actions:**
- Remove `brainsmith_explore`, `brainsmith_roofline`, `brainsmith_dataflow_analysis`, `brainsmith_generate`
- Remove `brainsmith_workflow` 
- Remove `explore_design_space` legacy wrapper
- Remove `_route_to_existing_legacy_system` and all legacy compatibility code

#### **1.3 Blueprint Validation - Hard Error**
**Current Problem**: Falls back to mock blueprint
**Solution**: Throw hard error if blueprint doesn't load/validate

```python
def _load_and_validate_blueprint(blueprint_path: str) -> Blueprint:
    """Load and validate blueprint - hard error if invalid."""
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    try:
        blueprint = Blueprint.from_yaml_file(Path(blueprint_path))
        is_valid, errors = blueprint.validate_library_config()
        if not is_valid:
            raise ValueError(f"Blueprint validation failed: {'; '.join(errors)}")
        return blueprint
    except Exception as e:
        raise ValueError(f"Failed to load or validate blueprint: {e}")
```

### **Phase 2: Move Roofline Analysis to Tools (Priority: HIGH)**

#### **2.1 Create Tools Interface**
**File**: `brainsmith/tools/__init__.py`

```python
"""
BrainSmith Supplementary Tools

Tools that are not part of the core toolflow but provide additional
analysis and profiling capabilities.
"""

from .profiling import roofline_analysis, RooflineProfiler
from .hw_kernel_gen import generate_hw_kernel

__all__ = [
    'roofline_analysis',
    'RooflineProfiler', 
    'generate_hw_kernel'
]
```

#### **2.2 Update Roofline Interface**
**File**: `brainsmith/tools/profiling/__init__.py`

```python
"""Model profiling and roofline analysis tools."""

from .roofline import roofline_analysis
from .model_profiling import RooflineModel

class RooflineProfiler:
    """High-level interface for roofline analysis."""
    
    def __init__(self):
        self.model = RooflineModel()
    
    def profile_model(self, model_path: str, hardware_config: Dict[str, Any]) -> Dict[str, Any]:
        """Profile model and generate roofline analysis."""
        # Implementation using existing roofline.py functionality
        pass
    
    def generate_report(self, profile_results: Dict[str, Any], output_path: str = None) -> str:
        """Generate roofline analysis report."""
        pass

def roofline_analysis(model_config: Dict, hw_config: Dict, dtypes: List[int]) -> Dict[str, Any]:
    """Wrapper for existing roofline analysis functionality."""
    # Use existing implementation from roofline.py
    pass
```

### **Phase 3: Simplify Public API (Priority: HIGH)**

#### **3.1 Update `__init__.py`**
**File**: `brainsmith/__init__.py`

**Remove:**
```python
# Remove all these imports
from .core.api import (
    brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
    brainsmith_generate, brainsmith_workflow, validate_blueprint,
    explore_design_space
)
```

**Add:**
```python
# New simplified core API
from .core.api import forge, validate_blueprint

# Tools (not part of core toolflow)  
from .tools import roofline_analysis, RooflineProfiler
```

#### **3.2 Update `__all__` exports**
```python
__all__ = [
    # Core toolchain
    'forge',
    'validate_blueprint',
    
    # Core components
    'DesignSpace',
    'DesignPoint', 
    'ParameterDefinition',
    'Blueprint',
    
    # Supplementary tools
    'roofline_analysis',
    'RooflineProfiler',
    
    # DSE system (optional)
    'DSEInterface',
    'DSEAnalyzer',
    'ParetoAnalyzer'
]
```

### **Phase 4: Implementation Details**

#### **4.1 `forge` Function Implementation**

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
    """Core BrainSmith toolchain implementation."""
    
    # 1. Input validation
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 2. Load and validate blueprint (hard error)
    blueprint = _load_and_validate_blueprint(blueprint_path)
    
    # 3. Setup DSE configuration
    dse_config = _setup_dse_configuration(
        blueprint, objectives, constraints, target_device
    )
    
    # 4. Branch based on is_hw_graph flag
    if is_hw_graph:
        # Input is already Dataflow Graph, skip to HW optimization
        dataflow_graph = _load_dataflow_graph(model_path)
        dse_results = _run_hw_optimization_dse(dataflow_graph, dse_config)
    else:
        # Standard flow: Model -> DSE -> Dataflow Graph
        dse_results = _run_full_dse(model_path, dse_config)
        dataflow_graph = dse_results.best_result.dataflow_graph
    
    # 5. Generate Dataflow Core if requested
    dataflow_core = None
    if build_core:
        dataflow_core = _generate_dataflow_core(dataflow_graph, dse_config)
    
    # 6. Prepare results
    results = {
        'dataflow_graph': dataflow_graph,
        'dataflow_core': dataflow_core,
        'dse_results': dse_results,
        'metrics': _extract_metrics(dse_results),
        'analysis': _generate_analysis(dse_results)
    }
    
    # 7. Save results if output directory specified
    if output_dir:
        _save_forge_results(results, output_dir)
    
    return results
```

#### **4.2 Helper Function Implementations**

```python
def _setup_dse_configuration(blueprint, objectives, constraints, target_device):
    """Setup DSE configuration from inputs."""
    from ..dse.interface import DSEConfiguration
    
    # Extract design space from blueprint
    design_space = blueprint.get_design_space()
    
    # Setup objectives
    dse_objectives = []
    if objectives:
        for obj_name, obj_config in objectives.items():
            dse_objectives.append(
                DSEObjective(obj_name, obj_config.get('direction', 'maximize'))
            )
    else:
        # Default objectives
        dse_objectives = [
            DSEObjective('throughput', 'maximize'),
            DSEObjective('latency', 'minimize')
        ]
    
    # Setup constraints
    dse_constraints = constraints or {}
    if target_device:
        dse_constraints['target_device'] = target_device
    
    return DSEConfiguration(
        design_space=design_space,
        objectives=dse_objectives,
        constraints=dse_constraints
    )

def _run_full_dse(model_path, dse_config):
    """Run full DSE: Model analysis -> Transform -> Kernel mapping -> HW optimization."""
    from ..dse.interface import DSEInterface
    
    dse_engine = DSEInterface(dse_config)
    return dse_engine.explore_design_space(model_path)

def _run_hw_optimization_dse(dataflow_graph, dse_config):
    """Run HW optimization DSE on existing Dataflow Graph."""
    from ..dse.interface import DSEInterface
    
    dse_engine = DSEInterface(dse_config)
    return dse_engine.optimize_dataflow_graph(dataflow_graph)

def _generate_dataflow_core(dataflow_graph, dse_config):
    """Generate stitched IP design from Dataflow Graph."""
    from ..finn.orchestration import FINNBuildOrchestrator
    
    orchestrator = FINNBuildOrchestrator(dse_config)
    return orchestrator.generate_ip_core(dataflow_graph)
```

### **Phase 5: Update Documentation and Tests**

#### **5.1 Update API Documentation**
- Update README.md with new `forge` API
- Update architecture docs to reflect simplified design
- Add migration guide for users of old API

#### **5.2 Update Tests**  
- Replace tests for old API functions with `forge` tests
- Add tests for roofline tools interface
- Test blueprint validation hard errors

#### **5.3 Example Usage Updates**

**New API Usage:**
```python
import brainsmith

# Core toolchain usage
results = brainsmith.forge(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml", 
    objectives={
        'throughput': {'direction': 'maximize', 'target': 1000},
        'latency': {'direction': 'minimize', 'target': 10}
    },
    constraints={
        'max_luts': 0.8,
        'max_dsps': 0.7,
        'target_device': 'xcvu9p'
    },
    build_core=True
)

# Access results
dataflow_graph = results['dataflow_graph']
dataflow_core = results['dataflow_core'] 
metrics = results['metrics']

# Use supplementary tools (not core toolflow)
from brainsmith.tools import roofline_analysis

roofline_results = roofline_analysis(
    model_config=model_params,
    hw_config=hardware_params,
    dtypes=[4, 8]
)
```

---

## 📊 **Implementation Timeline**

### **Week 1: Core API Simplification**
- [ ] Implement `forge` function
- [ ] Remove legacy functions  
- [ ] Update blueprint validation (hard error)
- [ ] Basic testing

### **Week 2: Tools Interface & Roofline Migration**
- [ ] Create tools interface structure
- [ ] Move roofline analysis to tools/
- [ ] Update imports and exports
- [ ] Integration testing

### **Week 3: Documentation & Migration**
- [ ] Update all documentation
- [ ] Create migration guide
- [ ] Update examples and demos
- [ ] Comprehensive testing

### **Week 4: Validation & Cleanup**
- [ ] Remove all unused legacy code
- [ ] Final testing and validation
- [ ] Performance benchmarking
- [ ] Release preparation

---

## ✅ **Success Criteria**

1. **Simplified API**: Single `forge` function replaces 5+ existing functions
2. **Clear Separation**: Core toolflow vs supplementary tools clearly separated  
3. **Blueprint Validation**: Hard errors instead of fallbacks to mocks
4. **No Legacy Code**: All unused legacy interfaces removed
5. **Maintained Functionality**: Core DSE and generation capabilities preserved
6. **Clear Documentation**: Updated docs and migration guide
7. **Robust Testing**: Comprehensive test coverage for new API

This implementation plan directly addresses all feedback points while maintaining the core functionality and improving the overall architecture clarity.