# Brainsmith Extensible Platform Implementation Plan

## Overview

This implementation plan transforms Brainsmith from its current monolithic architecture into an extensible platform for FPGA accelerator development and future design space exploration research. The plan focuses on structure, metrics, and extensibility rather than complex optimization algorithms.

## Implementation Scope

### **What We're Building**
- ✅ Extensible platform architecture with comprehensive metric collection
- ✅ Modular FINN interface with future-ready placeholder hooks
- ✅ Structured design space representation for research enablement
- ✅ Clean APIs for both basic compilation and parameter exploration
- ✅ CLI interface supporting simple builds and parameter sweeps
- ✅ Data export capabilities for external research tool integration

### **What We're NOT Building**
- ❌ Complex DSE optimization algorithms (genetic, Bayesian, etc.)
- ❌ Interactive CLI mode (per feedback)
- ❌ Detailed FINN hook implementations (await FINN evolution)
- ❌ Multi-objective optimization engines (future research)

## Phase 1: Core Platform Infrastructure ⭐ **PRIORITY**

### **1.1 Metric Collection System**

#### **Files to Create:**
- `brainsmith/core/metrics.py` - Comprehensive metric data structures
- `brainsmith/core/result.py` - Enhanced result classes with metrics

#### **Key Components:**
```python
@dataclass
class BrainsmithMetrics:
    """Comprehensive metrics for DSE research."""
    performance: PerformanceMetrics
    resources: ResourceMetrics
    quality: QualityMetrics
    build_info: BuildMetrics
    design_point: DesignPoint
    
    def to_research_dataset(self) -> Dict[str, Any]:
        """Export for external DSE research."""
        
    def get_optimization_features(self) -> np.ndarray:
        """Extract feature vector for ML-based DSE."""

@dataclass
class PerformanceMetrics:
    throughput_ops_sec: Optional[float] = None
    latency_ms: Optional[float] = None
    efficiency_ops_per_joule: Optional[float] = None
    clock_frequency_mhz: Optional[float] = None

@dataclass
class ResourceMetrics:
    lut_count: Optional[int] = None
    lut_utilization_percent: Optional[float] = None
    dsp_count: Optional[int] = None
    dsp_utilization_percent: Optional[float] = None
    bram_18k_count: Optional[int] = None
    estimated_power_w: Optional[float] = None
```

### **1.2 Design Space Definition System**

#### **Files to Create:**
- `brainsmith/core/design_space.py` - Design space representation
- `brainsmith/core/constraints.py` - Constraint system

#### **Key Components:**
```python
class DesignSpace:
    """Structured representation of design space."""
    
    def __init__(self, blueprint: Blueprint):
        self.dimensions = self._extract_dimensions(blueprint)
        self.constraints = self._extract_constraints(blueprint)
    
    def get_dimension_ranges(self) -> Dict[str, Any]:
        """Get ranges for each design dimension."""
        
    def validate_design_point(self, point: DesignPoint) -> bool:
        """Check if design point is valid."""
        
    def export_for_dse(self) -> Dict[str, Any]:
        """Export for external DSE tools."""

@dataclass
class DesignPoint:
    """Single point in design space."""
    platform: str
    configuration: Dict[str, Any]
    finn_hooks: Dict[str, Any]  # Placeholder
    
    def to_finn_config(self) -> Dict[str, Any]:
        """Convert to current FINN format."""
```

### **1.3 FINN Interface Layer**

#### **Files to Create:**
- `brainsmith/core/finn_interface.py` - FINN integration with placeholders
- `brainsmith/core/finn_hooks.py` - Future hook system placeholders

#### **Key Components:**
```python
class FINNInterfaceLayer:
    """Abstraction layer for FINN integration."""
    
    def __init__(self):
        self.hooks = FINNHooksPlaceholder()
        self.translator = ConfigTranslator()
        
    def execute_build(self, model: onnx.ModelProto, 
                     design_point: DesignPoint) -> BrainsmithResult:
        """Execute FINN build with comprehensive metric collection."""

class FINNHooksPlaceholder:
    """Placeholder system for future FINN hooks."""
    model_ops: ModelOpsPlaceholder = field(default_factory=ModelOpsPlaceholder)
    model_transforms: ModelTransformsPlaceholder = field(default_factory=ModelTransformsPlaceholder)
    hw_kernels: HWKernelsPlaceholder = field(default_factory=HWKernelsPlaceholder)
    hw_optimization: HWOptimizationPlaceholder = field(default_factory=HWOptimizationPlaceholder)
```

### **1.4 Enhanced Compiler Core**

#### **Files to Update:**
- `brainsmith/core/config.py` - Enhanced configuration system
- `brainsmith/core/compiler.py` - New modular compiler

#### **Key Components:**
```python
@dataclass
class CompilerConfig:
    """Enhanced configuration with DSE support."""
    blueprint: str
    output_dir: str
    
    # Metric collection
    collect_comprehensive_metrics: bool = True
    export_research_data: bool = False
    
    # Design space exploration
    dse_enabled: bool = False
    parameter_sweep: Optional[Dict[str, List[Any]]] = None
    
    # FINN hook placeholders
    finn_hooks_override: Optional[Dict[str, Any]] = None

class HardwareCompiler:
    """Main compiler with metric collection and extensibility."""
    
    def compile(self, model: onnx.ModelProto) -> BrainsmithResult:
        """Enhanced compilation with comprehensive metrics."""
        
    def parameter_sweep(self, model: onnx.ModelProto, 
                       parameters: Dict[str, List[Any]]) -> List[BrainsmithResult]:
        """Execute parameter sweep for optimization."""
```

## Phase 2: Enhanced Blueprint System

### **2.1 Design Space Blueprint Extensions**

#### **Files to Update:**
- `brainsmith/blueprints/yaml/bert_extensible.yaml` - Enhanced BERT blueprint
- `brainsmith/blueprints/manager.py` - Design space loading

#### **Enhanced Blueprint Structure:**
```yaml
name: "bert_extensible"
description: "BERT with extensible design space definition"

# Current build steps (unchanged)
build_steps:
  - "common.cleanup"
  - "transformer.remove_head"
  # ... existing steps

# Design space structure (for future DSE)
design_space:
  dimensions:
    platforms:
      type: "categorical"
      values: ["V80", "ZCU104", "U250"]
    target_fps:
      type: "continuous"
      range: [1000, 10000]
    clk_period_ns:
      type: "continuous"
      range: [2.0, 10.0]
      
  # Placeholder for future FINN hooks
  finn_hooks_config:
    model_ops:
      custom_ops_enabled: 
        type: "boolean"
        default: false
    hw_optimization:
      optimization_time_budget:
        type: "continuous"
        range: [60, 3600]
        default: 300

# Constraints
constraints:
  hard_constraints:
    max_lut_utilization: 0.8
    max_dsp_utilization: 0.8
  custom_constraints:
    - name: "throughput_per_lut"
      formula: "throughput / lut_count"
      operator: ">"
      value: 0.5

# Metric collection configuration
metrics_config:
  collect_intermediate_results: true
  export_for_research: true
  custom_metrics:
    - name: "efficiency_score"
      formula: "(throughput * accuracy) / power"
```

### **2.2 Blueprint Design Space Integration**

#### **Files to Update:**
- `brainsmith/blueprints/base.py` - Add design space methods
- `brainsmith/blueprints/__init__.py` - Export design space functions

## Phase 3: Library Interface Implementation

### **3.1 Simple Build Interface**

#### **Files to Create:**
- `brainsmith/api/simple.py` - Simple build functions
- `brainsmith/api/optimization.py` - Parameter optimization functions

#### **Simple Interface:**
```python
# In brainsmith/__init__.py
def build_model(model: onnx.ModelProto, blueprint: str, 
                output_dir: str, **kwargs) -> BrainsmithResult:
    """Simple model compilation interface."""

def optimize_model(model: onnx.ModelProto, blueprint: str,
                  output_dir: str, parameters: Dict[str, List[Any]]) -> List[BrainsmithResult]:
    """Parameter sweep optimization interface."""
```

### **3.2 Extensible DSE Interface**

#### **Files to Create:**
- `brainsmith/dse/interface.py` - Abstract DSE interface
- `brainsmith/dse/simple.py` - Basic implementation
- `brainsmith/dse/external.py` - External tool adapter

#### **DSE Interfaces:**
```python
class DSEInterface:
    """Abstract interface for future DSE engines."""
    
    @abstractmethod
    def explore(self, design_space: DesignSpace, 
                model: onnx.ModelProto) -> DSEResult:
        """Execute design space exploration."""

class SimpleDSEEngine(DSEInterface):
    """Basic grid-like exploration (no optimization)."""
    
class ExternalDSEAdapter(DSEInterface):
    """Adapter for external research tools."""
```

## Phase 4: CLI Interface Implementation

### **4.1 Core CLI Commands**

#### **Files to Create:**
- `brainsmith/cli/main.py` - Main CLI entry point
- `brainsmith/cli/commands/build.py` - Build command
- `brainsmith/cli/commands/optimize.py` - Optimization command (no interactive mode)
- `brainsmith/cli/commands/blueprints.py` - Blueprint management

#### **CLI Commands:**
```bash
# Basic build
brainsmith build model.onnx --blueprint bert --output ./build

# Parameter optimization
brainsmith optimize model.onnx --blueprint bert --output ./opt \
  --parameter target_fps=3000,5000,7500 \
  --parameter clk_period_ns=3.0,4.0,5.0

# Blueprint management  
brainsmith blueprints list
brainsmith blueprints show bert

# Design space analysis
brainsmith analyze-space --blueprint bert --export-schema space.json
```

### **4.2 Analysis and Export Commands**

#### **Files to Create:**
- `brainsmith/cli/commands/analyze.py` - Result analysis
- `brainsmith/cli/commands/export.py` - Data export

#### **Analysis Commands:**
```bash
# Result analysis
brainsmith analyze-results ./results/ --export-csv results.csv

# Research data export
brainsmith export-research-data ./builds/ --format csv,json
```

## Phase 5: Data Management and Export

### **5.1 Result Storage System**

#### **Files to Create:**
- `brainsmith/data/storage.py` - Result storage and retrieval
- `brainsmith/data/export.py` - Research data export
- `brainsmith/data/analysis.py` - Basic analysis tools

### **5.2 External Tool Integration**

#### **Files to Create:**
- `brainsmith/integration/adapters.py` - External tool adapters
- `brainsmith/integration/formats.py` - Standard export formats

## Implementation Order and Dependencies

### **Week 1-2: Core Infrastructure**
1. Metric collection system (`brainsmith/core/metrics.py`, `brainsmith/core/result.py`)
2. Design space representation (`brainsmith/core/design_space.py`)
3. FINN interface placeholders (`brainsmith/core/finn_interface.py`)

### **Week 3-4: Enhanced Compiler**
1. Enhanced configuration (`brainsmith/core/config.py`)
2. Modular compiler (`brainsmith/core/compiler.py`)
3. Update legacy interface for compatibility

### **Week 5-6: Blueprint System**
1. Enhanced BERT blueprint with design space
2. Blueprint manager design space integration
3. Constraint system implementation

### **Week 7-8: Library Interface**
1. Simple build API (`brainsmith/api/simple.py`)
2. Parameter optimization API (`brainsmith/api/optimization.py`)
3. Package exports update (`brainsmith/__init__.py`)

### **Week 9-10: CLI Interface**
1. Core CLI commands (`build`, `optimize`, `blueprints`)
2. Analysis and export commands
3. CLI entry point and packaging

### **Week 11-12: Data and Integration**
1. Result storage and export systems
2. External tool integration adapters
3. Documentation and examples

## Testing Strategy

### **Unit Tests**
- Comprehensive metric collection and calculation
- Design space validation and point generation
- Configuration parsing and validation
- FINN interface translation

### **Integration Tests**
- End-to-end BERT compilation with metric collection
- Parameter sweep execution and comparison
- CLI command validation
- Blueprint loading and design space extraction

### **Compatibility Tests**
- Existing BERT demo continues to work (legacy interface)
- All current workflows remain functional
- Performance regression testing

## Success Criteria

### **Phase 1 Success**
- ✅ BERT demo works with enhanced metrics collection
- ✅ Design space can be extracted from BERT blueprint
- ✅ FINN interface placeholders handle current configuration
- ✅ Comprehensive metrics collected and exportable

### **Full Implementation Success**
- ✅ Simple CLI: `brainsmith build model.onnx --blueprint bert --output ./build`
- ✅ Parameter optimization: Multiple configuration comparison
- ✅ Research data export: CSV/JSON datasets for external tools
- ✅ External tool integration: API for research framework connection
- ✅ Zero breaking changes: All existing workflows continue

## Risk Mitigation

### **Technical Risks**
- **FINN Interface Changes**: Use placeholder system, easy to update
- **Performance Regression**: Comprehensive benchmarking during development
- **Complexity Creep**: Focus on structure over algorithms, defer optimization research

### **Compatibility Risks**
- **Legacy Interface**: Maintain wrapper around new system
- **Blueprint Changes**: Extend rather than replace existing blueprints
- **Environment Dependencies**: Maintain current FINN integration patterns

This implementation plan provides a clear roadmap for transforming Brainsmith into an extensible platform while maintaining all current functionality and preparing for future DSE research capabilities.