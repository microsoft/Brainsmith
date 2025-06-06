# Brainsmith Extensible Platform Design

## Revised Mission Statement

**Brainsmith** is an extensible platform for FPGA dataflow accelerator development that provides structured interfaces for future global design space exploration (DSE) research. It creates a foundation that exposes key metrics, design parameters, and extensibility points while maintaining modular architecture for advanced DSE engines.

## Core Design Philosophy: Structure First

### **Platform Goals (Not Research Goals)**
1. **Extensible Structure**: Clean interfaces for future DSE research
2. **Metric Exposure**: Comprehensive data collection for optimization research  
3. **Modular Architecture**: Pluggable components for different optimization strategies
4. **FINN Integration**: Placeholder interfaces for evolving FINN hooks
5. **Data Foundation**: Rich dataset generation for DSE algorithm development

### **What We're NOT Building (Yet)**
- ❌ Advanced optimization algorithms (genetic, Bayesian, etc.)
- ❌ Multi-objective optimization engines
- ❌ Detailed FINN hook implementations
- ❌ Complex DSE strategies

### **What We ARE Building**
- ✅ Extensible architecture for future DSE engines
- ✅ Comprehensive metric collection and tracking
- ✅ Modular FINN interface with placeholder hooks
- ✅ Data structures for design space representation
- ✅ Clean APIs for future optimization research

## Architectural Structure

### **1. Core Platform Architecture**

```
Brainsmith Platform
├── Metric Collection Layer
│   ├── Performance Metrics (throughput, latency, efficiency)
│   ├── Resource Metrics (LUTs, DSPs, BRAM, power)
│   ├── Quality Metrics (accuracy, precision)
│   └── Build Metrics (time, success rate, convergence)
│
├── Design Space Definition Layer
│   ├── Parameter Space (platforms, configurations, strategies)
│   ├── Constraint System (resources, performance, custom)
│   ├── Design Point Representation
│   └── Space Exploration Interface (placeholder for future DSE)
│
├── FINN Interface Layer
│   ├── Hook Interface Placeholders (4 hooks)
│   ├── Configuration Translation
│   ├── Result Collection
│   └── Build Orchestration
│
├── Data Management Layer
│   ├── Result Storage (structured data, artifacts)
│   ├── Experiment Tracking
│   ├── Metric Analysis Tools
│   └── Export/Import (for external DSE tools)
│
└── Extensibility Layer
    ├── Plugin Architecture (custom metrics, strategies)
    ├── Hook System (pre/post build, analysis)
    ├── Custom DSE Interface
    └── Research API (for algorithm development)
```

### **2. Metric Collection System**

```python
@dataclass
class BrainsmithMetrics:
    """Comprehensive metrics collection for DSE research."""
    
    # Build identification
    build_id: str
    timestamp: datetime
    configuration_hash: str
    
    # Performance metrics
    performance: PerformanceMetrics
    resources: ResourceMetrics  
    quality: QualityMetrics
    build_info: BuildMetrics
    
    # Design space context
    design_point: DesignPoint
    constraints: ConstraintSet
    
    # Raw data for research
    raw_reports: Dict[str, Any]
    intermediate_results: List[IntermediateResult]
    
    def to_research_dataset(self) -> Dict[str, Any]:
        """Export for external DSE research."""
        pass
    
    def get_optimization_features(self) -> np.ndarray:
        """Extract feature vector for ML-based DSE."""
        pass

@dataclass 
class PerformanceMetrics:
    """Performance characteristics."""
    throughput_ops_sec: Optional[float] = None
    latency_ms: Optional[float] = None
    efficiency_ops_per_joule: Optional[float] = None
    clock_frequency_mhz: Optional[float] = None
    
@dataclass
class ResourceMetrics:
    """FPGA resource utilization."""
    lut_count: Optional[int] = None
    lut_utilization_percent: Optional[float] = None
    dsp_count: Optional[int] = None
    dsp_utilization_percent: Optional[float] = None
    bram_18k_count: Optional[int] = None
    bram_utilization_percent: Optional[float] = None
    estimated_power_w: Optional[float] = None
```

### **3. Design Space Definition System**

```python
class DesignSpace:
    """Structured representation of design space for future DSE."""
    
    def __init__(self, blueprint: Blueprint):
        self.dimensions = self._extract_dimensions(blueprint)
        self.constraints = self._extract_constraints(blueprint)
        self.parameters = self._extract_parameters(blueprint)
    
    def get_dimension_ranges(self) -> Dict[str, Any]:
        """Get ranges for each design dimension."""
        return {
            'platforms': self.dimensions['platforms'],
            'strategies': self.dimensions['strategies'], 
            'parameters': self.dimensions['parameters']
        }
    
    def validate_design_point(self, point: DesignPoint) -> bool:
        """Check if design point is valid within space."""
        pass
    
    def estimate_space_size(self) -> int:
        """Estimate total number of design points."""
        pass
    
    def export_for_dse(self) -> Dict[str, Any]:
        """Export format for external DSE tools."""
        pass

@dataclass
class DesignPoint:
    """Single point in design space."""
    platform: str
    configuration: Dict[str, Any]
    finn_hooks: Dict[str, Any]  # Placeholder for future hooks
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_finn_config(self) -> Dict[str, Any]:
        """Convert to FINN configuration (current format)."""
        pass
    
    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector for ML algorithms."""
        pass
```

### **4. FINN Interface Placeholder System**

```python
class FINNInterfaceLayer:
    """Abstraction layer for FINN integration with future hook support."""
    
    def __init__(self):
        self.hooks = FINNHooksPlaceholder()
        self.translator = ConfigTranslator()
        
    def execute_build(self, model: onnx.ModelProto, 
                     design_point: DesignPoint) -> BrainsmithResult:
        """Execute FINN build with design point configuration."""
        
        # Translate design point to current FINN format
        finn_config = self.translator.translate(design_point)
        
        # Execute with placeholder hook system
        result = self._run_finn_with_placeholders(model, finn_config, design_point)
        
        # Collect comprehensive metrics
        metrics = self._collect_metrics(result, design_point)
        
        return BrainsmithResult(
            success=result.success,
            metrics=metrics,
            design_point=design_point,
            artifacts=result.artifacts
        )

class FINNHooksPlaceholder:
    """Placeholder system for future FINN hooks."""
    
    def __init__(self):
        self.model_ops = ModelOpsPlaceholder()
        self.model_transforms = ModelTransformsPlaceholder()
        self.hw_kernels = HWKernelsPlaceholder()
        self.hw_optimization = HWOptimizationPlaceholder()
    
    def configure_from_design_point(self, point: DesignPoint) -> None:
        """Configure placeholders from design point (future implementation)."""
        # TODO: Implement when FINN hooks are defined
        pass

class ModelOpsPlaceholder:
    """Placeholder for Model Ops hook."""
    supported_ops: List[str] = field(default_factory=list)
    custom_mappings: Dict[str, str] = field(default_factory=dict)
    # More fields will be added as FINN interface evolves

class ModelTransformsPlaceholder:
    """Placeholder for Model Transforms hook."""
    transform_sequence: List[str] = field(default_factory=list)
    # More fields will be added as FINN interface evolves

class HWKernelsPlaceholder:
    """Placeholder for HW Kernels hook."""
    available_kernels: Dict[str, List[str]] = field(default_factory=dict)
    # More fields will be added as FINN interface evolves

class HWOptimizationPlaceholder:
    """Placeholder for HW Optimization hook."""
    optimization_strategy: str = "default"
    # More fields will be added as FINN interface evolves
```

### **5. Extensible DSE Interface**

```python
class DSEInterface:
    """Abstract interface for future DSE engines."""
    
    @abstractmethod
    def explore(self, design_space: DesignSpace, 
                model: onnx.ModelProto,
                constraints: ConstraintSet) -> DSEResult:
        """Execute design space exploration."""
        pass
    
    @abstractmethod
    def get_next_design_point(self, previous_results: List[BrainsmithResult]) -> DesignPoint:
        """Get next design point to evaluate."""
        pass

class SimpleDSEEngine(DSEInterface):
    """Basic implementation for current functionality."""
    
    def explore(self, design_space: DesignSpace, 
                model: onnx.ModelProto,
                constraints: ConstraintSet) -> DSEResult:
        """Simple grid-like exploration (placeholder for research)."""
        
        results = []
        for design_point in self._generate_design_points(design_space):
            if constraints.is_valid(design_point):
                result = self.finn_interface.execute_build(model, design_point)
                results.append(result)
                
                # Early stopping placeholder
                if len(results) >= self.max_evaluations:
                    break
        
        return DSEResult(
            results=results,
            pareto_frontier=None,  # Placeholder for future analysis
            best_point=self._find_best_result(results),
            analysis=self._basic_analysis(results)
        )

class ExternalDSEAdapter(DSEInterface):
    """Adapter for external DSE tools/research."""
    
    def __init__(self, external_tool_interface):
        self.external_tool = external_tool_interface
    
    def explore(self, design_space: DesignSpace, 
                model: onnx.ModelProto,
                constraints: ConstraintSet) -> DSEResult:
        """Delegate to external DSE tool."""
        
        # Export design space in standard format
        space_export = design_space.export_for_dse()
        
        # Let external tool generate design points
        design_points = self.external_tool.generate_points(space_export, constraints)
        
        # Evaluate each point through Brainsmith
        results = []
        for point in design_points:
            result = self.finn_interface.execute_build(model, point)
            results.append(result)
            
            # Provide feedback to external tool
            self.external_tool.update_with_result(point, result.metrics)
        
        return DSEResult(results=results)
```

## Enhanced Blueprint Structure

### **Structured Design Space Definition**

```yaml
# brainsmith/blueprints/yaml/bert_extensible.yaml
name: "bert_extensible"
description: "BERT with extensible design space definition"
architecture: "transformer"

# Current build steps (unchanged)
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
      
    streamlining_strategy:
      type: "categorical"
      values: ["conservative", "balanced", "aggressive"]
      
    folding_strategy:
      type: "categorical"
      values: ["auto", "manual", "template"]
      
  # Placeholder for future FINN hooks
  finn_hooks_config:
    model_ops:
      # Placeholder structure
      custom_ops_enabled: 
        type: "boolean"
        default: false
    
    model_transforms:
      # Placeholder structure  
      optimization_level:
        type: "categorical"
        values: ["conservative", "balanced", "aggressive"]
        default: "balanced"
    
    hw_kernels:
      # Placeholder structure
      kernel_preference:
        type: "categorical" 
        values: ["rtl", "hls", "mixed"]
        default: "mixed"
    
    hw_optimization:
      # Placeholder structure
      optimization_time_budget:
        type: "continuous"
        range: [60, 3600]  # seconds
        default: 300

# Constraints for design space exploration
constraints:
  hard_constraints:
    max_lut_utilization: 0.8
    max_dsp_utilization: 0.8
    max_bram_utilization: 0.8
    
  soft_constraints:
    preferred_throughput: 2000
    preferred_power_w: 50
    
  custom_constraints:
    # Extensible for future research
    - name: "efficiency_constraint"
      type: "formula"
      expression: "throughput / (lut_count + dsp_count)"
      operator: ">"
      value: 0.5

# Metric collection configuration
metrics_config:
  collect_intermediate_results: true
  export_for_research: true
  custom_metrics:
    - name: "throughput_per_lut" 
      formula: "throughput / lut_count"
    - name: "efficiency_score"
      formula: "(throughput * accuracy) / power"
```

## Library Interface: Extensibility Focus

### **Current Simple Interface**

```python
import brainsmith

# Simple single build (current functionality)
result = brainsmith.compile_model(
    model=model,
    blueprint="bert_extensible", 
    output_dir="./build"
)

# Access comprehensive metrics
print(f"Performance: {result.metrics.performance}")
print(f"Resources: {result.metrics.resources}")
print(f"Design point: {result.metrics.design_point}")

# Export for external DSE research
dataset = result.to_research_dataset()
np.save("build_data.npy", dataset)
```

### **Extensible DSE Interface**

```python
# Basic design space exploration (simple implementation)
explorer = brainsmith.SimpleDSEEngine(max_evaluations=20)
design_space = brainsmith.load_design_space("bert_extensible")

results = explorer.explore(
    design_space=design_space,
    model=model,
    constraints=brainsmith.ConstraintSet.from_blueprint("bert_extensible")
)

# Access structured results
best_result = results.best_point
all_metrics = [r.metrics for r in results.results]

# Export comprehensive dataset for DSE research
research_data = brainsmith.export_research_dataset(results)
research_data.to_csv("dse_experiment_data.csv")

# Plugin external DSE tool
external_tool = MyCustomDSETool()
external_explorer = brainsmith.ExternalDSEAdapter(external_tool)
external_results = external_explorer.explore(design_space, model, constraints)
```

## CLI Interface: Structure and Data Focus

### **Build and Metric Collection**

```bash
# Single build with comprehensive metrics
brainsmith build model.onnx --blueprint bert_extensible --output ./build \
  --collect-metrics --export-research-data

# Multiple builds for data collection
brainsmith sweep model.onnx --blueprint bert_extensible --output ./sweep \
  --parameter target_fps=1000,2000,3000,4000,5000 \
  --parameter clk_period_ns=3.0,4.0,5.0 \
  --export-dataset sweep_data.csv

# Design space analysis
brainsmith analyze-space --blueprint bert_extensible \
  --show-dimensions --estimate-size --export-schema space_schema.json
```

### **Research and Extension Commands**

```bash
# Export data for external DSE tools
brainsmith export-research-data ./build_results/ \
  --format csv,numpy,hdf5 \
  --include-intermediates \
  --anonymize

# Validate external DSE tool integration
brainsmith validate-dse-adapter my_dse_tool.py \
  --test-design-space bert_extensible

# Plugin management
brainsmith plugins list
brainsmith plugins install research_metrics_plugin
brainsmith plugins enable advanced_analysis
```

## Implementation Roadmap: Structure First

### **Phase 1: Foundation Structure** ⭐ **PRIORITY**
1. **Metric Collection System**: Comprehensive data structures and collection
2. **Design Space Representation**: Structured design space definitions
3. **FINN Interface Placeholders**: Future-ready hook system
4. **Result Management**: Storage, tracking, export capabilities

### **Phase 2: Extensibility Layer**
1. **Plugin Architecture**: Custom metrics, analysis tools
2. **External DSE Interface**: API for research tool integration  
3. **Data Export Systems**: Research dataset generation
4. **Basic DSE Engine**: Simple exploration for validation

### **Phase 3: Research Enablement**
1. **Research API**: Clean interfaces for algorithm development
2. **Advanced Metrics**: ML features, performance modeling
3. **Experiment Management**: Systematic study tracking
4. **Validation Tools**: DSE algorithm testing framework

### **Future Phases: Research Driven**
- Advanced DSE algorithms (genetic, Bayesian, etc.)
- Multi-objective optimization engines
- Detailed FINN hook implementations
- Real-time optimization capabilities

## Success Metrics: Platform Readiness

### **Technical Foundation**
1. **Data Completeness**: Capture 100% of relevant FINN build metrics
2. **Interface Stability**: Clean APIs that won't break with future DSE research
3. **Extensibility**: Support 3+ external DSE tool integrations
4. **Performance**: Handle 1000+ build evaluations efficiently

### **Research Enablement**
1. **Dataset Quality**: Generate research-grade datasets for algorithm development
2. **Tool Integration**: Seamless connection with external optimization frameworks
3. **Flexibility**: Support diverse DSE research approaches without code changes
4. **Reproducibility**: Deterministic results and experiment tracking

This revised design focuses on creating a solid, extensible platform that enables future DSE research rather than implementing specific optimization algorithms. The structure provides comprehensive metric collection, clean interfaces, and extensibility points while maintaining simplicity in the current implementation.