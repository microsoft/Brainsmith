# 🔧 Brainsmith Core Components
## Detailed Component Architecture and Implementation

---

## 📋 Component Overview

The Brainsmith platform is built on five core component categories, each with specific responsibilities and well-defined interfaces:

1. **Configuration System** - Parameter and setting management
2. **Design Space Management** - Optimization space definition and manipulation
3. **Result and Metrics System** - Performance tracking and data collection
4. **Workflow Orchestration** - Task coordination and execution
5. **Integration Layer** - External tool and legacy system interfaces

---

## ⚙️ Configuration System

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                CONFIGURATION SYSTEM                     │
├─────────────────────────────────────────────────────────┤
│                 Configuration Types                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│  │ CompilerConfig  │ │   DSEConfig     │ │ LibraryConfig│ │
│  │                 │ │                 │ │             │ │
│  │• Build settings │ │• Strategy opts  │ │• Lib params │ │
│  │• FINN params    │ │• Objectives     │ │• Capabilities│ │
│  │• Output control │ │• Constraints    │ │• Dependencies│ │
│  └─────────────────┘ └─────────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Configuration Loading                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Configuration Loader                   │ │
│  │  • YAML/JSON file parsing                          │ │
│  │  • Environment variable integration                │ │
│  │  • Template expansion and substitution             │ │
│  │  • Validation and type checking                    │ │
│  │  • Default value application                       │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Configuration Validation                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Validation Engine                      │ │
│  │  • Schema-based validation                         │ │
│  │  • Cross-parameter consistency checking            │ │
│  │  • Resource constraint validation                  │ │
│  │  • Legacy compatibility verification               │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Configuration Classes

#### CompilerConfig
Primary configuration for compilation and build operations:

```python
@dataclass
class CompilerConfig:
    """Enhanced configuration with DSE support and extensibility."""
    
    # Core compilation parameters
    blueprint: str = ""
    output_dir: str = "./build"
    model_path: str = ""
    
    # Design space exploration settings
    dse_enabled: bool = False
    parameter_sweep: Optional[Dict[str, List[Any]]] = None
    single_design_point: Optional[DesignPoint] = None
    
    # FINN integration settings (legacy compatibility)
    target_fps: int = 3000
    synth_clk_period_ns: float = 3.33
    board: str = "V80"
    folding_config_file: Optional[str] = None
    
    # Advanced features
    collect_comprehensive_metrics: bool = True
    export_research_data: bool = False
    parallel_builds: int = 1
    
    def to_design_point(self) -> DesignPoint:
        """Convert configuration to design point representation."""
        pass
    
    def validate(self) -> List[str]:
        """Validate configuration and return issues."""
        pass
```

#### DSEConfig
Specialized configuration for design space exploration:

```python
@dataclass
class DSEConfig:
    """Configuration for design space exploration operations."""
    
    # Strategy selection
    strategy: str = "random"  # random, adaptive, genetic, etc.
    max_evaluations: int = 50
    random_seed: Optional[int] = None
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ["throughput_ops_sec"])
    objective_directions: List[str] = field(default_factory=lambda: ["maximize"])
    
    # Constraint handling
    enforce_hard_constraints: bool = True
    constraint_violation_penalty: float = 1000.0
    
    # External tool integration
    external_tool_interface: Optional[str] = None
    external_tool_config: Dict[str, Any] = field(default_factory=dict)
    
    # Convergence and stopping criteria
    early_stopping: bool = False
    convergence_patience: int = 10
    min_improvement: float = 0.01
```

### Configuration Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   YAML/JSON │───▶│   Parser    │───▶│ Validator   │───▶│ Config      │
│   File      │    │             │    │             │    │ Object      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Template    │    │ Type        │    │ Cross-param │    │ Ready for   │
│ Variables   │    │ Checking    │    │ Validation  │    │ Execution   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🎯 Design Space Management

### Architecture Components

```
┌─────────────────────────────────────────────────────────┐
│              DESIGN SPACE MANAGEMENT                    │
├─────────────────────────────────────────────────────────┤
│                Parameter Definition                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              ParameterDefinition                    │ │
│  │  • Name and type specification                      │ │
│  │  • Value ranges and constraints                     │ │
│  │  • Validation rules                                 │ │
│  │  • Default values and metadata                      │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Design Space                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                DesignSpace                          │ │
│  │  • Parameter collection management                  │ │
│  │  • Space validation and consistency checking       │ │
│  │  • Point generation and sampling                    │ │
│  │  • Constraint evaluation                            │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Design Point                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                DesignPoint                          │ │
│  │  • Parameter value storage                          │ │
│  │  • Result and objective tracking                    │ │
│  │  • Metadata and provenance                          │ │
│  │  • Serialization support                            │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Parameter Types and Definitions

#### Supported Parameter Types

```python
class ParameterType(Enum):
    """Supported parameter types for optimization."""
    INTEGER = "integer"      # Discrete numeric values
    FLOAT = "float"         # Continuous numeric values  
    CATEGORICAL = "categorical"  # Discrete choice values
    BOOLEAN = "boolean"     # Binary choices

class ParameterDefinition:
    """Definition of a single optimization parameter."""
    
    def __init__(self, name: str, param_type: ParameterType, 
                 values: Any = None, range_min: float = None, 
                 range_max: float = None, default: Any = None):
        self.name = name
        self.type = param_type
        self.values = values        # For categorical
        self.range_min = range_min  # For numeric types
        self.range_max = range_max  # For numeric types
        self.default = default
    
    def validate_value(self, value: Any) -> bool:
        """Check if value is valid for this parameter."""
        pass
    
    def sample_value(self, rng: np.random.Generator) -> Any:
        """Generate random valid value."""
        pass
```

#### Design Space Construction

```python
class DesignSpace:
    """Container and manager for optimization parameter space."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.parameters = {}  # Dict[str, ParameterDefinition]
        self.constraints = []  # List[Constraint]
        self.metadata = {}
    
    def add_parameter(self, param_def: ParameterDefinition):
        """Add parameter definition to space."""
        pass
    
    def add_constraint(self, constraint: Constraint):
        """Add constraint to design space."""
        pass
    
    def generate_points(self, n_points: int, 
                       strategy: str = "random") -> List[DesignPoint]:
        """Generate design points using specified strategy."""
        pass
    
    def validate_point(self, point: DesignPoint) -> Tuple[bool, List[str]]:
        """Validate design point against space constraints."""
        pass
```

#### Design Point Management

```python
class DesignPoint:
    """Represents a single point in the design space."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.results = {}      # Execution results
        self.objectives = {}   # Objective function values
        self.metadata = {}     # Additional information
        self.timestamp = datetime.now()
    
    def set_result(self, key: str, value: Any):
        """Store execution result."""
        pass
    
    def set_objective(self, key: str, value: float):
        """Store objective function value."""
        pass
    
    def dominates(self, other: 'DesignPoint', 
                 objectives: List[str]) -> bool:
        """Check Pareto dominance relationship."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass
```

---

## 📊 Result and Metrics System

### Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│              RESULT AND METRICS SYSTEM                  │
├─────────────────────────────────────────────────────────┤
│                   Metrics Collection                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              BrainsmithMetrics                      │ │
│  │  • Performance metrics (throughput, latency)       │ │
│  │  • Resource utilization (LUT, DSP, BRAM)          │ │
│  │  • Power consumption estimates                     │ │
│  │  • Custom user-defined metrics                     │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Result Management                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              BrainsmithResult                       │ │
│  │  • Build success/failure status                    │ │
│  │  • Comprehensive metrics integration               │ │
│  │  • Artifact location tracking                      │ │
│  │  • Error and warning collection                    │ │
│  │  • Research data export support                    │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Aggregation and Analysis               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │          DSEResult / ParameterSweepResult           │ │
│  │  • Multi-point result aggregation                  │ │
│  │  • Pareto frontier computation                      │ │
│  │  • Statistical analysis and reporting              │ │
│  │  • Visualization data preparation                  │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Metrics Collection Framework

#### Core Metrics Structure

```python
@dataclass
class PerformanceMetrics:
    """Performance-related measurements."""
    throughput_ops_sec: Optional[float] = None
    latency_cycles: Optional[int] = None
    frequency_mhz: Optional[float] = None
    efficiency_ratio: Optional[float] = None

@dataclass  
class ResourceMetrics:
    """FPGA resource utilization measurements."""
    lut_count: Optional[int] = None
    lut_utilization_percent: Optional[float] = None
    dsp_count: Optional[int] = None
    dsp_utilization_percent: Optional[float] = None
    bram_count: Optional[int] = None
    bram_utilization_percent: Optional[float] = None
    estimated_power_w: Optional[float] = None

@dataclass
class BrainsmithMetrics:
    """Comprehensive metrics collection."""
    build_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Core metric categories
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    
    # Extensible custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_custom_metric(self, name: str, value: Any, unit: str = ""):
        """Add user-defined metric."""
        pass
    
    def to_research_dataset(self) -> Dict[str, Any]:
        """Export for academic research."""
        pass
```

### Result Management

#### Single Build Results

```python
@dataclass
class BrainsmithResult:
    """Comprehensive result object for single builds."""
    
    # Core result information
    success: bool = False
    output_dir: str = ""
    build_time: float = 0.0
    
    # Enhanced data
    metrics: Optional[BrainsmithMetrics] = None
    design_point: Optional[DesignPoint] = None
    
    # Build artifacts
    final_model_path: Optional[str] = None
    stitched_ip_path: Optional[str] = None
    reports_dir: Optional[str] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate result summary."""
        pass
    
    def to_research_dict(self) -> Dict[str, Any]:
        """Export research-ready data."""
        pass
    
    def save_result(self, filepath: str) -> str:
        """Persist result to file."""
        pass
```

#### Multi-Point Results

```python
@dataclass
class DSEResult:
    """Results from design space exploration."""
    
    results: List[BrainsmithResult] = field(default_factory=list)
    design_space_info: Dict[str, Any] = field(default_factory=dict)
    exploration_time: float = 0.0
    strategy_used: str = ""
    
    # Analysis results
    pareto_frontier: Optional[List[BrainsmithResult]] = None
    best_configurations: Dict[str, BrainsmithResult] = field(default_factory=dict)
    
    def get_successful_results(self) -> List[BrainsmithResult]:
        """Filter to successful builds only."""
        pass
    
    def get_pareto_frontier(self, objectives: List[str], 
                           directions: List[str]) -> List[BrainsmithResult]:
        """Compute Pareto-optimal results."""
        pass
    
    def export_research_dataset(self, filepath: str):
        """Export comprehensive research data."""
        pass
```

---

## 🔄 Workflow Orchestration

### Orchestration Architecture

```
┌─────────────────────────────────────────────────────────┐
│               WORKFLOW ORCHESTRATION                    │
├─────────────────────────────────────────────────────────┤
│                  Task Management                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │               Task Scheduler                        │ │
│  │  • Dependency resolution and ordering               │ │
│  │  • Resource allocation and management               │ │
│  │  • Parallel execution coordination                 │ │
│  │  • Progress tracking and status updates            │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Library Coordination                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Library Manager                        │ │
│  │  • Library discovery and loading                   │ │
│  │  • Capability matching and selection               │ │
│  │  • Data flow coordination between libraries        │ │
│  │  • Error handling and recovery                     │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Execution Engine                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Workflow Engine                        │ │
│  │  • Sequential and parallel task execution          │ │
│  │  • State management and checkpointing             │ │
│  │  • Resource cleanup and finalization              │ │
│  │  • Result collection and aggregation              │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Workflow Execution Patterns

#### Single Build Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Load        │───▶│ Validate    │───▶│ Transform   │───▶│ Compile     │
│ Blueprint   │    │ Config      │    │ Model       │    │ Hardware    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                             │                  │
                                             ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Generate    │◀───│ Collect     │◀───│ Optimize    │◀───│ Synthesize  │
│ Report      │    │ Metrics     │    │ Design      │    │ & P&R       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

#### DSE Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Initialize  │───▶│ Generate    │───▶│ Evaluate    │
│ DSE Engine  │    │ Design      │    │ Design      │◀─┐
└─────────────┘    │ Points      │    │ Points      │  │
                   └─────────────┘    └─────────────┘  │
                          ▲                  │         │
                          │                  ▼         │
                   ┌─────────────┐    ┌─────────────┐  │
                   │ Update      │◀───│ Collect     │  │
                   │ Strategy    │    │ Results     │  │
                   └─────────────┘    └─────────────┘  │
                          │                            │
                          ▼                            │
                   ┌─────────────┐                     │
                   │ Check       │────────────────────┘
                   │ Convergence │ No
                   └─────────────┘
                          │ Yes
                          ▼
                   ┌─────────────┐
                   │ Generate    │
                   │ Final       │
                   │ Analysis    │
                   └─────────────┘
```

---

## 🔌 Integration Layer

### Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 INTEGRATION LAYER                       │
├─────────────────────────────────────────────────────────┤
│                 Legacy Compatibility                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Legacy API Router                      │ │
│  │  • Function signature translation                   │ │
│  │  • Parameter mapping and validation                │ │
│  │  • Result format conversion                        │ │
│  │  • Error handling adaptation                       │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                External Tool Integration                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              External Adapters                      │ │
│  │  • FINN interface and process management           │ │
│  │  • Third-party DSE framework integration           │ │
│  │  • Custom tool plugin architecture                 │ │
│  │  • Process isolation and communication             │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Data Format Conversion                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Format Converters                      │ │
│  │  • Configuration format translation                │ │
│  │  • Result data standardization                     │ │
│  │  • Metadata preservation and mapping               │ │
│  │  • Version compatibility handling                  │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Legacy API Compatibility

```python
def explore_design_space(*args, **kwargs):
    """Legacy API compatibility function."""
    
    # Translate legacy parameters to new format
    enhanced_config = translate_legacy_params(args, kwargs)
    
    # Route to enhanced implementation
    result = brainsmith_explore(enhanced_config)
    
    # Convert result to legacy format if needed
    return convert_result_format(result, legacy=True)

def translate_legacy_params(args, kwargs) -> BrainsmithConfig:
    """Convert legacy parameter format to enhanced configuration."""
    pass

def convert_result_format(result: BrainsmithResult, 
                         legacy: bool = False) -> Any:
    """Convert result format for compatibility."""
    pass
```

---

*Next: [Library Ecosystem](04_LIBRARY_ECOSYSTEM.md)*