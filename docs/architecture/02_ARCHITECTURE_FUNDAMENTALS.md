# 🏗️ Brainsmith Architecture Fundamentals
## Core Design Principles and System Architecture

---

## 🔬 Dataflow Design Ethos

### Component Hierarchy (Fundamental to Brainsmith)

```
┌─────────────────────────────────────────────────────────┐
│                 DATAFLOW ACCELERATOR                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                DATAFLOW CORE                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │ HW KERNEL   │─▶│ HW KERNEL   │─▶│ HW KERNEL   │ │ │
│  │  │ (MatMul)    │  │ (Threshold) │  │ (LayerNorm) │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  │           ▲                ▲                ▲       │ │
│  │           │                │                │       │ │
│  │      ┌─────────┐    ┌─────────┐    ┌─────────┐     │ │
│  │      │Parameters│    │Parameters│    │Parameters│     │ │
│  │      │PE, SIMD  │    │PE, Steps │    │PE, SIMD  │     │ │
│  │      └─────────┘    └─────────┘    └─────────┘     │ │
│  └─────────────────────────────────────────────────────┘ │
│                            │                             │
│                    ┌─────────────┐                      │
│                    │   Shell     │                      │
│                    │ Integration │                      │
│                    └─────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### FINN Integration Model

**FINN Builder Role**: Optimizes within the *search space* - implementation variations of a given architecture
**Brainsmith DSE Role**: Optimizes within the *design space* - architectural choices and strategies

| FINN Search Space | Brainsmith Design Space |
|-------------------|-------------------------|
| Network optimizations | Platform selection |
| FIFO sizing | Kernel implementations |
| Kernel parallelism | DSE model transforms |
| Kernel variations | DSE HW transforms |

### Dataflow Accelerator Design Philosophy

Brainsmith is fundamentally designed around **dataflow accelerator principles** where:

- **Hardware Kernels** are the atomic units of computation (e.g., MatMul, Thresholding, LayerNorm)
- **Dataflow Cores** are composed by connecting kernels in a streaming pipeline
- **Parameters** (PE, SIMD, Steps) control the parallelism and resource utilization of each kernel
- **Shell Integration** provides the interface between the dataflow core and the FPGA platform

This hierarchy maps directly to FINN's architecture, where Brainsmith orchestrates the higher-level design decisions while FINN handles the low-level implementation details.

---

## 📋 Architectural Overview

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BRAINSMITH PLATFORM                        │
├─────────────────────────────────────────────────────────────────┤
│                         API Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Legacy API    │  │  Enhanced API   │  │   CLI Tools     │ │
│  │ explore_design_ │  │ brainsmith_     │  │   Command-line  │ │
│  │ space()         │  │ explore()       │  │   interface     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Core Orchestration                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Blueprint     │  │   Design Space  │  │   Workflow      │ │
│  │   Manager       │  │   Orchestrator  │  │   Manager       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Design Space Exploration                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   DSE Engine    │  │   Strategies    │  │   Analysis      │ │
│  │   Interface     │  │   (6+ algos)    │  │   Tools         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Library Ecosystem                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Transforms    │  │  HW Optimization│  │    Analysis     │ │
│  │   Library       │  │     Library     │  │    Library      │ │
│  │                 │  │                 │  │                 │ │
│  │ • Quantization  │  │ • Genetic Algo  │  │ • Roofline      │ │
│  │ • Folding       │  │ • Pareto Optim  │  │ • Resource Util │ │
│  │ • Streamlining  │  │ • Multi-obj     │  │ • Performance   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                        Core Infrastructure                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Configuration │  │     Results     │  │     Metrics     │ │
│  │   Management    │  │   & Reporting   │  │   Collection    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     External Integrations                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │      FINN       │  │   External DSE  │  │   Custom Tools  │ │
│  │   Interface     │  │   Frameworks    │  │   Integration   │ │
│  │  (Primary)      │  │   (Secondary)   │  │   (Optional)    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Design Principles

### 1. FINN-Centric Design

Brainsmith is fundamentally designed as a **wrapper and extension of FINN**:

```python
# FINN-first architecture principle
class BrainsmithCore:
    def __init__(self):
        self.finn_interface = FINNInterface()  # Primary integration
        self.dataflow_builder = DataflowCoreBuilder()
        self.kernel_library = HardwareKernelLibrary()
```

**Benefits:**
- Leverage FINN's proven dataflow acceleration capabilities
- Build on established hardware kernel implementations
- Maintain compatibility with FINN ecosystem
- Focus on higher-level optimization rather than reimplementation

### 2. Modularity and Separation of Concerns

Each component has clearly defined responsibilities and interfaces:

```python
# Clear component boundaries with dataflow focus
class APILayer:           # User interface and request handling
class CoreOrchestration:  # Dataflow workflow coordination and management  
class DSEEngine:          # Dataflow optimization algorithm execution
class LibraryEcosystem:   # Specialized dataflow functionality modules
class Infrastructure:     # Common services and utilities
```

**Benefits:**
- Independent development and testing
- Easy maintenance and debugging
- Clear upgrade and extension paths
- Reusable components across dataflow projects

### 3. Extensibility Through Interfaces

All major components implement well-defined interfaces:

```python
# Example: Library interface for dataflow extensions
class DataflowLibraryInterface(ABC):
    @abstractmethod
    def get_dataflow_capabilities(self) -> Dict[str, str]:
        pass
    
    @abstractmethod  
    def configure_for_dataflow(self, config: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def execute_dataflow_operation(self, inputs: Any) -> Any:
        pass
```

**Extension Points:**
- New dataflow optimization strategies
- Custom dataflow analysis algorithms
- Additional hardware kernel libraries
- FINN integration enhancements

### 4. Backward Compatibility

Legacy API preservation with automatic routing:

```python
# Legacy function automatically routes to new dataflow implementation
def explore_design_space(*args, **kwargs):
    # Automatic parameter translation for dataflow context
    enhanced_config = translate_legacy_params_for_dataflow(args, kwargs)
    # Route to enhanced dataflow implementation
    return brainsmith_explore(enhanced_config)
```

### 5. Configuration-Driven Behavior

Minimize hard-coded behavior through comprehensive configuration:

```yaml
# Example dataflow configuration structure
brainsmith:
  dataflow:
    finn_integration: true
    kernel_library: "standard"
    core_builder: "automatic"
  dse:
    strategy: "adaptive"
    max_evaluations: 100
    objectives: ["throughput", "power"]
  libraries:
    transforms:
      enabled: true
      pipeline: ["quantize", "fold", "streamline"]
    hw_optim:
      strategy: "genetic"
      population_size: 50
```

---

## 🔧 Core Component Architecture

### API Layer

The API layer provides multiple interfaces for different user needs:

```
┌─────────────────────────────────────────────────────────┐
│                      API LAYER                          │
├─────────────────────────────────────────────────────────┤
│                   Request Routing                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Request Router                         │ │
│  │  • Legacy API compatibility checking                │ │
│  │  • Parameter translation and validation             │ │
│  │  • Enhanced API feature detection                   │ │
│  │  • Dataflow-specific error handling                 │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Interface Types                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│  │   Python API    │ │   CLI Interface │ │ REST API    │ │
│  │                 │ │                 │ │ (Future)    │ │
│  │ • Function calls│ │ • Command line  │ │ • HTTP API  │ │
│  │ • Object-orient │ │ • Batch scripts │ │ • Web UI    │ │
│  │ • Interactive   │ │ • Automation    │ │ • Services  │ │
│  └─────────────────┘ └─────────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Orchestration

Central coordination of platform operations:

```
┌─────────────────────────────────────────────────────────┐
│                 CORE ORCHESTRATION                      │
├─────────────────────────────────────────────────────────┤
│                 Blueprint Manager                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  • YAML configuration loading and validation        │ │
│  │  • Dataflow template expansion and substitution     │ │
│  │  • Design space specification translation            │ │
│  │  • Multi-model support and kernel library mapping   │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Design Space Orchestrator                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  • Parameter space construction and validation       │ │
│  │  • Dataflow design point generation and management  │ │
│  │  • Constraint checking and feasibility analysis     │ │
│  │  • Result aggregation and analysis coordination     │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Workflow Manager                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  • Task scheduling and dependency management        │ │
│  │  • FINN integration and dataflow coordination       │ │
│  │  • Error recovery and retry logic                   │ │
│  │  • Progress tracking and status reporting           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Design Space Exploration Engine

Advanced optimization and search capabilities:

```
┌─────────────────────────────────────────────────────────┐
│            DESIGN SPACE EXPLORATION ENGINE              │
├─────────────────────────────────────────────────────────┤
│                    Strategy Engine                      │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Strategy Types:                                    │ │
│  │  • Random Sampling                                  │ │
│  │  • Latin Hypercube Sampling (LHS)                  │ │
│  │  • Sobol Sequences                                 │ │
│  │  • Adaptive Sampling                               │ │
│  │  • Bayesian Optimization                           │ │
│  │  • Genetic Algorithms                              │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Strategy Selection                      │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Automatic recommendation based on:                 │ │
│  │  • Problem size (parameter count, evaluation budget)│ │
│  │  • Objective count (single vs multi-objective)     │ │
│  │  • Dataflow search space characteristics            │ │
│  │  • Computational constraints                        │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Multi-Objective Optimization              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  • Pareto frontier computation                       │ │
│  │  • Dominance relationship analysis                  │ │
│  │  • Trade-off visualization and reporting            │ │
│  │  • Constraint handling and penalty methods          │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Architecture

### Information Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│   API       │───▶│ Blueprint   │
│   Request   │    │   Layer     │    │ Manager     │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Results   │◀───│ Workflow    │◀───│ Design Space│
│ & Reports   │    │ Manager     │    │Orchestrator │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                  │                   │
       │                  ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Analysis    │    │ Library     │    │ DSE Engine  │
│ Library     │    │ Ecosystem   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                  │                   │
       │                  ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Metrics &   │◀───│ Transform & │◀───│ Dataflow    │
│ Performance │    │ HW Optim    │    │ Design Pts  │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                  │
       │                  ▼
┌─────────────┐    ┌─────────────┐
│ FINN        │◀───│ FINN        │
│ Results     │    │ Interface   │
└─────────────┘    └─────────────┘
```

### Data Types and Structures

#### Configuration Data
```python
# Hierarchical configuration structure with dataflow focus
BrainsmithConfig = {
    'blueprint': str,           # Blueprint identifier
    'model': ModelConfig,       # Model specification
    'targets': TargetConfig,    # Performance targets
    'dataflow': DataflowConfig, # Dataflow-specific settings
    'dse': DSEConfig,          # Optimization settings
    'libraries': LibraryConfig, # Library configurations
    'finn': FINNConfig,        # FINN integration settings
    'output': OutputConfig      # Result settings
}
```

#### Design Space Data
```python
# Design space representation for dataflow accelerators
DataflowDesignSpace = {
    'parameters': Dict[str, ParameterDefinition],
    'constraints': List[Constraint],
    'objectives': List[Objective],
    'kernel_requirements': List[KernelRequirement],
    'metadata': Dict[str, Any]
}

DataflowDesignPoint = {
    'parameters': Dict[str, Any],
    'finn_config': FINNBuildConfig,
    'results': Dict[str, Any],
    'objectives': Dict[str, float],
    'metadata': Dict[str, Any]
}
```

#### Result Data
```python
# Comprehensive result structure with dataflow focus
BrainsmithResult = {
    'success': bool,
    'build_time': float,
    'finn_build_result': FINNBuildResult,
    'metrics': BrainsmithMetrics,
    'design_point': DataflowDesignPoint,
    'artifacts': Dict[str, str],
    'dataflow_analysis': DataflowAnalysis,
    'errors': List[str],
    'warnings': List[str]
}
```

---

## 🔌 Interface Contracts

### Library Interface Contract

All libraries must implement the base interface:

```python
class LibraryInterface(ABC):
    """Base interface for all Brainsmith libraries."""
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, str]:
        """Return dictionary of capability names and descriptions."""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure library with provided settings."""
        pass
    
    @abstractmethod
    def execute(self, inputs: Any) -> Any:
        """Execute library functionality with inputs."""
        pass
    
    def get_version(self) -> str:
        """Return library version string."""
        return "1.0.0"
    
    def is_available(self) -> bool:
        """Check if library dependencies are available."""
        return True
```

### DSE Strategy Interface

Optimization strategies implement a standard interface:

```python
class DSEStrategy(ABC):
    """Base interface for design space exploration strategies."""
    
    @abstractmethod
    def suggest(self, n_points: int) -> List[DataflowDesignPoint]:
        """Suggest next dataflow design points to evaluate."""
        pass
    
    @abstractmethod
    def update(self, point: DataflowDesignPoint, results: Dict[str, Any]):
        """Update strategy with evaluation results."""
        pass
    
    @abstractmethod
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        pass
```

---

## 🚀 Scalability Considerations

### Performance Optimization

- **Lazy Loading**: Components loaded only when needed
- **Caching**: Expensive FINN computations cached for reuse
- **Parallel Execution**: Multi-threaded dataflow evaluation support
- **Memory Management**: Efficient data structure usage for large design spaces

### Extensibility Patterns

- **Plugin Architecture**: Dynamic library discovery and loading
- **Event System**: Loose coupling through event-driven communication
- **Configuration Injection**: Runtime behavior modification
- **Interface Versioning**: Backward compatible evolution
- **FINN Integration Points**: Well-defined extension points for FINN enhancements

---

*Next: [Core Components](03_CORE_COMPONENTS.md)*