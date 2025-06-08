# 🏗️ Brainsmith Architecture Fundamentals
## Core Design Principles and System Architecture

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
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Design Principles

### 1. Modularity and Separation of Concerns

Each component has clearly defined responsibilities and interfaces:

```python
# Clear component boundaries
class APILayer:           # User interface and request handling
class CoreOrchestration:  # Workflow coordination and management  
class DSEEngine:          # Optimization algorithm execution
class LibraryEcosystem:   # Specialized functionality modules
class Infrastructure:     # Common services and utilities
```

**Benefits:**
- Independent development and testing
- Easy maintenance and debugging
- Clear upgrade and extension paths
- Reusable components across projects

### 2. Extensibility Through Interfaces

All major components implement well-defined interfaces:

```python
# Example: Library interface for extensions
class LibraryInterface(ABC):
    @abstractmethod
    def get_capabilities(self) -> Dict[str, str]:
        pass
    
    @abstractmethod  
    def configure(self, config: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def execute(self, inputs: Any) -> Any:
        pass
```

**Extension Points:**
- New optimization strategies
- Custom analysis algorithms
- Additional transformation libraries
- External tool integrations

### 3. Backward Compatibility

Legacy API preservation with automatic routing:

```python
# Legacy function automatically routes to new implementation
def explore_design_space(*args, **kwargs):
    # Automatic parameter translation
    enhanced_config = translate_legacy_params(args, kwargs)
    # Route to enhanced implementation
    return brainsmith_explore(enhanced_config)
```

### 4. Configuration-Driven Behavior

Minimize hard-coded behavior through comprehensive configuration:

```yaml
# Example configuration structure
brainsmith:
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
│  │  • Error handling and response formatting           │ │
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
│  │  • Template expansion and parameter substitution    │ │
│  │  │  • Design space specification translation        │ │
│  │  • Multi-model support and library mapping         │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Design Space Orchestrator                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  • Parameter space construction and validation       │ │
│  │  • Design point generation and management           │ │
│  │  • Constraint checking and feasibility analysis     │ │
│  │  • Result aggregation and analysis coordination     │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Workflow Manager                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  • Task scheduling and dependency management        │ │
│  │  • Library coordination and data flow               │ │
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
│  │  • Search space characteristics                     │ │
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
│ Metrics &   │◀───│ Transform & │◀───│ Design      │
│ Performance │    │ HW Optim    │    │ Points      │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Data Types and Structures

#### Configuration Data
```python
# Hierarchical configuration structure
BrainsmithConfig = {
    'blueprint': str,           # Blueprint identifier
    'model': ModelConfig,       # Model specification
    'targets': TargetConfig,    # Performance targets
    'dse': DSEConfig,          # Optimization settings
    'libraries': LibraryConfig, # Library configurations
    'output': OutputConfig      # Result settings
}
```

#### Design Space Data
```python
# Design space representation
DesignSpace = {
    'parameters': Dict[str, ParameterDefinition],
    'constraints': List[Constraint],
    'objectives': List[Objective],
    'metadata': Dict[str, Any]
}

DesignPoint = {
    'parameters': Dict[str, Any],
    'results': Dict[str, Any],
    'objectives': Dict[str, float],
    'metadata': Dict[str, Any]
}
```

#### Result Data
```python
# Comprehensive result structure
BrainsmithResult = {
    'success': bool,
    'build_time': float,
    'metrics': BrainsmithMetrics,
    'design_point': DesignPoint,
    'artifacts': Dict[str, str],
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
    def suggest(self, n_points: int) -> List[DesignPoint]:
        """Suggest next design points to evaluate."""
        pass
    
    @abstractmethod
    def update(self, point: DesignPoint, results: Dict[str, Any]):
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
- **Caching**: Expensive computations cached for reuse
- **Parallel Execution**: Multi-threaded evaluation support
- **Memory Management**: Efficient data structure usage

### Extensibility Patterns

- **Plugin Architecture**: Dynamic library discovery and loading
- **Event System**: Loose coupling through event-driven communication
- **Configuration Injection**: Runtime behavior modification
- **Interface Versioning**: Backward compatible evolution

---

*Next: [Core Components](03_CORE_COMPONENTS.md)*