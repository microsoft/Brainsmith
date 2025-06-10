# 🏗️ **BrainSmith Repository Structure & Workflow Guide**
## Comprehensive Visual Documentation with Detailed Diagrams

---

## 📋 **Document Overview**

This guide provides comprehensive visual documentation of the BrainSmith repository structure, component relationships, and operational workflows. It includes detailed ASCII diagrams, flowcharts, and structural visualizations to help developers, reviewers, and maintainers understand the codebase organization and execution flows.

---

## 🗂️ **Repository Structure Overview**

### **Top-Level Directory Structure**
```
brainsmith/                           # 🏠 Root Repository
├── 📁 brainsmith/                   # 🐍 Main Python Package
│   ├── 📁 core/                     # 🎯 Core Platform Components
│   ├── 📁 finn/                     # 🔧 FINN Integration Engine
│   ├── 📁 dse/                      # 🎲 Design Space Exploration Engine
│   ├── 📁 blueprints/               # 📋 Blueprint Management System
│   ├── 📁 analysis/                 # 📈 Analysis & Reporting Framework
│   ├── 📁 selection/                # 🎯 Selection Strategy Engine
│   ├── 📁 automation/               # 🧠 Automation & Learning Framework
│   ├── 📁 metrics/                  # 📊 Metrics Collection Infrastructure
│   ├── 📁 transformation/           # ⚙️ Model Transformation Pipeline
│   ├── 📁 custom_op/                # 🛠️ Custom Operator Definitions
│   ├── 📁 hw_kernels/               # 💾 Hardware Kernel Implementations
│   ├── 📁 libraries/                # 📚 Transform & Analysis Libraries
│   └── 📁 tools/                    # 🔨 Development & Generation Tools
├── 📁 tests/                        # 🧪 Comprehensive Test Suite
│   ├── 📁 functional/               # ✅ Functional Testing Framework
│   ├── 📁 performance/              # ⚡ Performance Benchmarking
│   ├── 📁 integration/              # 🔗 Integration Testing
│   ├── 📁 configs/                  # ⚙️ Test Configuration
│   └── 📁 end2end/                  # 🎯 End-to-End Testing
├── 📁 docs/                         # 📚 Documentation Repository
│   ├── 📁 architecture/             # 🏛️ Architectural Specifications
│   └── 📁 implementation/           # 🛠️ Implementation Guides
├── 📁 demos/                        # 🎬 Demonstration Examples
├── 📁 examples/                     # 💡 Usage Examples
├── 📁 docker/                       # 🐳 Container Configuration
└── 📁 ssh_keys/                     # 🔐 SSH Key Management
```

---

## 🎯 **Core Module Architecture**

### **Core Platform Components (`brainsmith/core/`)**

```
📁 brainsmith/core/                  # Central Platform Logic
├── 🔥 api.py                       # 🚪 Main User-Facing API
│   ├── brainsmith_explore()        #   ├─ Primary exploration function
│   ├── brainsmith_roofline()       #   ├─ Performance roofline analysis
│   ├── brainsmith_dataflow()       #   ├─ Dataflow graph analysis
│   ├── brainsmith_generate()       #   ├─ Code generation interface
│   └── brainsmith_workflow()       #   └─ Complete workflow orchestration
│
├── 🌌 design_space.py              # 🎛️ Design Space Management
│   ├── DesignSpace                 #   ├─ Design space definition
│   ├── DesignPoint                 #   ├─ Individual design points
│   ├── ParameterDefinition         #   ├─ Parameter specifications
│   └── sample_design_space()       #   └─ Sampling algorithms
│
├── 🏭 compiler.py                  # ⚙️ Model Compilation Engine
│   ├── BrainsmithCompiler          #   ├─ Main compilation orchestrator
│   ├── compile_model()             #   ├─ Model compilation workflow
│   └── optimize_model()            #   └─ Model optimization pipeline
│
├── 🔗 finn_interface.py            # 🤝 FINN Framework Integration
│   ├── FINNInterface               #   ├─ FINN API wrapper
│   ├── FINNBuilder                 #   ├─ Build orchestration
│   └── FINNAnalyzer                #   └─ FINN analysis integration
│
├── 📊 metrics.py                   # 📈 Core Metrics Infrastructure
│   ├── BrainsmithMetrics           #   ├─ Metrics collection framework
│   ├── PerformanceMetrics          #   ├─ Performance tracking
│   └── ResourceMetrics             #   └─ Resource utilization tracking
│
├── ⚙️ config.py                    # 🔧 Configuration Management
│   ├── BrainsmithConfig            #   ├─ Global configuration
│   ├── load_config()               #   ├─ Configuration loading
│   └── validate_config()           #   └─ Configuration validation
│
├── 🏗️ workflow.py                  # 🔄 Workflow Orchestration
│   ├── WorkflowEngine              #   ├─ Main workflow coordinator
│   ├── StepExecutor                #   ├─ Individual step execution
│   └── DependencyResolver          #   └─ Step dependency resolution
│
└── 📋 result.py                    # 📊 Result Management
    ├── BrainsmithResult            #   ├─ Generic result container
    ├── DSEResult                   #   ├─ DSE-specific results
    └── BuildResult                 #   └─ Build result container
```

### **API Call Flow Diagram**

```
🚀 User API Call Flow
┌─────────────────┐
│   User Script   │
└─────────┬───────┘
          │ import brainsmith
          │ brainsmith.optimize_model(...)
          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  brainsmith/    │────▶│  core/api.py    │────▶│ Core Functions  │
│  __init__.py    │     │                 │     │                 │
│                 │     │ ┌─────────────┐ │     │ ┌─────────────┐ │
│ ┌─────────────┐ │     │ │brainsmith_  │ │     │ │Design Space │ │
│ │Public API   │ │     │ │explore()    │ │     │ │Management   │ │
│ │Functions    │ │     │ └─────────────┘ │     │ └─────────────┘ │
│ └─────────────┘ │     │ ┌─────────────┐ │     │ ┌─────────────┐ │
│ ┌─────────────┐ │     │ │validate_    │ │     │ │FINN         │ │
│ │Legacy       │ │     │ │blueprint()  │ │     │ │Integration  │ │
│ │Compatibility│ │     │ └─────────────┘ │     │ └─────────────┘ │
│ └─────────────┘ │     └─────────────────┘     └─────────────────┘
└─────────────────┘              │                       │
          ▲                      ▼                       ▼
          │              ┌─────────────────┐     ┌─────────────────┐
          │              │   DSE Engine    │     │  FINN Workflow  │
          │              │   Integration   │     │   Orchestration │
          │              └─────────────────┘     └─────────────────┘
          │                       │                       │
          │                       ▼                       ▼
          │              ┌─────────────────┐     ┌─────────────────┐
          │              │   Results &     │     │  Build Artifacts│
          │              │   Analysis      │     │  & Reports      │
          │              └─────────────────┘     └─────────────────┘
          │                       │                       │
          └───────────────────────┴───────────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  User Results   │
                         │   (Returned)    │
                         └─────────────────┘
```

---

## 🔧 **FINN Integration Engine**

### **FINN Module Structure (`brainsmith/finn/`)**

```
📁 brainsmith/finn/                  # FINN Framework Integration
├── 🏗️ orchestration.py             # 🎭 Build Orchestration Engine
│   ├── FINNBuildOrchestrator        #   ├─ Main orchestration controller
│   ├── ParallelBuildManager         #   ├─ Parallel build coordination
│   ├── BuildDependencyResolver      #   ├─ Dependency management
│   └── ArtifactCacheManager         #   └─ Build artifact caching
│
├── ⚙️ workflow.py                   # 🔄 FINN Workflow Management
│   ├── ModelOpsManager              #   ├─ Model operations management
│   ├── ModelTransformsManager       #   ├─ Transform pipeline management
│   ├── HwOptimizationManager        #   ├─ Hardware optimization directives
│   └── WorkflowValidator            #   └─ Workflow validation engine
│
├── 🚀 hw_kernels_manager.py         # 💎 Hardware Kernel Management
│   ├── HwKernelsManager             #   ├─ Kernel discovery & management
│   ├── KernelPerformanceModeler     #   ├─ Performance prediction models
│   ├── OptimalKernelSelector        #   ├─ Kernel selection algorithms
│   └── KernelCompatibilityChecker   #   └─ Compatibility validation
│
├── 🔧 model_ops_manager.py          # 🛠️ Model Operations Manager
│   ├── ModelOpsManager              #   ├─ FINN ModelOps coordination
│   ├── OperationValidator           #   ├─ Operation validation
│   └── OperationOptimizer           #   └─ Operation optimization
│
├── 🔄 model_transforms_manager.py   # 🔀 Model Transforms Manager
│   ├── ModelTransformsManager       #   ├─ Transform pipeline coordination
│   ├── TransformValidator           #   ├─ Transform validation
│   └── TransformOptimizer           #   └─ Transform optimization
│
├── ⚡ hw_optimization_manager.py    # ⚡ Hardware Optimization Manager
│   ├── HwOptimizationManager        #   ├─ HW optimization coordination
│   ├── OptimizationValidator        #   ├─ Optimization validation
│   └── OptimizationTuner            #   └─ Optimization parameter tuning
│
└── 📊 monitoring.py                 # 👁️ Build Process Monitoring
    ├── BuildProgressMonitor         #   ├─ Real-time build monitoring
    ├── ResourceUsageTracker         #   ├─ Resource utilization tracking
    ├── ErrorDetectionSystem         #   ├─ Build error detection
    └── PerformanceProfiler          #   └─ Performance profiling
```

### **FINN Integration Workflow**

```
🔧 FINN Integration Workflow
┌─────────────────┐
│   Input Model   │
│   (ONNX/PyTorch)│
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FINN Workflow Management                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │Model Ops    │  │Transforms   │  │HW Kernels   │  │HW Opt   ││
│  │Manager      │  │Manager      │  │Manager      │  │Manager  ││
│  │             │  │             │  │             │  │         ││
│  │┌───────────┐│  │┌───────────┐│  │┌───────────┐│  │┌───────┐││
│  ││Operation  ││  ││Transform  ││  ││Kernel     ││  ││Opt    │││
│  ││Validation ││  ││Pipeline   ││  ││Selection  ││  ││Config │││
│  │└───────────┘│  │└───────────┘│  │└───────────┘│  │└───────┘││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘│
└─────────┬───────────────┬───────────────┬───────────────┬─────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Build Orchestration Engine                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │Parallel     │  │Dependency   │  │Artifact     │  │Progress ││
│  │Build        │  │Resolution   │  │Caching      │  │Monitor  ││
│  │Manager      │  │             │  │             │  │         ││
│  │             │  │             │  │             │  │         ││
│  │┌───────────┐│  │┌───────────┐│  │┌───────────┐│  │┌───────┐││
│  ││Build      ││  ││Dep        ││  ││Cache      ││  ││Real   │││
│  ││Queue      ││  ││Graph      ││  ││Manager    ││  ││Time   │││
│  │└───────────┘│  │└───────────┘│  │└───────────┘│  │└───────┘││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘│
└─────────┬───────────────┬───────────────┬───────────────┬─────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FINN Build Process                         │
│                                                                 │
│  Input Model → Transforms → Kernel Selection → HW Synthesis    │
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │ONNX     │───▶│Transform│───▶│Kernel   │───▶│Vivado   │     │
│  │Model    │    │Pipeline │    │Mapping  │    │Synthesis│     │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
└─────────┬───────────────┬───────────────┬───────────────┬─────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Bitstream     │ │   Driver        │ │  Performance    │ │   Analysis      │
│   Generation    │ │   Generation    │ │  Reports        │ │   Reports       │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
          │               │               │               │
          └───────────────┴───────────────┴───────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Build Results  │
                         │   & Artifacts   │
                         └─────────────────┘
```

This document will be continued with additional sections including DSE Engine, Test Suite, and complete system workflows.