# Brainsmith Repository Restructuring Plan (Revised)
**Comprehensive Design for Stakeholder-Extensible FPGA Accelerator Platform**

---

## 📋 Executive Summary

This document outlines a revised repository restructuring plan for Brainsmith, transforming the current feature-complete toolchain into a streamlined, extensible foundation optimized for stakeholder development. The goal is to provide a solid base with rich core libraries that enables easy addition of kernels, transforms, analysis tools, and optimization strategies while maintaining the robust capabilities already implemented.

**Key Revision**: Components like kernels, transforms, and analysis tools are **core libraries** essential to the compiler, not optional extensions. The structure reflects their fundamental importance.

---

## 🔍 Current State Analysis

### Platform Maturity Assessment

**Brainsmith is a mature, feature-complete platform** with the following strengths:

#### **Core Capabilities** ✅
- **Unified API**: [`forge(model, blueprint)`](brainsmith/core/api.py:27) function provides single entry point
- **Design Space Exploration**: Comprehensive DSE with multi-objective optimization
- **FINN Integration**: Clean interface with preparation for 4-hooks architecture
- **Blueprint System**: YAML-based design templates with validation
- **Performance Modeling**: Analytical and empirical models for kernel performance
- **Automation Helpers**: Parameter sweeps, batch processing, result analysis

#### **Rich Library Ecosystem** ✅
- **Kernel Library**: FPGA-specific operations and performance models
- **Transform Library**: Comprehensive transformation pipeline
- **Analysis Library**: Performance analysis and visualization tools
- **Hardware Optimization Library**: Multi-objective optimization strategies
- **Custom Operations**: Framework for domain-specific operators

#### **Production Features** ✅
- **CLI Interface**: Command-line tools for automation
- **Comprehensive Testing**: 100% validation success rate
- **Documentation**: Complete implementation guides
- **Legacy Compatibility**: Backward-compatible APIs

### Key Insight

**The platform has rich core libraries that need better organization and clearer extension points for stakeholder additions.**

---

## 🏗️ Proposed Repository Structure

### **Design Philosophy**

1. **Essential Core**: Minimal API and orchestration layer
2. **Rich Libraries**: Comprehensive component libraries that are core to the compiler
3. **Extensible Infrastructure**: Services that enable and support the libraries
4. **Clear Addition Points**: Obvious places for stakeholder contributions
5. **Backward Compatibility**: Existing code continues to work

### **New Directory Structure**

```
brainsmith/
├── core/                          # Essential APIs and orchestration
│   ├── api.py                     # Main forge() function
│   ├── cli.py                     # Command-line interface
│   ├── types.py                   # Core data types
│   └── __init__.py                # Core exports
│
├── infrastructure/                # Compilation pipeline infrastructure
│   ├── finn/                      # FINN integration layer
│   │   ├── interface.py           # FINN wrapper and integration
│   │   ├── types.py               # FINN-specific data types
│   │   └── hooks.py               # 4-hooks preparation
│   ├── dse/                       # Design space exploration engine
│   │   ├── engine.py              # DSE orchestration
│   │   ├── strategies.py          # Built-in optimization strategies
│   │   ├── types.py               # DSE data types
│   │   └── analysis.py            # Result analysis
│   ├── blueprints/                # Blueprint management system
│   │   ├── manager.py             # Blueprint lifecycle management
│   │   ├── functions.py           # Blueprint utilities
│   │   ├── validation.py          # Blueprint validation
│   │   └── templates/             # Standard blueprint templates
│   ├── data/                      # Data management and export
│   │   ├── collection.py          # Metrics collection
│   │   ├── export.py              # Data export capabilities
│   │   ├── types.py               # Data structures
│   │   └── adapters/              # External tool adapters
│   └── hooks/                     # Event and monitoring system
│       ├── events.py              # Event management
│       ├── handlers.py            # Built-in event handlers
│       ├── registry.py            # Plugin registry
│       └── types.py               # Hook system types
│
├── libraries/                     # Core component libraries
│   ├── kernels/                   # Hardware kernel library
│   │   ├── registry.py            # Kernel discovery and management
│   │   ├── performance.py         # Performance modeling framework
│   │   ├── base.py                # Base kernel interfaces
│   │   ├── fpga/                  # FPGA-specific kernel implementations
│   │   │   ├── matmul.py          # Matrix multiplication kernels
│   │   │   ├── layernorm.py       # Layer normalization kernels
│   │   │   ├── softmax.py         # Softmax kernels
│   │   │   └── thresholding.py    # Thresholding kernels
│   │   ├── rtl/                   # RTL implementations
│   │   ├── hls/                   # HLS implementations
│   │   └── contrib/               # Community-contributed kernels
│   ├── transforms/                # Transformation library
│   │   ├── registry.py            # Transform discovery and ordering
│   │   ├── base.py                # Base transformation interfaces
│   │   ├── optimization/          # Optimization transformations
│   │   │   ├── streamlining.py    # Model streamlining
│   │   │   ├── folding.py         # Folding optimizations
│   │   │   └── fusion.py          # Operation fusion
│   │   ├── conversion/            # Format conversion transforms
│   │   │   ├── qonnx_to_finn.py   # QONNX to FINN conversion
│   │   │   └── finn_to_hw.py      # FINN to hardware conversion
│   │   ├── hardware/              # Hardware-specific transforms
│   │   │   ├── inference.py       # Hardware inference
│   │   │   └── validation.py      # Hardware validation
│   │   └── contrib/               # Community-contributed transforms
│   ├── analysis/                  # Analysis and visualization library
│   │   ├── registry.py            # Analysis tool discovery
│   │   ├── base.py                # Base analysis interfaces
│   │   ├── performance/           # Performance analysis tools
│   │   │   ├── roofline.py        # Roofline analysis
│   │   │   ├── profiling.py       # Performance profiling
│   │   │   └── estimation.py      # Performance estimation
│   │   ├── resource/              # Resource analysis tools
│   │   │   ├── utilization.py     # Resource utilization analysis
│   │   │   └── optimization.py    # Resource optimization analysis
│   │   ├── visualization/         # Visualization tools
│   │   │   ├── plotting.py        # Standard plots
│   │   │   ├── reports.py         # Report generation
│   │   │   └── dashboards.py      # Interactive dashboards
│   │   └── contrib/               # Community-contributed analysis
│   └── hardware/                  # Hardware optimization library
│       ├── registry.py            # Optimizer discovery
│       ├── base.py                # Base optimizer interfaces
│       ├── genetic/               # Genetic algorithm optimizers
│       │   ├── nsga2.py           # NSGA-II implementation
│       │   └── moea.py            # Multi-objective evolutionary algorithms
│       ├── bayesian/              # Bayesian optimization
│       │   ├── gaussian_process.py # Gaussian process optimization
│       │   └── acquisition.py     # Acquisition functions
│       ├── gradient/              # Gradient-based optimization
│       ├── heuristic/             # Heuristic optimization methods
│       └── contrib/               # Community-contributed optimizers
│
├── examples/                      # Complete examples and tutorials
│   ├── quickstart/                # Getting started examples
│   │   ├── basic_forge.py         # Simple forge example
│   │   ├── parameter_sweep.py     # Basic DSE example
│   │   └── custom_blueprint.py    # Blueprint creation
│   ├── tutorials/                 # Step-by-step tutorials
│   │   ├── 01_first_accelerator/  # Complete tutorial series
│   │   ├── 02_custom_kernels/     # Kernel development tutorial
│   │   ├── 03_optimization/       # DSE optimization tutorial
│   │   └── 04_analysis/           # Result analysis tutorial
│   ├── reference/                 # Reference implementations
│   │   ├── bert_optimization/     # Complete BERT example
│   │   ├── cnn_acceleration/      # CNN acceleration example
│   │   └── custom_workflows/      # Custom workflow examples
│   └── contrib/                   # Community-contributed examples
│
├── docs/                          # Comprehensive documentation
│   ├── user-guide/                # User documentation
│   │   ├── getting-started.md     # Quick start guide
│   │   ├── api-reference.md       # Complete API documentation
│   │   ├── blueprints.md          # Blueprint system guide
│   │   └── workflows.md           # Common workflow patterns
│   ├── developer-guide/           # Developer documentation
│   │   ├── architecture.md        # Platform architecture
│   │   ├── contributing.md        # Contribution guidelines
│   │   ├── testing.md             # Testing framework
│   │   └── debugging.md           # Debugging and profiling
│   ├── library-guide/             # Library development documentation
│   │   ├── kernels.md             # Kernel development guide
│   │   ├── transforms.md          # Transform development guide
│   │   ├── analysis.md            # Analysis tool development
│   │   └── hardware.md            # Hardware optimizer development
│   └── reference/                 # Reference documentation
│       ├── finn-integration.md    # FINN integration details
│       ├── performance-models.md  # Performance modeling
│       └── data-formats.md        # Data format specifications
│
└── tests/                         # Comprehensive test suite
    ├── unit/                      # Unit tests for all components
    ├── integration/               # Integration tests
    ├── libraries/                 # Library-specific tests
    ├── examples/                  # Example validation tests
    └── benchmarks/                # Performance benchmarks
```

---

## 🏛️ Architecture Layer Explanation

### **Core Layer**
**Purpose**: Minimal, stable API surface that orchestrates the entire toolchain

**Components**:
- **`api.py`**: The main [`forge()`](brainsmith/core/api.py:27) function that serves as the unified entry point
- **`cli.py`**: Command-line interface for automation and scripting
- **`types.py`**: Essential data types used throughout the platform

**Characteristics**:
- **Minimal**: Only the essential orchestration logic
- **Stable**: Rarely changes, providing stability for stakeholders
- **High-level**: Abstracts away complexity of underlying systems

### **Infrastructure Layer**
**Purpose**: The compilation pipeline infrastructure that enables and supports the libraries

**Components**:
- **FINN Integration**: Manages interaction with the FINN framework, handles 4-hooks preparation
- **DSE Engine**: Orchestrates design space exploration, manages optimization strategies
- **Blueprint System**: Handles blueprint lifecycle, validation, and template management
- **Data Management**: Collects metrics, exports data, provides adapters for external tools
- **Hooks System**: Provides event monitoring, plugin registration, and extensibility

**Characteristics**:
- **Service-oriented**: Provides services that libraries depend on
- **Configurable**: Supports different modes and configurations
- **Extensible**: Plugin architecture allows adding new capabilities

### **Libraries Layer**
**Purpose**: Rich, comprehensive libraries containing the core components that do the actual compilation work

**This is NOT "extensions" - these are essential compiler components**:

- **Kernel Library**: Hardware-specific operations, performance models, code generation
- **Transform Library**: Model transformations, optimizations, format conversions
- **Analysis Library**: Performance analysis, resource analysis, visualization
- **Hardware Library**: Optimization algorithms, multi-objective optimization, strategy selection

**Characteristics**:
- **Core functionality**: Essential to the compiler's operation
- **Rich and comprehensive**: Extensive libraries with many implementations
- **Stakeholder-extensible**: Clear points for adding new components
- **Well-organized**: Categorized by function with discovery mechanisms

---

## 🔌 Library Extension Points

### **1. Kernel Library Extension**

**Location**: `brainsmith/libraries/kernels/contrib/`

**Purpose**: Enable stakeholders to add custom FPGA kernels and operations

**Integration**: New kernels are automatically discovered by the registry and integrated into the compilation flow

**Example**:
```python
# brainsmith/libraries/kernels/contrib/my_custom_kernel.py
from brainsmith.libraries.kernels.base import KernelBase
from brainsmith.libraries.kernels.registry import register_kernel

@register_kernel("my_custom_op")
class MyCustomKernel(KernelBase):
    def estimate_performance(self, params, platform):
        # Custom performance model
        pass
    
    def generate_code(self, params):
        # Custom code generation
        pass
```

### **2. Transform Library Extension**

**Location**: `brainsmith/libraries/transforms/contrib/`

**Purpose**: Enable custom transformation steps in the compilation pipeline

**Integration**: Transforms are discovered by the registry and automatically integrated into the pipeline with proper dependency resolution

**Example**:
```python
# brainsmith/libraries/transforms/contrib/my_optimization.py
from brainsmith.libraries.transforms.base import TransformBase
from brainsmith.libraries.transforms.registry import register_transform

@register_transform("my_optimization", category="optimization")
class MyOptimizationTransform(TransformBase):
    dependencies = ["streamlining"]
    
    def apply(self, model, config):
        # Custom transformation logic
        return transformed_model
```

### **3. Analysis Library Extension**

**Location**: `brainsmith/libraries/analysis/contrib/`

**Purpose**: Enable custom analysis and visualization tools

**Integration**: Analysis tools are discovered and can be invoked through the standard analysis interface

**Example**:
```python
# brainsmith/libraries/analysis/contrib/my_analyzer.py
from brainsmith.libraries.analysis.base import AnalyzerBase
from brainsmith.libraries.analysis.registry import register_analyzer

@register_analyzer("my_analysis")
class MyCustomAnalyzer(AnalyzerBase):
    def analyze(self, results):
        # Custom analysis logic
        return analysis_results
```

### **4. Hardware Library Extension**

**Location**: `brainsmith/libraries/hardware/contrib/`

**Purpose**: Enable custom hardware optimization strategies

**Integration**: Optimizers are discovered and can be selected through the strategy selection system

**Example**:
```python
# brainsmith/libraries/hardware/contrib/my_optimizer.py
from brainsmith.libraries.hardware.base import OptimizerBase
from brainsmith.libraries.hardware.registry import register_optimizer

@register_optimizer("my_algorithm")
class MyCustomOptimizer(OptimizerBase):
    def optimize(self, design_space, objectives):
        # Custom optimization algorithm
        return optimal_configurations
```

---

## 📋 Implementation Plan

### **Phase 1: Core and Infrastructure Reorganization (Week 1)**

#### **Objectives**
- Create minimal, stable core layer
- Organize infrastructure services clearly
- Establish library structure foundation

#### **Tasks**
1. **Create new directory structure**
   - Move essential APIs to `brainsmith/core/`
   - Organize services in `brainsmith/infrastructure/`
   - Create library directories in `brainsmith/libraries/`

2. **Infrastructure service organization**
   - FINN integration as infrastructure service
   - DSE engine as infrastructure service  
   - Blueprint system as infrastructure service
   - Data management as infrastructure service
   - Hooks system as infrastructure service

3. **Update imports and maintain compatibility**
   - Create import aliases for backward compatibility
   - Update `__init__.py` files for new structure
   - Ensure all existing code continues to work

#### **Deliverables**
- ✅ New three-layer structure implemented
- ✅ Infrastructure services clearly separated
- ✅ Backward-compatible imports maintained

### **Phase 2: Library Organization and Registry Development (Week 2)**

#### **Objectives**
- Organize existing components into rich libraries
- Implement discovery and registry systems
- Create clear addition points for stakeholders

#### **Tasks**
1. **Kernel Library organization**
   - Move existing kernels to organized structure
   - Implement kernel registry and discovery
   - Create performance modeling framework

2. **Transform Library organization**
   - Organize existing transforms by category
   - Implement transform registry with dependency resolution
   - Create pipeline integration framework

3. **Analysis Library organization**
   - Organize existing analysis tools
   - Implement analysis registry
   - Create visualization framework

4. **Hardware Library organization**
   - Organize existing optimization strategies
   - Implement optimizer registry
   - Create strategy selection framework

#### **Deliverables**
- ✅ All libraries organized and categorized
- ✅ Registry systems implemented for discovery
- ✅ Clear addition points established

### **Phase 3: Documentation and Examples (Week 3)**

#### **Objectives**
- Create comprehensive documentation for new structure
- Build complete tutorial examples
- Provide development guides for each library

#### **Tasks**
1. **Architecture documentation**
   - Explain the three-layer architecture
   - Document infrastructure services
   - Describe library organization

2. **Library development guides**
   - Kernel development documentation
   - Transform development guide
   - Analysis tool development
   - Hardware optimizer development

3. **Complete examples and tutorials**
   - Getting started examples
   - Step-by-step tutorials for each library
   - Reference implementations

#### **Deliverables**
- ✅ Complete architecture documentation
- ✅ Library development guides
- ✅ Comprehensive examples and tutorials

### **Phase 4: Stakeholder Enablement (Week 4)**

#### **Objectives**
- Provide tools for library development
- Create contribution frameworks
- Implement validation systems

#### **Tasks**
1. **Development tools**
   - Library component scaffolding tools
   - Validation utilities for new components
   - Testing frameworks for libraries

2. **Community framework**
   - Contribution guidelines for each library
   - Quality assurance procedures
   - Integration validation

3. **Advanced features**
   - Performance benchmarking for libraries
   - Automated testing for contributed components
   - Documentation generation tools

#### **Deliverables**
- ✅ Complete development toolchain
- ✅ Community contribution framework
- ✅ Automated validation and testing

---

## 🎯 Benefits for Stakeholders

### **1. Rich Core Libraries**
- **Comprehensive**: Extensive libraries covering all aspects of FPGA compilation
- **Well-organized**: Clear categorization and discovery mechanisms
- **Production-ready**: Already validated with 100% test success
- **Extensible**: Clear points for adding new components

### **2. Clear Architecture**
- **Three-layer design**: Core, Infrastructure, Libraries with clear responsibilities
- **Service-oriented**: Infrastructure provides services that libraries depend on
- **Stable foundation**: Core layer provides stability while libraries evolve
- **Organized complexity**: Complex functionality is well-organized and discoverable

### **3. Stakeholder Addition Points**
- **Library-specific contrib directories**: Clear places to add new components
- **Registry systems**: Automatic discovery of new components
- **Standardized interfaces**: Consistent APIs across all libraries
- **Comprehensive guides**: Complete documentation for adding to each library

### **4. Production Excellence**
- **Battle-tested**: Current implementation has 100% validation success
- **Performance-focused**: Built-in performance modeling and optimization
- **Data-driven**: Rich data export and analysis capabilities
- **Community-ready**: Framework for sharing and collaboration

---

## 🔄 Backward Compatibility Strategy

### **Import Aliases**
All existing imports continue to work:

```python
# All existing imports work unchanged
from brainsmith import forge
from brainsmith.dse import parameter_sweep
from brainsmith.automation import batch_process

# New recommended imports (optional migration)
from brainsmith.core import forge
from brainsmith.infrastructure.dse import parameter_sweep
from brainsmith.examples.automation import batch_process
```

### **Gradual Migration**
- **Optional adoption**: New structure is recommended but not required
- **Migration tools**: Automated tools to help transition code
- **Clear benefits**: New structure provides obvious advantages
- **Long-term support**: Legacy APIs supported for multiple versions

---

## 📊 Success Metrics

### **Technical Metrics**
- **Library Development Time**: < 2 days for new library components
- **Discovery Effectiveness**: 100% of library components automatically discoverable
- **Integration Success**: New components integrate seamlessly
- **Performance**: No degradation in compilation performance

### **Stakeholder Metrics**
- **Time to Contribution**: < 1 week for stakeholders to add new components
- **Component Quality**: Automated validation ensures quality
- **Library Growth**: Number of components in each library
- **Developer Satisfaction**: Feedback from library developers

### **Platform Metrics**
- **Backward Compatibility**: 100% of existing code continues to work
- **Library Richness**: Comprehensive components in all categories
- **Documentation Coverage**: Complete guides for all libraries
- **Community Adoption**: Usage of contribution frameworks

---

## 🚀 Next Steps

### **Immediate Actions**
1. **Review and Approve Plan**: Stakeholder review of this revised restructuring plan
2. **Create Implementation Timeline**: Detailed schedule for the 4-week implementation
3. **Assign Implementation Teams**: Technical teams for each phase
4. **Prepare Development Environment**: Tools and infrastructure for restructuring

### **Implementation Readiness**
- **Architecture Design**: ✅ Three-layer architecture with clear responsibilities
- **Library Organization**: ✅ Rich libraries with clear extension points
- **Infrastructure Services**: ✅ Well-defined service layer
- **Documentation Plan**: ✅ Comprehensive documentation strategy

---

## 📋 Conclusion

This revised repository restructuring plan transforms Brainsmith into a **platform with rich core libraries** rather than treating essential components as optional extensions. The new structure provides:

✅ **Rich core libraries** that are essential to the compiler  
✅ **Clear infrastructure services** that support the libraries  
✅ **Minimal stable core** that provides orchestration  
✅ **Organized complexity** with clear categorization  
✅ **Stakeholder addition points** in each library  
✅ **Comprehensive documentation** for all layers  
✅ **Backward compatibility** with existing code  

The platform recognizes that kernels, transforms, analysis tools, and hardware optimizers are **core to the compiler's functionality**, not optional add-ons, while still providing clear ways for stakeholders to contribute to these rich libraries.

---

*Document prepared: June 11, 2025*  
*Status: Ready for stakeholder review and implementation*  
*Implementation timeline: 4 weeks*