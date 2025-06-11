# Brainsmith Repository Restructuring Plan
**Comprehensive Design for Stakeholder-Extensible FPGA Accelerator Platform**

---

## 📋 Executive Summary

This document outlines a comprehensive repository restructuring plan for Brainsmith, transforming the current feature-complete toolchain into a streamlined, extensible foundation optimized for stakeholder development. The goal is to provide a solid base that enables easy addition of kernels, transforms, analysis tools, and optimization strategies while maintaining the robust capabilities already implemented.

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

#### **Extension Infrastructure** ✅
- **Hooks System**: Event-driven architecture for optimization tracking
- **Custom Operations**: Framework for FPGA-specific operators
- **Transform Steps**: Pluggable transformation pipeline
- **Data Export**: Analysis-ready data structures for external tools
- **Modular Architecture**: Clean separation of concerns

#### **Production Features** ✅
- **CLI Interface**: Command-line tools for automation
- **Comprehensive Testing**: 100% validation success rate
- **Documentation**: Complete implementation guides
- **Legacy Compatibility**: Backward-compatible APIs

### Key Insight

**The platform doesn't need more features - it needs organization and clarity for extension.**

---

## 🏗️ Proposed Repository Structure

### **Design Philosophy**

1. **Stable Core**: Essential functionality that rarely changes
2. **Extensible Platform**: Services that enable extensions
3. **Clear Extension Points**: Obvious places for stakeholder additions
4. **Rich Documentation**: Comprehensive guides for all extension types
5. **Backward Compatibility**: Existing code continues to work

### **New Directory Structure**

```
brainsmith/
├── core/                          # Stable core functionality
│   ├── api.py                     # Main forge() function
│   ├── design_space.py            # Design space management
│   ├── metrics.py                 # Performance metrics
│   ├── cli.py                     # Command-line interface
│   └── __init__.py                # Core exports
│
├── platform/                     # Platform services
│   ├── finn/                      # FINN integration
│   │   ├── interface.py           # FINN wrapper
│   │   ├── types.py               # FINN data types
│   │   └── hooks.py               # 4-hooks preparation
│   ├── blueprints/                # Blueprint system
│   │   ├── manager.py             # Blueprint management
│   │   ├── functions.py           # Blueprint utilities
│   │   └── templates/             # Standard templates
│   ├── dse/                       # Design space exploration
│   │   ├── engine.py              # DSE orchestration
│   │   ├── strategies.py          # Optimization strategies
│   │   └── analysis.py            # Result analysis
│   ├── data/                      # Data management
│   │   ├── collection.py          # Metrics collection
│   │   ├── export.py              # Data export
│   │   └── types.py               # Data structures
│   └── hooks/                     # Event system
│       ├── events.py              # Event management
│       ├── handlers.py            # Built-in handlers
│       └── registry.py            # Plugin registry
│
├── extensions/                    # Extension points
│   ├── kernels/                   # Custom kernels
│   │   ├── registry.py            # Kernel discovery
│   │   ├── performance.py         # Performance modeling
│   │   ├── fpga/                  # FPGA-specific kernels
│   │   └── examples/              # Reference implementations
│   ├── transforms/                # Transformation steps
│   │   ├── registry.py            # Transform discovery
│   │   ├── optimization/          # Optimization transforms
│   │   ├── conversion/            # Conversion transforms
│   │   └── examples/              # Reference implementations
│   ├── analysis/                  # Analysis tools
│   │   ├── registry.py            # Analysis discovery
│   │   ├── performance/           # Performance analysis
│   │   ├── visualization/         # Visualization tools
│   │   └── examples/              # Reference implementations
│   ├── hardware/                  # Hardware optimizers
│   │   ├── registry.py            # Optimizer discovery
│   │   ├── genetic/               # Genetic algorithms
│   │   ├── bayesian/              # Bayesian optimization
│   │   └── examples/              # Reference implementations
│   └── community/                 # Community extensions
│       ├── contrib/               # Contributed extensions
│       ├── experimental/          # Experimental features
│       └── templates/             # Extension templates
│
├── examples/                      # Complete examples
│   ├── quickstart/                # Getting started examples
│   │   ├── basic_forge.py         # Simple forge example
│   │   ├── parameter_sweep.py     # Basic DSE example
│   │   └── custom_blueprint.py    # Blueprint creation
│   ├── tutorials/                 # Step-by-step tutorials
│   │   ├── 01_first_accelerator/  # Complete tutorial
│   │   ├── 02_custom_kernels/     # Kernel development
│   │   ├── 03_optimization/       # DSE optimization
│   │   └── 04_analysis/           # Result analysis
│   └── reference/                 # Reference implementations
│       ├── bert_optimization/     # Complete BERT example
│       ├── custom_transforms/     # Transform examples
│       └── analysis_workflows/    # Analysis examples
│
├── docs/                          # Documentation
│   ├── user-guide/                # User documentation
│   │   ├── getting-started.md     # Quick start guide
│   │   ├── api-reference.md       # API documentation
│   │   └── blueprints.md          # Blueprint guide
│   ├── developer-guide/           # Developer documentation
│   │   ├── architecture.md        # Platform architecture
│   │   ├── contributing.md        # Contribution guide
│   │   └── testing.md             # Testing guide
│   └── extension-guide/           # Extension development
│       ├── kernels.md             # Kernel development
│       ├── transforms.md          # Transform development
│       ├── analysis.md            # Analysis tool development
│       └── hardware.md            # Hardware optimizer development
│
└── tests/                         # Test suite
    ├── unit/                      # Unit tests
    ├── integration/               # Integration tests
    ├── extension/                 # Extension tests
    └── examples/                  # Example tests
```

---

## 🔌 Extension Points Architecture

### **Extension Registry System**

```mermaid
graph TB
    subgraph "Core Platform"
        A[forge API] --> B[Extension Registry]
        B --> C[Discovery Engine]
        C --> D[Plugin Loader]
    end
    
    subgraph "Extension Categories"
        E[Custom Kernels]
        F[Transform Steps]
        G[Analysis Tools]
        H[Hardware Optimizers]
        I[Blueprint Templates]
    end
    
    D --> E
    D --> F
    D --> G
    D --> H
    D --> I
    
    subgraph "Stakeholder Development"
        J[Add Custom Kernels]
        K[Implement Transforms]
        L[Create Analysis Tools]
        M[Build Optimizers]
        N[Design Blueprints]
    end
    
    E --> J
    F --> K
    G --> L
    H --> M
    I --> N
```

### **1. Custom Kernels Extension Point**

**Location**: `brainsmith/extensions/kernels/`

**Purpose**: Enable stakeholders to add custom FPGA kernels and operations

**Key Components**:
- **Kernel Registry**: Automatic discovery of custom kernels
- **Performance Models**: Analytical and empirical performance modeling
- **Code Generation**: RTL/HLS code generation templates
- **Integration**: Seamless integration with FINN flow

**Example Extension**:
```python
# brainsmith/extensions/kernels/custom/my_kernel.py
from brainsmith.extensions.kernels import KernelBase

@register_kernel("my_custom_op")
class MyCustomKernel(KernelBase):
    def estimate_performance(self, params, platform):
        # Custom performance model
        pass
    
    def generate_code(self, params):
        # Custom code generation
        pass
```

### **2. Transform Steps Extension Point**

**Location**: `brainsmith/extensions/transforms/`

**Purpose**: Enable custom transformation steps in the compilation pipeline

**Key Components**:
- **Transform Registry**: Discovery and ordering of transforms
- **Pipeline Integration**: Seamless integration with existing flow
- **Dependency Management**: Automatic dependency resolution
- **Validation**: Input/output validation for transforms

**Example Extension**:
```python
# brainsmith/extensions/transforms/optimization/my_transform.py
from brainsmith.extensions.transforms import TransformBase

@register_transform("my_optimization", category="optimization")
class MyOptimizationTransform(TransformBase):
    dependencies = ["streamlining"]
    
    def apply(self, model, config):
        # Custom transformation logic
        return transformed_model
```

### **3. Analysis Tools Extension Point**

**Location**: `brainsmith/extensions/analysis/`

**Purpose**: Enable custom analysis and visualization tools

**Key Components**:
- **Analysis Registry**: Discovery of analysis tools
- **Data Adapters**: Integration with platform data structures
- **Export Formats**: Support for various output formats
- **Visualization**: Plotting and reporting capabilities

**Example Extension**:
```python
# brainsmith/extensions/analysis/custom/my_analyzer.py
from brainsmith.extensions.analysis import AnalyzerBase

@register_analyzer("my_analysis")
class MyCustomAnalyzer(AnalyzerBase):
    def analyze(self, results):
        # Custom analysis logic
        return analysis_results
    
    def visualize(self, analysis):
        # Custom visualization
        return plots
```

### **4. Hardware Optimizers Extension Point**

**Location**: `brainsmith/extensions/hardware/`

**Purpose**: Enable custom hardware optimization strategies

**Key Components**:
- **Optimizer Registry**: Discovery of optimization algorithms
- **Multi-objective Support**: Pareto frontier computation
- **Strategy Selection**: Automatic algorithm recommendation
- **Convergence Tracking**: Progress monitoring and early stopping

**Example Extension**:
```python
# brainsmith/extensions/hardware/custom/my_optimizer.py
from brainsmith.extensions.hardware import OptimizerBase

@register_optimizer("my_algorithm")
class MyCustomOptimizer(OptimizerBase):
    def optimize(self, design_space, objectives):
        # Custom optimization algorithm
        return optimal_configurations
```

---

## 📋 Implementation Plan

### **Phase 1: Core Reorganization (Week 1)**

#### **Objectives**
- Consolidate stable core functionality
- Create clean platform layer
- Establish extension structure

#### **Tasks**
1. **Create new directory structure**
   - Move core components to `brainsmith/core/`
   - Organize platform services in `brainsmith/platform/`
   - Create extension directories in `brainsmith/extensions/`

2. **Update imports and exports**
   - Maintain backward compatibility with import aliases
   - Update `__init__.py` files for new structure
   - Create migration guides for developers

3. **Extension registry foundation**
   - Implement basic plugin discovery system
   - Create registration decorators
   - Add configuration management

#### **Deliverables**
- ✅ New directory structure implemented
- ✅ Backward-compatible imports maintained
- ✅ Basic extension registry functional

### **Phase 2: Extension Point Development (Week 2)**

#### **Objectives**
- Implement comprehensive extension system
- Create developer APIs for extensions
- Build reference implementations

#### **Tasks**
1. **Kernel extension system**
   - Registry for custom kernels
   - Performance modeling interface
   - Code generation templates

2. **Transform extension system**
   - Pipeline integration framework
   - Dependency resolution
   - Validation mechanisms

3. **Analysis extension system**
   - Data adapter interfaces
   - Export format support
   - Visualization framework

4. **Hardware optimizer extensions**
   - Algorithm registry
   - Multi-objective interfaces
   - Strategy selection logic

#### **Deliverables**
- ✅ Complete extension APIs implemented
- ✅ Reference implementations created
- ✅ Validation and testing framework

### **Phase 3: Documentation & Examples (Week 3)**

#### **Objectives**
- Create comprehensive documentation
- Build complete tutorial examples
- Provide extension templates

#### **Tasks**
1. **User documentation**
   - Getting started guide
   - API reference
   - Blueprint documentation

2. **Developer documentation**
   - Architecture overview
   - Contributing guidelines
   - Testing documentation

3. **Extension guides**
   - Step-by-step tutorials for each extension type
   - Best practices and patterns
   - Troubleshooting guides

4. **Example implementations**
   - Complete tutorials with working code
   - Reference implementations
   - Extension templates

#### **Deliverables**
- ✅ Comprehensive documentation suite
- ✅ Complete tutorial examples
- ✅ Extension development templates

### **Phase 4: Stakeholder Enablement (Week 4)**

#### **Objectives**
- Provide tools for stakeholder development
- Create community structure
- Implement validation systems

#### **Tasks**
1. **Development tools**
   - Extension scaffolding tools
   - Validation utilities
   - Testing frameworks

2. **Community structure**
   - Contribution guidelines
   - Extension submission process
   - Quality assurance procedures

3. **Validation and testing**
   - Automated extension testing
   - Integration validation
   - Performance benchmarking

#### **Deliverables**
- ✅ Complete development toolchain
- ✅ Community contribution framework
- ✅ Automated validation system

---

## 🎯 Benefits for Stakeholders

### **1. Clear Extension Points**
- **Obvious Integration**: Well-defined places to add functionality
- **Standardized Interfaces**: Consistent APIs across all extension types
- **Automatic Discovery**: Extensions are automatically found and integrated
- **Rich Examples**: Complete reference implementations for common patterns

### **2. Stable Foundation**
- **Core Stability**: Core platform remains unchanged while extensions evolve
- **Backward Compatibility**: Existing code continues to work
- **Future-Proof**: Architecture designed for long-term evolution
- **Production Ready**: Already validated with 100% test success

### **3. Developer Experience**
- **Comprehensive Documentation**: Complete guides for all extension types
- **Extension Templates**: Scaffolding for quick development
- **Validation Tools**: Automated testing and validation
- **Community Support**: Guidelines for sharing and collaboration

### **4. Technical Excellence**
- **Performance Modeling**: Built-in performance estimation
- **Multi-Objective Optimization**: Advanced DSE capabilities
- **Data Export**: Analysis-ready data structures
- **Event System**: Hooks for monitoring and analysis

---

## 🔄 Backward Compatibility Strategy

### **Import Aliases**
Maintain existing import paths through aliases:

```python
# Old import (continues to work)
from brainsmith import forge
from brainsmith.dse import parameter_sweep

# New import (recommended)
from brainsmith.core import forge
from brainsmith.platform.dse import parameter_sweep
```

### **API Wrappers**
Preserve current function signatures:

```python
# Legacy API wrapper
def legacy_optimize_model(*args, **kwargs):
    return forge(*args, **kwargs)
```

### **Gradual Migration**
- **Optional Adoption**: New structure is recommended but not required
- **Migration Tools**: Automated tools to help transition code
- **Deprecation Warnings**: Gentle guidance toward new APIs
- **Long-term Support**: Legacy APIs supported for multiple versions

---

## 📊 Success Metrics

### **Technical Metrics**
- **Extension Development Time**: < 1 day for simple extensions
- **Documentation Coverage**: 100% of extension APIs documented
- **Example Completeness**: Working examples for all extension types
- **Test Coverage**: > 90% coverage for extension framework

### **Stakeholder Metrics**
- **Time to First Extension**: < 1 week for new stakeholders
- **Extension Quality**: Automated validation ensures quality
- **Community Growth**: Number of contributed extensions
- **Developer Satisfaction**: Feedback from extension developers

### **Platform Metrics**
- **Backward Compatibility**: 100% of existing code continues to work
- **Performance**: No degradation in core platform performance
- **Stability**: Extension failures don't impact core platform
- **Scalability**: Support for large numbers of extensions

---

## 🚀 Next Steps

### **Immediate Actions**
1. **Review and Approve Plan**: Stakeholder review of this restructuring plan
2. **Create Implementation Timeline**: Detailed schedule for the 4-week implementation
3. **Assign Implementation Teams**: Technical teams for each phase
4. **Prepare Development Environment**: Tools and infrastructure for restructuring

### **Implementation Readiness**
- **Codebase Analysis**: ✅ Complete understanding of current implementation
- **Architecture Design**: ✅ Comprehensive restructuring plan
- **Extension Framework**: ✅ Detailed design for all extension points
- **Documentation Plan**: ✅ Complete documentation strategy

### **Risk Mitigation**
- **Backward Compatibility**: Comprehensive testing ensures no breaking changes
- **Incremental Implementation**: Phase-by-phase approach reduces risk
- **Validation Framework**: Automated testing catches issues early
- **Rollback Plan**: Ability to revert changes if needed

---

## 📋 Conclusion

This repository restructuring plan transforms Brainsmith from a feature-complete toolchain into a **stakeholder-extensible platform**. The new structure provides:

✅ **Clear extension points** for all types of functionality  
✅ **Stable core foundation** that doesn't change  
✅ **Comprehensive documentation** for all extension types  
✅ **Rich examples** and reference implementations  
✅ **Backward compatibility** with existing code  
✅ **Community framework** for sharing extensions  

The platform is **ready for implementation** with a clear 4-week plan that will deliver a production-ready, stakeholder-extensible foundation for FPGA accelerator design and optimization.

---

*Document prepared: June 11, 2025*  
*Status: Ready for stakeholder review and implementation*  
*Implementation timeline: 4 weeks*