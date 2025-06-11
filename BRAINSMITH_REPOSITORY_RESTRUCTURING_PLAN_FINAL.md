# Brainsmith Repository Restructuring Plan (Final)
**Extensible Foundation for FPGA Accelerator Platform**

---

## 📋 Executive Summary

This document outlines the final repository restructuring plan for Brainsmith, organizing the current feature-complete toolchain into a clear, extensible foundation optimized for stakeholder development. The restructuring focuses on organizing existing components into logical layers while providing clear extension points for future development.

**Key Principle**: Organize existing components into core libraries rather than treating them as optional extensions, while maintaining 100% backward compatibility.

---

## 🔍 Current State Analysis

### **Existing Component Inventory**

**Core Infrastructure** (Already Implemented):
- [`brainsmith/core/api.py`](brainsmith/core/api.py) - Main [`forge()`](brainsmith/core/api.py:27) function
- [`brainsmith/core/design_space.py`](brainsmith/core/design_space.py) - Design space management
- [`brainsmith/core/metrics.py`](brainsmith/core/metrics.py) - Performance metrics
- [`brainsmith/core/finn_interface.py`](brainsmith/core/finn_interface.py) - FINN integration

**DSE System** (Already Implemented):
- [`brainsmith/dse/functions.py`](brainsmith/dse/functions.py) - DSE functionality
- [`brainsmith/dse/helpers.py`](brainsmith/dse/helpers.py) - DSE utilities
- [`brainsmith/dse/types.py`](brainsmith/dse/types.py) - DSE data types

**FINN Integration** (Already Implemented):
- [`brainsmith/finn/interface.py`](brainsmith/finn/interface.py) - FINN wrapper
- [`brainsmith/finn/types.py`](brainsmith/finn/types.py) - FINN data types

**Transform Library** (Already Implemented):
- [`brainsmith/steps/conversion.py`](brainsmith/steps/conversion.py) - QONNX to FINN conversion
- [`brainsmith/steps/streamlining.py`](brainsmith/steps/streamlining.py) - Model streamlining
- [`brainsmith/steps/hardware.py`](brainsmith/steps/hardware.py) - Hardware inference
- [`brainsmith/steps/optimizations.py`](brainsmith/steps/optimizations.py) - Optimization steps
- [`brainsmith/steps/validation.py`](brainsmith/steps/validation.py) - Validation steps
- [`brainsmith/steps/cleanup.py`](brainsmith/steps/cleanup.py) - Cleanup operations
- [`brainsmith/steps/bert.py`](brainsmith/steps/bert.py) - BERT-specific steps
- [`brainsmith/transformation/expand_norms.py`](brainsmith/transformation/expand_norms.py) - LayerNorm expansion
- [`brainsmith/transformation/convert_to_hw_layers.py`](brainsmith/transformation/convert_to_hw_layers.py) - Hardware layer conversion
- [`brainsmith/transformation/shuffle_helpers.py`](brainsmith/transformation/shuffle_helpers.py) - Shuffle operations

**Kernel Library** (Already Implemented):
- [`brainsmith/kernels/conv2d_hls/`](brainsmith/kernels/conv2d_hls/) - Conv2D HLS implementation
- [`brainsmith/hw_kernels/hls/`](brainsmith/hw_kernels/hls/) - HLS kernel utilities
- [`brainsmith/hw_kernels/rtl/`](brainsmith/hw_kernels/rtl/) - RTL implementations

**Analysis Tools** (Already Implemented):
- [`brainsmith/tools/profiling/roofline.py`](brainsmith/tools/profiling/roofline.py) - Roofline analysis
- [`brainsmith/tools/profiling/model_profiling.py`](brainsmith/tools/profiling/model_profiling.py) - Model profiling
- [`brainsmith/tools/profiling/roofline_runner.py`](brainsmith/tools/profiling/roofline_runner.py) - Roofline execution
- [`brainsmith/tools/hw_kernel_gen/`](brainsmith/tools/hw_kernel_gen/) - Hardware kernel generation tools

**Event System** (Already Implemented):
- [`brainsmith/hooks/events.py`](brainsmith/hooks/events.py) - Event management
- [`brainsmith/hooks/types.py`](brainsmith/hooks/types.py) - Hook system types

---

## 🏗️ Proposed Repository Structure

### **Three-Layer Architecture**

```
brainsmith/
├── core/                          # Essential APIs and orchestration
│   ├── api.py                     # Main forge() function [EXISTING]
│   ├── cli.py                     # Command-line interface [NEW]
│   ├── types.py                   # Core data types [NEW]
│   └── __init__.py                # Core exports [UPDATED]
│
├── infrastructure/                # Platform services and frameworks
│   ├── finn/                      # FINN integration layer
│   │   ├── interface.py           # [MOVED FROM brainsmith/finn/]
│   │   ├── types.py               # [MOVED FROM brainsmith/finn/]
│   │   └── __init__.py            # [NEW]
│   ├── dse/                       # Design space exploration engine
│   │   ├── engine.py              # [MOVED FROM brainsmith/dse/functions.py]
│   │   ├── helpers.py             # [MOVED FROM brainsmith/dse/helpers.py]
│   │   ├── types.py               # [MOVED FROM brainsmith/dse/types.py]
│   │   └── __init__.py            # [UPDATED]
│   ├── blueprints/                # Blueprint management system
│   │   ├── manager.py             # [MOVED FROM brainsmith/blueprints/]
│   │   ├── functions.py           # [MOVED FROM brainsmith/blueprints/]
│   │   ├── templates/             # [MOVED FROM brainsmith/blueprints/yaml/]
│   │   └── __init__.py            # [UPDATED]
│   ├── data/                      # Data management and export
│   │   ├── collection.py          # [MOVED FROM brainsmith/data/functions.py]
│   │   ├── export.py              # [MOVED FROM brainsmith/data/export.py]
│   │   ├── types.py               # [MOVED FROM brainsmith/data/types.py]
│   │   └── __init__.py            # [UPDATED]
│   ├── hooks/                     # Event and monitoring system
│   │   ├── events.py              # [MOVED FROM brainsmith/hooks/events.py]
│   │   ├── types.py               # [MOVED FROM brainsmith/hooks/types.py]
│   │   ├── registry.py            # [NEW - plugin registry]
│   │   └── __init__.py            # [UPDATED]
│   └── metrics/                   # Metrics infrastructure
│       ├── collection.py          # [MOVED FROM brainsmith/core/metrics.py]
│       ├── design_space.py        # [MOVED FROM brainsmith/core/design_space.py]
│       ├── finn_interface.py      # [MOVED FROM brainsmith/core/finn_interface.py]
│       └── __init__.py            # [NEW]
│
├── libraries/                     # Core component libraries
│   ├── kernels/                   # Hardware kernel library
│   │   ├── registry.py            # [NEW - kernel discovery]
│   │   ├── base.py                # [NEW - base kernel interfaces]
│   │   ├── conv2d_hls/            # [MOVED FROM brainsmith/kernels/conv2d_hls/]
│   │   ├── hls/                   # [MOVED FROM brainsmith/hw_kernels/hls/]
│   │   ├── rtl/                   # [MOVED FROM brainsmith/hw_kernels/rtl/]
│   │   ├── contrib/               # [NEW - community contributions]
│   │   └── __init__.py            # [NEW]
│   ├── transforms/                # Transformation library
│   │   ├── registry.py            # [NEW - transform discovery]
│   │   ├── base.py                # [NEW - base transform interfaces]
│   │   ├── steps/                 # [MOVED FROM brainsmith/steps/]
│   │   │   ├── conversion.py      # [EXISTING]
│   │   │   ├── streamlining.py    # [EXISTING]
│   │   │   ├── hardware.py        # [EXISTING]
│   │   │   ├── optimizations.py   # [EXISTING]
│   │   │   ├── validation.py      # [EXISTING]
│   │   │   ├── cleanup.py         # [EXISTING]
│   │   │   └── bert.py            # [EXISTING]
│   │   ├── operations/            # [MOVED FROM brainsmith/transformation/]
│   │   │   ├── expand_norms.py    # [EXISTING]
│   │   │   ├── convert_to_hw_layers.py # [EXISTING]
│   │   │   └── shuffle_helpers.py # [EXISTING]
│   │   ├── contrib/               # [NEW - community contributions]
│   │   └── __init__.py            # [NEW]
│   ├── analysis/                  # Analysis and profiling library
│   │   ├── registry.py            # [NEW - analysis tool discovery]
│   │   ├── base.py                # [NEW - base analysis interfaces]
│   │   ├── profiling/             # [MOVED FROM brainsmith/tools/profiling/]
│   │   │   ├── roofline.py        # [EXISTING]
│   │   │   ├── model_profiling.py # [EXISTING]
│   │   │   └── roofline_runner.py # [EXISTING]
│   │   ├── generation/            # [MOVED FROM brainsmith/tools/hw_kernel_gen/]
│   │   ├── contrib/               # [NEW - community contributions]
│   │   └── __init__.py            # [NEW]
│   └── automation/                # Automation and utilities library
│       ├── batch.py               # [MOVED FROM brainsmith/automation/batch.py]
│       ├── sweep.py               # [MOVED FROM brainsmith/automation/sweep.py]
│       ├── contrib/               # [NEW - community contributions]
│       └── __init__.py            # [UPDATED]
│
├── examples/                      # Examples and tutorials [EXISTING]
├── docs/                          # Documentation [EXISTING]
└── tests/                         # Test suite [EXISTING]
```

---

## 🏛️ Architecture Layer Details

### **Core Layer**
**Purpose**: Minimal, stable API that orchestrates the toolchain

**Components**:
- **`api.py`**: The main [`forge()`](brainsmith/core/api.py:27) function [EXISTING]
- **`cli.py`**: Command-line interface [NEW - consolidated from existing CLI components]
- **`types.py`**: Essential data types [NEW - consolidated from various type definitions]

**Characteristics**:
- **Minimal**: Only essential orchestration logic
- **Stable**: Changes infrequently, providing stability
- **High-level**: Abstracts underlying complexity

### **Infrastructure Layer**
**Purpose**: Platform services that enable and support the libraries

**Services**:
- **FINN Integration**: Manages FINN framework interaction
- **DSE Engine**: Orchestrates design space exploration
- **Blueprint System**: Handles blueprint management and validation
- **Data Management**: Collects metrics and manages data export
- **Hooks System**: Provides event monitoring and extensibility
- **Metrics Infrastructure**: Core metrics collection and management

**Characteristics**:
- **Service-oriented**: Provides services that libraries depend on
- **Configurable**: Supports different operational modes
- **Extensible**: Plugin architecture for new capabilities

### **Libraries Layer**
**Purpose**: Rich, comprehensive libraries containing core compiler components

**Libraries**:
- **Kernel Library**: Hardware-specific operations and implementations
- **Transform Library**: Model transformations and compilation steps
- **Analysis Library**: Profiling, analysis, and generation tools
- **Automation Library**: Batch processing and sweep utilities

**Characteristics**:
- **Core functionality**: Essential to compiler operation
- **Well-organized**: Clear categorization and discovery
- **Extensible**: `contrib/` directories for stakeholder additions
- **Rich**: Comprehensive implementations in each category

---

## 📋 Implementation Plan

### **Phase 1: Infrastructure Reorganization (Week 1)**

#### **Objectives**
- Create three-layer directory structure
- Move components to appropriate layers
- Establish registry systems

#### **Tasks**
1. **Create new directory structure**
   ```bash
   mkdir -p brainsmith/{infrastructure,libraries}
   mkdir -p brainsmith/infrastructure/{finn,dse,blueprints,data,hooks,metrics}
   mkdir -p brainsmith/libraries/{kernels,transforms,analysis,automation}
   ```

2. **Move existing components**
   - `brainsmith/finn/` → `brainsmith/infrastructure/finn/`
   - `brainsmith/dse/` → `brainsmith/infrastructure/dse/`
   - `brainsmith/steps/` → `brainsmith/libraries/transforms/steps/`
   - `brainsmith/transformation/` → `brainsmith/libraries/transforms/operations/`
   - `brainsmith/kernels/` → `brainsmith/libraries/kernels/`
   - `brainsmith/tools/profiling/` → `brainsmith/libraries/analysis/profiling/`
   - `brainsmith/tools/hw_kernel_gen/` → `brainsmith/libraries/analysis/generation/`

3. **Update import statements**
   - Create compatibility aliases in old locations
   - Update `__init__.py` files for new structure
   - Ensure backward compatibility

#### **Deliverables**
- ✅ New directory structure implemented
- ✅ All components moved to appropriate layers
- ✅ Import compatibility maintained

### **Phase 2: Registry and Discovery Systems (Week 2)**

#### **Objectives**
- Implement discovery systems for each library
- Create base interfaces for extensibility
- Add contribution frameworks

#### **Tasks**
1. **Kernel Library registry**
   - `brainsmith/libraries/kernels/registry.py` - automatic kernel discovery
   - `brainsmith/libraries/kernels/base.py` - base kernel interfaces
   - `brainsmith/libraries/kernels/contrib/` - contribution directory

2. **Transform Library registry**
   - `brainsmith/libraries/transforms/registry.py` - transform discovery with dependencies
   - `brainsmith/libraries/transforms/base.py` - base transform interfaces
   - `brainsmith/libraries/transforms/contrib/` - contribution directory

3. **Analysis Library registry**
   - `brainsmith/libraries/analysis/registry.py` - analysis tool discovery
   - `brainsmith/libraries/analysis/base.py` - base analysis interfaces
   - `brainsmith/libraries/analysis/contrib/` - contribution directory

4. **Infrastructure enhancements**
   - `brainsmith/infrastructure/hooks/registry.py` - plugin registry system
   - Enhanced blueprint system in `brainsmith/infrastructure/blueprints/`
   - Improved data export in `brainsmith/infrastructure/data/`

#### **Deliverables**
- ✅ Registry systems for all libraries
- ✅ Base interfaces for extensibility
- ✅ Contribution directories established

### **Phase 3: Documentation and Integration (Week 3)**

#### **Objectives**
- Document new architecture
- Create integration guides
- Validate restructured system

#### **Tasks**
1. **Architecture documentation**
   - Update existing documentation for new structure
   - Create architecture overview explaining three layers
   - Document infrastructure services

2. **Library documentation**
   - Document existing components in new structure
   - Create contribution guides for each library
   - Document registry and discovery systems

3. **Integration validation**
   - Verify all existing functionality works
   - Test discovery systems
   - Validate contribution frameworks

#### **Deliverables**
- ✅ Complete architecture documentation
- ✅ Library contribution guides
- ✅ Validated system integration

### **Phase 4: Stakeholder Enablement (Week 4)**

#### **Objectives**
- Provide development tools
- Create contribution templates
- Establish community framework

#### **Tasks**
1. **Development tools**
   - Scaffolding tools for new library components
   - Validation utilities for contributions
   - Testing frameworks for each library

2. **Community framework**
   - Contribution guidelines for each library
   - Quality assurance procedures
   - Integration testing for contributions

3. **Examples and templates**
   - Reference implementations in each `contrib/` directory
   - Templates for common contribution patterns
   - Complete examples showing extension points

#### **Deliverables**
- ✅ Development and scaffolding tools
- ✅ Community contribution framework
- ✅ Reference implementations and templates

---

## 🔌 Library Extension Points

### **1. Kernel Library Contributions**

**Location**: `brainsmith/libraries/kernels/contrib/`

**Existing Foundation**: Build on [`brainsmith/kernels/conv2d_hls/`](brainsmith/kernels/conv2d_hls/) structure

**Example Structure**:
```
brainsmith/libraries/kernels/contrib/my_kernel/
├── my_kernel_custom_op.py      # FINN custom operation
├── my_kernel_rtl_backend.py    # RTL backend
├── my_kernel.sv               # Hardware implementation
└── kernel.yaml               # Kernel metadata
```

### **2. Transform Library Contributions**

**Location**: `brainsmith/libraries/transforms/contrib/`

**Existing Foundation**: Follow patterns from [`brainsmith/steps/conversion.py`](brainsmith/steps/conversion.py) and [`brainsmith/transformation/expand_norms.py`](brainsmith/transformation/expand_norms.py)

**Example Structure**:
```
brainsmith/libraries/transforms/contrib/
├── my_optimization.py          # Transform implementation
└── __init__.py                # Registration
```

### **3. Analysis Library Contributions**

**Location**: `brainsmith/libraries/analysis/contrib/`

**Existing Foundation**: Extend [`brainsmith/tools/profiling/roofline.py`](brainsmith/tools/profiling/roofline.py) patterns

**Example Structure**:
```
brainsmith/libraries/analysis/contrib/
├── my_analyzer.py             # Analysis implementation
└── __init__.py               # Registration
```

---

## 🔄 Backward Compatibility Strategy

### **Import Aliases**
All existing imports continue to work:

```python
# All existing imports work unchanged
from brainsmith.core.api import forge                    # Works
from brainsmith.dse.functions import parameter_sweep     # Works via alias
from brainsmith.steps.conversion import qonnx_to_finn_step # Works via alias
from brainsmith.tools.profiling.roofline import roofline_analysis # Works via alias

# New recommended imports
from brainsmith.core import forge                        # New location
from brainsmith.infrastructure.dse import parameter_sweep # New location
from brainsmith.libraries.transforms.steps import qonnx_to_finn_step # New location
from brainsmith.libraries.analysis.profiling import roofline_analysis # New location
```

### **Gradual Migration**
- **No breaking changes**: All existing code continues to work
- **Import aliases**: Old import paths redirect to new locations
- **Deprecation warnings**: Gentle guidance toward new structure
- **Documentation**: Clear migration guides for new development

---

## 🎯 Benefits for Stakeholders

### **1. Organized Foundation**
- **Clear structure**: Three-layer architecture with obvious responsibilities
- **Rich libraries**: Comprehensive components organized by function
- **Stable core**: Minimal API surface that changes infrequently
- **Production-ready**: Already validated with existing implementation

### **2. Extension Points**
- **`contrib/` directories**: Clear places for stakeholder additions
- **Registry systems**: Automatic discovery of new components
- **Base interfaces**: Standardized APIs for all component types
- **Rich examples**: Existing implementations as reference

### **3. Developer Experience**
- **Minimal learning curve**: Build on existing component patterns
- **Clear organization**: Easy to find and understand components
- **Comprehensive tooling**: Scaffolding and validation tools
- **Community framework**: Guidelines for contribution and collaboration

---

## 📊 Success Metrics

### **Organizational Metrics**
- **Component findability**: < 30 seconds to locate any component
- **Structure clarity**: 100% of components in logical locations
- **Backward compatibility**: 100% of existing code continues to work
- **Documentation coverage**: Complete guides for all libraries

### **Extensibility Metrics**
- **Contribution ease**: < 2 days to add new library component
- **Discovery effectiveness**: 100% automatic discovery of contributions
- **Quality assurance**: Automated validation for all contributions
- **Community adoption**: Usage of contribution frameworks

---

## 🚀 Next Steps

### **Implementation Readiness**
- **Component inventory**: ✅ Complete understanding of existing components
- **Architecture design**: ✅ Three-layer structure with clear responsibilities
- **Migration plan**: ✅ Detailed steps with backward compatibility
- **Extension framework**: ✅ Clear contribution points and processes

### **Immediate Actions**
1. **Approve restructuring plan**: Review and approve this final plan
2. **Begin Phase 1**: Start infrastructure reorganization
3. **Validate approach**: Ensure restructured components work correctly
4. **Document progress**: Track implementation against plan

---

## 📋 Conclusion

This restructuring plan organizes Brainsmith's existing components into a clear, extensible foundation:

✅ **Three-layer architecture**: Core, Infrastructure, Libraries with clear separation  
✅ **Organized existing components**: All current functionality properly categorized  
✅ **Rich core libraries**: Essential compiler components well-organized  
✅ **Clear extension points**: Obvious places for stakeholder contributions  
✅ **100% backward compatibility**: All existing code continues to work  
✅ **Production-ready foundation**: Built on validated, working implementation  

The platform transforms from a feature-complete toolchain into an organized, stakeholder-extensible foundation while preserving all existing capabilities and maintaining complete backward compatibility.

---

*Document prepared: June 11, 2025*  
*Status: Final plan ready for implementation*  
*Implementation timeline: 4 weeks*