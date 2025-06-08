# Brainsmith Architectural Alignment Analysis

## Executive Summary

After comparing the high-level design document (`docs/brainsmith-high-level.md`) with the current implementation, there are significant opportunities to better align the codebase with the intended architectural vision. The current implementation has excellent DSE capabilities but lacks the modular library structure and clear separation of concerns envisioned in the high-level design.

## Key Gaps Identified

### 1. **Library Structure Mismatch**

**Vision**: Clean separation into specialized libraries (Hardware Kernels, Model Transforms, Hardware Optimization, Blueprints)
**Current**: Mixed concerns with DSE logic, blueprints, and compilation logic intermingled

**Proposed Directory Realignment**:
```
brainsmith/
├── blueprints/           # ✅ Exists but needs enhancement
├── core/                 # ✅ Good but needs DSE engine separation  
├── kernels/              # ❌ Missing - currently scattered in custom_op/
├── model_transforms/     # ❌ Missing - currently in steps/
├── hw_optim/            # ❌ Missing - optimization strategies scattered
├── analysis/            # ❌ Missing - currently in dse/analysis.py
├── interfaces/          # ❌ Missing - CLI/API separation needed
└── dse/                 # ✅ Exists but should be core DSE engine only
```

### 2. **DSE Engine Architecture Gap**

**Vision**: Meta-aware DSE engine that orchestrates libraries
**Current**: DSE engines are strategy-specific rather than meta-orchestrators

**Required Changes**:
- Create unified `DSEEngine` that coordinates across libraries
- Move strategy-specific logic to `hw_optim/` library
- Implement hierarchical exit points (Roofline → Dataflow → RTL)

### 3. **Blueprint System Enhancement**

**Vision**: Declarative YAML driving entire toolchain configuration
**Current**: Limited blueprint system focused on parameters

**Enhancement Needed**:
- Blueprint-driven kernel selection
- Transform pipeline specification
- Search strategy configuration
- Exit point selection

### 4. **Library Modularity**

**Vision**: Pluggable libraries with clear interfaces
**Current**: Tight coupling between components

**Modularization Required**:
- Hardware Kernels Library with uniform interface
- Model Transform Library with declarative configuration
- Hardware Optimization Strategies Library
- Clean plugin architecture

## Detailed Refactoring Plan

### Phase 1: Library Structure Reorganization

#### 1.1 Create Hardware Kernels Library
```
brainsmith/kernels/
├── __init__.py              # Kernel registry and discovery
├── registry.py              # Central kernel registration
├── base.py                  # Base kernel interfaces
├── rtl/                     # RTL kernel implementations
│   ├── __init__.py
│   ├── conv.py
│   ├── relu.py
│   └── ...
├── hls/                     # HLS kernel implementations  
│   ├── __init__.py
│   ├── conv_hls.py
│   └── ...
└── ops/                     # ONNX custom operations
    ├── __init__.py
    ├── hardware_conv.py
    └── ...
```

#### 1.2 Create Model Transforms Library
```
brainsmith/model_transforms/
├── __init__.py              # Transform registry
├── registry.py              # Transform registration system
├── base.py                  # Base transform interfaces
├── fusions.py               # Layer fusion transforms
├── streamlining.py          # Streamlining transforms
├── layout.py                # Layout optimization transforms
├── quantization.py          # Quantization transforms
└── search/                  # Meta-search strategies
    ├── __init__.py
    ├── bayesian.py
    ├── evolutionary.py
    └── hierarchical.py
```

#### 1.3 Create Hardware Optimization Library
```
brainsmith/hw_optim/
├── __init__.py              # Optimization registry
├── registry.py              # Strategy registration
├── base.py                  # Base optimization interfaces
├── param_opt.py             # Parameter optimization
├── impl_styles.py           # Implementation style optimization
├── scheduling.py            # Global scheduling optimization
├── resource_allocation.py   # Resource allocation strategies
└── strategies/              # Specific optimization algorithms
    ├── __init__.py
    ├── bayesian_opt.py
    ├── genetic_opt.py
    └── adaptive_opt.py
```

#### 1.4 Create Analysis Library
```
brainsmith/analysis/
├── __init__.py              # Analysis registry
├── roofline.py              # Roofline analysis
├── performance.py           # Performance modeling
├── reporting.py             # Report generation
├── visualization.py         # Visualization tools
└── export.py                # Export utilities
```

#### 1.5 Create Interfaces Library
```
brainsmith/interfaces/
├── __init__.py
├── cli.py                   # Command-line interface
├── api.py                   # Python API wrappers
├── dashboard/               # Web dashboard (future)
└── integration/             # External tool integration
```

### Phase 2: Core DSE Engine Refactoring

#### 2.1 Meta-DSE Engine
```python
# brainsmith/core/dse_engine.py
class MetaDSEEngine:
    """Meta-aware DSE engine that orchestrates libraries."""
    
    def __init__(self, blueprint: Blueprint):
        self.blueprint = blueprint
        self.kernel_library = KernelLibrary()
        self.transform_library = TransformLibrary()
        self.hw_optim_library = HardwareOptimizationLibrary()
        self.analysis_library = AnalysisLibrary()
    
    def construct_design_space(self):
        """Construct design space from blueprint configuration."""
        # Use blueprint to determine available:
        # - Kernels and their parameters
        # - Transform sequences
        # - Optimization strategies
        pass
    
    def explore_design_space(self, exit_point: str = "dataflow_generation"):
        """Explore design space with specified exit point."""
        if exit_point == "roofline":
            return self.roofline_analysis()
        elif exit_point == "dataflow_analysis":
            return self.dataflow_analysis()
        elif exit_point == "dataflow_generation":
            return self.dataflow_generation()
    
    def roofline_analysis(self):
        """Exit Point 1: Analytical model-only profiling."""
        pass
    
    def dataflow_analysis(self):
        """Exit Point 2: Hardware-abstracted performance estimation."""
        pass
    
    def dataflow_generation(self):
        """Exit Point 3: Generate RTL/HLS IP."""
        pass
```

#### 2.2 Design Space Construction
```python
# brainsmith/core/design_space.py (enhanced)
class DesignSpace:
    """Enhanced design space with library integration."""
    
    def __init__(self, blueprint: Blueprint):
        self.blueprint = blueprint
        self.kernel_space = None
        self.transform_space = None
        self.optimization_space = None
    
    def construct_from_blueprint(self):
        """Construct design space from blueprint libraries."""
        # Parse blueprint YAML to determine:
        # - Available kernels and parameters
        # - Transform sequences
        # - Optimization strategies
        pass
    
    def get_cartesian_product(self):
        """Get Cartesian product of all configuration spaces."""
        pass
```

### Phase 3: Blueprint System Enhancement

#### 3.1 Enhanced Blueprint Schema
```yaml
# Enhanced blueprint YAML schema
model_type: transformer
version: "1.0"

# Kernel Library Configuration
kernels:
  available:
    - name: qlinear_conv
      impl: [hls, rtl]
      params:
        parallelism: {type: integer, range: [1, 16]}
        quant: {type: categorical, values: [int4, int8, int16]}
    - name: quantized_relu
      impl: [rtl]
      params:
        threshold: {type: continuous, range: [0.0, 1.0]}

# Model Transform Configuration  
transforms:
  pipeline:
    - name: fuse_layernorm
      enabled: true
      search: false
    - name: streamline
      enabled: true
      search: false
    - name: layout_optimization
      enabled: true
      search: true
      strategies: [greedy, optimal]

# Hardware Optimization Configuration
hw_optimization:
  strategies:
    - param_optimization:
        algorithm: bayesian
        budget: 100
    - impl_style_optimization:
        algorithm: genetic
        budget: 50
  global_optimization:
    scheduling: enabled
    resource_allocation: enabled

# Search Strategy
search_strategy:
  meta_algorithm: hierarchical
  exit_points: [roofline, dataflow_analysis, dataflow_generation]
  
# Objectives and Constraints
objectives:
  primary:
    - metric: throughput
      target: maximize
      weight: 1.0
  secondary:
    - metric: latency
      target: minimize
      weight: 0.5
    - metric: resource_utilization
      target: minimize
      weight: 0.3

constraints:
  target_device: xcvu9p
  max_latency: 100us
  min_throughput: 5000
  resource_limits:
    lut_utilization: 0.8
    bram_utilization: 0.9
```

#### 3.2 Blueprint-Driven Workflow
```python
# brainsmith/core/workflow.py
class BrainsmithWorkflow:
    """High-level workflow orchestration."""
    
    def __init__(self, model_path: str, blueprint: Blueprint):
        self.model_path = model_path
        self.blueprint = blueprint
        self.dse_engine = MetaDSEEngine(blueprint)
    
    def execute(self, exit_point: str = "dataflow_generation"):
        """Execute complete workflow with blueprint configuration."""
        # 1. Validate inputs
        self.validate_inputs()
        
        # 2. Construct design space from blueprint
        design_space = self.dse_engine.construct_design_space()
        
        # 3. Execute DSE with specified exit point
        results = self.dse_engine.explore_design_space(exit_point)
        
        # 4. Generate reports and analysis
        analysis = self.generate_analysis(results)
        
        return results, analysis
```

### Phase 4: Interface Standardization

#### 4.1 Unified Python API
```python
# brainsmith/interfaces/api.py
def brainsmith_explore(model_path: str, 
                      blueprint_path: str,
                      exit_point: str = "dataflow_generation",
                      **kwargs):
    """Unified high-level API matching architectural vision."""
    blueprint = Blueprint.from_yaml_file(blueprint_path)
    workflow = BrainsmithWorkflow(model_path, blueprint)
    return workflow.execute(exit_point)

def brainsmith_roofline(model_path: str, blueprint_path: str):
    """Quick roofline analysis."""
    return brainsmith_explore(model_path, blueprint_path, "roofline")

def brainsmith_dataflow_analysis(model_path: str, blueprint_path: str):
    """Dataflow-level analysis."""
    return brainsmith_explore(model_path, blueprint_path, "dataflow_analysis")

def brainsmith_generate(model_path: str, blueprint_path: str):
    """Full RTL/HLS generation."""
    return brainsmith_explore(model_path, blueprint_path, "dataflow_generation")
```

#### 4.2 CLI Interface
```python
# brainsmith/interfaces/cli.py
import click

@click.group()
def cli():
    """Brainsmith: Meta-toolchain for FPGA accelerator synthesis."""
    pass

@cli.command()
@click.argument('model_path')
@click.argument('blueprint_path')
@click.option('--exit-point', default='dataflow_generation')
def explore(model_path, blueprint_path, exit_point):
    """Explore design space with specified exit point."""
    results, analysis = brainsmith_explore(model_path, blueprint_path, exit_point)
    # Display results
    pass

@cli.command()
@click.argument('model_path')
@click.argument('blueprint_path')
def roofline(model_path, blueprint_path):
    """Perform roofline analysis."""
    results = brainsmith_roofline(model_path, blueprint_path)
    # Display roofline analysis
    pass
```

## Implementation Priority

### High Priority (Immediate)
1. **Library Structure Reorganization** - Move existing code to proper library structure
2. **Meta-DSE Engine** - Create unified orchestration engine
3. **Enhanced Blueprint System** - Support library-driven configuration

### Medium Priority (Next Phase)
1. **CLI Interface** - Implement command-line tools
2. **Analysis Library** - Separate analysis from DSE logic
3. **Plugin Architecture** - Enable dynamic library registration

### Low Priority (Future)
1. **Web Dashboard** - Visual interface for exploration
2. **Advanced Meta-Search** - Hierarchical and multi-level optimization
3. **External Tool Integration** - Enhanced FINN and third-party integration

## Migration Strategy

### Phase 1: Non-Breaking Reorganization
- Move files to new library structure
- Maintain existing API compatibility
- Add new library interfaces alongside existing code

### Phase 2: Enhanced Blueprint Support
- Extend existing blueprint system
- Add library-driven configuration
- Maintain backward compatibility with existing blueprints

### Phase 3: Unified Interface
- Implement meta-DSE engine
- Add CLI interface
- Provide migration path for existing users

### Phase 4: Full Architecture Realization
- Complete plugin architecture
- Advanced meta-search strategies
- Full architectural vision implementation

## Quality Assurance

### Testing Strategy
- Maintain all existing tests during reorganization
- Add integration tests for library interfaces
- Performance regression testing
- Backward compatibility validation

### Documentation Updates
- Update all documentation to reflect new architecture
- Provide migration guides
- Create library-specific documentation
- Update examples and tutorials

## Expected Benefits

### For Users
- **Clearer Workflow**: Blueprint-driven configuration matches mental model
- **Better Modularity**: Easy to understand and extend
- **More Powerful**: Access to full library capabilities through blueprints
- **Consistent Interface**: Unified CLI and API experience

### For Developers
- **Clean Architecture**: Clear separation of concerns
- **Easy Extension**: Plugin architecture for new capabilities
- **Better Testing**: Modular components easier to test
- **Clear Interfaces**: Well-defined library boundaries

### For Research
- **Systematic Exploration**: Library-driven design space construction
- **Reproducible Results**: Blueprint-based configuration management
- **Easy Experimentation**: Pluggable optimization strategies
- **Rich Analysis**: Comprehensive analysis library

This refactoring plan aligns the implementation with the architectural vision while maintaining backward compatibility and providing a clear migration path for existing users.