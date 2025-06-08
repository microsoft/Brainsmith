# Brainsmith Phase 4: Architectural Alignment Implementation Plan

## Overview

Phase 4 focuses on aligning the current Brainsmith implementation with the high-level architectural vision defined in `docs/brainsmith-high-level.md`. This phase will restructure the codebase into the envisioned library architecture while maintaining backward compatibility.

## Goals

1. **Restructure into Library Architecture**: Organize code into specialized libraries (Kernels, Transforms, Hardware Optimization, Analysis)
2. **Implement Meta-DSE Engine**: Create unified orchestration engine that coordinates libraries
3. **Enhance Blueprint System**: Support library-driven configuration through enhanced YAML schemas
4. **Create Unified Interfaces**: Implement CLI and enhanced API matching architectural vision
5. **Maintain Backward Compatibility**: Ensure existing code continues to work

## Phase 4 Components

### Component 1: Library Structure Reorganization

#### 1.1 Hardware Kernels Library
**Goal**: Consolidate scattered kernel implementations into unified library

**Current State**: 
- Kernels scattered in `brainsmith/custom_op/`
- HLS implementations in various locations
- No unified registration system

**Target Structure**:
```
brainsmith/kernels/
├── __init__.py              # Public API and registry
├── registry.py              # Kernel registration system
├── base.py                  # Base interfaces
├── rtl/                     # RTL implementations
├── hls/                     # HLS implementations
└── ops/                     # ONNX custom operations
```

**Implementation Steps**:
1. Create `brainsmith/kernels/` directory structure
2. Move existing kernel code from `custom_op/` to appropriate subdirectories
3. Implement kernel registry system
4. Create base interfaces for kernel standardization
5. Update imports throughout codebase

#### 1.2 Model Transforms Library
**Goal**: Create dedicated library for model-level transformations

**Current State**:
- Transform logic scattered in `brainsmith/steps/`
- No systematic transform discovery
- Limited search strategy support

**Target Structure**:
```
brainsmith/model_transforms/
├── __init__.py              # Public API
├── registry.py              # Transform registration
├── base.py                  # Base transform interfaces
├── fusions.py               # Layer fusion transforms
├── streamlining.py          # Streamlining operations
├── layout.py                # Layout optimizations
├── quantization.py          # Quantization transforms
└── search/                  # Meta-search strategies
    ├── __init__.py
    ├── bayesian.py
    ├── evolutionary.py
    └── hierarchical.py
```

**Implementation Steps**:
1. Create `brainsmith/model_transforms/` directory
2. Extract and refactor transform logic from `steps/`
3. Implement transform registry and discovery
4. Create base interfaces for transform standardization
5. Implement meta-search strategies

#### 1.3 Hardware Optimization Library
**Goal**: Centralize hardware optimization strategies

**Current State**:
- Optimization logic mixed with DSE engines
- No systematic strategy registry
- Limited parameter optimization support

**Target Structure**:
```
brainsmith/hw_optim/
├── __init__.py              # Public API
├── registry.py              # Strategy registration
├── base.py                  # Base optimization interfaces
├── param_opt.py             # Parameter optimization
├── impl_styles.py           # Implementation styles
├── scheduling.py            # Global scheduling
├── resource_allocation.py   # Resource allocation
└── strategies/              # Specific algorithms
    ├── __init__.py
    ├── bayesian_opt.py
    ├── genetic_opt.py
    └── adaptive_opt.py
```

**Implementation Steps**:
1. Create `brainsmith/hw_optim/` directory structure
2. Extract optimization logic from `dse/` modules
3. Implement strategy registry system
4. Create base interfaces for optimization strategies
5. Implement specific optimization algorithms

#### 1.4 Analysis Library
**Goal**: Separate analysis capabilities into dedicated library

**Current State**:
- Analysis mixed with DSE logic in `dse/analysis.py`
- Limited analysis capabilities
- No systematic reporting framework

**Target Structure**:
```
brainsmith/analysis/
├── __init__.py              # Public API
├── roofline.py              # Roofline analysis
├── performance.py           # Performance modeling
├── reporting.py             # Report generation
├── visualization.py         # Visualization tools
└── export.py                # Export utilities
```

**Implementation Steps**:
1. Create `brainsmith/analysis/` directory
2. Extract analysis logic from `dse/analysis.py`
3. Implement roofline analysis capabilities
4. Create comprehensive reporting framework
5. Add visualization and export utilities

### Component 2: Meta-DSE Engine Implementation

#### 2.1 Core DSE Engine Refactoring
**Goal**: Create meta-aware DSE engine that orchestrates libraries

**Current State**:
- DSE engines are strategy-specific
- No unified orchestration
- Limited library coordination

**Target Implementation**:
```python
# brainsmith/core/dse_engine.py
class MetaDSEEngine:
    """Meta-aware DSE engine orchestrating all libraries."""
    
    def __init__(self, blueprint: Blueprint):
        self.blueprint = blueprint
        self.libraries = self._initialize_libraries()
        self.design_space = None
    
    def _initialize_libraries(self):
        """Initialize all libraries based on blueprint."""
        return {
            'kernels': KernelLibrary(self.blueprint),
            'transforms': TransformLibrary(self.blueprint),
            'hw_optim': HWOptimLibrary(self.blueprint),
            'analysis': AnalysisLibrary(self.blueprint)
        }
    
    def construct_design_space(self):
        """Construct design space from blueprint and libraries."""
        # Combine design spaces from all libraries
        pass
    
    def explore_design_space(self, exit_point: str = "dataflow_generation"):
        """Execute exploration with hierarchical exit points."""
        if exit_point == "roofline":
            return self._roofline_analysis()
        elif exit_point == "dataflow_analysis":
            return self._dataflow_analysis()
        elif exit_point == "dataflow_generation":
            return self._dataflow_generation()
    
    def _roofline_analysis(self):
        """Exit Point 1: Analytical model profiling."""
        pass
    
    def _dataflow_analysis(self):
        """Exit Point 2: Hardware-abstracted analysis."""
        pass
    
    def _dataflow_generation(self):
        """Exit Point 3: Complete RTL/HLS generation."""
        pass
```

**Implementation Steps**:
1. Create `MetaDSEEngine` class in `brainsmith/core/dse_engine.py`
2. Implement library coordination logic
3. Create hierarchical exit point system
4. Integrate with existing DSE infrastructure
5. Add comprehensive testing

#### 2.2 Design Space Integration
**Goal**: Enable blueprint-driven design space construction

**Enhancement to Existing**:
```python
# brainsmith/core/design_space.py (enhanced)
class DesignSpace:
    def construct_from_blueprint(self, blueprint: Blueprint, libraries: dict):
        """Construct design space from blueprint and available libraries."""
        # Parse blueprint to determine:
        # - Available kernels and parameters
        # - Transform sequences and options
        # - Optimization strategies and parameters
        # - Constraints and objectives
        pass
    
    def get_library_spaces(self):
        """Get individual design spaces from each library."""
        return {
            'kernel_space': self.kernel_design_space,
            'transform_space': self.transform_design_space,
            'optimization_space': self.optimization_design_space
        }
    
    def get_cartesian_product(self):
        """Get full Cartesian product of all design dimensions."""
        pass
```

### Component 3: Enhanced Blueprint System

#### 3.1 Extended Blueprint Schema
**Goal**: Support comprehensive library-driven configuration

**Enhanced YAML Schema**:
```yaml
# Enhanced blueprint supporting full architectural vision
name: "transformer_optimized"
version: "1.0"
description: "Optimized transformer with full library support"

# Kernel Library Configuration
kernels:
  registry: "transformer_kernels"
  available:
    - name: "quantized_linear"
      implementations: ["hls", "rtl"]
      parameters:
        parallelism: {type: "integer", range: [1, 16], default: 8}
        quantization: {type: "categorical", values: ["int4", "int8", "int16"], default: "int8"}
        tiling: {type: "integer", range: [1, 32], default: 4}
    - name: "layer_norm"
      implementations: ["hls"]
      parameters:
        precision: {type: "categorical", values: ["float16", "bfloat16"], default: "float16"}

# Model Transform Library Configuration
transforms:
  registry: "transformer_transforms"
  pipeline:
    - name: "fuse_layernorm"
      enabled: true
      searchable: false
    - name: "streamline_graph"
      enabled: true
      searchable: false
    - name: "optimize_layout"
      enabled: true
      searchable: true
      parameters:
        strategy: {type: "categorical", values: ["greedy", "optimal", "heuristic"], default: "optimal"}

# Hardware Optimization Configuration
hw_optimization:
  registry: "fpga_optimization"
  strategies:
    - name: "parameter_optimization"
      algorithm: "bayesian"
      budget: 100
      parameters:
        acquisition: {type: "categorical", values: ["EI", "UCB", "PI"], default: "EI"}
    - name: "implementation_optimization"
      algorithm: "genetic"
      budget: 50
      parameters:
        population_size: {type: "integer", range: [20, 100], default: 50}
        mutation_rate: {type: "continuous", range: [0.01, 0.1], default: 0.05}

# Analysis Configuration
analysis:
  exit_points: ["roofline", "dataflow_analysis", "dataflow_generation"]
  metrics:
    primary: ["throughput", "latency", "resource_utilization"]
    secondary: ["power_consumption", "memory_bandwidth"]
  reporting:
    formats: ["json", "yaml", "html"]
    visualization: true

# Global Configuration
search_strategy:
  meta_algorithm: "hierarchical"
  coordination: "parallel"
  
objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0
    constraint: {min: 1000}
  - name: "resource_utilization"
    direction: "minimize"
    weight: 0.5
    constraint: {max: 0.8}

constraints:
  target_device: "xcvu9p"
  clock_frequency: "300MHz"
  resource_limits:
    lut_utilization: 0.85
    bram_utilization: 0.9
    dsp_utilization: 0.95
```

#### 3.2 Blueprint Processing Enhancement
**Goal**: Enable comprehensive blueprint-driven workflow

**Enhanced Blueprint Class**:
```python
# brainsmith/blueprints/base.py (enhanced)
class Blueprint:
    def get_library_configs(self) -> Dict[str, Dict]:
        """Get library-specific configurations from blueprint."""
        return {
            'kernels': self.yaml_data.get('kernels', {}),
            'transforms': self.yaml_data.get('transforms', {}),
            'hw_optimization': self.yaml_data.get('hw_optimization', {}),
            'analysis': self.yaml_data.get('analysis', {})
        }
    
    def get_search_strategy_config(self) -> Dict:
        """Get meta-search strategy configuration."""
        return self.yaml_data.get('search_strategy', {})
    
    def get_exit_points(self) -> List[str]:
        """Get configured exit points for exploration."""
        analysis_config = self.yaml_data.get('analysis', {})
        return analysis_config.get('exit_points', ['dataflow_generation'])
    
    def supports_library_driven_dse(self) -> bool:
        """Check if blueprint supports full library-driven DSE."""
        required_sections = ['kernels', 'transforms', 'hw_optimization']
        return all(section in self.yaml_data for section in required_sections)
```

### Component 4: Unified Interface Implementation

#### 4.1 Enhanced Python API
**Goal**: Provide API matching architectural vision

**New API Implementation**:
```python
# brainsmith/interfaces/api.py
def brainsmith_explore(model_path: str, 
                      blueprint_path: str,
                      exit_point: str = "dataflow_generation",
                      **kwargs) -> Tuple[DSEResult, Dict]:
    """
    Main exploration API matching architectural vision.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        exit_point: Exit point ('roofline', 'dataflow_analysis', 'dataflow_generation')
        **kwargs: Additional configuration options
    
    Returns:
        Tuple of (DSE results, comprehensive analysis)
    """
    # Load blueprint
    blueprint = Blueprint.from_yaml_file(blueprint_path)
    
    # Create meta-DSE engine
    meta_engine = MetaDSEEngine(blueprint)
    
    # Execute exploration with specified exit point
    results = meta_engine.explore_design_space(exit_point)
    
    # Generate comprehensive analysis
    analysis = meta_engine.libraries['analysis'].analyze_results(results)
    
    return results, analysis

# Convenience functions for each exit point
def brainsmith_roofline(model_path: str, blueprint_path: str):
    """Quick roofline analysis."""
    return brainsmith_explore(model_path, blueprint_path, "roofline")

def brainsmith_dataflow_analysis(model_path: str, blueprint_path: str):
    """Dataflow-level performance analysis."""
    return brainsmith_explore(model_path, blueprint_path, "dataflow_analysis")

def brainsmith_generate(model_path: str, blueprint_path: str):
    """Complete RTL/HLS generation."""
    return brainsmith_explore(model_path, blueprint_path, "dataflow_generation")
```

#### 4.2 CLI Interface Implementation
**Goal**: Provide command-line interface matching vision

**CLI Implementation**:
```python
# brainsmith/interfaces/cli.py
import click
from pathlib import Path

@click.group()
@click.version_option()
def brainsmith():
    """
    Brainsmith: Meta-toolchain for FPGA accelerator synthesis.
    
    A comprehensive platform for neural network accelerator design space
    exploration with hierarchical analysis capabilities.
    """
    pass

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--exit-point', '-e', 
              type=click.Choice(['roofline', 'dataflow_analysis', 'dataflow_generation']),
              default='dataflow_generation',
              help='Analysis exit point')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def explore(model_path, blueprint_path, exit_point, output, verbose):
    """Explore design space with specified exit point."""
    if verbose:
        click.echo(f"Exploring {model_path} with {blueprint_path}")
        click.echo(f"Exit point: {exit_point}")
    
    results, analysis = brainsmith_explore(model_path, blueprint_path, exit_point)
    
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save results and analysis
        
    # Display summary
    click.echo("Exploration complete!")
    # Display key metrics

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
def roofline(model_path, blueprint_path):
    """Perform quick roofline analysis."""
    results, analysis = brainsmith_roofline(model_path, blueprint_path)
    # Display roofline-specific output

@brainsmith.command()
@click.argument('blueprint_path', type=click.Path(exists=True))
def validate(blueprint_path):
    """Validate blueprint configuration."""
    try:
        blueprint = Blueprint.from_yaml_file(blueprint_path)
        click.echo("✅ Blueprint is valid")
        # Display blueprint summary
    except Exception as e:
        click.echo(f"❌ Blueprint validation failed: {e}")

@brainsmith.command()
def list_kernels():
    """List available hardware kernels."""
    # Display available kernels from registry

@brainsmith.command()
def list_transforms():
    """List available model transforms."""
    # Display available transforms from registry
```

### Component 5: Backward Compatibility Layer

#### 5.1 Compatibility Wrapper
**Goal**: Ensure existing code continues to work

**Implementation**:
```python
# brainsmith/__init__.py (enhanced)
# Maintain all existing exports
from .legacy import *  # Import all Phase 3 functions

# Enhanced functions with backward compatibility
def explore_design_space(model_path: str, blueprint_name: str, **kwargs):
    """Enhanced with backward compatibility."""
    # If blueprint_name is a string, assume old-style blueprint
    if isinstance(blueprint_name, str):
        # Load blueprint by name (existing behavior)
        blueprint = get_blueprint(blueprint_name)
        
        # Check if blueprint supports new architecture
        if blueprint.supports_library_driven_dse():
            # Use new meta-DSE engine
            meta_engine = MetaDSEEngine(blueprint)
            return meta_engine.explore_design_space()
        else:
            # Fall back to existing DSE logic
            return _legacy_explore_design_space(model_path, blueprint_name, **kwargs)
    else:
        # Assume blueprint_name is actually a Blueprint object or path
        return brainsmith_explore(model_path, blueprint_name, **kwargs)
```

## Implementation Timeline

### Week 1-2: Library Structure Setup
- [ ] Create directory structures for all libraries
- [ ] Move existing code to appropriate libraries
- [ ] Implement basic registry systems
- [ ] Update imports throughout codebase

### Week 3-4: Meta-DSE Engine
- [ ] Implement `MetaDSEEngine` class
- [ ] Create library coordination logic
- [ ] Implement hierarchical exit points
- [ ] Integration testing with existing systems

### Week 5-6: Enhanced Blueprint System
- [ ] Extend blueprint YAML schema
- [ ] Implement library-driven configuration
- [ ] Create blueprint validation system
- [ ] Update existing blueprints

### Week 7-8: Interface Implementation
- [ ] Implement enhanced Python API
- [ ] Create CLI interface
- [ ] Add comprehensive documentation
- [ ] Create usage examples

### Week 9-10: Integration and Testing
- [ ] Comprehensive integration testing
- [ ] Backward compatibility validation
- [ ] Performance regression testing
- [ ] Documentation updates

## Quality Assurance

### Testing Strategy
1. **Unit Tests**: Test each library component independently
2. **Integration Tests**: Test library coordination and meta-DSE engine
3. **Backward Compatibility Tests**: Ensure existing code works
4. **CLI Tests**: Test command-line interface
5. **Performance Tests**: Validate no performance regressions

### Documentation Requirements
1. **Library Documentation**: Complete API documentation for each library
2. **Blueprint Guide**: Comprehensive blueprint configuration guide
3. **CLI Reference**: Complete command-line reference
4. **Migration Guide**: Guide for upgrading to new architecture
5. **Examples**: Working examples for all major use cases

## Success Criteria

### Functional Requirements
- [ ] All existing functionality preserved
- [ ] New library architecture fully functional
- [ ] Blueprint-driven configuration working
- [ ] CLI interface operational
- [ ] Hierarchical exit points implemented

### Quality Requirements
- [ ] All existing tests pass
- [ ] New functionality >95% test coverage
- [ ] No performance regressions
- [ ] Complete documentation coverage
- [ ] Successful migration examples

### Architectural Requirements
- [ ] Clean separation of concerns
- [ ] Modular, extensible architecture
- [ ] Clear library interfaces
- [ ] Plugin-ready architecture
- [ ] Future-proof design

## Risk Mitigation

### Technical Risks
- **Complex Refactoring**: Mitigate with incremental approach and comprehensive testing
- **Performance Impact**: Address with performance monitoring and optimization
- **Integration Issues**: Prevent with thorough integration testing

### Schedule Risks
- **Scope Creep**: Control with clear requirements and phased implementation
- **Testing Overhead**: Address with automated testing and CI/CD
- **Documentation Load**: Manage with parallel documentation development

### Quality Risks
- **Backward Compatibility**: Ensure with comprehensive compatibility testing
- **User Experience**: Validate with beta testing and feedback
- **Code Quality**: Maintain with code reviews and quality gates

This Phase 4 implementation will successfully align Brainsmith with its architectural vision while maintaining all existing capabilities and providing a clear path for future enhancements.