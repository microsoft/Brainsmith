# Brainsmith Final Architectural Design

## Executive Summary

This document presents the comprehensive architectural design for Brainsmith's transformation into a modular, extensible meta-toolchain for FPGA accelerator synthesis. The design incorporates feedback-driven refinements and addresses the transition from legacy FINN interfaces to future 4-hook systems while maintaining clean extensibility and good coding practices.

**IMPORTANT**: This architecture focuses purely on creating a high-quality extensible structure using only existing components as examples. No new library additions are made - the focus is on organizing and structuring existing capabilities for maximum extensibility.

## 1. Architectural Vision

Brainsmith is a meta-toolchain that superimposes advanced design space exploration, optimization, and modularity atop the FINN framework. It provides a blueprint-driven, library-based architecture that abstracts complexity while exposing powerful configuration capabilities through declarative YAML specifications.

### 1.1 Core Principles

- **Modular Library Architecture**: Specialized libraries with clear interfaces using existing components
- **Blueprint-Driven Workflow**: Declarative YAML configuration controls entire pipeline
- **Hierarchical Exit Points**: Multiple analysis levels (Roofline → Dataflow → RTL)
- **Clean Extensibility**: Plugin architecture for easy addition of future capabilities
- **Legacy Compatibility**: Seamless transition from current FINN interfaces to future systems
- **Structure Over Content**: Focus on extensible structure rather than adding new functionality

## 2. Final Architecture Overview

### 2.1 Directory Structure

```
brainsmith/
├── core/                           # Core orchestration and interfaces
│   ├── design_space_orchestrator.py   # Main orchestration engine
│   ├── workflow.py                     # High-level workflow management
│   ├── finn_interface.py               # FINN integration layer
│   ├── api.py                          # Python API (merged from interfaces/)
│   ├── cli.py                          # Command-line interface (merged from interfaces/)
│   └── legacy_support.py               # Legacy compatibility layer
├── blueprints/                     # Enhanced blueprint system
│   ├── base.py                         # Enhanced Blueprint class
│   ├── manager.py                      # Blueprint management
│   ├── validator.py                    # Blueprint validation
│   └── yaml/                           # Blueprint templates
├── kernels/                        # AI layer-based kernel library (existing components only)
│   ├── __init__.py                     # Public API and registry
│   ├── registry.py                     # Kernel registration system
│   ├── base.py                         # Base interfaces
│   ├── conv/                          # Convolution operations (from existing custom_op)
│   │   └── hw_custom_op.py            # Existing HWConv2D wrapper
│   ├── linear/                        # Linear operations (from existing custom_op)
│   │   └── hw_custom_op.py            # Existing linear layer wrapper
│   └── activation/                    # Activation functions (from existing custom_op)
│       └── hw_custom_op.py            # Existing activation wrapper
├── model_transforms/               # Model transformation library (existing components only)
│   ├── __init__.py                     # Public API
│   ├── registry.py                     # Transform registration
│   ├── base.py                         # Base transform interfaces
│   ├── streamlining.py                 # Existing streamlining transforms
│   ├── folding.py                      # Existing folding optimizations
│   └── partitioning.py                 # Existing graph partitioning
├── hw_optim/                      # Hardware optimization library (existing components only)
│   ├── __init__.py                     # Public API
│   ├── registry.py                     # Strategy registration
│   ├── base.py                         # Base optimization interfaces
│   ├── param_opt.py                    # Existing parameter optimization
│   ├── resource_opt.py                 # Existing resource optimization
│   └── strategies/                     # Existing optimization algorithms
│       ├── bayesian_opt.py             # Existing Bayesian optimization
│       ├── genetic_opt.py              # Existing genetic algorithms
│       └── random_opt.py               # Existing random search
├── analysis/                      # Analysis and reporting library
│   ├── __init__.py                     # Public API
│   ├── performance.py                  # Existing performance analysis
│   ├── reporting.py                    # Existing report generation
│   └── visualization.py                # Existing visualization tools
├── coordination/                  # DSE coordination layer
│   ├── __init__.py                     # Public API
│   ├── orchestration.py               # Cross-library coordination
│   ├── workflow.py                     # DSE workflow management
│   └── result_aggregation.py           # Multi-library results
└── legacy/                        # Legacy compatibility
    ├── __init__.py                     # Legacy API exports
    ├── compatibility.py               # Compatibility wrappers
    └── migration.py                    # Migration utilities
```

### 2.2 Key Architectural Changes

1. **Merged interfaces/ into core/**: Unified core orchestration with API/CLI interfaces
2. **AI layer-based kernels**: Organized by AI operation using existing custom operations
3. **Legacy FINN support**: Maintains DataflowBuildConfig while preparing for 4-hook interface
4. **Clear coordination layer**: Handles cross-library orchestration distinct from optimization algorithms
5. **Existing components only**: No new functionality, focus on clean extensible structure

## 3. Core Components

### 3.1 Design Space Orchestrator

**Purpose**: Central orchestration engine coordinating all libraries for design space exploration.

```python
# brainsmith/core/design_space_orchestrator.py
class DesignSpaceOrchestrator:
    """
    Orchestrates design space exploration across all Brainsmith libraries.
    
    Coordinates Hardware Kernels, Model Transforms, Hardware Optimization,
    and Analysis libraries to execute comprehensive design space exploration
    with hierarchical exit points using existing components only.
    """
    
    def __init__(self, blueprint: Blueprint):
        self.blueprint = blueprint
        self.libraries = self._initialize_libraries()
        self.finn_interface = self._initialize_finn_interface()
        self.design_space = None
    
    def _initialize_libraries(self) -> Dict[str, Any]:
        """Initialize all libraries based on blueprint configuration."""
        library_configs = self.blueprint.get_library_configs()
        
        return {
            'kernels': KernelLibrary(
                # Use existing custom operations
                kernels=kernel_registry.get_existing_kernels_for_blueprint(
                    library_configs.get('kernels', {})
                )
            ),
            'transforms': TransformLibrary(
                # Use existing transforms from steps/
                pipeline=transform_registry.get_existing_transforms_for_blueprint(
                    library_configs.get('transforms', {})
                )
            ),
            'hw_optim': HardwareOptimLibrary(
                # Use existing optimization strategies from dse/
                strategies=optimization_registry.get_existing_strategies_for_blueprint(
                    library_configs.get('hw_optimization', {})
                )
            ),
            'analysis': AnalysisLibrary(
                # Use existing analysis capabilities
                config=library_configs.get('analysis', {})
            )
        }
    
    def _initialize_finn_interface(self) -> FINNInterface:
        """Initialize FINN interface with legacy and future support."""
        return FINNInterface(
            legacy_config=self.blueprint.get_finn_legacy_config(),
            future_hooks=FINNHooksPlaceholder()  # Placeholder for 4-hook interface
        )
    
    def orchestrate_exploration(self, exit_point: str = "dataflow_generation") -> DSEResult:
        """
        Orchestrate complete design space exploration workflow.
        
        Args:
            exit_point: Analysis exit point ('roofline', 'dataflow_analysis', 'dataflow_generation')
            
        Returns:
            Complete DSE results with analysis
        """
        # Construct unified design space from existing libraries
        design_space = self.construct_design_space()
        
        # Execute hierarchical exploration based on exit point
        if exit_point == "roofline":
            return self._execute_roofline_analysis(design_space)
        elif exit_point == "dataflow_analysis":
            return self._execute_dataflow_analysis(design_space)
        elif exit_point == "dataflow_generation":
            return self._execute_dataflow_generation(design_space)
        else:
            raise ValueError(f"Invalid exit point: {exit_point}")
    
    def construct_design_space(self) -> DesignSpace:
        """
        Construct unified design space from existing library components.
        Implements the Cartesian product concept from the vision.
        """
        if self.design_space is not None:
            return self.design_space
        
        # Get design spaces from each library using existing components
        library_spaces = {
            'kernels': self.libraries['kernels'].get_design_space_from_existing(),
            'transforms': self.libraries['transforms'].get_design_space_from_existing(),
            'hw_optim': self.libraries['hw_optim'].get_design_space_from_existing()
        }
        
        # Combine into unified design space
        self.design_space = DesignSpace.combine(library_spaces.values())
        
        # Apply blueprint constraints
        constraints = self.blueprint.get_constraints_config()
        self.design_space.apply_constraints(constraints)
        
        return self.design_space
    
    def _execute_roofline_analysis(self, design_space: DesignSpace) -> DSEResult:
        """
        Exit Point 1: Analytical model-only profiling.
        Quick performance bounds estimation without hardware generation.
        """
        # Use existing analysis capabilities
        analyzer = self.libraries['analysis'].get_existing_analyzer()
        
        # Analyze model computational characteristics using existing methods
        model_analysis = analyzer.analyze_model_existing_methods(
            self.blueprint.model_path,
            self.blueprint
        )
        
        return DSEResult(
            results=[],
            analysis={
                'exit_point': 'roofline',
                'model_analysis': model_analysis,
                'method': 'existing_analysis_tools'
            },
            design_space=design_space
        )
    
    def _execute_dataflow_analysis(self, design_space: DesignSpace) -> DSEResult:
        """
        Exit Point 2: Hardware-abstracted ONNX lowering and performance estimation.
        Dataflow-level analysis without RTL generation using existing transforms.
        """
        # Execute transform optimization using existing transforms
        transform_coordinator = self.libraries['transforms']
        transform_results = transform_coordinator.apply_existing_transforms(design_space)
        
        # Map to existing hardware kernels
        kernel_mapper = self.libraries['kernels'].get_existing_mapper()
        kernel_mapping = kernel_mapper.map_model_to_existing_kernels(
            transform_results['transformed_model']
        )
        
        return DSEResult(
            results=[],
            analysis={
                'exit_point': 'dataflow_analysis',
                'transformed_model': transform_results['transformed_model'],
                'kernel_mapping': kernel_mapping,
                'method': 'existing_dataflow_tools'
            },
            design_space=design_space
        )
    
    def _execute_dataflow_generation(self, design_space: DesignSpace) -> DSEResult:
        """
        Exit Point 3: Complete stitched, parameterized RTL or HLS IP generation.
        Full RTL/HLS generation workflow using existing FINN interface.
        """
        # Execute optimization using existing strategies
        coordination_engine = CoordinationEngine(self.libraries)
        optimization_results = coordination_engine.execute_existing_optimization(
            design_space, self.blueprint.get_objectives_config()
        )
        
        best_design_point = optimization_results['best_point']
        
        # Generate RTL/HLS using existing FINN interface
        generation_results = self.finn_interface.generate_implementation(
            model_path=self.blueprint.model_path,
            design_point=best_design_point,
            target_device=self.blueprint.get_target_device()
        )
        
        return DSEResult(
            results=optimization_results['all_results'],
            best_result=optimization_results['best_result'],
            analysis={
                'exit_point': 'dataflow_generation',
                'design_point': best_design_point,
                'generation_results': generation_results,
                'method': 'existing_generation_tools'
            },
            design_space=design_space
        )
```

### 3.2 FINN Interface with Legacy Support

**Purpose**: Manage transition from DataflowBuildConfig to future 4-hook interface.

```python
# brainsmith/core/finn_interface.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
from finn.util.fpgadataflow import DataflowBuildConfig  # Legacy

@dataclass
class FINNHooksPlaceholder:
    """
    Placeholder for future 4-hook FINN interface.
    
    This will be replaced with the actual 4-hook interface when available.
    For now, it serves as a structured placeholder to ensure clean transition.
    """
    
    # Placeholder hook definitions (will be replaced)
    preprocessing_hook: Optional[Any] = None
    transformation_hook: Optional[Any] = None
    optimization_hook: Optional[Any] = None
    generation_hook: Optional[Any] = None
    
    # Configuration for future interface
    hook_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.hook_config is None:
            self.hook_config = {}
    
    def is_available(self) -> bool:
        """Check if 4-hook interface is available."""
        # For now, always False until 4-hook interface is implemented
        return False
    
    def prepare_for_future_interface(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for future 4-hook interface."""
        return {
            'preprocessing_config': design_point.get('preprocessing', {}),
            'transformation_config': design_point.get('transforms', {}),
            'optimization_config': design_point.get('hw_optimization', {}),
            'generation_config': design_point.get('generation', {})
        }

class FINNInterface:
    """
    FINN integration layer supporting both legacy and future interfaces.
    
    Maintains support for current DataflowBuildConfig while preparing for
    the upcoming 4-hook interface. Provides clean transition path.
    """
    
    def __init__(self, legacy_config: Dict[str, Any], future_hooks: FINNHooksPlaceholder):
        self.legacy_config = legacy_config
        self.future_hooks = future_hooks
        self.use_legacy = not future_hooks.is_available()
    
    def generate_implementation(self, model_path: str, design_point: Dict[str, Any], 
                              target_device: str) -> Dict[str, Any]:
        """
        Generate RTL/HLS implementation using appropriate FINN interface.
        
        Uses existing DataflowBuildConfig flow while providing placeholder
        for future 4-hook interface.
        """
        if self.use_legacy:
            return self._generate_with_legacy_interface(model_path, design_point, target_device)
        else:
            return self._generate_with_future_interface(model_path, design_point, target_device)
    
    def _generate_with_legacy_interface(self, model_path: str, design_point: Dict[str, Any], 
                                      target_device: str) -> Dict[str, Any]:
        """Generate implementation using existing legacy DataflowBuildConfig."""
        # Create DataflowBuildConfig from design point using existing patterns
        build_config = self._create_legacy_build_config(design_point, target_device)
        
        # Execute existing FINN build process
        from finn.builder.build_dataflow import build_dataflow
        
        build_results = build_dataflow(
            model=model_path,
            cfg=build_config
        )
        
        # Extract and format results using existing result format
        return {
            'rtl_files': build_results.get('rtl_files', []),
            'hls_files': build_results.get('hls_files', []),
            'synthesis_results': build_results.get('synthesis_results', {}),
            'performance_metrics': self._extract_performance_metrics(build_results),
            'resource_utilization': build_results.get('resource_utilization', {}),
            'interface_type': 'legacy_dataflow_build_config'
        }
    
    def _create_legacy_build_config(self, design_point: Dict[str, Any], target_device: str) -> DataflowBuildConfig:
        """Create DataflowBuildConfig from design point using existing configuration patterns."""
        config = DataflowBuildConfig(
            # Core configuration using existing patterns
            target_device=target_device,
            
            # Kernel configuration from existing custom ops
            kernel_params=design_point.get('kernels', {}),
            
            # Transform configuration from existing transforms
            transform_params=design_point.get('transforms', {}),
            
            # Optimization configuration from existing optimizers
            optimization_params=design_point.get('hw_optimization', {}),
            
            # Legacy-specific settings from existing configuration
            **self.legacy_config
        )
        
        return config
```

### 3.3 Existing Component-Based Kernel Library

**Purpose**: Organize existing kernels by AI operation for extensibility.

```python
# brainsmith/kernels/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class ExistingKernelWrapper(ABC):
    """
    Base wrapper for existing kernel implementations.
    
    Provides extensible structure around existing custom operations
    without adding new functionality.
    """
    
    def __init__(self, existing_kernel_class: Any, operation_name: str):
        self.existing_kernel = existing_kernel_class
        self.operation_name = operation_name
        self.existing_params = {}
    
    @abstractmethod
    def get_existing_parameter_space(self) -> Dict[str, Any]:
        """Extract parameter space from existing kernel implementation."""
        pass
    
    @abstractmethod
    def wrap_existing_generation(self, parameters: Dict[str, Any]) -> Any:
        """Wrap existing kernel generation with parameter mapping."""
        pass

# brainsmith/kernels/conv/hw_custom_op.py
class ConvolutionHWCustomOpWrapper(ExistingKernelWrapper):
    """
    Wrapper for existing HWConv2D custom operation.
    
    Provides extensible structure around existing convolution implementation
    without modifying the underlying functionality.
    """
    
    def __init__(self):
        # Import existing custom operation
        from ...custom_op.hw_conv2d import HWConv2D
        super().__init__(HWConv2D, "Convolution")
    
    def get_existing_parameter_space(self) -> Dict[str, Any]:
        """Extract parameter space from existing HWConv2D implementation."""
        # Use existing parameter definitions from HWConv2D
        return {
            'simd': {
                'type': 'integer',
                'range': [1, 64],  # From existing implementation
                'default': 1,
                'description': 'SIMD parallelism from existing HWConv2D'
            },
            'pe': {
                'type': 'integer',
                'range': [1, 64],  # From existing implementation
                'default': 1,
                'description': 'PE parallelism from existing HWConv2D'
            }
            # Additional parameters extracted from existing implementation
        }
    
    def wrap_existing_generation(self, parameters: Dict[str, Any]) -> Any:
        """Wrap existing HWConv2D generation."""
        # Map parameters to existing HWConv2D format
        existing_params = {
            'simd': parameters.get('simd', 1),
            'pe': parameters.get('pe', 1)
            # Map other parameters to existing format
        }
        
        # Use existing HWConv2D generation
        return self.existing_kernel(**existing_params)

# brainsmith/kernels/registry.py
class ExistingKernelRegistry:
    """
    Registry for existing kernel implementations.
    
    Provides extensible structure for organizing existing custom operations
    without adding new kernels.
    """
    
    def __init__(self):
        self._existing_kernels = {}
        self._register_existing_kernels()
    
    def _register_existing_kernels(self):
        """Register all existing custom operations in extensible structure."""
        # Register existing HWConv2D
        from .conv.hw_custom_op import ConvolutionHWCustomOpWrapper
        self.register_existing_kernel(ConvolutionHWCustomOpWrapper())
        
        # Register existing linear operations
        from .linear.hw_custom_op import LinearHWCustomOpWrapper
        self.register_existing_kernel(LinearHWCustomOpWrapper())
        
        # Register existing activation functions
        from .activation.hw_custom_op import ActivationHWCustomOpWrapper
        self.register_existing_kernel(ActivationHWCustomOpWrapper())
    
    def register_existing_kernel(self, wrapper: ExistingKernelWrapper):
        """Register an existing kernel wrapper for extensibility."""
        self._existing_kernels[wrapper.operation_name] = wrapper
    
    def get_existing_kernels_for_blueprint(self, blueprint_config: Dict) -> List[ExistingKernelWrapper]:
        """Get existing kernels specified in blueprint configuration."""
        kernels = []
        for kernel_config in blueprint_config.get('available', []):
            kernel_name = kernel_config['name']
            if kernel_name in self._existing_kernels:
                kernels.append(self._existing_kernels[kernel_name])
        return kernels

# Global registry for existing kernels
existing_kernel_registry = ExistingKernelRegistry()
```

### 3.4 Existing Transform Library Structure

**Purpose**: Organize existing transforms for extensibility without adding new functionality.

```python
# brainsmith/model_transforms/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class ExistingTransformWrapper(ABC):
    """
    Base wrapper for existing transform implementations.
    
    Provides extensible structure around existing transforms from steps/
    without adding new functionality.
    """
    
    def __init__(self, existing_transform_module: Any, transform_name: str):
        self.existing_transform = existing_transform_module
        self.transform_name = transform_name
        self.existing_config = {}
    
    @abstractmethod
    def get_existing_parameter_space(self) -> Dict[str, Any]:
        """Extract parameter space from existing transform."""
        pass
    
    @abstractmethod
    def wrap_existing_application(self, model: Any, parameters: Dict[str, Any]) -> Any:
        """Wrap existing transform application."""
        pass

# brainsmith/model_transforms/streamlining.py
class ExistingStreamliningWrapper(ExistingTransformWrapper):
    """
    Wrapper for existing streamlining transforms.
    
    Provides extensible structure around existing streamlining functionality
    from steps/ without modification.
    """
    
    def __init__(self):
        # Import existing streamlining from steps
        from ...steps import streamlining
        super().__init__(streamlining, "streamlining")
    
    def get_existing_parameter_space(self) -> Dict[str, Any]:
        """Extract parameter space from existing streamlining implementation."""
        # Use existing streamlining parameters
        return {
            'fold_constants': {
                'type': 'boolean',
                'default': True,
                'description': 'Fold constants (from existing streamlining)'
            },
            'remove_unused': {
                'type': 'boolean', 
                'default': True,
                'description': 'Remove unused nodes (from existing streamlining)'
            }
            # Additional parameters from existing implementation
        }
    
    def wrap_existing_application(self, model: Any, parameters: Dict[str, Any]) -> Any:
        """Wrap existing streamlining application."""
        # Map parameters to existing streamlining format
        existing_config = {
            'fold_constants': parameters.get('fold_constants', True),
            'remove_unused': parameters.get('remove_unused', True)
        }
        
        # Use existing streamlining application
        return self.existing_transform.apply_streamlining(model, existing_config)

# brainsmith/model_transforms/registry.py
class ExistingTransformRegistry:
    """
    Registry for existing transform implementations.
    
    Provides extensible structure for organizing existing transforms
    from steps/ without adding new transforms.
    """
    
    def __init__(self):
        self._existing_transforms = {}
        self._register_existing_transforms()
    
    def _register_existing_transforms(self):
        """Register all existing transforms in extensible structure."""
        # Register existing streamlining
        from .streamlining import ExistingStreamliningWrapper
        self.register_existing_transform(ExistingStreamliningWrapper())
        
        # Register existing folding optimizations
        from .folding import ExistingFoldingWrapper
        self.register_existing_transform(ExistingFoldingWrapper())
        
        # Register existing partitioning
        from .partitioning import ExistingPartitioningWrapper
        self.register_existing_transform(ExistingPartitioningWrapper())
    
    def register_existing_transform(self, wrapper: ExistingTransformWrapper):
        """Register an existing transform wrapper for extensibility."""
        self._existing_transforms[wrapper.transform_name] = wrapper
    
    def get_existing_transforms_for_blueprint(self, blueprint_config: Dict) -> List[ExistingTransformWrapper]:
        """Get existing transforms specified in blueprint configuration."""
        transforms = []
        pipeline_config = blueprint_config.get('pipeline', [])
        
        for transform_config in pipeline_config:
            if transform_config.get('enabled', True):
                transform_name = transform_config['name']
                if transform_name in self._existing_transforms:
                    transforms.append(self._existing_transforms[transform_name])
        
        return transforms

# Global registry for existing transforms
existing_transform_registry = ExistingTransformRegistry()
```

### 3.5 Enhanced Blueprint System (Structure Focus)

**Purpose**: Support comprehensive library-driven configuration using existing components only.

```yaml
# Example blueprint using existing components only: brainsmith/blueprints/yaml/existing_components_blueprint.yaml
name: "existing_components_structure"
version: "1.0" 
description: "Blueprint demonstrating extensible structure using existing components only"
architecture: "transformer"

# Legacy compatibility
build_steps:
  - "model_conversion"
  - "transform_application"
  - "kernel_mapping"
  - "hardware_generation"

# Kernel Library Configuration (existing components only)
kernels:
  registry: "existing_kernels"
  available:
    - name: "convolution"
      operation_type: "Conv"
      implementations:
        - type: "hw_conv2d_existing"
          source: "custom_op.hw_conv2d"
          parameters:
            simd: {type: "integer", range: [1, 64], default: 1}
            pe: {type: "integer", range: [1, 64], default: 1}
    
    - name: "linear"
      operation_type: "MatMul"
      implementations:
        - type: "hw_linear_existing"
          source: "custom_op.hw_linear"
          parameters:
            simd: {type: "integer", range: [1, 64], default: 1}
            pe: {type: "integer", range: [1, 64], default: 1}

# Model Transform Library Configuration (existing transforms only)
transforms:
  registry: "existing_transforms"
  pipeline:
    - name: "streamlining"
      enabled: true
      searchable: false
      source: "steps.streamlining"
    - name: "folding"
      enabled: true
      searchable: false
      source: "steps.folding"
    - name: "partitioning"
      enabled: true
      searchable: true
      source: "steps.partitioning"
      parameters:
        partition_strategy: {type: "categorical", values: ["dataflow", "layer"], default: "dataflow"}

# Hardware Optimization Configuration (existing strategies only)
hw_optimization:
  registry: "existing_optimization"
  strategies:
    - name: "parameter_optimization"
      algorithm: "bayesian"
      budget: 100
      source: "dse.external.skopt"
    - name: "random_search"
      algorithm: "random"
      budget: 50
      source: "dse.simple.random"
    - name: "genetic_optimization"
      algorithm: "genetic"
      budget: 200
      source: "dse.external.genetic"

# Analysis Configuration (existing tools only)
analysis:
  exit_points: ["roofline", "dataflow_analysis", "dataflow_generation"]
  tools:
    performance_analysis:
      source: "existing_analysis_tools"
      enabled: true
    reporting:
      source: "existing_reporting"
      enabled: true

# FINN Interface Configuration (existing DataflowBuildConfig)
finn_interface:
  # Legacy DataflowBuildConfig support (existing)
  legacy_config:
    auto_fifo_depths: true
    fpga_part: "xcvu9p-flga2104-2-i"
    generate_outputs: ["estimate", "bitfile", "deployment"]
    
  # Future 4-hook interface preparation (placeholder only)
  future_hooks:
    preprocessing_hook:
      enabled: false  # Not available yet
      placeholder: true
    transformation_hook:
      enabled: false
      placeholder: true
    optimization_hook:
      enabled: false
      placeholder: true
    generation_hook:
      enabled: false
      placeholder: true

# Search Strategy Configuration
search_strategy:
  meta_algorithm: "sequential"  # Use existing coordination
  coordination: "existing_coordination"

# Objectives and Constraints
objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0
    metric_path: "performance.throughput_ops_sec"

constraints:
  target_device: "xcvu9p-flga2104-2-i"
  resource_limits:
    lut_utilization: 0.85
    bram_utilization: 0.90
    dsp_utilization: 0.95

# Metadata
metadata:
  version: "1.0"
  components_used: "existing_only"
  extensibility_focus: "structure_over_content"
  note: "Uses only existing components in extensible structure"
```

## 4. Implementation Execution Plan

### 4.1 Phase 1: Core Infrastructure and Workflow (Weeks 1-2)

**Focus**: Runner and core toolflow establishment

#### Week 1: Core Orchestration Foundation
**Deliverables**:
- [ ] `brainsmith/core/design_space_orchestrator.py` - Main orchestration engine
- [ ] `brainsmith/core/workflow.py` - High-level workflow management
- [ ] `brainsmith/core/finn_interface.py` - Legacy + future FINN support
- [ ] Directory structure creation with proper `__init__.py` files

**Key Implementation Details**:
```python
# Priority 1: DesignSpaceOrchestrator with hierarchical exit points
class DesignSpaceOrchestrator:
    def orchestrate_exploration(self, exit_point: str):
        """Core workflow with 3 exit points using existing components"""
        if exit_point == "roofline":
            return self._analyze_existing_model_characteristics()
        elif exit_point == "dataflow_analysis":
            return self._apply_existing_transforms_and_estimate()
        elif exit_point == "dataflow_generation":
            return self._generate_using_existing_finn_flow()

# Priority 2: FINNInterface with legacy support
class FINNInterface:
    def generate_implementation(self, model_path, design_point, target_device):
        """Use existing DataflowBuildConfig + placeholder for 4-hook"""
        if self.use_legacy:  # Always True for now
            return self._generate_with_existing_dataflow_build_config()
        else:
            return self._prepare_for_future_4_hook_interface()
```

#### Week 2: Core API and CLI Integration
**Deliverables**:
- [ ] `brainsmith/core/api.py` - Python API with hierarchical exit points
- [ ] `brainsmith/core/cli.py` - Command-line interface
- [ ] `brainsmith/core/legacy_support.py` - Backward compatibility layer

**Key Implementation Details**:
```python
# Priority 1: Main API functions
def brainsmith_explore(model_path, blueprint_path, exit_point="dataflow_generation"):
    """Main API using existing components in extensible structure"""
    orchestrator = DesignSpaceOrchestrator(blueprint)
    return orchestrator.orchestrate_exploration(exit_point)

def brainsmith_roofline(model_path, blueprint_path):
    """Exit Point 1: Analytical analysis using existing tools"""
    return brainsmith_explore(model_path, blueprint_path, "roofline")

# Priority 2: CLI commands
@click.command()
def explore(model_path, blueprint_path, exit_point):
    """CLI for hierarchical exploration"""
    results, analysis = brainsmith_explore(model_path, blueprint_path, exit_point)
```

### 4.2 Phase 2: Library Structure Implementation (Weeks 3-6)

**Focus**: Extensible structure for existing components

#### Week 3-4: Kernels Library Structure
**Deliverables**:
- [ ] `brainsmith/kernels/base.py` - Base interfaces and wrappers
- [ ] `brainsmith/kernels/registry.py` - Registration system for existing kernels
- [ ] `brainsmith/kernels/conv/hw_custom_op.py` - Wrapper for existing HWConv2D
- [ ] `brainsmith/kernels/linear/hw_custom_op.py` - Wrapper for existing linear ops
- [ ] `brainsmith/kernels/activation/hw_custom_op.py` - Wrapper for existing activations

**Key Implementation Focus**:
- **Structure over content**: Create extensible wrappers around existing custom operations
- **No new kernels**: Only organize existing `custom_op/` components
- **AI layer organization**: Group by operation type, not implementation type

**Implementation Priority**:
```python
# Priority 1: ExistingKernelWrapper base class
class ExistingKernelWrapper:
    def __init__(self, existing_kernel_class, operation_name):
        self.existing_kernel = existing_kernel_class  # Import from custom_op/
        
# Priority 2: Registry for existing kernels
class ExistingKernelRegistry:
    def _register_existing_kernels(self):
        # Register existing HWConv2D, HWLinear, etc.
        from ...custom_op.hw_conv2d import HWConv2D
        self.register_existing_kernel(ConvolutionWrapper(HWConv2D))
```

#### Week 5: Model Transforms Library Structure
**Deliverables**:
- [ ] `brainsmith/model_transforms/base.py` - Base transform interfaces
- [ ] `brainsmith/model_transforms/registry.py` - Registration for existing transforms
- [ ] `brainsmith/model_transforms/streamlining.py` - Wrapper for existing streamlining
- [ ] `brainsmith/model_transforms/folding.py` - Wrapper for existing folding
- [ ] `brainsmith/model_transforms/partitioning.py` - Wrapper for existing partitioning

**Key Implementation Focus**:
- **No quantization exploration**: Transforms cannot change model weights (per feedback)
- **Existing transforms only**: Organize existing transforms from `steps/`
- **Extensible structure**: Create wrappers that enable future extensibility

#### Week 6: Hardware Optimization Library Structure
**Deliverables**:
- [ ] `brainsmith/hw_optim/base.py` - Base optimization interfaces
- [ ] `brainsmith/hw_optim/registry.py` - Registration for existing strategies
- [ ] `brainsmith/hw_optim/strategies/bayesian_opt.py` - Wrapper for existing Bayesian
- [ ] `brainsmith/hw_optim/strategies/genetic_opt.py` - Wrapper for existing genetic
- [ ] `brainsmith/hw_optim/strategies/random_opt.py` - Wrapper for existing random

**Key Implementation Focus**:
- **Existing strategies only**: Organize existing optimization from `dse/`
- **Clean separation**: Distinguish coordination (dse/) from strategies (hw_optim/)

### 4.3 Phase 3: Enhanced Blueprint and Coordination (Weeks 7-8)

#### Week 7: Blueprint System Enhancement
**Deliverables**:
- [ ] `brainsmith/blueprints/base.py` - Enhanced Blueprint class
- [ ] `brainsmith/blueprints/validator.py` - Blueprint validation framework
- [ ] `brainsmith/blueprints/yaml/existing_components_blueprint.yaml` - Example blueprint
- [ ] Enhanced blueprint methods for library configuration

#### Week 8: Coordination Layer Implementation
**Deliverables**:
- [ ] `brainsmith/coordination/orchestration.py` - Cross-library coordination
- [ ] `brainsmith/coordination/workflow.py` - DSE workflow management
- [ ] `brainsmith/coordination/result_aggregation.py` - Multi-library results
- [ ] Integration between orchestrator and libraries

### 4.4 Phase 4: Integration and Legacy Support (Weeks 9-10)

#### Week 9: Legacy Compatibility and Testing
**Deliverables**:
- [ ] `brainsmith/legacy/compatibility.py` - Full backward compatibility
- [ ] `brainsmith/legacy/migration.py` - Migration utilities
- [ ] Comprehensive unit and integration tests
- [ ] Performance regression testing

#### Week 10: Documentation and Finalization
**Deliverables**:
- [ ] Complete API documentation
- [ ] Blueprint configuration reference
- [ ] Migration guide from current system
- [ ] Working examples demonstrating extensible structure

## 5. Expected Outcomes

### 5.1 Structural Benefits

- **Clean Extensible Architecture**: Modular structure ready for future additions
- **Existing Component Organization**: All current functionality organized in extensible manner
- **Clear Extension Points**: Well-defined interfaces for adding new capabilities
- **Legacy Preservation**: 100% backward compatibility maintained

### 5.2 Future Extensibility

- **Plugin Architecture**: Clear patterns for adding new kernels, transforms, optimizers
- **Library Independence**: Each library can be extended without affecting others
- **Blueprint-Driven**: Declarative configuration enables easy experimentation
- **4-Hook Ready**: Structured transition path for future FINN interface

This final design provides a comprehensive transformation focused on creating high-quality extensible structure using existing components only, with clear implementation priorities and execution timeline.