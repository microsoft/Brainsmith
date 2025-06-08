# Phase 4 Architectural Specification: Complete Implementation Guide

## Overview

This document provides the detailed architectural specification for implementing Phase 4 of Brainsmith, which will align the current implementation with the high-level vision defined in `docs/brainsmith-high-level.md`. This specification serves as a blueprint for developers to implement the modular library architecture.

## Core Architectural Changes

### 1. Library Structure Implementation

#### 1.1 Hardware Kernels Library Structure

**Directory**: `brainsmith/kernels/`

**File Structure**:
```
brainsmith/kernels/
├── __init__.py              # Public API and kernel registry
├── registry.py              # Kernel registration and discovery system
├── base.py                  # Base kernel interfaces and abstract classes
├── rtl/                     # RTL kernel implementations
│   ├── __init__.py
│   ├── conv_rtl.py         # Convolution RTL kernels
│   ├── relu_rtl.py         # ReLU RTL kernels
│   ├── linear_rtl.py       # Linear layer RTL kernels
│   └── norm_rtl.py         # Normalization RTL kernels
├── hls/                     # HLS kernel implementations
│   ├── __init__.py
│   ├── conv_hls.py         # Convolution HLS kernels
│   ├── relu_hls.py         # ReLU HLS kernels
│   ├── linear_hls.py       # Linear layer HLS kernels
│   └── norm_hls.py         # Normalization HLS kernels
└── ops/                     # ONNX custom operations
    ├── __init__.py
    ├── hardware_conv.py    # Hardware convolution ONNX op
    ├── hardware_relu.py    # Hardware ReLU ONNX op
    ├── hardware_linear.py  # Hardware linear ONNX op
    └── hardware_norm.py    # Hardware normalization ONNX op
```

**Key Classes to Implement**:

```python
# brainsmith/kernels/base.py
class KernelInterface(ABC):
    """Base interface for all hardware kernels."""
    
    def __init__(self, name: str, implementation_type: str):
        self.name = name
        self.implementation_type = implementation_type  # 'rtl', 'hls', 'onnx'
        self.parameters = {}
        self.constraints = {}
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return parameter space for this kernel."""
        pass
    
    @abstractmethod
    def generate_code(self, parameters: Dict[str, Any]) -> str:
        """Generate kernel code with given parameters."""
        pass
    
    @abstractmethod
    def get_resource_estimate(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resource usage for given parameters."""
        pass

class RTLKernel(KernelInterface):
    """Base class for RTL kernels."""
    
    def __init__(self, name: str):
        super().__init__(name, 'rtl')

class HLSKernel(KernelInterface):
    """Base class for HLS kernels."""
    
    def __init__(self, name: str):
        super().__init__(name, 'hls')

class ONNXCustomOp(KernelInterface):
    """Base class for ONNX custom operations."""
    
    def __init__(self, name: str):
        super().__init__(name, 'onnx')

# brainsmith/kernels/registry.py
class KernelRegistry:
    """Central registry for all available kernels."""
    
    def __init__(self):
        self._kernels = {}
        self._implementations = {}
    
    def register_kernel(self, kernel: KernelInterface):
        """Register a kernel implementation."""
        key = f"{kernel.name}_{kernel.implementation_type}"
        self._kernels[key] = kernel
        
        if kernel.name not in self._implementations:
            self._implementations[kernel.name] = []
        self._implementations[kernel.name].append(kernel.implementation_type)
    
    def get_kernel(self, name: str, implementation_type: str) -> Optional[KernelInterface]:
        """Get specific kernel implementation."""
        key = f"{name}_{implementation_type}"
        return self._kernels.get(key)
    
    def list_kernels(self) -> Dict[str, List[str]]:
        """List all available kernels and their implementations."""
        return self._implementations
    
    def get_kernels_for_blueprint(self, blueprint_config: Dict) -> List[KernelInterface]:
        """Get kernels specified in blueprint configuration."""
        kernels = []
        for kernel_config in blueprint_config.get('available', []):
            kernel_name = kernel_config['name']
            for impl_type in kernel_config.get('implementations', ['hls']):
                kernel = self.get_kernel(kernel_name, impl_type)
                if kernel:
                    kernels.append(kernel)
        return kernels

# Global registry instance
kernel_registry = KernelRegistry()
```

#### 1.2 Model Transforms Library Structure

**Directory**: `brainsmith/model_transforms/`

**File Structure**:
```
brainsmith/model_transforms/
├── __init__.py              # Public API and transform registry
├── registry.py              # Transform registration system
├── base.py                  # Base transform interfaces
├── fusions.py               # Layer fusion transforms
├── streamlining.py          # Graph streamlining transforms
├── layout.py                # Layout optimization transforms
├── quantization.py          # Quantization transforms
├── partitioning.py          # Graph partitioning transforms
└── search/                  # Meta-search strategies
    ├── __init__.py
    ├── bayesian.py         # Bayesian transform sequence optimization
    ├── evolutionary.py     # Evolutionary transform optimization
    ├── hierarchical.py     # Hierarchical transform search
    └── greedy.py          # Greedy transform selection
```

**Key Classes to Implement**:

```python
# brainsmith/model_transforms/base.py
class TransformInterface(ABC):
    """Base interface for model transformations."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}
        self.prerequisites = []
        self.conflicts = []
    
    @abstractmethod
    def can_apply(self, model: Any) -> bool:
        """Check if transform can be applied to model."""
        pass
    
    @abstractmethod
    def apply(self, model: Any, parameters: Dict[str, Any] = None) -> Any:
        """Apply transformation to model."""
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for this transform."""
        pass
    
    def estimate_impact(self, model: Any) -> Dict[str, float]:
        """Estimate performance impact of applying this transform."""
        return {'estimated_improvement': 0.0}

class TransformPipeline:
    """Manages sequence of transformations."""
    
    def __init__(self, transforms: List[TransformInterface]):
        self.transforms = transforms
        self.applied_transforms = []
    
    def validate_pipeline(self) -> bool:
        """Validate that transform sequence is valid."""
        # Check prerequisites and conflicts
        pass
    
    def apply_pipeline(self, model: Any, parameters: Dict[str, Dict] = None) -> Any:
        """Apply complete transform pipeline."""
        pass
    
    def optimize_sequence(self, model: Any, strategy: str = 'greedy') -> List[TransformInterface]:
        """Optimize transform sequence for given model."""
        pass

# brainsmith/model_transforms/registry.py
class TransformRegistry:
    """Registry for model transformations."""
    
    def __init__(self):
        self._transforms = {}
        self._categories = {}
    
    def register_transform(self, transform: TransformInterface, category: str = 'general'):
        """Register a transformation."""
        self._transforms[transform.name] = transform
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(transform.name)
    
    def get_transform(self, name: str) -> Optional[TransformInterface]:
        """Get transform by name."""
        return self._transforms.get(name)
    
    def get_transforms_for_blueprint(self, blueprint_config: Dict) -> TransformPipeline:
        """Create transform pipeline from blueprint configuration."""
        pipeline_config = blueprint_config.get('pipeline', [])
        transforms = []
        
        for transform_config in pipeline_config:
            if transform_config.get('enabled', True):
                transform = self.get_transform(transform_config['name'])
                if transform:
                    transforms.append(transform)
        
        return TransformPipeline(transforms)

# Global registry instance
transform_registry = TransformRegistry()
```

#### 1.3 Hardware Optimization Library Structure

**Directory**: `brainsmith/hw_optim/`

**File Structure**:
```
brainsmith/hw_optim/
├── __init__.py              # Public API and optimization registry
├── registry.py              # Optimization strategy registration
├── base.py                  # Base optimization interfaces
├── param_opt.py             # Parameter optimization strategies
├── impl_styles.py           # Implementation style optimization
├── scheduling.py            # Global scheduling optimization
├── resource_allocation.py   # Resource allocation strategies
├── memory_opt.py            # Memory optimization strategies
└── strategies/              # Specific optimization algorithms
    ├── __init__.py
    ├── bayesian_opt.py     # Bayesian optimization
    ├── genetic_opt.py      # Genetic algorithms
    ├── adaptive_opt.py     # Adaptive optimization
    ├── simulated_annealing.py # Simulated annealing
    └── multi_objective.py  # Multi-objective optimization
```

**Key Classes to Implement**:

```python
# brainsmith/hw_optim/base.py
class OptimizationStrategy(ABC):
    """Base class for hardware optimization strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}
        self.supports_multi_objective = False
        self.supports_constraints = False
    
    @abstractmethod
    def optimize(self, design_space: Any, objectives: List[str], 
                constraints: Dict = None, budget: int = 100) -> Any:
        """Execute optimization strategy."""
        pass
    
    @abstractmethod
    def suggest_next_points(self, n_points: int = 1) -> List[Dict]:
        """Suggest next points to evaluate."""
        pass
    
    @abstractmethod
    def update_with_result(self, point: Dict, result: Any):
        """Update strategy with evaluation result."""
        pass

class ParameterOptimizer(OptimizationStrategy):
    """Optimizes kernel parameters."""
    
    def __init__(self, algorithm: str = 'bayesian'):
        super().__init__(f'param_opt_{algorithm}', f'Parameter optimization using {algorithm}')
        self.algorithm = algorithm

class ImplementationOptimizer(OptimizationStrategy):
    """Optimizes implementation choices (RTL vs HLS, etc.)."""
    
    def __init__(self, algorithm: str = 'genetic'):
        super().__init__(f'impl_opt_{algorithm}', f'Implementation optimization using {algorithm}')
        self.algorithm = algorithm

class GlobalOptimizer(OptimizationStrategy):
    """Performs global cross-layer optimization."""
    
    def __init__(self, algorithm: str = 'hierarchical'):
        super().__init__(f'global_opt_{algorithm}', f'Global optimization using {algorithm}')
        self.algorithm = algorithm
        self.supports_multi_objective = True
        self.supports_constraints = True

# brainsmith/hw_optim/registry.py
class OptimizationRegistry:
    """Registry for optimization strategies."""
    
    def __init__(self):
        self._strategies = {}
        self._algorithms = {}
    
    def register_strategy(self, strategy: OptimizationStrategy):
        """Register optimization strategy."""
        self._strategies[strategy.name] = strategy
        if strategy.name not in self._algorithms:
            self._algorithms[strategy.name] = []
    
    def get_strategies_for_blueprint(self, blueprint_config: Dict) -> List[OptimizationStrategy]:
        """Get optimization strategies from blueprint configuration."""
        strategies = []
        strategy_configs = blueprint_config.get('strategies', [])
        
        for config in strategy_configs:
            strategy_name = config.get('name')
            algorithm = config.get('algorithm', 'bayesian')
            
            if strategy_name == 'parameter_optimization':
                strategies.append(ParameterOptimizer(algorithm))
            elif strategy_name == 'implementation_optimization':
                strategies.append(ImplementationOptimizer(algorithm))
            elif strategy_name == 'global_optimization':
                strategies.append(GlobalOptimizer(algorithm))
        
        return strategies

# Global registry instance
optimization_registry = OptimizationRegistry()
```

#### 1.4 Analysis Library Structure

**Directory**: `brainsmith/analysis/`

**File Structure**:
```
brainsmith/analysis/
├── __init__.py              # Public API
├── roofline.py              # Roofline analysis implementation
├── performance.py           # Performance modeling and prediction
├── reporting.py             # Report generation and formatting
├── visualization.py         # Visualization tools and plots
├── export.py                # Export utilities for various formats
├── metrics.py               # Metrics collection and aggregation
└── comparison.py            # Design point comparison utilities
```

**Key Classes to Implement**:

```python
# brainsmith/analysis/roofline.py
class RooflineAnalyzer:
    """Implements roofline analysis for FPGA accelerators."""
    
    def __init__(self, target_device: str):
        self.target_device = target_device
        self.device_specs = self._load_device_specs(target_device)
    
    def analyze_model(self, model_path: str, blueprint: Any) -> Dict:
        """Perform roofline analysis on model."""
        # Analyze computational intensity
        # Estimate memory bandwidth requirements
        # Calculate theoretical performance bounds
        pass
    
    def generate_roofline_plot(self, analysis_results: Dict) -> Any:
        """Generate roofline visualization."""
        pass

# brainsmith/analysis/performance.py
class PerformanceModeler:
    """Models and predicts accelerator performance."""
    
    def __init__(self, target_device: str):
        self.target_device = target_device
        self.performance_models = {}
    
    def model_dataflow_performance(self, design_point: Dict, 
                                 kernel_implementations: List) -> Dict:
        """Model performance at dataflow level."""
        # Estimate throughput, latency, resource usage
        # Consider pipeline stalls, memory bottlenecks
        pass
    
    def predict_synthesis_results(self, design_point: Dict) -> Dict:
        """Predict synthesis results without running synthesis."""
        # Use ML models or analytical models
        pass

# brainsmith/analysis/reporting.py
class AnalysisReporter:
    """Generates comprehensive analysis reports."""
    
    def __init__(self, output_format: str = 'html'):
        self.output_format = output_format
        self.template_engine = None
    
    def generate_exploration_report(self, dse_results: Any, 
                                  analysis_results: Dict) -> str:
        """Generate complete exploration report."""
        # Include design space summary
        # Show optimization progress
        # Display Pareto frontiers
        # Include recommendations
        pass
    
    def generate_comparison_report(self, design_points: List[Dict]) -> str:
        """Generate design point comparison report."""
        pass
```

### 2. Meta-DSE Engine Implementation

#### 2.1 Core Meta-DSE Engine

**File**: `brainsmith/core/dse_engine.py`

**Implementation Specification**:

```python
# brainsmith/core/dse_engine.py
class MetaDSEEngine:
    """
    Meta-aware DSE engine that orchestrates all libraries according
    to the architectural vision.
    """
    
    def __init__(self, blueprint: Blueprint):
        self.blueprint = blueprint
        self.libraries = self._initialize_libraries()
        self.design_space = None
        self.current_exit_point = None
        self.results_cache = {}
    
    def _initialize_libraries(self) -> Dict[str, Any]:
        """Initialize all libraries based on blueprint configuration."""
        from ..kernels import kernel_registry
        from ..model_transforms import transform_registry  
        from ..hw_optim import optimization_registry
        from ..analysis import AnalysisLibrary
        
        library_configs = self.blueprint.get_library_configs()
        
        return {
            'kernels': KernelLibrary(
                kernels=kernel_registry.get_kernels_for_blueprint(
                    library_configs.get('kernels', {})
                )
            ),
            'transforms': TransformLibrary(
                pipeline=transform_registry.get_transforms_for_blueprint(
                    library_configs.get('transforms', {})
                )
            ),
            'hw_optim': HardwareOptimLibrary(
                strategies=optimization_registry.get_strategies_for_blueprint(
                    library_configs.get('hw_optimization', {})
                )
            ),
            'analysis': AnalysisLibrary(
                config=library_configs.get('analysis', {})
            )
        }
    
    def construct_design_space(self) -> DesignSpace:
        """
        Construct unified design space from all libraries.
        This implements the Cartesian product concept from the vision.
        """
        if self.design_space is not None:
            return self.design_space
        
        # Get design spaces from each library
        kernel_space = self.libraries['kernels'].get_design_space()
        transform_space = self.libraries['transforms'].get_design_space()
        optimization_space = self.libraries['hw_optim'].get_design_space()
        
        # Combine into unified design space
        self.design_space = DesignSpace.combine([
            kernel_space,
            transform_space, 
            optimization_space
        ])
        
        # Apply blueprint constraints
        constraints = self.blueprint.get_constraints()
        self.design_space.apply_constraints(constraints)
        
        return self.design_space
    
    def explore_design_space(self, exit_point: str = "dataflow_generation") -> DSEResult:
        """
        Execute design space exploration with specified exit point.
        Implements the hierarchical exit points from the vision.
        """
        self.current_exit_point = exit_point
        
        # Construct design space
        design_space = self.construct_design_space()
        
        # Select meta-search strategy
        search_config = self.blueprint.get_search_strategy_config()
        meta_algorithm = search_config.get('meta_algorithm', 'hierarchical')
        
        if meta_algorithm == 'hierarchical':
            return self._hierarchical_exploration(design_space, exit_point)
        elif meta_algorithm == 'parallel':
            return self._parallel_exploration(design_space, exit_point)
        else:
            return self._sequential_exploration(design_space, exit_point)
    
    def _hierarchical_exploration(self, design_space: DesignSpace, 
                                exit_point: str) -> DSEResult:
        """Implement hierarchical exploration strategy."""
        results = DSEResult()
        
        # Stage 1: Transform-level optimization
        if exit_point in ['roofline', 'dataflow_analysis', 'dataflow_generation']:
            transform_results = self._optimize_transforms(design_space)
            results.add_stage_results('transforms', transform_results)
        
        # Stage 2: Kernel-level optimization  
        if exit_point in ['dataflow_analysis', 'dataflow_generation']:
            kernel_results = self._optimize_kernels(design_space, transform_results)
            results.add_stage_results('kernels', kernel_results)
        
        # Stage 3: Global hardware optimization
        if exit_point == 'dataflow_generation':
            hw_results = self._optimize_hardware(design_space, kernel_results)
            results.add_stage_results('hardware', hw_results)
        
        return results
    
    def _roofline_analysis(self, design_space: DesignSpace) -> Dict:
        """
        Exit Point 1: Analytical model-only profiling.
        Implements roofline analysis without hardware generation.
        """
        analyzer = self.libraries['analysis'].get_roofline_analyzer()
        
        # Analyze model computational characteristics
        model_analysis = analyzer.analyze_model(
            self.blueprint.model_path,
            self.blueprint
        )
        
        # Estimate performance bounds for design space
        performance_bounds = {}
        for point in design_space.sample(n_samples=100):
            bounds = analyzer.estimate_bounds(point)
            performance_bounds[str(point)] = bounds
        
        return {
            'model_analysis': model_analysis,
            'performance_bounds': performance_bounds,
            'roofline_plot': analyzer.generate_roofline_plot(model_analysis)
        }
    
    def _dataflow_analysis(self, design_space: DesignSpace) -> Dict:
        """
        Exit Point 2: Hardware-abstracted ONNX lowering and performance estimation.
        Implements dataflow-level analysis without RTL generation.
        """
        # Apply model transforms
        transform_results = self._optimize_transforms(design_space)
        best_transforms = transform_results['best_pipeline']
        
        # Apply transforms to model
        transformed_model = self.libraries['transforms'].apply_pipeline(
            self.blueprint.model_path, 
            best_transforms
        )
        
        # Map to hardware kernels (abstractly)
        kernel_mapping = self.libraries['kernels'].map_model_to_kernels(
            transformed_model
        )
        
        # Estimate performance without RTL generation
        performance_estimator = self.libraries['analysis'].get_performance_modeler()
        performance_estimates = {}
        
        for point in design_space.sample(n_samples=50):
            estimate = performance_estimator.model_dataflow_performance(
                point, kernel_mapping
            )
            performance_estimates[str(point)] = estimate
        
        return {
            'transformed_model': transformed_model,
            'kernel_mapping': kernel_mapping,
            'performance_estimates': performance_estimates,
            'recommended_points': self._select_pareto_optimal(performance_estimates)
        }
    
    def _dataflow_generation(self, design_space: DesignSpace) -> Dict:
        """
        Exit Point 3: Complete stitched, parameterized RTL or HLS IP generation.
        Implements full RTL/HLS generation workflow.
        """
        # Execute complete optimization pipeline
        transform_results = self._optimize_transforms(design_space)
        kernel_results = self._optimize_kernels(design_space, transform_results)
        hw_results = self._optimize_hardware(design_space, kernel_results)
        
        # Generate final RTL/HLS
        best_design_point = hw_results['best_point']
        
        # Apply transforms
        transformed_model = self.libraries['transforms'].apply_pipeline(
            self.blueprint.model_path,
            best_design_point['transforms']
        )
        
        # Generate kernel implementations
        kernel_implementations = self.libraries['kernels'].generate_implementations(
            transformed_model,
            best_design_point['kernels']
        )
        
        # Generate system integration
        system_rtl = self.libraries['kernels'].generate_system_integration(
            kernel_implementations,
            best_design_point['system']
        )
        
        return {
            'transformed_model': transformed_model,
            'kernel_implementations': kernel_implementations,
            'system_rtl': system_rtl,
            'design_point': best_design_point,
            'performance_analysis': hw_results['performance_analysis']
        }

class KernelLibrary:
    """Wrapper for kernel registry with design space integration."""
    
    def __init__(self, kernels: List[KernelInterface]):
        self.kernels = kernels
    
    def get_design_space(self) -> DesignSpace:
        """Get combined design space from all kernels."""
        pass
    
    def map_model_to_kernels(self, model: Any) -> Dict:
        """Map model operations to available kernels."""
        pass
    
    def generate_implementations(self, model: Any, kernel_params: Dict) -> Dict:
        """Generate kernel implementations with specified parameters."""
        pass

class TransformLibrary:
    """Wrapper for transform registry with pipeline management."""
    
    def __init__(self, pipeline: TransformPipeline):
        self.pipeline = pipeline
    
    def get_design_space(self) -> DesignSpace:
        """Get design space for transform parameters."""
        pass
    
    def apply_pipeline(self, model_path: str, transform_params: Dict) -> Any:
        """Apply transform pipeline with specified parameters."""
        pass

class HardwareOptimLibrary:
    """Wrapper for optimization registry with strategy coordination."""
    
    def __init__(self, strategies: List[OptimizationStrategy]):
        self.strategies = strategies
    
    def get_design_space(self) -> DesignSpace:
        """Get design space for optimization parameters."""
        pass
    
    def optimize(self, design_space: DesignSpace, objectives: List[str]) -> Dict:
        """Execute optimization strategies."""
        pass

class AnalysisLibrary:
    """Analysis library with comprehensive analysis capabilities."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.roofline_analyzer = None
        self.performance_modeler = None
        self.reporter = None
    
    def get_roofline_analyzer(self) -> RooflineAnalyzer:
        """Get roofline analyzer instance."""
        if self.roofline_analyzer is None:
            target_device = self.config.get('target_device', 'xcvu9p')
            self.roofline_analyzer = RooflineAnalyzer(target_device)
        return self.roofline_analyzer
    
    def get_performance_modeler(self) -> PerformanceModeler:
        """Get performance modeler instance."""
        if self.performance_modeler is None:
            target_device = self.config.get('target_device', 'xcvu9p')
            self.performance_modeler = PerformanceModeler(target_device)
        return self.performance_modeler
    
    def analyze_results(self, dse_results: DSEResult) -> Dict:
        """Generate comprehensive analysis of DSE results."""
        pass
```

### 3. Enhanced Blueprint System

#### 3.1 Blueprint Enhancement Specification

**File**: `brainsmith/blueprints/base.py` (enhancements)

**Additional Methods to Add**:

```python
class Blueprint:
    # Existing implementation...
    
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
        return self.yaml_data.get('search_strategy', {
            'meta_algorithm': 'hierarchical',
            'coordination': 'sequential'
        })
    
    def get_exit_points(self) -> List[str]:
        """Get configured exit points for exploration."""
        analysis_config = self.yaml_data.get('analysis', {})
        return analysis_config.get('exit_points', ['dataflow_generation'])
    
    def get_objectives_config(self) -> List[Dict]:
        """Get objectives configuration."""
        return self.yaml_data.get('objectives', [
            {'name': 'throughput', 'direction': 'maximize', 'weight': 1.0}
        ])
    
    def get_constraints_config(self) -> Dict:
        """Get constraints configuration."""
        return self.yaml_data.get('constraints', {})
    
    def supports_library_driven_dse(self) -> bool:
        """Check if blueprint supports full library-driven DSE."""
        required_sections = ['kernels', 'transforms', 'hw_optimization']
        return all(section in self.yaml_data for section in required_sections)
    
    def validate_library_config(self) -> Tuple[bool, List[str]]:
        """Validate blueprint library configuration."""
        errors = []
        
        # Validate kernel configuration
        kernels_config = self.yaml_data.get('kernels', {})
        if 'available' not in kernels_config:
            errors.append("Missing 'available' section in kernels configuration")
        
        # Validate transforms configuration
        transforms_config = self.yaml_data.get('transforms', {})
        if 'pipeline' not in transforms_config:
            errors.append("Missing 'pipeline' section in transforms configuration")
        
        # Validate optimization configuration
        hw_opt_config = self.yaml_data.get('hw_optimization', {})
        if 'strategies' not in hw_opt_config:
            errors.append("Missing 'strategies' section in hw_optimization configuration")
        
        return len(errors) == 0, errors
```

#### 3.2 Enhanced Blueprint YAML Schema

**Example Enhanced Blueprint**: `brainsmith/blueprints/yaml/transformer_architectural.yaml`

```yaml
name: "transformer_architectural"
version: "1.0"
description: "Transformer blueprint implementing full architectural vision"
architecture: "transformer"

# Build steps (backward compatibility)
build_steps:
  - "model_conversion"
  - "transform_application"
  - "kernel_mapping"
  - "hardware_generation"

# Kernel Library Configuration
kernels:
  registry: "transformer_kernels"
  available:
    - name: "quantized_linear"
      implementations: ["hls", "rtl"]
      parameters:
        parallelism:
          type: "integer"
          range: [1, 16]
          default: 8
          description: "Parallel processing elements"
        quantization:
          type: "categorical"
          values: ["int4", "int8", "int16"]
          default: "int8"
          description: "Quantization bit width"
        tiling_factor:
          type: "integer"
          range: [1, 32]
          default: 4
          description: "Memory tiling factor"
      constraints:
        resource_limit: "lut < 1000"
        timing_constraint: "clock_period > 3ns"
    
    - name: "layer_norm"
      implementations: ["hls"]
      parameters:
        precision:
          type: "categorical"
          values: ["float16", "bfloat16"]
          default: "float16"
        pipeline_depth:
          type: "integer"
          range: [1, 8]
          default: 2
      constraints:
        resource_limit: "dsp < 100"
    
    - name: "attention_mechanism"
      implementations: ["hls", "rtl"]
      parameters:
        num_heads:
          type: "integer"
          range: [1, 16]
          default: 8
        head_dimension:
          type: "integer"
          range: [32, 128]
          default: 64
        softmax_implementation:
          type: "categorical"
          values: ["lut", "cordic", "piecewise"]
          default: "lut"

# Model Transform Library Configuration
transforms:
  registry: "transformer_transforms"
  pipeline:
    - name: "fuse_layernorm"
      enabled: true
      searchable: false
      description: "Fuse layer normalization with adjacent operations"
    
    - name: "streamline_graph"
      enabled: true
      searchable: false
      description: "Streamline computation graph"
    
    - name: "optimize_attention"
      enabled: true
      searchable: true
      parameters:
        attention_fusion:
          type: "categorical"
          values: ["none", "qkv_fusion", "full_fusion"]
          default: "qkv_fusion"
        sequence_tiling:
          type: "integer"
          range: [1, 512]
          default: 64
      description: "Optimize attention mechanism implementation"
    
    - name: "quantization_aware_optimization"
      enabled: true
      searchable: true
      parameters:
        quantization_strategy:
          type: "categorical"
          values: ["uniform", "per_channel", "mixed_precision"]
          default: "per_channel"
      description: "Apply quantization-aware optimizations"

# Hardware Optimization Configuration
hw_optimization:
  registry: "fpga_optimization"
  strategies:
    - name: "parameter_optimization"
      algorithm: "bayesian"
      budget: 100
      parameters:
        acquisition_function:
          type: "categorical"
          values: ["EI", "UCB", "PI"]
          default: "EI"
        initial_samples: 10
      description: "Optimize kernel parameters using Bayesian optimization"
    
    - name: "implementation_optimization"
      algorithm: "genetic"
      budget: 50
      parameters:
        population_size:
          type: "integer"
          range: [20, 100]
          default: 50
        mutation_rate:
          type: "continuous"
          range: [0.01, 0.1]
          default: 0.05
        crossover_rate:
          type: "continuous"
          range: [0.5, 0.9]
          default: 0.7
      description: "Optimize implementation choices using genetic algorithm"
    
    - name: "global_optimization"
      algorithm: "hierarchical"
      budget: 200
      parameters:
        levels: 3
        coordination: "parallel"
      description: "Global cross-layer optimization"
  
  global_settings:
    resource_allocation: "enabled"
    scheduling_optimization: "enabled"
    memory_optimization: "enabled"

# Analysis Configuration
analysis:
  exit_points: ["roofline", "dataflow_analysis", "dataflow_generation"]
  metrics:
    primary:
      - "throughput"
      - "latency"
      - "resource_utilization"
    secondary:
      - "power_consumption"
      - "memory_bandwidth"
      - "energy_efficiency"
  reporting:
    formats: ["json", "yaml", "html", "pdf"]
    visualization: true
    comparison: true
  roofline:
    enabled: true
    compute_intensity_analysis: true
    memory_bandwidth_analysis: true
  performance_modeling:
    enabled: true
    analytical_models: true
    ml_models: false

# Search Strategy Configuration
search_strategy:
  meta_algorithm: "hierarchical"
  coordination: "parallel"
  early_stopping:
    enabled: true
    patience: 10
    min_improvement: 0.01
  checkpointing:
    enabled: true
    frequency: 10

# Objectives and Constraints
objectives:
  - name: "throughput"
    direction: "maximize"
    weight: 1.0
    metric_path: "performance.throughput_ops_sec"
    constraint:
      min: 1000
  
  - name: "resource_utilization"  
    direction: "minimize"
    weight: 0.5
    metric_path: "hardware.resource_utilization"
    constraint:
      max: 0.8
  
  - name: "latency"
    direction: "minimize"
    weight: 0.7
    metric_path: "performance.latency_ms"
    constraint:
      max: 100

constraints:
  target_device: "xcvu9p"
  clock_frequency: "300MHz"
  resource_limits:
    lut_utilization: 0.85
    bram_utilization: 0.90
    dsp_utilization: 0.95
  timing_constraints:
    max_clock_period: "3.33ns"
    setup_margin: "0.5ns"
  power_constraints:
    max_power: "25W"

# Metadata
metadata:
  version: "1.0"
  author: "Brainsmith"
  created: "2025-01-01"
  target_models: ["BERT", "GPT", "T5"]
  validated_devices: ["xcvu9p", "xczu7ev"]
  
# Research Configuration
research_config:
  dse_enabled: true
  experiment_tracking: true
  reproducibility:
    seed: 42
    deterministic: true
  publication_ready: true
```

### 4. Interface Implementation

#### 4.1 Enhanced Python API

**File**: `brainsmith/interfaces/api.py`

```python
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from ..blueprints.base import Blueprint
from ..core.dse_engine import MetaDSEEngine
from ..core.result import DSEResult

def brainsmith_explore(model_path: str, 
                      blueprint_path: str,
                      exit_point: str = "dataflow_generation",
                      output_dir: Optional[str] = None,
                      **kwargs) -> Tuple[DSEResult, Dict[str, Any]]:
    """
    Main exploration API implementing the architectural vision.
    
    This function implements the complete workflow described in the high-level
    architecture: Design Space Construction → DSE Engine → Exit Points.
    
    Args:
        model_path: Path to quantized ONNX model
        blueprint_path: Path to blueprint YAML file
        exit_point: Analysis exit point ('roofline', 'dataflow_analysis', 'dataflow_generation')
        output_dir: Optional output directory for results
        **kwargs: Additional configuration options
    
    Returns:
        Tuple of (DSE results, comprehensive analysis)
    
    Raises:
        ValueError: If blueprint is invalid or exit_point is unsupported
        FileNotFoundError: If model or blueprint files don't exist
    """
    # Validate inputs
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    valid_exit_points = ["roofline", "dataflow_analysis", "dataflow_generation"]
    if exit_point not in valid_exit_points:
        raise ValueError(f"Invalid exit point: {exit_point}. Must be one of {valid_exit_points}")
    
    # Load and validate blueprint
    blueprint = Blueprint.from_yaml_file(Path(blueprint_path))
    is_valid, errors = blueprint.validate_library_config()
    if not is_valid:
        raise ValueError(f"Blueprint validation failed: {'; '.join(errors)}")
    
    # Store model path in blueprint for engine access
    blueprint.model_path = model_path
    
    # Create meta-DSE engine
    meta_engine = MetaDSEEngine(blueprint)
    
    # Execute exploration with specified exit point
    logger.info(f"Starting exploration with exit point: {exit_point}")
    results = meta_engine.explore_design_space(exit_point)
    
    # Generate comprehensive analysis
    logger.info("Generating analysis...")
    analysis = meta_engine.libraries['analysis'].analyze_results(results)
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results and analysis
        results.save(output_path / "dse_results.json")
        
        with open(output_path / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate reports
        reporter = meta_engine.libraries['analysis'].get_reporter()
        report = reporter.generate_exploration_report(results, analysis)
        
        with open(output_path / "report.html", 'w') as f:
            f.write(report)
    
    return results, analysis

def brainsmith_roofline(model_path: str, blueprint_path: str, 
                       output_dir: Optional[str] = None) -> Tuple[DSEResult, Dict[str, Any]]:
    """
    Perform roofline analysis (Exit Point 1).
    
    Quick analytical model-only profiling without hardware generation.
    """
    return brainsmith_explore(model_path, blueprint_path, "roofline", output_dir)

def brainsmith_dataflow_analysis(model_path: str, blueprint_path: str,
                                output_dir: Optional[str] = None) -> Tuple[DSEResult, Dict[str, Any]]:
    """
    Perform dataflow-level analysis (Exit Point 2).
    
    Hardware-abstracted ONNX lowering and performance estimation.
    """
    return brainsmith_explore(model_path, blueprint_path, "dataflow_analysis", output_dir)

def brainsmith_generate(model_path: str, blueprint_path: str,
                       output_dir: Optional[str] = None) -> Tuple[DSEResult, Dict[str, Any]]:
    """
    Complete RTL/HLS generation (Exit Point 3).
    
    Full dataflow core generation with stitched, parameterized RTL or HLS IP.
    """
    return brainsmith_explore(model_path, blueprint_path, "dataflow_generation", output_dir)

# Convenience functions for blueprint management
def validate_blueprint(blueprint_path: str) -> Tuple[bool, List[str]]:
    """Validate blueprint configuration."""
    try:
        blueprint = Blueprint.from_yaml_file(Path(blueprint_path))
        return blueprint.validate_library_config()
    except Exception as e:
        return False, [str(e)]

def list_available_kernels() -> Dict[str, List[str]]:
    """List all available hardware kernels."""
    from ..kernels import kernel_registry
    return kernel_registry.list_kernels()

def list_available_transforms() -> Dict[str, List[str]]:
    """List all available model transforms."""
    from ..model_transforms import transform_registry
    return transform_registry.list_transforms()

def list_available_optimizers() -> Dict[str, List[str]]:
    """List all available optimization strategies."""
    from ..hw_optim import optimization_registry
    return optimization_registry.list_strategies()

# Backward compatibility layer
def explore_design_space(model_path: str, blueprint_name: str, **kwargs):
    """Backward compatibility wrapper for existing API."""
    # Check if blueprint_name is a path or name
    if Path(blueprint_name).exists():
        # It's a path to blueprint file
        return brainsmith_explore(model_path, blueprint_name, **kwargs)
    else:
        # It's a blueprint name - use legacy system
        from ..legacy_api import legacy_explore_design_space
        return legacy_explore_design_space(model_path, blueprint_name, **kwargs)
```

#### 4.2 CLI Interface

**File**: `brainsmith/interfaces/cli.py`

```python
import click
import json
from pathlib import Path
from typing import Optional
from .api import (
    brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
    brainsmith_generate, validate_blueprint, list_available_kernels,
    list_available_transforms, list_available_optimizers
)

@click.group()
@click.version_option(version="0.4.0", prog_name="brainsmith")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def brainsmith(ctx, verbose):
    """
    Brainsmith: Meta-toolchain for FPGA accelerator synthesis.
    
    A comprehensive platform implementing the modular library architecture
    for neural network accelerator design space exploration with hierarchical
    analysis capabilities.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--exit-point', '-e', 
              type=click.Choice(['roofline', 'dataflow_analysis', 'dataflow_generation']),
              default='dataflow_generation',
              help='Analysis exit point for hierarchical exploration')
@click.option('--output', '-o', type=click.Path(), 
              help='Output directory for results and reports')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'yaml', 'html']),
              default='html',
              help='Output format for reports')
@click.pass_context
def explore(ctx, model_path, blueprint_path, exit_point, output, output_format):
    """
    Explore design space with specified exit point.
    
    Implements the complete architectural workflow:
    Input → Design Space Construction → DSE Engine → Exit Point Analysis
    
    Examples:
        brainsmith explore model.onnx blueprint.yaml --exit-point roofline
        brainsmith explore model.onnx blueprint.yaml -e dataflow_analysis -o results/
        brainsmith explore model.onnx blueprint.yaml -o results/ --format json
    """
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"📊 Exploring {model_path} with {blueprint_path}")
        click.echo(f"🎯 Exit point: {exit_point}")
        if output:
            click.echo(f"📁 Output directory: {output}")
    
    try:
        # Execute exploration
        with click.progressbar(
            length=100, 
            label=f'Executing {exit_point} analysis'
        ) as bar:
            results, analysis = brainsmith_explore(
                model_path, blueprint_path, exit_point, output
            )
            bar.update(100)
        
        # Display summary
        click.echo("\n🎉 Exploration complete!")
        
        if exit_point == "roofline":
            _display_roofline_summary(analysis)
        elif exit_point == "dataflow_analysis":
            _display_dataflow_summary(analysis)
        else:
            _display_generation_summary(analysis)
        
        if output and verbose:
            click.echo(f"\n📋 Detailed results saved to: {output}")
            
    except Exception as e:
        click.echo(f"❌ Exploration failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def roofline(model_path, blueprint_path, output):
    """
    Perform quick roofline analysis (Exit Point 1).
    
    Analytical model-only profiling for rapid performance bounds estimation.
    """
    click.echo("🏠 Performing roofline analysis...")
    
    try:
        results, analysis = brainsmith_roofline(model_path, blueprint_path, output)
        click.echo("✅ Roofline analysis complete!")
        _display_roofline_summary(analysis)
        
    except Exception as e:
        click.echo(f"❌ Roofline analysis failed: {e}", err=True)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def dataflow(model_path, blueprint_path, output):
    """
    Perform dataflow-level analysis (Exit Point 2).
    
    Hardware-abstracted ONNX lowering and performance estimation without RTL generation.
    """
    click.echo("⚡ Performing dataflow analysis...")
    
    try:
        results, analysis = brainsmith_dataflow_analysis(model_path, blueprint_path, output)
        click.echo("✅ Dataflow analysis complete!")
        _display_dataflow_summary(analysis)
        
    except Exception as e:
        click.echo(f"❌ Dataflow analysis failed: {e}", err=True)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def generate(model_path, blueprint_path, output):
    """
    Complete RTL/HLS generation (Exit Point 3).
    
    Full dataflow core generation with stitched, parameterized RTL or HLS IP.
    """
    click.echo("🔧 Generating RTL/HLS implementation...")
    
    try:
        results, analysis = brainsmith_generate(model_path, blueprint_path, output)
        click.echo("✅ RTL/HLS generation complete!")
        _display_generation_summary(analysis)
        
    except Exception as e:
        click.echo(f"❌ RTL/HLS generation failed: {e}", err=True)

@brainsmith.command()
@click.argument('blueprint_path', type=click.Path(exists=True))
def validate(blueprint_path):
    """Validate blueprint configuration."""
    click.echo(f"🔍 Validating blueprint: {blueprint_path}")
    
    try:
        is_valid, errors = validate_blueprint(blueprint_path)
        
        if is_valid:
            click.echo("✅ Blueprint is valid")
            
            # Display blueprint summary
            from ..blueprints.base import Blueprint
            blueprint = Blueprint.from_yaml_file(Path(blueprint_path))
            _display_blueprint_summary(blueprint)
            
        else:
            click.echo("❌ Blueprint validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
                
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)

@brainsmith.group()
def list():
    """List available components in libraries."""
    pass

@list.command()
def kernels():
    """List available hardware kernels."""
    click.echo("🔧 Available hardware kernels:")
    
    try:
        kernels = list_available_kernels()
        for kernel_name, implementations in kernels.items():
            click.echo(f"  • {kernel_name}: {', '.join(implementations)}")
            
    except Exception as e:
        click.echo(f"❌ Failed to list kernels: {e}", err=True)

@list.command()
def transforms():
    """List available model transforms."""
    click.echo("🔄 Available model transforms:")
    
    try:
        transforms = list_available_transforms()
        for category, transform_list in transforms.items():
            click.echo(f"  {category}:")
            for transform in transform_list:
                click.echo(f"    • {transform}")
                
    except Exception as e:
        click.echo(f"❌ Failed to list transforms: {e}", err=True)

@list.command()
def optimizers():
    """List available optimization strategies."""
    click.echo("⚡ Available optimization strategies:")
    
    try:
        optimizers = list_available_optimizers()
        for category, optimizer_list in optimizers.items():
            click.echo(f"  {category}:")
            for optimizer in optimizer_list:
                click.echo(f"    • {optimizer}")
                
    except Exception as e:
        click.echo(f"❌ Failed to list optimizers: {e}", err=True)

# Helper functions for displaying summaries
def _display_roofline_summary(analysis: Dict):
    """Display roofline analysis summary."""
    roofline = analysis.get('roofline_analysis', {})
    
    click.echo("\n📊 Roofline Analysis Summary:")
    click.echo(f"  Computational Intensity: {roofline.get('compute_intensity', 'N/A')}")
    click.echo(f"  Memory Bandwidth Bound: {roofline.get('memory_bound', 'N/A')}")
    click.echo(f"  Compute Bound: {roofline.get('compute_bound', 'N/A')}")
    
    bounds = roofline.get('performance_bounds', {})
    if bounds:
        click.echo(f"  Performance Bounds:")
        click.echo(f"    Min Throughput: {bounds.get('min_throughput', 'N/A')}")
        click.echo(f"    Max Throughput: {bounds.get('max_throughput', 'N/A')}")

def _display_dataflow_summary(analysis: Dict):
    """Display dataflow analysis summary."""
    dataflow = analysis.get('dataflow_analysis', {})
    
    click.echo("\n⚡ Dataflow Analysis Summary:")
    click.echo(f"  Kernel Mapping: {len(dataflow.get('kernel_mapping', {}))} operations mapped")
    
    estimates = dataflow.get('performance_estimates', {})
    if estimates:
        click.echo(f"  Performance Estimates:")
        best_estimate = max(estimates.values(), key=lambda x: x.get('throughput', 0))
        click.echo(f"    Best Throughput: {best_estimate.get('throughput', 'N/A')}")
        click.echo(f"    Estimated Latency: {best_estimate.get('latency', 'N/A')}")
        click.echo(f"    Resource Usage: {best_estimate.get('resource_usage', 'N/A')}")

def _display_generation_summary(analysis: Dict):
    """Display generation summary."""
    generation = analysis.get('generation_results', {})
    
    click.echo("\n🔧 Generation Summary:")
    click.echo(f"  Generated Kernels: {len(generation.get('kernel_implementations', {}))}")
    click.echo(f"  System RTL: {'Generated' if generation.get('system_rtl') else 'Not Generated'}")
    
    performance = generation.get('performance_analysis', {})
    if performance:
        click.echo(f"  Final Performance:")
        click.echo(f"    Throughput: {performance.get('throughput', 'N/A')}")
        click.echo(f"    Latency: {performance.get('latency', 'N/A')}")
        click.echo(f"    Resource Utilization: {performance.get('resource_utilization', 'N/A')}")

def _display_blueprint_summary(blueprint):
    """Display blueprint summary."""
    click.echo(f"\n📋 Blueprint Summary:")
    click.echo(f"  Name: {blueprint.name}")
    click.echo(f"  Architecture: {blueprint.architecture}")
    click.echo(f"  Library Support: {'Full' if blueprint.supports_library_driven_dse() else 'Basic'}")
    
    library_configs = blueprint.get_library_configs()
    kernels = library_configs.get('kernels', {}).get('available', [])
    transforms = library_configs.get('transforms', {}).get('pipeline', [])
    strategies = library_configs.get('hw_optimization', {}).get('strategies', [])
    
    click.echo(f"  Components:")
    click.echo(f"    Kernels: {len(kernels)}")
    click.echo(f"    Transforms: {len(transforms)}")
    click.echo(f"    Optimization Strategies: {len(strategies)}")

if __name__ == '__main__':
    brainsmith()
```

## Implementation Notes

### Development Priority

1. **Week 1-2**: Implement library directory structures and base classes
2. **Week 3-4**: Implement Meta-DSE Engine and library coordination
3. **Week 5-6**: Enhance blueprint system with library configuration support
4. **Week 7-8**: Implement CLI and enhanced API interfaces
5. **Week 9-10**: Integration testing and backward compatibility validation

### Backward Compatibility Strategy

- Maintain all existing APIs in `brainsmith/__init__.py`
- Add compatibility layer that detects blueprint type (legacy vs library-driven)
- Route to appropriate implementation based on blueprint capabilities
- Preserve all existing function signatures and behavior

### Quality Assurance

- Comprehensive unit tests for each library component
- Integration tests for Meta-DSE Engine coordination
- CLI interface testing with example blueprints
- Performance regression testing
- Documentation with working examples

This architectural specification provides the complete implementation guide for aligning Brainsmith with its high-level vision while maintaining full backward compatibility and providing a clear migration path for users.