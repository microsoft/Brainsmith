# Phase 4 Execution Plan: Extensible Structure Implementation

## Overview

This execution plan implements the final architectural design with a focus on creating high-quality extensible structure using only existing components. The plan prioritizes the runner and core toolflow first, then systematically implements library structures around existing functionality.

**Key Constraints**:
- **NO new library additions**: Use existing components only
- **Structure over content**: Focus on extensible architecture
- **Existing transforms**: No quantization exploration (transforms cannot change weights)
- **Legacy compatibility**: Maintain all existing functionality

## Phase 1: Core Infrastructure and Workflow (Weeks 1-2)

### Week 1: Core Orchestration Foundation

#### Day 1-2: Directory Structure and Core Orchestrator

**Create core directory structure**:
```bash
mkdir -p brainsmith/core
touch brainsmith/core/__init__.py
touch brainsmith/core/design_space_orchestrator.py
touch brainsmith/core/workflow.py
touch brainsmith/core/finn_interface.py
touch brainsmith/core/legacy_support.py
```

**Priority 1: `brainsmith/core/design_space_orchestrator.py`**
```python
"""
Core orchestration engine - highest priority implementation.
Coordinates existing libraries in extensible structure.
"""

from typing import Dict, Any, Optional
from ..blueprints.base import Blueprint
from ..core.result import DSEResult

class DesignSpaceOrchestrator:
    """
    Main orchestration engine using existing components only.
    
    Provides extensible structure around current functionality
    with hierarchical exit points.
    """
    
    def __init__(self, blueprint: Blueprint):
        self.blueprint = blueprint
        self.libraries = self._initialize_existing_libraries()
        self.finn_interface = self._initialize_finn_interface()
        self.design_space = None
    
    def _initialize_existing_libraries(self) -> Dict[str, Any]:
        """Initialize libraries using existing components only."""
        return {
            'kernels': ExistingKernelLibrary(),
            'transforms': ExistingTransformLibrary(), 
            'hw_optim': ExistingOptimizationLibrary(),
            'analysis': ExistingAnalysisLibrary()
        }
    
    def orchestrate_exploration(self, exit_point: str = "dataflow_generation") -> DSEResult:
        """
        Main orchestration method with hierarchical exit points.
        Uses existing components in extensible structure.
        """
        if exit_point == "roofline":
            return self._execute_roofline_analysis_existing()
        elif exit_point == "dataflow_analysis":
            return self._execute_dataflow_analysis_existing()
        elif exit_point == "dataflow_generation":
            return self._execute_dataflow_generation_existing()
        else:
            raise ValueError(f"Invalid exit point: {exit_point}")
    
    def _execute_roofline_analysis_existing(self) -> DSEResult:
        """Exit Point 1: Use existing analysis tools."""
        # Use existing analysis capabilities without modification
        existing_analyzer = self.libraries['analysis']
        results = existing_analyzer.analyze_model_characteristics(
            self.blueprint.model_path
        )
        
        return DSEResult(
            results=[],
            analysis={'exit_point': 'roofline', 'results': results}
        )
    
    def _execute_dataflow_analysis_existing(self) -> DSEResult:
        """Exit Point 2: Use existing transforms and estimation."""
        # Apply existing transforms without modification
        existing_transforms = self.libraries['transforms']
        transformed_model = existing_transforms.apply_existing_pipeline(
            self.blueprint.model_path
        )
        
        # Use existing kernel mapping
        existing_kernels = self.libraries['kernels']
        kernel_mapping = existing_kernels.map_to_existing_kernels(transformed_model)
        
        return DSEResult(
            results=[],
            analysis={
                'exit_point': 'dataflow_analysis',
                'transformed_model': transformed_model,
                'kernel_mapping': kernel_mapping
            }
        )
    
    def _execute_dataflow_generation_existing(self) -> DSEResult:
        """Exit Point 3: Use existing FINN generation flow."""
        # Use existing optimization strategies
        existing_optimizer = self.libraries['hw_optim']
        optimization_results = existing_optimizer.optimize_using_existing_strategies(
            self.blueprint
        )
        
        # Use existing FINN interface
        generation_results = self.finn_interface.generate_implementation_existing(
            self.blueprint.model_path,
            optimization_results['best_point']
        )
        
        return DSEResult(
            results=optimization_results['all_results'],
            analysis={
                'exit_point': 'dataflow_generation',
                'generation_results': generation_results
            }
        )

# Placeholder classes for existing component libraries
class ExistingKernelLibrary:
    """Wrapper for existing custom operations."""
    pass

class ExistingTransformLibrary:
    """Wrapper for existing transforms from steps/."""
    pass

class ExistingOptimizationLibrary:
    """Wrapper for existing optimization from dse/."""
    pass

class ExistingAnalysisLibrary:
    """Wrapper for existing analysis capabilities."""
    pass
```

#### Day 3-4: FINN Interface with Legacy Support

**Priority 2: `brainsmith/core/finn_interface.py`**
```python
"""
FINN interface supporting existing DataflowBuildConfig + future 4-hook placeholder.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from finn.util.fpgadataflow import DataflowBuildConfig

@dataclass
class FINNHooksPlaceholder:
    """
    Placeholder for future 4-hook FINN interface.
    Provides structure without implementation.
    """
    
    preprocessing_hook: Optional[Any] = None
    transformation_hook: Optional[Any] = None
    optimization_hook: Optional[Any] = None
    generation_hook: Optional[Any] = None
    
    def is_available(self) -> bool:
        """Always False until 4-hook interface exists."""
        return False
    
    def prepare_for_future_interface(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration structure for future interface."""
        return {
            'preprocessing_config': design_point.get('preprocessing', {}),
            'transformation_config': design_point.get('transforms', {}),
            'optimization_config': design_point.get('hw_optimization', {}),
            'generation_config': design_point.get('generation', {})
        }

class FINNInterface:
    """
    FINN integration using existing DataflowBuildConfig flow.
    Provides placeholder structure for future 4-hook interface.
    """
    
    def __init__(self, legacy_config: Dict[str, Any], future_hooks: FINNHooksPlaceholder):
        self.legacy_config = legacy_config
        self.future_hooks = future_hooks
        self.use_legacy = True  # Always use legacy for now
    
    def generate_implementation_existing(self, model_path: str, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate implementation using existing DataflowBuildConfig flow.
        Maintains current functionality while providing future structure.
        """
        # Create DataflowBuildConfig using existing patterns
        build_config = self._create_existing_build_config(design_point)
        
        # Use existing FINN build process
        from finn.builder.build_dataflow import build_dataflow
        
        build_results = build_dataflow(
            model=model_path,
            cfg=build_config
        )
        
        # Return results in existing format
        return {
            'rtl_files': build_results.get('rtl_files', []),
            'hls_files': build_results.get('hls_files', []),
            'synthesis_results': build_results.get('synthesis_results', {}),
            'interface_type': 'existing_dataflow_build_config'
        }
    
    def _create_existing_build_config(self, design_point: Dict[str, Any]) -> DataflowBuildConfig:
        """Create DataflowBuildConfig using existing configuration patterns."""
        # Use existing configuration structure
        config = DataflowBuildConfig(
            **self.legacy_config,  # Existing legacy settings
            # Map design point to existing config format
            **design_point.get('finn_config', {})
        )
        return config
```

#### Day 5: Core Workflow Management

**Priority 3: `brainsmith/core/workflow.py`**
```python
"""
High-level workflow management using existing components.
"""

from typing import Dict, Any, List
from .design_space_orchestrator import DesignSpaceOrchestrator

class WorkflowManager:
    """
    Manages high-level workflows using existing components.
    Provides extensible structure for workflow coordination.
    """
    
    def __init__(self, orchestrator: DesignSpaceOrchestrator):
        self.orchestrator = orchestrator
        self.workflow_history = []
    
    def execute_existing_workflow(self, workflow_type: str) -> Dict[str, Any]:
        """
        Execute workflow using existing components only.
        
        Args:
            workflow_type: Type of workflow ('standard', 'fast', 'comprehensive')
        """
        if workflow_type == "fast":
            return self._execute_fast_workflow_existing()
        elif workflow_type == "comprehensive":
            return self._execute_comprehensive_workflow_existing()
        else:
            return self._execute_standard_workflow_existing()
    
    def _execute_fast_workflow_existing(self) -> Dict[str, Any]:
        """Fast workflow using roofline analysis only."""
        result = self.orchestrator.orchestrate_exploration("roofline")
        self.workflow_history.append({"type": "fast", "result": result})
        return {"workflow": "fast", "result": result}
    
    def _execute_standard_workflow_existing(self) -> Dict[str, Any]:
        """Standard workflow using dataflow analysis."""
        result = self.orchestrator.orchestrate_exploration("dataflow_analysis")
        self.workflow_history.append({"type": "standard", "result": result})
        return {"workflow": "standard", "result": result}
    
    def _execute_comprehensive_workflow_existing(self) -> Dict[str, Any]:
        """Comprehensive workflow using full generation."""
        result = self.orchestrator.orchestrate_exploration("dataflow_generation")
        self.workflow_history.append({"type": "comprehensive", "result": result})
        return {"workflow": "comprehensive", "result": result}
```

### Week 2: Core API and CLI Implementation

#### Day 6-7: Python API Implementation

**Priority 1: `brainsmith/core/api.py`**
```python
"""
Python API using existing components in extensible structure.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from ..blueprints.base import Blueprint
from .design_space_orchestrator import DesignSpaceOrchestrator
from ..core.result import DSEResult

def brainsmith_explore(model_path: str, 
                      blueprint_path: str,
                      exit_point: str = "dataflow_generation",
                      output_dir: Optional[str] = None,
                      **kwargs) -> Tuple[DSEResult, Dict[str, Any]]:
    """
    Main exploration API using existing components only.
    
    Provides hierarchical exit points with extensible structure
    around current functionality.
    """
    # Validate inputs using existing validation
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    # Load blueprint using existing Blueprint class
    blueprint = Blueprint.from_yaml_file(Path(blueprint_path))
    blueprint.model_path = model_path
    
    # Create orchestrator using existing components
    orchestrator = DesignSpaceOrchestrator(blueprint)
    results = orchestrator.orchestrate_exploration(exit_point)
    
    # Generate analysis using existing analysis tools
    analysis = {"exit_point": exit_point, "method": "existing_tools"}
    
    # Save results if requested
    if output_dir:
        _save_results_existing(results, analysis, output_dir)
    
    return results, analysis

def brainsmith_roofline(model_path: str, blueprint_path: str, 
                       output_dir: Optional[str] = None) -> Tuple[DSEResult, Dict[str, Any]]:
    """Roofline analysis using existing tools."""
    return brainsmith_explore(model_path, blueprint_path, "roofline", output_dir)

def brainsmith_dataflow_analysis(model_path: str, blueprint_path: str,
                                output_dir: Optional[str] = None) -> Tuple[DSEResult, Dict[str, Any]]:
    """Dataflow analysis using existing transforms and estimation."""
    return brainsmith_explore(model_path, blueprint_path, "dataflow_analysis", output_dir)

def brainsmith_generate(model_path: str, blueprint_path: str,
                       output_dir: Optional[str] = None) -> Tuple[DSEResult, Dict[str, Any]]:
    """Full generation using existing FINN flow."""
    return brainsmith_explore(model_path, blueprint_path, "dataflow_generation", output_dir)

# Backward compatibility using existing API
def explore_design_space(model_path: str, blueprint_name: str, **kwargs):
    """
    Backward compatibility wrapper using existing API.
    Maintains 100% compatibility with current usage.
    """
    # Check if using new blueprint path or legacy name
    if Path(blueprint_name).exists():
        return brainsmith_explore(model_path, blueprint_name, **kwargs)
    else:
        # Route to existing legacy system
        from ..legacy.compatibility import existing_explore_design_space
        return existing_explore_design_space(model_path, blueprint_name, **kwargs)

def _save_results_existing(results: DSEResult, analysis: Dict, output_dir: str):
    """Save results using existing save functionality."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use existing result save method
    results.save(output_path / "dse_results.json")
    
    # Use existing analysis export
    import json
    with open(output_path / "analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
```

#### Day 8-9: CLI Implementation

**Priority 2: `brainsmith/core/cli.py`**
```python
"""
Command-line interface using existing components.
"""

import click
from pathlib import Path
from .api import (
    brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
    brainsmith_generate
)

@click.group()
@click.version_option(version="0.4.0", prog_name="brainsmith")
def brainsmith():
    """
    Brainsmith: Meta-toolchain for FPGA accelerator synthesis.
    
    Extensible structure using existing components.
    """
    pass

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--exit-point', '-e', 
              type=click.Choice(['roofline', 'dataflow_analysis', 'dataflow_generation']),
              default='dataflow_generation',
              help='Hierarchical exit point for exploration')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def explore(model_path, blueprint_path, exit_point, output):
    """
    Explore design space using existing components with hierarchical exit points.
    
    Examples:
        brainsmith explore model.onnx blueprint.yaml --exit-point roofline
        brainsmith explore model.onnx blueprint.yaml -e dataflow_analysis
        brainsmith explore model.onnx blueprint.yaml  # Full generation
    """
    try:
        results, analysis = brainsmith_explore(model_path, blueprint_path, exit_point, output)
        click.echo(f"✅ Exploration complete using existing tools!")
        click.echo(f"Exit point: {exit_point}")
        _display_summary_existing(analysis, exit_point)
    except Exception as e:
        click.echo(f"❌ Exploration failed: {e}", err=True)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path())
def roofline(model_path, blueprint_path, output):
    """Quick roofline analysis using existing analysis tools."""
    try:
        results, analysis = brainsmith_roofline(model_path, blueprint_path, output)
        click.echo("✅ Roofline analysis complete using existing tools!")
        _display_roofline_summary_existing(analysis)
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path())
def dataflow(model_path, blueprint_path, output):
    """Dataflow analysis using existing transforms and estimation."""
    try:
        results, analysis = brainsmith_dataflow_analysis(model_path, blueprint_path, output)
        click.echo("✅ Dataflow analysis complete using existing tools!")
        _display_dataflow_summary_existing(analysis)
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)

@brainsmith.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('blueprint_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path())
def generate(model_path, blueprint_path, output):
    """Full RTL/HLS generation using existing FINN flow."""
    try:
        results, analysis = brainsmith_generate(model_path, blueprint_path, output)
        click.echo("✅ Generation complete using existing FINN flow!")
        _display_generation_summary_existing(analysis)
    except Exception as e:
        click.echo(f"❌ Generation failed: {e}", err=True)

def _display_summary_existing(analysis: Dict, exit_point: str):
    """Display summary based on exit point using existing tools."""
    click.echo(f"  📊 Method: {analysis.get('method', 'existing_tools')}")
    click.echo(f"  🔧 Components: Existing only")

def _display_roofline_summary_existing(analysis: Dict):
    """Display roofline summary using existing analysis."""
    click.echo("  📊 Roofline analysis using existing tools")

def _display_dataflow_summary_existing(analysis: Dict):
    """Display dataflow summary using existing components."""
    click.echo("  ⚡ Dataflow analysis using existing transforms")

def _display_generation_summary_existing(analysis: Dict):
    """Display generation summary using existing FINN flow."""
    click.echo("  🔧 RTL/HLS generation using existing DataflowBuildConfig")

if __name__ == '__main__':
    brainsmith()
```

#### Day 10: Legacy Compatibility Layer

**Priority 3: `brainsmith/core/legacy_support.py`**
```python
"""
Legacy compatibility layer maintaining existing functionality.
"""

from typing import Dict, Any
import warnings

def maintain_existing_api_compatibility():
    """
    Ensure all existing API functions continue to work.
    This function validates that no existing functionality is broken.
    """
    # Import existing functions to ensure they still work
    try:
        from ..dse import explore_design_space as existing_explore
        from ..blueprints import get_blueprint as existing_get_blueprint
        return True
    except ImportError as e:
        warnings.warn(f"Legacy API compatibility issue: {e}")
        return False

def route_to_existing_implementation(function_name: str, *args, **kwargs):
    """
    Route function calls to existing implementations when needed.
    Provides fallback to maintain compatibility.
    """
    if function_name == "explore_design_space":
        from ..dse import explore_design_space
        return explore_design_space(*args, **kwargs)
    elif function_name == "get_blueprint":
        from ..blueprints import get_blueprint
        return get_blueprint(*args, **kwargs)
    else:
        raise ValueError(f"Unknown legacy function: {function_name}")

class LegacyAPIWarning(UserWarning):
    """Warning for deprecated but supported legacy API usage."""
    pass

def warn_legacy_usage(function_name: str, new_function_name: str):
    """Warn users about legacy API usage while maintaining support."""
    warnings.warn(
        f"Using legacy function {function_name}. "
        f"Consider migrating to {new_function_name} for enhanced features.",
        LegacyAPIWarning,
        stacklevel=3
    )
```

## Phase 2: Library Structure Implementation (Weeks 3-6)

### Week 3: Kernels Library Structure (Existing Components Only)

#### Day 11-13: Kernel Base Structure and Registry

**Create kernels directory structure**:
```bash
mkdir -p brainsmith/kernels/{conv,linear,activation}
touch brainsmith/kernels/__init__.py
touch brainsmith/kernels/base.py
touch brainsmith/kernels/registry.py
touch brainsmith/kernels/conv/__init__.py
touch brainsmith/kernels/conv/hw_custom_op.py
touch brainsmith/kernels/linear/__init__.py
touch brainsmith/kernels/linear/hw_custom_op.py
touch brainsmith/kernels/activation/__init__.py
touch brainsmith/kernels/activation/hw_custom_op.py
```

**Priority 1: `brainsmith/kernels/base.py`**
```python
"""
Base interfaces for existing kernel components.
Provides extensible structure without adding new kernels.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ExistingKernelWrapper(ABC):
    """
    Base wrapper for existing kernel implementations.
    Provides extensible structure around existing custom operations.
    """
    
    def __init__(self, existing_kernel_class: Any, operation_name: str):
        self.existing_kernel_class = existing_kernel_class
        self.operation_name = operation_name
        self.existing_config = {}
    
    @abstractmethod
    def get_existing_parameter_space(self) -> Dict[str, Any]:
        """Extract parameter space from existing kernel."""
        pass
    
    @abstractmethod
    def wrap_existing_instantiation(self, parameters: Dict[str, Any]) -> Any:
        """Wrap existing kernel instantiation."""
        pass
    
    def get_existing_attributes(self) -> Dict[str, Any]:
        """Get attributes from existing kernel class."""
        return {
            'class_name': self.existing_kernel_class.__name__,
            'module': self.existing_kernel_class.__module__,
            'operation_name': self.operation_name
        }

class ExistingHWCustomOp(ExistingKernelWrapper):
    """
    Wrapper for existing HW custom operations.
    Provides extensible structure around existing implementations.
    """
    
    def get_existing_onnx_signature(self) -> Dict[str, Any]:
        """Get ONNX signature from existing implementation."""
        # Extract from existing custom op if available
        if hasattr(self.existing_kernel_class, 'op_type'):
            return {
                'op_type': self.existing_kernel_class.op_type,
                'domain': getattr(self.existing_kernel_class, 'domain', 'brainsmith.custom')
            }
        return {'op_type': self.operation_name, 'domain': 'brainsmith.custom'}
```

**Priority 2: `brainsmith/kernels/registry.py`**
```python
"""
Registry for existing kernel implementations.
Provides extensible structure for organizing existing custom operations.
"""

from typing import Dict, List, Any
from .base import ExistingKernelWrapper

class ExistingKernelRegistry:
    """
    Registry organizing existing custom operations in extensible structure.
    NO new kernels added - only existing components organized.
    """
    
    def __init__(self):
        self._existing_kernels = {}
        self._operation_types = {}
        self._register_existing_kernels()
    
    def _register_existing_kernels(self):
        """Register all existing custom operations found in custom_op/."""
        try:
            self._register_existing_conv_kernels()
            self._register_existing_linear_kernels()
            self._register_existing_activation_kernels()
        except ImportError as e:
            # Handle missing existing components gracefully
            print(f"Warning: Could not register some existing kernels: {e}")
    
    def _register_existing_conv_kernels(self):
        """Register existing convolution custom operations."""
        try:
            from ...custom_op.hw_conv2d import HWConv2D
            from .conv.hw_custom_op import ConvolutionWrapper
            wrapper = ConvolutionWrapper(HWConv2D)
            self.register_existing_kernel(wrapper)
        except ImportError:
            pass  # Skip if not available
    
    def _register_existing_linear_kernels(self):
        """Register existing linear custom operations."""
        try:
            # Import existing linear custom operations
            from .linear.hw_custom_op import LinearWrapper
            # Register if exists
        except ImportError:
            pass  # Skip if not available
    
    def _register_existing_activation_kernels(self):
        """Register existing activation custom operations.""" 
        try:
            # Import existing activation custom operations
            from .activation.hw_custom_op import ActivationWrapper
            # Register if exists
        except ImportError:
            pass  # Skip if not available
    
    def register_existing_kernel(self, wrapper: ExistingKernelWrapper):
        """Register existing kernel wrapper in extensible structure."""
        self._existing_kernels[wrapper.operation_name] = wrapper
        
        # Organize by operation type for extensibility
        op_type = wrapper.get_existing_onnx_signature().get('op_type', 'unknown')
        if op_type not in self._operation_types:
            self._operation_types[op_type] = []
        self._operation_types[op_type].append(wrapper.operation_name)
    
    def get_existing_kernels_for_blueprint(self, blueprint_config: Dict) -> List[ExistingKernelWrapper]:
        """Get existing kernels specified in blueprint."""
        kernels = []
        for kernel_config in blueprint_config.get('available', []):
            kernel_name = kernel_config['name']
            if kernel_name in self._existing_kernels:
                kernels.append(self._existing_kernels[kernel_name])
        return kernels
    
    def list_existing_kernels(self) -> Dict[str, List[str]]:
        """List all existing kernels organized by operation type."""
        return self._operation_types.copy()

# Global registry instance for existing kernels
existing_kernel_registry = ExistingKernelRegistry()
```

#### Day 14-15: Specific Kernel Wrappers

**Priority 1: `brainsmith/kernels/conv/hw_custom_op.py`**
```python
"""
Wrapper for existing convolution custom operations.
Provides extensible structure around existing HWConv2D.
"""

from typing import Dict, Any
from ..base import ExistingHWCustomOp

class ConvolutionWrapper(ExistingHWCustomOp):
    """
    Wrapper for existing HWConv2D custom operation.
    Provides extensible structure without modifying existing implementation.
    """
    
    def __init__(self, existing_hw_conv2d_class):
        super().__init__(existing_hw_conv2d_class, "Convolution")
    
    def get_existing_parameter_space(self) -> Dict[str, Any]:
        """Extract parameter space from existing HWConv2D."""
        # Extract parameters that existing HWConv2D supports
        return {
            'simd': {
                'type': 'integer',
                'range': [1, 64],  # From existing implementation limits
                'default': 1,
                'description': 'SIMD parallelism (from existing HWConv2D)',
                'source': 'existing_hw_conv2d'
            },
            'pe': {
                'type': 'integer',
                'range': [1, 64],  # From existing implementation limits
                'default': 1,
                'description': 'PE parallelism (from existing HWConv2D)',
                'source': 'existing_hw_conv2d'
            }
            # Add other parameters that existing HWConv2D supports
        }
    
    def wrap_existing_instantiation(self, parameters: Dict[str, Any]) -> Any:
        """Wrap existing HWConv2D instantiation."""
        # Map parameters to existing HWConv2D constructor format
        existing_params = {
            'simd': parameters.get('simd', 1),
            'pe': parameters.get('pe', 1)
            # Map other parameters as supported by existing implementation
        }
        
        # Instantiate existing HWConv2D with mapped parameters
        return self.existing_kernel_class(**existing_params)
    
    def get_existing_onnx_signature(self) -> Dict[str, Any]:
        """Get ONNX signature from existing HWConv2D."""
        return {
            'op_type': 'Conv',
            'domain': 'brainsmith.custom',
            'inputs': [{'name': 'X', 'type': 'tensor(float)'}],
            'outputs': [{'name': 'Y', 'type': 'tensor(float)'}],
            'source': 'existing_hw_conv2d'
        }
```

### Week 4: Model Transforms Library Structure (Existing Components Only)

#### Day 16-18: Transform Base Structure

**Create transforms directory structure**:
```bash
mkdir -p brainsmith/model_transforms
touch brainsmith/model_transforms/__init__.py
touch brainsmith/model_transforms/base.py
touch brainsmith/model_transforms/registry.py
touch brainsmith/model_transforms/streamlining.py
touch brainsmith/model_transforms/folding.py
touch brainsmith/model_transforms/partitioning.py
```

**Priority 1: `brainsmith/model_transforms/base.py`**
```python
"""
Base interfaces for existing transform components.
NO quantization transforms (transforms cannot change model weights).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ExistingTransformWrapper(ABC):
    """
    Base wrapper for existing transform implementations from steps/.
    Provides extensible structure without adding new transforms.
    """
    
    def __init__(self, existing_transform_module: Any, transform_name: str):
        self.existing_transform_module = existing_transform_module
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
    
    def get_existing_attributes(self) -> Dict[str, Any]:
        """Get attributes from existing transform."""
        return {
            'module_name': getattr(self.existing_transform_module, '__name__', 'unknown'),
            'transform_name': self.transform_name,
            'type': 'existing_transform'
        }
    
    def can_apply_to_existing_model(self, model: Any) -> bool:
        """Check if existing transform can be applied to model."""
        # Use existing transform's applicability check if available
        if hasattr(self.existing_transform_module, 'can_apply'):
            return self.existing_transform_module.can_apply(model)
        return True  # Assume applicable if no check available

class ExistingTransformPipeline:
    """
    Pipeline for existing transforms.
    Coordinates existing transform sequence without adding new functionality.
    """
    
    def __init__(self, existing_transforms: List[ExistingTransformWrapper]):
        self.existing_transforms = existing_transforms
        self.applied_transforms = []
    
    def apply_existing_pipeline(self, model: Any, config: Dict[str, Any] = None) -> Any:
        """Apply pipeline of existing transforms."""
        current_model = model
        
        for transform in self.existing_transforms:
            if transform.can_apply_to_existing_model(current_model):
                transform_config = config.get(transform.transform_name, {}) if config else {}
                current_model = transform.wrap_existing_application(current_model, transform_config)
                self.applied_transforms.append(transform.transform_name)
        
        return current_model
```

**Priority 2: `brainsmith/model_transforms/streamlining.py`**
```python
"""
Wrapper for existing streamlining transforms from steps/.
"""

from typing import Dict, Any
from .base import ExistingTransformWrapper

class ExistingStreamliningWrapper(ExistingTransformWrapper):
    """
    Wrapper for existing streamlining functionality.
    Uses existing streamlining from steps/ without modification.
    """
    
    def __init__(self):
        try:
            # Import existing streamlining from steps
            from ...steps import streamlining as existing_streamlining
            super().__init__(existing_streamlining, "streamlining")
        except ImportError:
            # Handle case where existing streamlining is not available
            super().__init__(None, "streamlining")
            self.available = False
    
    def get_existing_parameter_space(self) -> Dict[str, Any]:
        """Extract parameters from existing streamlining."""
        if not hasattr(self, 'available'):
            return {
                'fold_constants': {
                    'type': 'boolean',
                    'default': True,
                    'description': 'Fold constants (existing streamlining)',
                    'source': 'steps.streamlining'
                },
                'remove_unused_nodes': {
                    'type': 'boolean',
                    'default': True,
                    'description': 'Remove unused nodes (existing streamlining)',
                    'source': 'steps.streamlining'
                },
                'simplify_graph': {
                    'type': 'boolean',
                    'default': True,
                    'description': 'Simplify graph structure (existing streamlining)',
                    'source': 'steps.streamlining'
                }
            }
        return {}
    
    def wrap_existing_application(self, model: Any, parameters: Dict[str, Any]) -> Any:
        """Apply existing streamlining with parameter mapping."""
        if not hasattr(self, 'available') and self.existing_transform_module:
            # Map parameters to existing streamlining format
            streamlining_config = {
                'fold_constants': parameters.get('fold_constants', True),
                'remove_unused_nodes': parameters.get('remove_unused_nodes', True),
                'simplify_graph': parameters.get('simplify_graph', True)
            }
            
            # Apply existing streamlining (adapt to actual existing API)
            if hasattr(self.existing_transform_module, 'apply_streamlining'):
                return self.existing_transform_module.apply_streamlining(model, streamlining_config)
            elif hasattr(self.existing_transform_module, 'streamline_model'):
                return self.existing_transform_module.streamline_model(model, streamlining_config)
            else:
                # Fallback if existing API is different
                return model
        
        return model  # Return unchanged if streamlining not available
```

### Week 5: Hardware Optimization Library Structure (Existing Components Only)

#### Day 19-21: Optimization Base Structure

**Create hw_optim directory structure**:
```bash
mkdir -p brainsmith/hw_optim/strategies
touch brainsmith/hw_optim/__init__.py
touch brainsmith/hw_optim/base.py
touch brainsmith/hw_optim/registry.py
touch brainsmith/hw_optim/strategies/__init__.py
touch brainsmith/hw_optim/strategies/bayesian_opt.py
touch brainsmith/hw_optim/strategies/genetic_opt.py
touch brainsmith/hw_optim/strategies/random_opt.py
```

**Priority 1: `brainsmith/hw_optim/base.py`**
```python
"""
Base interfaces for existing optimization strategies from dse/.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ExistingOptimizationWrapper(ABC):
    """
    Base wrapper for existing optimization strategies from dse/.
    Provides extensible structure without adding new optimization algorithms.
    """
    
    def __init__(self, existing_optimizer_module: Any, strategy_name: str):
        self.existing_optimizer_module = existing_optimizer_module
        self.strategy_name = strategy_name
        self.existing_config = {}
    
    @abstractmethod
    def get_existing_parameter_space(self) -> Dict[str, Any]:
        """Extract parameter space from existing optimizer."""
        pass
    
    @abstractmethod
    def wrap_existing_optimization(self, design_space: Any, objectives: List[str], budget: int) -> Dict[str, Any]:
        """Wrap existing optimization execution."""
        pass
    
    def get_existing_attributes(self) -> Dict[str, Any]:
        """Get attributes from existing optimizer."""
        return {
            'module_name': getattr(self.existing_optimizer_module, '__name__', 'unknown'),
            'strategy_name': self.strategy_name,
            'type': 'existing_optimization_strategy'
        }

class ExistingOptimizationCoordinator:
    """
    Coordinates existing optimization strategies.
    Uses existing optimization from dse/ without adding new algorithms.
    """
    
    def __init__(self, existing_strategies: List[ExistingOptimizationWrapper]):
        self.existing_strategies = existing_strategies
        self.optimization_history = []
    
    def optimize_using_existing_strategies(self, blueprint: Any) -> Dict[str, Any]:
        """Optimize using existing strategies from dse/."""
        results = {}
        
        for strategy in self.existing_strategies:
            try:
                strategy_result = strategy.wrap_existing_optimization(
                    design_space=blueprint.get_design_space(),
                    objectives=blueprint.get_objectives(),
                    budget=blueprint.get_optimization_budget()
                )
                results[strategy.strategy_name] = strategy_result
                self.optimization_history.append({
                    'strategy': strategy.strategy_name,
                    'result': strategy_result
                })
            except Exception as e:
                # Handle strategy failures gracefully
                results[strategy.strategy_name] = {'error': str(e)}
        
        # Select best result using existing logic
        best_result = self._select_best_from_existing_results(results)
        
        return {
            'all_results': results,
            'best_point': best_result.get('best_point', {}),
            'best_result': best_result
        }
    
    def _select_best_from_existing_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Select best result using existing selection logic."""
        # Use existing result selection if available
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'best_point': {}, 'error': 'No valid optimization results'}
        
        # Simple selection - use first valid result (can be enhanced with existing logic)
        return list(valid_results.values())[0]
```

### Week 6: Analysis Library and Coordination

#### Day 22-24: Analysis Library Structure

**Create analysis directory structure**:
```bash
mkdir -p brainsmith/analysis
touch brainsmith/analysis/__init__.py
touch brainsmith/analysis/performance.py
touch brainsmith/analysis/reporting.py
touch brainsmith/analysis/visualization.py
```

**Priority 1: `brainsmith/analysis/performance.py`**
```python
"""
Performance analysis using existing analysis tools.
"""

from typing import Dict, Any
from ..core.result import DSEResult

class ExistingPerformanceAnalyzer:
    """
    Performance analyzer using existing analysis capabilities.
    Wraps existing analysis tools in extensible structure.
    """
    
    def __init__(self):
        self.existing_analyzers = self._load_existing_analyzers()
    
    def _load_existing_analyzers(self) -> Dict[str, Any]:
        """Load existing analysis tools."""
        analyzers = {}
        
        try:
            # Import existing analysis from dse/analysis.py
            from ..dse.analysis import DSEAnalyzer
            analyzers['dse_analyzer'] = DSEAnalyzer
        except ImportError:
            pass
        
        return analyzers
    
    def analyze_model_characteristics(self, model_path: str) -> Dict[str, Any]:
        """Analyze model using existing tools."""
        results = {
            'model_path': model_path,
            'analysis_method': 'existing_tools'
        }
        
        if 'dse_analyzer' in self.existing_analyzers:
            try:
                analyzer = self.existing_analyzers['dse_analyzer']()
                existing_results = analyzer.analyze_model(model_path)
                results.update(existing_results)
            except Exception as e:
                results['analysis_error'] = str(e)
        
        return results
    
    def analyze_results(self, dse_results: DSEResult) -> Dict[str, Any]:
        """Analyze DSE results using existing analysis."""
        analysis = {
            'results_count': len(dse_results.results) if dse_results.results else 0,
            'analysis_method': 'existing_tools'
        }
        
        if 'dse_analyzer' in self.existing_analyzers:
            try:
                analyzer = self.existing_analyzers['dse_analyzer']()
                if hasattr(analyzer, 'analyze_results'):
                    existing_analysis = analyzer.analyze_results(dse_results)
                    analysis.update(existing_analysis)
            except Exception as e:
                analysis['analysis_error'] = str(e)
        
        return analysis
```

#### Day 25-27: Coordination Layer

**Create coordination directory structure**:
```bash
mkdir -p brainsmith/coordination
touch brainsmith/coordination/__init__.py
touch brainsmith/coordination/orchestration.py
touch brainsmith/coordination/workflow.py
touch brainsmith/coordination/result_aggregation.py
```

**Priority 1: `brainsmith/coordination/orchestration.py`**
```python
"""
Cross-library coordination using existing components.
Coordinates existing libraries without adding new functionality.
"""

from typing import Dict, Any, List

class CoordinationEngine:
    """
    Coordinates existing libraries in extensible structure.
    Uses existing coordination logic from dse/ where available.
    """
    
    def __init__(self, libraries: Dict[str, Any]):
        self.libraries = libraries
        self.coordination_history = []
    
    def execute_existing_optimization(self, design_space: Any, objectives: List[Dict]) -> Dict[str, Any]:
        """Execute optimization using existing coordination logic."""
        coordination_result = {
            'libraries_used': list(self.libraries.keys()),
            'coordination_method': 'existing_sequential'
        }
        
        # Use existing coordination if available
        try:
            # Sequential coordination using existing libraries
            results = self._coordinate_existing_sequential(design_space, objectives)
            coordination_result.update(results)
        except Exception as e:
            coordination_result['coordination_error'] = str(e)
            # Fallback to simple coordination
            results = self._simple_existing_coordination(design_space, objectives)
            coordination_result.update(results)
        
        self.coordination_history.append(coordination_result)
        return coordination_result
    
    def _coordinate_existing_sequential(self, design_space: Any, objectives: List[Dict]) -> Dict[str, Any]:
        """Sequential coordination using existing libraries."""
        # Step 1: Apply existing transforms
        if 'transforms' in self.libraries:
            transform_result = self.libraries['transforms'].apply_existing_pipeline(design_space)
        else:
            transform_result = design_space
        
        # Step 2: Map to existing kernels
        if 'kernels' in self.libraries:
            kernel_result = self.libraries['kernels'].map_to_existing_kernels(transform_result)
        else:
            kernel_result = transform_result
        
        # Step 3: Optimize using existing strategies
        if 'hw_optim' in self.libraries:
            optimization_result = self.libraries['hw_optim'].optimize_using_existing_strategies({
                'design_space': kernel_result,
                'objectives': objectives
            })
        else:
            optimization_result = {'best_point': {}}
        
        return {
            'transform_result': transform_result,
            'kernel_result': kernel_result,
            'optimization_result': optimization_result,
            'all_results': [optimization_result],
            'best_point': optimization_result.get('best_point', {}),
            'best_result': optimization_result
        }
    
    def _simple_existing_coordination(self, design_space: Any, objectives: List[Dict]) -> Dict[str, Any]:
        """Simple fallback coordination."""
        return {
            'best_point': {},
            'all_results': [],
            'best_result': {},
            'coordination_method': 'simple_fallback'
        }
```

## Phase 3: Enhanced Blueprint and Coordination (Weeks 7-8)

### Week 7: Blueprint System Enhancement

#### Day 28-30: Enhanced Blueprint Support

**Priority 1: Enhanced `brainsmith/blueprints/base.py`**
```python
"""
Enhanced Blueprint class supporting library-driven configuration.
Uses existing blueprint functionality with extensions for new structure.
"""

# Add to existing Blueprint class:
def get_library_configs(self) -> Dict[str, Dict]:
    """Get library-specific configurations for existing components."""
    return {
        'kernels': self.yaml_data.get('kernels', {}),
        'transforms': self.yaml_data.get('transforms', {}),
        'hw_optimization': self.yaml_data.get('hw_optimization', {}),
        'analysis': self.yaml_data.get('analysis', {})
    }

def get_finn_legacy_config(self) -> Dict[str, Any]:
    """Get FINN legacy configuration for existing DataflowBuildConfig."""
    finn_config = self.yaml_data.get('finn_interface', {})
    return finn_config.get('legacy_config', {
        'auto_fifo_depths': True,
        'fpga_part': 'xcvu9p-flga2104-2-i',
        'generate_outputs': ['estimate', 'bitfile']
    })

def supports_library_driven_dse(self) -> bool:
    """Check if blueprint supports library-driven DSE using existing components."""
    return 'kernels' in self.yaml_data or 'transforms' in self.yaml_data

def validate_library_config(self) -> Tuple[bool, List[str]]:
    """Validate blueprint library configuration for existing components."""
    errors = []
    
    # Validate that only existing components are referenced
    kernels_config = self.yaml_data.get('kernels', {})
    if 'available' in kernels_config:
        for kernel in kernels_config['available']:
            if kernel.get('source') and 'custom_op' not in kernel.get('source', ''):
                errors.append(f"Kernel {kernel['name']} references non-existing component")
    
    transforms_config = self.yaml_data.get('transforms', {})
    if 'pipeline' in transforms_config:
        for transform in transforms_config['pipeline']:
            if transform.get('source') and 'steps' not in transform.get('source', ''):
                errors.append(f"Transform {transform['name']} references non-existing component")
    
    return len(errors) == 0, errors
```

### Week 8: Integration and Testing Framework

#### Day 31-35: Comprehensive Testing Setup

**Create testing structure**:
```bash
mkdir -p tests/integration/existing_components
touch tests/integration/existing_components/test_orchestrator.py
touch tests/integration/existing_components/test_libraries.py
touch tests/integration/existing_components/test_api.py
touch tests/integration/existing_components/test_cli.py
```

**Priority 1: Integration Tests**
```python
# tests/integration/existing_components/test_orchestrator.py
"""
Integration tests for DesignSpaceOrchestrator using existing components only.
"""

import pytest
from brainsmith.core.design_space_orchestrator import DesignSpaceOrchestrator
from brainsmith.blueprints.base import Blueprint

def test_orchestrator_with_existing_components():
    """Test orchestrator using existing components only."""
    # Create minimal blueprint for existing components
    blueprint_data = {
        'name': 'test_existing',
        'kernels': {'available': []},
        'transforms': {'pipeline': []},
        'hw_optimization': {'strategies': []},
        'finn_interface': {'legacy_config': {}}
    }
    
    blueprint = Blueprint.from_dict(blueprint_data)
    orchestrator = DesignSpaceOrchestrator(blueprint)
    
    # Test that orchestrator initializes with existing components
    assert orchestrator.libraries is not None
    assert 'kernels' in orchestrator.libraries
    assert 'transforms' in orchestrator.libraries
    assert 'hw_optim' in orchestrator.libraries

def test_hierarchical_exit_points():
    """Test all three exit points work with existing components."""
    blueprint_data = {'name': 'test_exit_points'}
    blueprint = Blueprint.from_dict(blueprint_data)
    orchestrator = DesignSpaceOrchestrator(blueprint)
    
    # Test roofline exit point
    result_roofline = orchestrator.orchestrate_exploration("roofline")
    assert result_roofline.analysis['exit_point'] == 'roofline'
    
    # Test dataflow analysis exit point
    result_dataflow = orchestrator.orchestrate_exploration("dataflow_analysis")
    assert result_dataflow.analysis['exit_point'] == 'dataflow_analysis'
    
    # Test full generation exit point
    result_generation = orchestrator.orchestrate_exploration("dataflow_generation")
    assert result_generation.analysis['exit_point'] == 'dataflow_generation'
```

## Phase 4: Integration and Legacy Support (Weeks 9-10)

### Week 9: Legacy Compatibility and Testing

#### Day 36-40: Comprehensive Testing and Validation

**Create comprehensive test suite**:
```bash
mkdir -p tests/legacy_compatibility
touch tests/legacy_compatibility/test_backward_compatibility.py
touch tests/legacy_compatibility/test_existing_api.py
touch tests/legacy_compatibility/test_migration.py
```

**Priority 1: Backward Compatibility Tests**
```python
# tests/legacy_compatibility/test_backward_compatibility.py
"""
Comprehensive backward compatibility testing.
Ensures all existing functionality continues to work.
"""

def test_existing_explore_design_space_still_works():
    """Test that existing explore_design_space function still works."""
    from brainsmith import explore_design_space
    
    # Test with existing blueprint name
    try:
        result = explore_design_space("dummy_model.onnx", "existing_blueprint")
        # Should not raise exception (may fail due to missing files, but API should work)
    except FileNotFoundError:
        pass  # Expected for dummy files
    except AttributeError:
        pytest.fail("explore_design_space API broken")

def test_existing_blueprint_functions_work():
    """Test that existing blueprint functions still work."""
    from brainsmith.blueprints import get_blueprint
    
    try:
        blueprint = get_blueprint("existing_blueprint_name")
    except Exception:
        pass  # Expected if blueprint doesn't exist, but function should be callable

def test_new_api_maintains_compatibility():
    """Test that new API maintains compatibility with existing patterns."""
    from brainsmith.core.api import brainsmith_explore, explore_design_space
    
    # Test new API
    assert callable(brainsmith_explore)
    
    # Test compatibility wrapper
    assert callable(explore_design_space)
```

### Week 10: Documentation and Finalization

#### Day 41-45: Documentation and Examples

**Create documentation structure**:
```bash
mkdir -p docs/examples/existing_components
touch docs/examples/existing_components/basic_usage.md
touch docs/examples/existing_components/blueprint_migration.md
touch docs/examples/existing_components/api_migration.md
```

**Priority 1: Usage Examples**
```markdown
# docs/examples/existing_components/basic_usage.md

# Using Brainsmith with Existing Components

This guide shows how to use the new Brainsmith architecture with existing components only.

## Quick Start with Existing Components

```python
from brainsmith.core.api import brainsmith_explore

# Use hierarchical exit points with existing components
results, analysis = brainsmith_explore(
    model_path="model.onnx",
    blueprint_path="existing_components_blueprint.yaml",
    exit_point="roofline"  # Fast analysis using existing tools
)

print(f"Analysis method: {analysis['method']}")  # Shows "existing_tools"
```

## CLI Usage with Existing Components

```bash
# Quick roofline analysis using existing tools
brainsmith roofline model.onnx blueprint.yaml

# Dataflow analysis using existing transforms
brainsmith dataflow model.onnx blueprint.yaml

# Full generation using existing FINN flow
brainsmith generate model.onnx blueprint.yaml
```

## Blueprint for Existing Components Only

```yaml
name: "existing_components_example"
description: "Example using only existing components"

# Existing kernels from custom_op/
kernels:
  available:
    - name: "convolution"
      source: "custom_op.hw_conv2d"
      parameters:
        simd: {type: "integer", range: [1, 64], default: 1}
        pe: {type: "integer", range: [1, 64], default: 1}

# Existing transforms from steps/
transforms:
  pipeline:
    - name: "streamlining"
      source: "steps.streamlining"
      enabled: true

# Existing optimization from dse/
hw_optimization:
  strategies:
    - name: "bayesian"
      source: "dse.external.skopt"
      budget: 50

# Existing FINN interface
finn_interface:
  legacy_config:
    fpga_part: "xcvu9p-flga2104-2-i"
    generate_outputs: ["estimate"]
```
```

## Success Criteria and Validation

### Functional Requirements
- [ ] All three hierarchical exit points operational
- [ ] Existing components organized in extensible structure  
- [ ] CLI interface functional with existing components
- [ ] 100% backward compatibility maintained
- [ ] Blueprint validation for existing components working

### Quality Requirements  
- [ ] All existing tests pass
- [ ] New integration tests >90% coverage
- [ ] No performance regressions
- [ ] Documentation covers all usage patterns
- [ ] Example blueprints work out-of-the-box

### Extensibility Requirements
- [ ] Clear extension points documented
- [ ] Library interfaces support future additions
- [ ] Plugin architecture ready for new components
- [ ] 4-hook FINN interface placeholder functional

This execution plan provides a concrete roadmap for implementing the extensible structure using existing components only, with clear priorities and measurable success criteria.