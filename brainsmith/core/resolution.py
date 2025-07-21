"""
Plugin Resolution Functions

This module handles resolving plugin names from the blueprint to actual
classes using the plugin registry.
"""

from typing import Dict, List, Optional, Tuple, Type, Union

from qonnx.transformation.base import Transformation

from .plugins.registry import get_registry
from .execution_tree import TransformStage


def resolve_transform_spec(spec: Union[str, List]) -> List[Optional[Type[Transformation]]]:
    """
    Resolve transform specification to list of transform class options.
    
    Args:
        spec: Either a string (single required transform) or list (multiple options)
        
    Returns:
        List of transform classes, where None represents skip option
    """
    registry = get_registry()
    
    if isinstance(spec, list):
        # List = multiple options (mutually exclusive)
        options = []
        for name in spec:
            if name == "~":
                options.append(None)  # Skip option
            else:
                transform_class = registry.get_transform(name)
                if not transform_class:
                    raise ValueError(f"Transform '{name}' not found in registry")
                options.append(transform_class)
        return options
    else:
        # Single string = required transform
        transform_class = registry.get_transform(spec)
        if not transform_class:
            raise ValueError(f"Transform '{spec}' not found in registry")
        return [transform_class]


def resolve_kernel_spec(spec: Union[str, Dict]) -> Tuple[str, List[Type]]:
    """
    Resolve kernel specification to kernel name and backend classes.
    
    Args:
        spec: Either kernel name string or dict with kernel: backends mapping
        
    Returns:
        Tuple of (kernel_name, [backend_classes])
    """
    registry = get_registry()
    
    if isinstance(spec, str):
        # Just kernel name - get all available backends
        kernel_name = spec
        backend_names = registry.list_backends_by_kernel(kernel_name)
        
        if not backend_names:
            raise ValueError(f"No backends found for kernel '{kernel_name}'")
        
        # Get backend classes
        backend_classes = []
        for backend_name in backend_names:
            backend_class = registry.get_backend(backend_name)
            if backend_class:
                backend_classes.append(backend_class)
        
        return (kernel_name, backend_classes)
    
    elif isinstance(spec, dict):
        # Kernel with specific backends
        if len(spec) != 1:
            raise ValueError(f"Kernel spec must have exactly one key: {spec}")
        
        kernel_name, backend_specs = next(iter(spec.items()))
        
        # Normalize to list
        if isinstance(backend_specs, str):
            backend_specs = [backend_specs]
        
        # Resolve backend names to classes
        backend_classes = []
        for backend_name in backend_specs:
            backend_class = registry.get_backend(backend_name)
            if not backend_class:
                raise ValueError(f"Backend '{backend_name}' not found in registry")
            
            # Verify backend supports this kernel
            available = registry.list_backends_by_kernel(kernel_name)
            if backend_name not in available:
                raise ValueError(
                    f"Backend '{backend_name}' does not support kernel '{kernel_name}'. "
                    f"Available backends: {available}"
                )
            
            backend_classes.append(backend_class)
        
        return (kernel_name, backend_classes)
    
    else:
        raise ValueError(f"Invalid kernel spec type: {type(spec)}")


def parse_transform_stage(stage_name: str, stage_spec: List) -> TransformStage:
    """
    Parse a transform stage specification into a TransformStage object.
    
    Args:
        stage_name: Name of the stage
        stage_spec: List of transform specifications for this stage
        
    Returns:
        TransformStage with resolved transform classes
    """
    transform_steps = []
    
    for spec in stage_spec:
        options = resolve_transform_spec(spec)
        transform_steps.append(options)
    
    return TransformStage(stage_name, transform_steps)


def validate_pipeline_steps(steps: List[str], transform_stages: Dict[str, TransformStage]) -> None:
    """
    Validate that all transform stage references in pipeline exist.
    
    Args:
        steps: List of pipeline steps
        transform_stages: Available transform stages
        
    Raises:
        ValueError: If a referenced stage doesn't exist
    """
    for step in steps:
        if step.startswith("{") and step.endswith("}"):
            stage_name = step[1:-1]
            if stage_name not in transform_stages:
                raise ValueError(
                    f"Pipeline references stage '{stage_name}' which is not defined. "
                    f"Available stages: {list(transform_stages.keys())}"
                )