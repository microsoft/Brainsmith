# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Validation utilities for blueprint parsing.
"""

from typing import Any, Dict, List, Optional, Union

from .design_space import ForgeConfig, OutputStage


# Constants
SKIP_INDICATORS = {None, "~", ""}


def validate_step(step: Optional[str]) -> str:
    """
    Validate a step name against the registry.
    
    Args:
        step: Step name to validate
        
    Returns:
        Validated step name or skip indicator
        
    Raises:
        ValueError: If step not found in registry
    """
    if step in SKIP_INDICATORS:
        return "~"
    
    from .plugins.registry import has_step
    if not has_step(step):
        raise ValueError(f"Step '{step}' not found in registry")
    
    return step


def validate_finn_config(forge_config: ForgeConfig, finn_config: Dict[str, Any]) -> None:
    """
    Validate required FINN fields for synthesis.
    
    Args:
        forge_config: Forge configuration
        finn_config: FINN-specific configuration
        
    Raises:
        ValueError: If required fields are missing
    """
    if forge_config.output_stage != OutputStage.GENERATE_REPORTS:
        if 'synth_clk_period_ns' not in finn_config:
            raise ValueError("Hardware synthesis requires synth_clk_period_ns (or target_clk)")
        if 'board' not in finn_config:
            raise ValueError("Hardware synthesis requires board (or platform)")


def validate_and_resolve_kernels(
    kernel_name: str, 
    backend_names: List[str]
) -> List[Any]:
    """
    Validate and resolve backend classes for a kernel.
    
    Args:
        kernel_name: Name of the kernel
        backend_names: List of backend names to validate
        
    Returns:
        List of backend classes
        
    Raises:
        ValueError: If backend not found or doesn't support kernel
    """
    from .plugins.registry import list_backends_by_kernel, get_backend
    
    available = list_backends_by_kernel(kernel_name)
    backend_classes = []
    
    for name in backend_names:
        backend_class = get_backend(name)
        if not backend_class:
            raise ValueError(f"Backend '{name}' not found in registry")
        if name not in available:
            raise ValueError(
                f"Backend '{name}' does not support kernel '{kernel_name}'. "
                f"Available backends: {available}"
            )
        backend_classes.append(backend_class)
    
    return backend_classes