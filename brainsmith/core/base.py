# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base classes for Brainsmith components.

This module defines the base classes that all plugins should inherit from:
- Step: Transformation/compilation steps
- CustomOp: Custom ONNX operations (kernels)
- Backend: Hardware backend implementations

Using these base classes enables:
- Type-safe metadata (class attributes instead of dicts)
- IDE autocomplete and type checking
- Explicit, refactorable code
- Natural inheritance and composition
"""

from typing import Any, Optional, Type
from abc import ABC, abstractmethod


class Step(ABC):
    """Base class for transformation steps.

    Steps are callables that transform a model, typically used in compilation
    pipelines. They can be functions or classes with __call__.

    Attributes:
        name: Step name for registration (defaults to __name__ if None)

    Examples:
        >>> class MyOptimization(Step):
        ...     name = "my_optimization"
        ...
        ...     def __call__(self, model, **kwargs):
        ...         # Transform model
        ...         return model

        >>> # Or as a function
        >>> def streamline_step(model, **kwargs):
        ...     return model
    """

    name: Optional[str] = None

    @abstractmethod
    def __call__(self, model: Any, **kwargs: Any) -> Any:
        """Apply transformation to model.

        Args:
            model: QONNX ModelWrapper to transform
            **kwargs: Step-specific parameters

        Returns:
            Transformed model
        """
        raise NotImplementedError


class CustomOp(ABC):
    """Base class for custom ONNX operations (kernels).

    Custom operations represent ONNX nodes with custom semantics beyond
    standard ONNX ops. They define shape inference, execution, and
    optionally hardware backends.

    Attributes:
        op_type: ONNX operation type (e.g., "LayerNorm", "MVAU")
        infer_transform: Associated InferTransform class for shape inference
        domain: ONNX domain (default: "finn.custom")

    Examples:
        >>> class LayerNorm(CustomOp):
        ...     op_type = "LayerNorm"
        ...     infer_transform = InferLayerNorm
        ...
        ...     def __init__(self, onnx_node):
        ...         super().__init__(onnx_node)
        ...         self.epsilon = get_attribute(onnx_node, "epsilon")
        ...
        ...     def execute(self, context):
        ...         # Execute operation
        ...         pass
    """

    op_type: Optional[str] = None
    infer_transform: Optional[Type] = None
    domain: str = "finn.custom"

    def __init__(self, onnx_node: Any):
        """Initialize custom op from ONNX node.

        Args:
            onnx_node: ONNX NodeProto defining this operation
        """
        self.onnx_node = onnx_node

    @abstractmethod
    def execute(self, context: Any) -> Any:
        """Execute the operation.

        Args:
            context: Execution context with inputs/outputs

        Returns:
            Execution result
        """
        raise NotImplementedError


class Backend(ABC):
    """Base class for hardware backend implementations.

    Backends generate hardware-specific code (HLS, RTL, etc.) for custom
    operations. Multiple backends can target the same kernel, enabling
    users to provide optimized implementations.

    Attributes:
        target_kernel: Full name of kernel this backend implements (e.g., 'brainsmith:LayerNorm')
        language: Backend language ('hls', 'rtl', 'chisel', etc.)
        variant: Optional variant name for multiple backends of same language

    Examples:
        >>> class LayerNormHLS(Backend):
        ...     target_kernel = 'brainsmith:LayerNorm'
        ...     language = 'hls'
        ...
        ...     def __init__(self, onnx_node):
        ...         super().__init__(onnx_node)
        ...
        ...     def generate_code(self):
        ...         # Generate HLS code
        ...         return hls_code

        >>> # Multiple backends for same kernel
        >>> class LayerNormHLS_Fast(Backend):
        ...     target_kernel = 'brainsmith:LayerNorm'
        ...     language = 'hls'
        ...     variant = 'fast'  # Distinguishes from default
    """

    target_kernel: Optional[str] = None
    language: Optional[str] = None
    variant: Optional[str] = None

    def __init__(self, onnx_node: Any):
        """Initialize backend from ONNX node.

        Args:
            onnx_node: ONNX NodeProto to generate code for
        """
        self.onnx_node = onnx_node

    @abstractmethod
    def generate_code(self) -> str:
        """Generate hardware implementation code.

        Returns:
            Generated code as string
        """
        raise NotImplementedError


# Convenience: Allow plain functions as Steps
def step_from_function(func):
    """Wrap a function as a Step.

    This allows using plain functions as steps without explicit class definition.
    The registry automatically wraps functions when registered.

    Args:
        func: Function that takes (model, **kwargs) and returns model

    Returns:
        Step instance wrapping the function
    """

    class FunctionStep(Step):
        name = func.__name__

        def __call__(self, model, **kwargs):
            return func(model, **kwargs)

    # Preserve function metadata
    FunctionStep.__name__ = func.__name__
    FunctionStep.__doc__ = func.__doc__
    FunctionStep.__module__ = func.__module__

    return FunctionStep()
