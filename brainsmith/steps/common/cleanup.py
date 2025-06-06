"""
Common cleanup steps that can be used across different model architectures.
"""

from brainsmith.steps import register_step
from qonnx.transformation.general import (
    SortCommutativeInputsInitializerLast, 
    RemoveUnusedTensors, 
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    ConvertDivToMul 
)
from qonnx.transformation.remove import RemoveIdentityOps


@register_step(
    name="common.cleanup",
    category="common",
    description="Basic cleanup operations for ONNX models"
)
def cleanup_step(model, cfg):
    """Basic cleanup steps for ONNX models."""
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model


@register_step(
    name="common.cleanup_advanced",
    category="common", 
    description="Advanced cleanup with tensor naming and unused tensor removal"
)
def cleanup_advanced_step(model, cfg):
    """Advanced cleanup including tensor naming."""
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueNodeNames())
    return model