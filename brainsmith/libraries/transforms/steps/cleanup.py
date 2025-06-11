"""ONNX model cleanup operations."""

from qonnx.transformation.general import (
    SortCommutativeInputsInitializerLast, 
    RemoveUnusedTensors, 
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    ConvertDivToMul 
)
from qonnx.transformation.remove import RemoveIdentityOps


def cleanup_step(model, cfg):
    """
    Basic cleanup operations for ONNX models.
    
    Category: cleanup
    Dependencies: []
    Description: Removes identity operations and sorts commutative inputs
    """
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model


def cleanup_advanced_step(model, cfg):
    """
    Advanced cleanup with tensor naming and unused tensor removal.
    
    Category: cleanup
    Dependencies: []
    Description: Extended cleanup including readable naming and tensor pruning
    """
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueNodeNames())
    return model