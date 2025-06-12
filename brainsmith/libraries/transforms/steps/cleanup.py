"""ONNX model cleanup operations with fail-fast dependency checking."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fail-fast imports - clear error if qonnx not available
try:
    from qonnx.transformation.general import (
        SortCommutativeInputsInitializerLast, 
        RemoveUnusedTensors, 
        GiveReadableTensorNames,
        GiveUniqueNodeNames,
        ConvertDivToMul 
    )
    from qonnx.transformation.remove import RemoveIdentityOps
    QONNX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"QONNX not available: {e}")
    
    # Placeholder implementations for Docker-only deployment
    class PlaceholderTransform:
        """Placeholder transform for missing qonnx dependency."""
        def __init__(self, name: str):
            self.name = name
        
        def __call__(self):
            raise RuntimeError(
                f"QONNX transform '{self.name}' not available. "
                f"Install qonnx package: pip install qonnx"
            )
    
    # Create placeholder classes that fail at usage time with clear errors
    SortCommutativeInputsInitializerLast = lambda: PlaceholderTransform("SortCommutativeInputsInitializerLast")
    RemoveUnusedTensors = lambda: PlaceholderTransform("RemoveUnusedTensors")
    GiveReadableTensorNames = lambda: PlaceholderTransform("GiveReadableTensorNames") 
    GiveUniqueNodeNames = lambda: PlaceholderTransform("GiveUniqueNodeNames")
    ConvertDivToMul = lambda: PlaceholderTransform("ConvertDivToMul")
    RemoveIdentityOps = lambda: PlaceholderTransform("RemoveIdentityOps")
    QONNX_AVAILABLE = False


def cleanup_step(model: Any, cfg: Any) -> Any:
    """
    Basic cleanup operations for ONNX models.
    
    Category: cleanup
    Dependencies: qonnx
    Description: Removes identity operations and sorts commutative inputs
    
    Args:
        model: ONNX model object
        cfg: Configuration object
        
    Returns:
        Transformed model
        
    Raises:
        RuntimeError: If qonnx dependency not available
    """
    if not QONNX_AVAILABLE:
        raise RuntimeError(
            "cleanup_step requires qonnx package. "
            "Install with: pip install qonnx"
        )
    
    logger.info("Applying basic cleanup transformations")
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model


def cleanup_advanced_step(model: Any, cfg: Any) -> Any:
    """
    Advanced cleanup with tensor naming and unused tensor removal.
    
    Category: cleanup
    Dependencies: qonnx
    Description: Extended cleanup including readable naming and tensor pruning
    
    Args:
        model: ONNX model object
        cfg: Configuration object
        
    Returns:
        Transformed model
        
    Raises:
        RuntimeError: If qonnx dependency not available
    """
    if not QONNX_AVAILABLE:
        raise RuntimeError(
            "cleanup_advanced_step requires qonnx package. "
            "Install with: pip install qonnx"
        )
    
    logger.info("Applying advanced cleanup transformations")
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueNodeNames())
    return model