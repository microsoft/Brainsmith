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


def fix_dynamic_dimensions_step(model, cfg):
    """
    Fix all dynamic dimensions in the model to concrete values.
    
    Category: cleanup
    Dependencies: []
    Description: Converts all dynamic dimensions to batch size 1
    
    This step is crucial for hardware inference which requires concrete dimensions.
    It converts any remaining dynamic dimensions (like 'unk__0') to the value 1.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    changes_made = 0
    
    # Fix graph inputs
    for inp in model.graph.input:
        for i, dim in enumerate(inp.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in input {inp.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    # Fix value_info tensors (intermediate tensors)
    for vi in model.graph.value_info:
        for i, dim in enumerate(vi.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in tensor {vi.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    # Fix graph outputs
    for out in model.graph.output:
        for i, dim in enumerate(out.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in output {out.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    logger.info(f"Fixed {changes_made} dynamic dimensions")
    return model