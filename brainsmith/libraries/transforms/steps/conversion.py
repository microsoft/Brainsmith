"""QONNX to FINN conversion operations with fail-fast dependency checking."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fail-fast imports - clear error if dependencies not available
try:
    from qonnx.transformation.general import ConvertDivToMul
    from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
    from qonnx.transformation.fold_constants import FoldConstants
    QONNX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"QONNX not available: {e}")
    
    # Placeholder implementations
    class PlaceholderTransform:
        def __init__(self, name: str):
            self.name = name
        
        def __call__(self):
            raise RuntimeError(
                f"QONNX transform '{self.name}' not available. "
                f"Install qonnx package: pip install qonnx"
            )
    
    ConvertDivToMul = lambda: PlaceholderTransform("ConvertDivToMul")
    ExtractQuantScaleZeroPt = lambda: PlaceholderTransform("ExtractQuantScaleZeroPt")
    FoldConstants = lambda: PlaceholderTransform("FoldConstants")
    QONNX_AVAILABLE = False

try:
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
    FINN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"FINN not available: {e}")
    ConvertQONNXtoFINN = lambda: PlaceholderTransform("ConvertQONNXtoFINN")
    FINN_AVAILABLE = False

try:
    from brainsmith.transforms.topology_optimization.expand_norms import ExpandNorms
    BRAINSMITH_TRANSFORMS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BrainSmith transforms not available: {e}")
    ExpandNorms = lambda: PlaceholderTransform("ExpandNorms")
    BRAINSMITH_TRANSFORMS_AVAILABLE = False


def qonnx_to_finn_step(model: Any, cfg: Any) -> Any:
    """
    Convert QONNX to FINN with special handling for SoftMax operations.
    
    Category: conversion
    Dependencies: qonnx, finn, brainsmith.transformation
    Description: Converts QONNX models to FINN with SoftMax transformations
    
    Args:
        model: QONNX model object
        cfg: Configuration object
        
    Returns:
        Transformed model
        
    Raises:
        RuntimeError: If required dependencies not available
    
    The SoftMax custom op requires some extra care here, hence
    the requirement for this plugin step.

    QuantSoftMax makes use of the fact that the output
    of SoftMax is well defined between [0,1] so we can
    specify the output as a fixed-point number with 0
    integer bits and N fractional bits (where N is the
    bitwidth of the output datatype).

    For an INT8 model this means we will have:
        SoftMax -> Quant node (scale=1/255)
    in the ONNX model.

    We then call ExtractQuantScaleZeroPt to pull the
    scale calculation out of the Quant. which gives us

        SoftMax -> Div(1/255) -> Quant (scale=1) -> Mul(1/255)

    Then we convert the Div node to a Mul node with
    ConvertDivToMul :
        
        SoftMax -> Mul(255) -> Quant (scale=1) -> Mul(1/255)

    Then we call ConvertQONNXtoFINN to get:
        
        SoftMax -> Mul(255) -> MultiThreshold -> Mul(1/255)

    By having these steps we can have a scale factor of 1
    in the Quant node, then we can deal with the leftover 
    mul nodes later in the streamlining_step streamlining it into
    a MultiThreshold node. (see streamlining.py) 
    """
    missing_deps = []
    if not QONNX_AVAILABLE:
        missing_deps.append("qonnx")
    if not FINN_AVAILABLE:
        missing_deps.append("finn")
    if not BRAINSMITH_TRANSFORMS_AVAILABLE:
        missing_deps.append("brainsmith.transformation")
    
    if missing_deps:
        raise RuntimeError(
            f"qonnx_to_finn_step requires: {', '.join(missing_deps)}. "
            f"Install with: pip install {' '.join(missing_deps)}"
        )
    
    logger.info("Applying QONNX to FINN conversion transformations")
    model = model.transform(ExpandNorms())
    #model = model.transform(ExtractQuantScaleZeroPt())
    model = model.transform(FoldConstants())
    model = model.transform(ConvertDivToMul())
    model = model.transform(ConvertQONNXtoFINN())
    return model