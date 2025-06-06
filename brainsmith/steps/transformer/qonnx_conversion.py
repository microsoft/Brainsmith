"""
Transformer-specific QONNX to FINN conversion operations.
"""

from brainsmith.steps import register_step
from qonnx.transformation.general import ConvertDivToMul
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.fold_constants import FoldConstants
from brainsmith.transformation.expand_norms import ExpandNorms


@register_step(
    name="transformer.qonnx_to_finn",
    category="transformer",
    description="Convert QONNX to FINN with special handling for SoftMax operations"
)
def qonnx_to_finn_step(model, cfg):
    """
    BERT custom step for converting between QONNX and FINN-ONNX

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
    a MultiThreshold node. (see custom_streamlining_step below) 

    """
    model = model.transform(ExpandNorms())
    #model = model.transform(ExtractQuantScaleZeroPt())
    model = model.transform(FoldConstants())
    model = model.transform(ConvertDivToMul())
    model = model.transform(ConvertQONNXtoFINN())
    return model