"""
Transformer-specific streamlining operations.
"""

from brainsmith.steps import register_step
import finn.transformation.streamline as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveUniqueNodeNames


@register_step(
    name="transformer.streamlining",
    category="transformer",
    description="Custom streamlining for transformer models with SoftMax handling",
    dependencies=["transformer.qonnx_to_finn"]
)
def streamlining_step(model, cfg):
    """
    BERT custom step for streamlining

    Some additional streamlining steps are required here
    to handle the Mul nodes leftover from the SoftMax
    transformations done in custom_step_qonnx2finn.

    In particular, we need to move the Mul operation
    at the output of the QuantSoftMax lower in the graph
    so that it has the option to be merged into a MultiThreshold 
    node. In particular:

        * MoveScalarMulPastMatMul : moves the Mul past the DynMatMul
        * ModeScalarLinearPartInvariants : moves the Mul over the
          reshape and transpose
        * AbsorbMulIntoMultiThreshold : absorbs the Mul into the MT

    """
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(reorder.MoveOpPastFork(["Mul"]))
    model = model.transform(reorder.MoveScalarMulPastMatMul())
    model = model.transform(reorder.MoveScalarLinearPastInvariants())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(InferDataTypes(allow_scaledint_dtypes=False))
    model = model.transform(GiveUniqueNodeNames())
    return model