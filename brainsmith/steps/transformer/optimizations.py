"""
Transformer-specific optimization steps.
"""

from brainsmith.steps import register_step
from qonnx.transformation.base import Transformation
import qonnx.custom_op.registry as registry


class SetPumpedCompute(Transformation):
    """ For all MVAUs and DynMatMuls set the pumped compute attribute """
    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph

        for node in graph.node:
            if (node.op_type == "MVAU_rtl"):
                inst = registry.getCustomOp(node)
                inst.set_nodeattr("pumpedCompute", 1)
        return (model, False)


class TempShuffleFixer(Transformation):
    """ A temporary transformation that ensures that shuffles are sized correctly for the
    initial BERT builds """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph

        for node in graph.node:
            if node.op_type == "Shuffle_hls":
                inst = registry.getCustomOp(node)
                inner_moves = inst.get_nodeattr("inner_moves")
                simd = inst.get_nodeattr("SIMD")
                if (inner_moves == 1) and (simd > 1):
                    print(f"WARNING: as a safety precaution changing the shuffle where the inner dimension moves to SIMD=1 \n{node=}")
                    inst.set_nodeattr("SIMD", 1)
        return (model, False)


@register_step(
    name="transformer.constrain_folding_and_set_pumped_compute",
    category="transformer",
    description="Apply temporary optimizations for BERT builds"
)
def constrain_folding_and_set_pumped_compute_step(model, cfg):
    """Apply temporary optimizations for BERT builds."""
    model = model.transform(TempShuffleFixer())
    model = model.transform(SetPumpedCompute())
    return model