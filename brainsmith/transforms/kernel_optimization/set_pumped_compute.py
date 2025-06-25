"""Set pumped compute attribute for hardware operations."""

from qonnx.transformation.base import Transformation
import qonnx.custom_op.registry as registry
from brainsmith.plugin.decorators import transform


@transform(
    name="SetPumpedCompute",
    stage="kernel_optimization",
    description="Set pumped compute attribute for MVAUs and DynMatMuls",
    author="brainsmith-team",
    version="1.0.0",
    requires=["qonnx"]
)
class SetPumpedCompute(Transformation):
    """For all MVAUs and DynMatMuls set the pumped compute attribute"""
    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph

        for node in graph.node:
            if (node.op_type == "MVAU_rtl"):
                inst = registry.getCustomOp(node)
                inst.set_nodeattr("pumpedCompute", 1)
        return (model, False)