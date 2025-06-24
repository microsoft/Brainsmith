"""BERT tail removal transform."""

from qonnx.transformation.base import Transformation
from brainsmith.plugin.decorators import transform


def _recurse_model_tail_removal(model, to_remove, node):
    """Helper function for recursively walking the BERT graph from the second
    output up to the last LayerNorm to remove it"""
    if node is not None:
        if node.op_type != "LayerNormalization":
            to_remove.append(node)
            for tensor in node.input:
                _recurse_model_tail_removal(model, to_remove, model.find_producer(tensor))
    return


@transform(
    name="RemoveBertTail",
    stage="model_specific",
    description="Remove from global_out_1 all the way back to the first LayerNorm",
    author="shane.fleming",
    version="1.0.0",
    requires=["qonnx"]
)
class RemoveBertTail(Transformation):
    """Remove from global_out_1 all the way back to the first LayerNorm."""
    
    def apply(self, model):
        out_names = [x.name for x in model.graph.output]
        assert "global_out_1" in out_names, "Error: expected one of the outputs to be called global_out_1, we might need better pattern matching logic here"

        to_remove = []
        current_node = model.find_producer('global_out_1')
        _recurse_model_tail_removal(model, to_remove, current_node)

        for node in to_remove:
            model.graph.node.remove(node)
        del model.graph.output[out_names.index('global_out_1')]

        return (model, len(to_remove) > 0)