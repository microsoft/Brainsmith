"""BERT head removal transform."""

from qonnx.transformation.base import Transformation
from qonnx.transformation.general import RemoveUnusedTensors, GiveReadableTensorNames
from brainsmith.plugin.decorators import transform


@transform(
    name="RemoveBertHead",
    stage="model_specific",
    description="Remove all nodes up to the first LayerNormalization node and rewire input",
    author="shane.fleming",
    version="1.0.0",
    requires=["qonnx"]
)
class RemoveBertHead(Transformation):
    """Remove all nodes up to the first LayerNormalization node and rewire input."""
    
    def apply(self, model):
        assert len(model.graph.input) == 1, "Error the graph has more inputs than expected"
        tensor_to_node = {output: node for node in model.graph.node for output in node.output}

        to_remove = []

        current_tensor = model.graph.input[0].name
        current_node = model.find_consumer(current_tensor)
        while current_node.op_type != "LayerNormalization":
            to_remove.append(current_node)
            assert len(current_node.output) == 1, "Error expected an linear path to the first LN"
            current_tensor = current_node.output[0]
            current_node = model.find_consumer(current_tensor)

        # Send the global input to the consumers of the layernorm output
        LN_output = current_node.output[0]
        consumers = model.find_consumers(LN_output)

        # Remove nodes
        to_remove.append(current_node)
        for node in to_remove:
            model.graph.node.remove(node)

        in_vi = model.get_tensor_valueinfo(LN_output)
        model.graph.input.pop()
        model.graph.input.append(in_vi)
        model.graph.value_info.remove(in_vi)
        
        # Fix dynamic batch dimension to concrete value
        # The hardware inference needs concrete dimensions
        if model.graph.input[0].type.tensor_type.shape.dim[0].HasField('dim_param'):
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

        # Reconnect input
        for con in consumers:
            for i,ip in enumerate(con.input):
                if ip == LN_output:
                    con.input[i] = model.graph.input[0].name

        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())

        return (model, True)