"""BERT-specific graph surgery operations."""

from qonnx.transformation.general import RemoveUnusedTensors, GiveReadableTensorNames


def remove_head_step(model, cfg):
    """
    Remove all nodes up to the first LayerNormalization node and rewire input.
    
    Category: bert
    Dependencies: []
    Description: BERT-specific head removal for models
    """
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

    # Reconnect input
    for con in consumers:
        for i,ip in enumerate(con.input):
            if ip == LN_output:
                con.input[i] = model.graph.input[0].name

    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())

    return model


def _recurse_model_tail_removal(model, to_remove, node):
    """Helper function for recursively walking the BERT graph from the second
    output up to the last LayerNorm to remove it"""
    if node is not None:
        if node.op_type != "LayerNormalization":
            to_remove.append(node)
            for tensor in node.input:
                _recurse_model_tail_removal(model, to_remove, model.find_producer(tensor))
    return


def remove_tail_step(model, cfg):
    """
    Remove from global_out_1 all the way back to the first LayerNorm.
    
    Category: bert
    Dependencies: []
    Description: BERT-specific tail removal for models
    """
    out_names = [x.name for x in model.graph.output]
    assert "global_out_1" in out_names, "Error: expected one of the outputs to be called global_out_1, we might need better pattern matching logic here"

    to_remove = []
    current_node = model.find_producer('global_out_1')
    _recurse_model_tail_removal(model, to_remove, current_node)

    for node in to_remove:
        model.graph.node.remove(node)
    del model.graph.output[out_names.index('global_out_1')]

    return model