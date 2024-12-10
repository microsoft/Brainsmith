import onnx  
import argparse
from onnxsim import simplify  
from qonnx.util.cleanup import cleanup
from qonnx.transformation.general import SortCommutativeInputsInitializerLast
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.remove import remove_node_and_rewire
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

def custom_step_remove_head(model, cfg):
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

    # Reconnect input
    for con in consumers:
        for i,ip in enumerate(con.input):
            if ip == LN_output:
                con.input[i] = model.graph.input[0].name

    return model

def recurse_model_tail_removal(model, to_remove, node):
    """ Helper function for recursively walking the BERT graph from the second
    output up to the last LayerNorm to remove it """
    if node is not None:
        if node.op_type != "LayerNormalization":
            to_remove.append(node)
            for tensor in node.input:
                recurse_model_tail_removal(model, to_remove, model.find_producer(tensor))
    return

def custom_step_remove_tail(model, cfg):
    """ Removes from global_out_1 all the way back to the first LayerNorm """
    out_names = [x.name for x in model.graph.output]
    assert "global_out_1" in out_names, "Error: expected one of the outputs to be called global_out_1, we might need better pattern matching logic here"
    
    to_remove = []
    current_node = model.find_producer('global_out_1')
    recurse_model_tail_removal(model, to_remove, current_node)
    
    for node in to_remove:
        model.graph.node.remove(node)
    del model.graph.output[out_names.index('global_out_1')]
    
    return model


def custom_step_cleanup(model, cfg):
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model

def main(model_path:str):
    model = onnx.load(model_path)  
    
    model_simp, check = simplify(model)  
    if check:  
        print("Simplification successful.")  
        onnx.save(model_simp, "simp_"+model_path)  
    else:  
        print("Simplification failed.")  
    
    cleanup(in_file="simp_"+model_path, out_file="qonnx_cleanup_"+model_path)
    
    
    steps = [ custom_step_cleanup, custom_step_remove_head, custom_step_remove_tail ]
    
    cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=steps,
        output_dir="./",
        synth_clk_period_ns=5,
        fpga_part="xcv80-lsva4737-2MHP-e-S",
        generate_outputs=[],
    )
    
    _ = build.build_dataflow_cfg("qonnx_cleanup_"+model_path, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBERT cleanup script')
    parser.add_argument('-i', '--input', help='Input ONNX file path', required=True)

    args = parser.parse_args()
    main(args.input)
