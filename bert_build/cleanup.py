import onnx  
import argparse
from onnxsim import simplify  
from qonnx.util.cleanup import cleanup
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.remove import remove_node_and_rewire
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

def custom_step_cleanup(model, cfg):
    model = model.transform(RemoveIdentityOps())
    
    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type == "Dropout":
            print(f"Found a Dropout node {node.name} to remove")
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        remove_node_and_rewire(model, node)

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
    
    
    steps = [ custom_step_cleanup ]
    
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
