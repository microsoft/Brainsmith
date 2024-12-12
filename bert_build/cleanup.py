import onnx  
import os
import argparse
from onnxsim import simplify  
from qonnx.util.cleanup import cleanup
from qonnx.transformation.general import SortCommutativeInputsInitializerLast
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.remove import remove_node_and_rewire
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finnbrainsmith.util.bert import custom_step_remove_head, custom_step_remove_tail, custom_step_cleanup

def main(model_path:str):
    tmp = "./intermediate_models"
    os.makedirs(tmp, exist_ok=True)

    # Initial model cleanup
    model = onnx.load(model_path)  
    model_simp, check = simplify(model)  
    if check:  
        onnx.save(model_simp, f"{tmp}/simp.onnx")  
    else:  
        raise RuntimeError(f"Unable to simplify the Brevitas bert model")
    cleanup(in_file=f"{tmp}/simp.onnx", out_file=f"{tmp}/qonnx_cleanup.onnx")
    
    steps = [ custom_step_cleanup, custom_step_remove_head, custom_step_remove_tail ]
    cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=steps,
        output_dir=tmp,
        synth_clk_period_ns=5,
        fpga_part="xcv80-lsva4737-2MHP-e-S",
        generate_outputs=[],
    )
    
    cleaned_up_model = build.build_dataflow_cfg(f"{tmp}/qonnx_cleanup.onnx", cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBERT cleanup script')
    parser.add_argument('-i', '--input', help='Input ONNX file path', required=True)

    args = parser.parse_args()
    main(args.input)
