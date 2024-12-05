import onnx  
import argparse
from onnxsim import simplify  
from qonnx.util.cleanup import cleanup
from qonnx.transformation.general import SortCommutativeInputsInitializerLast
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.remove import remove_node_and_rewire
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
import finn.transformation.streamline as absorb
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finnbrainsmith.transformation.convert_to_hw_layers as to_bs_hw
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

def custom_infer_shuffle(model, cfg):
    model = model.transform(to_bs_hw.InferShuffle())
    return model

def custom_infer_quantsoftmax(model, cfg):
    model = model.transform(to_bs_hw.InferQuantSoftmax())
    return model

def custom_step_cleanup(model, cfg):
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())

    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type == "Dropout":
            print(f"Found a Dropout node {node.name} to remove")
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        remove_node_and_rewire(model, node)

    return model

def custom_streamlining_step(model,cfg):
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    return model

def attempt_convert_step(model, cfg):
    model = model.transform(ConvertQONNXtoFINN())
    return model

def attempt_specialise_layers(model, cfg):
    model = model.transform(SpecializeLayers(fpgapart=cfg.fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model

def custom_step_infer_hardware(model, cfg):
    # infer duplicate streams
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferStreamingEltwise())
    model = model.transform(to_hw.InferLookupLayer())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    return model

def custom_step_create_ip(model, cfg):
    model = model.transform(PrepareIP(cfg.fpga_part, cfg.synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(cfg.fpga_part, cfg.synth_clk_period_ns))
    return model

def main(model_path:str):
    model = onnx.load(model_path)  
    
    steps = [ custom_step_cleanup, 
              custom_streamlining_step, 
              custom_step_infer_hardware, 
              custom_infer_shuffle, 
              custom_infer_quantsoftmax, 
              attempt_specialise_layers ]
    
    cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=steps,
        output_dir="./",
        synth_clk_period_ns=5,
        fpga_part="xcv80-lsva4737-2MHP-e-S",
        generate_outputs=[],
    )
    
    _ = build.build_dataflow_cfg(model_path, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBERT pipecleaner builder script')
    parser.add_argument('-i', '--input', help='Input ONNX file path', required=True)

    args = parser.parse_args()
    main(args.input)
