import onnx  
import os
import pytest
import shutil
import argparse
import math
import tempfile

from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
import finn.transformation.streamline as absorb
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
import finn.builder.build_dataflow_config as build_cfg

from finn.builder.build_dataflow_steps import step_hw_codegen, step_hw_ipgen 

from finnbrainsmith.util.bert import custom_step_remove_head, custom_step_remove_tail, custom_step_cleanup
import finnbrainsmith.transformation.convert_to_hw_layers as to_bs_hw

from bert_testing_utils import create_dynamic_fixtures, model, save_dashboard  

test_cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=[],
        output_dir='./',
        synth_clk_period_ns=5,
        fpga_part="xcv80-lsva4737-2MHP-e-S",
        generate_outputs=[],
    )

def custom_streamlining_step(model, cfg):
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    return model

def qonnx2finn_convert_step(model, cfg):
    model = model.transform(ConvertQONNXtoFINN())
    return model

def specialise_layers_step(model,cfg):
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model

def custom_step_infer_hardware(model, cfg):
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferStreamingEltwise())
    model = model.transform(to_hw.InferLookupLayer())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_bs_hw.InferShuffle())
    model = model.transform(to_bs_hw.InferQuantSoftmax())
    return model

def custom_step_create_ip(model, cfg):
    model = model.transform(PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    return model
    
def get_non_specialised_nodes(model)->list:
    """ Returns the list of nodes in the model that have not been specialised """
    specialised = []
    for node in model.graph.node:
        if node.op_type.endswith("rtl") or node.op_type.endswith("hls"):
            specialised.append(node)
    return specialised

def calculate_specialised_layers_ratio(model)->float:
    """ Returns the percentage of layers that were sucessfully specialised """
    return len(get_non_specialised_nodes(model))/len(model.graph.node)

steps = [  
    custom_step_cleanup,  
    custom_step_remove_head,  
    custom_step_remove_tail,  
    qonnx2finn_convert_step,  
    custom_streamlining_step,  
    custom_step_infer_hardware,  
    specialise_layers_step,  
]  
  
create_dynamic_fixtures(steps, globals(), test_cfg)

##############################################
#    Do custom steps complete  
##############################################
def test_model_initial_model_soundness(model):
    """ Test to make sure that the model is sound """
    _ = model.transform(InferShapes())

def test_model_head_removal_completes(custom_step_remove_head):
    _ = custom_step_remove_head.transform(InferShapes()) 

def test_model_tail_removal_completes(custom_step_remove_tail):
    _ = custom_step_remove_tail.transform(InferShapes()) 

def test_qonnx_conversion_completes(qonnx2finn_convert_step):
    _ = qonnx2finn_convert_step.transform(InferShapes()) 

def test_streamlining_completes(custom_streamlining_step):
    _ = custom_streamlining_step.transform(InferShapes()) 

def test_infer_hw_completes(custom_step_infer_hardware):
    _ = custom_step_infer_hardware.transform(InferShapes())

def test_specialise_step_completes(specialise_layers_step):
    _ = specialise_layers_step.transform(InferShapes())

##############################################
#    Specialised layers testing
##############################################
def get_specialised_nodes(specialise_layers_step)->list:
    """ Returns the list of nodes in the model that have not been specialised """
    model = specialise_layers_step
    specialised = []
    for node in model.graph.node:
        if node.op_type.endswith("rtl") or node.op_type.endswith("hls"):
            specialised.append(node)
    return specialised

def calculate_specialised_layers_ratio(specialise_layers_step)->float:
    """ Returns the percentage of layers that were sucessfully specialised """
    model = specialise_layers_step
    return len(get_specialised_nodes(model))/len(model.graph.node)


def test_all_layers_specialised(specialise_layers_step, save_dashboard):
    """ Test to determine if all the layers in the model have been specialised """
    model = specialise_layers_step
    ratio = calculate_specialised_layers_ratio(model)
    dashboard["specialised_ratio"] = ratio
    dashboard["specialised_layers"] = [x.name for x in get_specialised_nodes(model)]
    dashboard["non_specialised_layers"] = [x.name for x in model.graph.node if x not in get_specialised_nodes(model)]
    if ratio < 1.0:
        raise RuntimeError(f"Not all layers were specialised only {ratio*100}% were")

##############################################
#       Generate Hardware Testing 
##############################################
def test_hw_generation_step(specialise_layers_step, save_dashboard):
    model = specialise_layers_step
    #step_hw_codegen, step_hw_ipgen
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["FINN_HOST_BUILD_DIR"] = temp_dir
        try:
            model = step_hw_codegen(model, cfg)
        except:
            pass

        dashboard['gen_hw_list'] = os.listdir(temp_dir)
        dashboard['gen_hw_ratio'] = len(model.graph.node) / len(os.listdir(temp_dir))

        if len(model.graph.node) > len(os.listdir(temp_dir)):
            raise RuntimeError(f"Only {len(oslistdir(temp_dir))/len(model.graph.node):.2f}% of layers have generated hardware")

##############################################
#       Create IP testing 
##############################################
def test_create_ip(specialise_layers_step):
    """ Test to see if we can create IP from the specialised model """
    model = specialise_layers_step
    model = model.transform(PrepareIP(test_cfg._resolve_fpga_part(), test_cfg._resolve_hls_clk_period()))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_cfg._resolve_fpga_part(), test_cfg._resolve_hls_clk_period()))


