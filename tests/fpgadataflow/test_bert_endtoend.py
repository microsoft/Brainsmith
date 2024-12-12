import onnx  
import os
import pytest
import shutil
import argparse
import math
import tempfile

from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, ConvertDivToMul
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from qonnx.transformation.infer_shapes import InferShapes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
import finn.transformation.streamline as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
import finn.builder.build_dataflow_config as build_cfg

from finnbrainsmith.util.bert import custom_step_remove_head, custom_step_remove_tail, custom_step_cleanup
import finnbrainsmith.transformation.convert_to_hw_layers as to_bs_hw
from bert_testing_utils import create_dynamic_fixtures, model 

# The default steps
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    step_convert_to_hw,
    step_create_dataflow_partition,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_create_stitched_ip,
    step_measure_rtlsim_performance,
    step_out_of_context_synthesis,
    step_synthesize_bitfile,
    step_make_pynq_driver,
    step_deployment_package,
)

test_cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=[],
        output_dir='./',
        synth_clk_period_ns=5,
        fpga_part="xcv80-lsva4737-2MHP-e-S",
        generate_outputs=[],
    )

# Save a json file with the current status of the endtoend flow for tracking
import json
dashboard = {}

@pytest.fixture
def save_dashboard():
    """ save the dashboard to a file at the end of a test.
        runs at the end of all tests.
    """
    yield
    with open("end2end_test_dashboard.json", "w") as fp:
        json.dump(dashboard, fp, indent=4)

def custom_step_qonnx2finn(model, cfg):
    model = model.transform(ExtractQuantScaleZeroPt())
    model = model.transform(ConvertDivToMul())
    model = model.transform(ConvertQONNXtoFINN())
    return model

def custom_streamlining_step(model, cfg):
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(reorder.MoveScalarMulPastMatMul())
    model = model.transform(reorder.MoveScalarLinearPastInvariants())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    return model

def custom_step_infer_hardware(model, cfg):
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(to_hw.InferLookupLayer())
    model = model.transform(to_bs_hw.InferShuffle())
    model = model.transform(to_bs_hw.InferQuantSoftmax())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    return model

def custom_step_specialise_layers(model, cfg):
    model = model.transform(SpecializeLayers(fpgapart=cfg.fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model

def custom_step_create_ip(model, cfg):
    model = model.transform(PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    return model
    
steps = [  

    # Cleanup and custom graph surgery
    custom_step_cleanup,  
    custom_step_remove_head,  
    custom_step_remove_tail,  

    # Conversion
    custom_step_qonnx2finn, 

    # Streamlining
    custom_streamlining_step,  

    # Infer Hardware
    custom_step_infer_hardware,  

    # dataflow partition
    #step_create_dataflow_partition,

    # Specialise the hardware layers
    custom_step_specialise_layers,

    # How far do we get
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_create_stitched_ip,
    step_measure_rtlsim_performance,
]  
  
create_dynamic_fixtures(steps, globals(), test_cfg)

##############################################
#    Test buildflow steps 
##############################################
# Generate tests for each step and at the start a complete model generation
for step_func in steps:
    def test_model_generation(request, step_func=step_func):
        step_fixture = request.getfixturevalue(step_func.__name__)
        _ = step_fixture.transform(InferShapes())

    test_func_name = f"test_{step_func.__name__}"
    test_model_generation.__name__ = test_func_name

    globals()[test_func_name] = pytest.mark.usefixtures(step_func.__name__)(test_model_generation)

##############################################
#    Specialised layers testing
##############################################

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

def get_specialised_nodes(custom_step_specialise_layers)->list:
    """ Returns the list of nodes in the model that have not been specialised """
    model = custom_step_specialise_layers
    specialised = []
    for node in model.graph.node:
        if node.op_type.endswith("rtl") or node.op_type.endswith("hls"):
            specialised.append(node)
    return specialised

def calculate_specialised_layers_ratio(custom_step_specialise_layers)->float:
    """ Returns the percentage of layers that were sucessfully specialised """
    model = custom_step_specialise_layers
    return len(get_specialised_nodes(model))/len(model.graph.node)

def test_is_every_layer_specialised(custom_step_specialise_layers, save_dashboard):
    """ Test to determine if all the layers in the model have been specialised """
    model = custom_step_specialise_layers
    ratio = calculate_specialised_layers_ratio(model)
    d = {}
    d["specialised_ratio"] = ratio
    d["specialised_layers"] = [x.name for x in get_specialised_nodes(model)]
    d["non_specialised_layers"] = [x.name for x in model.graph.node if x not in get_specialised_nodes(model)]
    dashboard["step_specialize_layers"] = d
    if ratio < 1.0:
        raise RuntimeError(f"Not all layers were specialised only {ratio*100}% were")

##############################################
#       How many layers produce hardware 
##############################################
def test_how_many_layers_produce_hardware(custom_step_specialise_layers):
    """ Test to see if we can create IP from the specialised model """
    model = custom_step_specialise_layers
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["FINN_BUILD_DIR"] = temp_dir
        try:
            model = model.transform(PrepareIP(test_cfg._resolve_fpga_part(), test_cfg._resolve_hls_clk_period()))
            model = model.transform(HLSSynthIP())
            model = model.transform(CreateStitchedIP(test_cfg._resolve_fpga_part(), test_cfg._resolve_hls_clk_period()))
        except:
            pass

        dashboard['gen_hw_list'] = os.listdir(temp_dir)
        dashboard['gen_hw_ratio'] = len(model.graph.node) / len(os.listdir(temp_dir))

        if len(model.graph.node) > len(os.listdir(temp_dir)):
            raise RuntimeError(f"Only {len(os.listdir(temp_dir))/len(model.graph.node):.2f}% of layers have generated hardware")


