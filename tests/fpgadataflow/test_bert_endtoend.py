import onnx  
import os
from pathlib import Path  
import json
import pytest
import shutil
import argparse
import math
import tempfile

from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
import finn.builder.build_dataflow_config as build_cfg

from finnbrainsmith.util.bert import (
        custom_step_remove_head, 
        custom_step_remove_tail, 
        custom_step_cleanup,
        custom_step_create_ip, 
        custom_step_specialise_layers, 
        custom_step_infer_hardware, 
        custom_streamlining_step, 
        custom_step_qonnx2finn
)

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
dashboard = {}

@pytest.fixture
def save_dashboard():
    """ save the dashboard to a file at the end of a test.
        runs at the end of all tests.
    """
    yield
    with open("end2end_test_dashboard.json", "w") as fp:
        json.dump(dashboard, fp, indent=4)

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
    #step_set_fifo_depths,
    #step_create_stitched_ip,
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
def get_attribute_by_name(node, attr:str):
    for a in node.attribute:
        if a.name == attr:
            return a
    return None

def test_hardware_generation_progress(step_hw_ipgen, save_dashboard):
    """ Examines the model after the hwipgen step and determines how far along
    each layer is from being fully implemented. """
    mod = step_hw_ipgen
    d = {}
    for node in mod.graph.node:
        d[node.name] = {}
        if get_attribute_by_name(node, "code_gen_dir_ipgen"):
            d[node.name]["HWGEN"] = True
            if node.domain.endswith("hls"):
                # parse the hls solution
                d[node.name]['specialised'] = True
                hls_path = get_attribute_by_name(node, "code_gen_dir_ipgen")
                d[node.name]["HLS_SYNTH"] = Path(f"{hls_path.s.decode('utf-8')}/project_{node.name}/sol1/sol1_data.json").is_file()
                #with open(f"{hls_path.s.decode('utf-8')}/project_{node.name}/sol1/sol1_data.json", "r") as fp:
                #    d[node.name]['hls_synth_log'] = json.load(fp)
            elif node.domain.endswith("rtl"):
                # parse the rtl solution
                d[node.name]['specialised'] = True
            else:
                d[node.name]['specialised'] = False
        else:
            d[node.name]["HWGEN"] = False
        d[node.name]["RTLSIM"] = False
    dashboard['progress'] = d
                


