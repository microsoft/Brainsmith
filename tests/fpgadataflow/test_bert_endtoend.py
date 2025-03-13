############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import onnx  
import os
from pathlib import Path  
import json
import pytest
import shutil
import argparse
import math
import tempfile
import numpy as np

from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.core.datatype import DataType

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe

from brainsmith.finnlib.util.bert import (
        custom_step_remove_head, 
        custom_step_remove_tail, 
        custom_step_cleanup,
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
        synth_clk_period_ns=3.33,
        stitched_ip_gen_dcp=False,
        folding_config_file="./config/l_1_n_12_z_384_i_1536.json",
        auto_fifo_depths=False,
        #split_large_fifos=True,
        fpga_part="xcv80-lsva4737-2MHP-e-S",
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            ],
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
    custom_step_qonnx2finn, 
    custom_streamlining_step,  
    custom_step_infer_hardware,  
    step_create_dataflow_partition,
    step_specialize_layers,

    # How far do we get
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_create_stitched_ip,
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
#          Validate steps 
##############################################

def _compare_contexts(y_ref, y_out):
    both = set(y_ref.keys()).intersection(set(y_out.keys()))
    for tensor in both:
        print(f"{tensor}  : ref shape {y_ref[tensor].shape}   out shape {y_out[tensor].shape}")
        if (y_ref[tensor].shape == y_out[tensor].shape) :
            print(f"\t{tensor}  :  {np.allclose(y_ref[tensor], y_out[tensor])}")
        print("")
    return

def _save_context(arrays_dict, dict_name):  
    if not os.path.exists(dict_name):  
        os.makedirs(dict_name)  
      
    for key, array in arrays_dict.items():  
        filename = os.path.join(dict_name, f"{key}.npy")  
        np.save(filename, array)  

def test_validate_custom_step_infer_hardware(custom_step_remove_tail, custom_step_infer_hardware):
    """ Using the pruned model produced by Brevitas as a reference
    perform validation of the custom_step_infer_hardware """

    input_m = custom_step_remove_tail.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)

    input_t = { input_m.name : in_tensor}
    out_name = custom_step_remove_tail.graph.output[0].name

    custom_step_remove_tail.save("custom_step_remove_tail.onnx")
    custom_step_infer_hardware.save("custom_step_infer_hardware.onnx")
    y_ref = oxe.execute_onnx(custom_step_remove_tail, input_t, return_full_exec_context=True) 
    y_out = oxe.execute_onnx(custom_step_infer_hardware, input_t, return_full_exec_context=True)

    if not np.allclose(y_ref[out_name], y_out[out_name], atol=1e-1):
        _compare_contexts(y_ref, y_out)
        raise RuntimeError(f"y_ref != y_out")

def test_validate_step_specialize_layers_cppsim(custom_step_remove_tail, step_specialize_layers):
    """ Using the pruned model produced by Brevitas as a reference
    perform validation of the step_specialize_layers """

    input_m = custom_step_remove_tail.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)

    input_t = { input_m.name : in_tensor}
    out_name = custom_step_remove_tail.graph.output[0].name

    y_ref = oxe.execute_onnx(custom_step_remove_tail, input_t)[out_name] 

    cppsim_model = step_specialize_layers.transform(SetExecMode("cppsim"))
    cppsim_model = cppsim_model.transform(PrepareCppSim())
    cppsim_model = cppsim_model.transform(CompileCppSim())
    y_out = oxe.execute_onnx(cppsim_model, input_t)[out_name] 

    assert np.allclose(y_ref, y_out, atol=1e-1), "step_specialize_layers(cppsim) output does not match custom_step_remove_tail"

def test_validate_stitched_ip_rtlsim(custom_step_remove_tail, step_create_stitched_ip):
    """ Using the pruned model produced by Brevitas as a reference
    perform  """

    input_m = custom_step_remove_tail.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)

    input_t = { input_m.name : in_tensor}
    out_name = custom_step_remove_tail.graph.output[0].name

    y_ref = oxe.execute_onnx(custom_step_remove_tail, input_t) 

    rtlsim_model = step_create_stitched_ip.transform(SetExecMode("rtlsim"))
    rtlsim_model = rtlsim_model.transform(PrepareRTLSim())
    y_out = oxe.execute_onnx(rtlsim_model, input_t) 

    if not np.allclose(y_ref[out_name], y_out[out_name], atol=1e-1):
        _compare_contexts(y_ref, y_out)
        _save_context(y_ref, "stitched_ip_rtlsim_context/y_ref")
        _save_context(y_out, "stitched_ip_rtlsim_context/y_out")
        raise RuntimeError(f"y_ref != y_out")

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

def calculate_specialised_layers_ratio(model)->float:
    """ Returns the percentage of layers that were sucessfully specialised """
    return len(get_specialised_nodes(model))/len(model.graph.node)

def test_is_every_layer_specialised(step_specialize_layers, save_dashboard):
    """ Test to determine if all the layers in the model have been specialised """
    model = step_specialize_layers 
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

        if node.domain.endswith("hls") or node.domain.endswith("rtl"):
            d[node.name]['specialised'] = True
        else:
            d[node.name]['specialised'] = False

        if get_attribute_by_name(node, "code_gen_dir_ipgen"):
            d[node.name]["HWGEN"] = True
            if node.domain.endswith("hls"):
                # parse the hls solution
                hls_path = get_attribute_by_name(node, "code_gen_dir_ipgen")
                d[node.name]["HLS_SYNTH"] = Path(f"{hls_path.s.decode('utf-8')}/project_{node.name}/sol1/sol1_data.json").is_file()
                #with open(f"{hls_path.s.decode('utf-8')}/project_{node.name}/sol1/sol1_data.json", "r") as fp:
                #    d[node.name]['hls_synth_log'] = json.load(fp)
        else:
            d[node.name]["HWGEN"] = False
        d[node.name]["RTLSIM"] = False
    dashboard['progress'] = d
                


