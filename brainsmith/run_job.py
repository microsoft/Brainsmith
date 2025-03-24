############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import warnings  
warnings.simplefilter("ignore")  

import onnx  
import os
import shutil
import argparse
import math
import torch
from torch import nn
from transformers import BertConfig, BertModel
from transformers import AutoModel
from transformers.utils.fx import symbolic_trace

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
import brevitas.onnx as bo
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.calibrate import calibration_mode

from onnxsim import simplify  
from qonnx.util.cleanup import cleanup
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, ConvertDivToMul
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from brainsmith.jobs import JOB_REGISTRY


def run_job(job_name, model, steps):
        
    cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=job_steps,
        target_fps=args.fps,
        output_dir=build_path,
        synth_clk_period_ns=args.clk,
        folding_config_file=args.param,
        stop_step=args.stop_step,
        auto_fifo_depths=args.fifodepth,
        fifosim_n_inferences=2,
        verification_atol=1e-1,
        split_large_fifos=True,
        stitched_ip_gen_dcp=args.dcp,
        board="V80",
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            ],
        verify_input_npy=build_path+"/input.npy",
        verify_expected_output_npy=build_path+"/expected_output.npy",
        verify_save_full_context=True,
        verify_steps=[
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
        ],
    )


def main(args):

    # Find Job steps
    job_name = args.job
    # Check if the job name is registered
    if job_name in JOB_REGISTRY.keys():
        job_steps = JOB_REGISTRY[job_name]
    # TODO: Add functionality to handle custom jobs

    # Find model
    # Initial model generation
    gen_initial_bert_model(
        outfile=f"{tmp_path}/initial.onnx",
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        bitwidth=args.bitwidth,
        seqlen=args.seqlen
    )

    # Initial model cleanup
    model = onnx.load(f"{tmp_path}/initial.onnx")
    model_simp, check = simplify(model)
    if check:
        onnx.save(model_simp, f"{tmp_path}/simp.onnx")
    else:
        raise RuntimeError(f"Unable to simplify the Brevitas bert model")
    cleanup(in_file=f"{tmp_path}/simp.onnx", out_file=f"{tmp_path}/qonnx_cleanup.onnx")
    



    
    _ = build.build_dataflow_cfg(f"{tmp_path}/qonnx_cleanup.onnx", cfg)
    if args.stop_step is None:
        shutil.copy2(f"{tmp_path}/{steps[-1].__name__}.onnx", args.model)
    else:
        shutil.copy2(f"{tmp_path}/{args.stop_step}.onnx", args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBERT FINN demo script')
    parser.add_argument('-m', '--model', help='Output ONNX model name', required=True)
    parser.add_argument('-o', '--output', type=str, default='./builds/', help='Output build path', required=True)
    parser.add_argument('-z', '--hidden_size', type=int, default=384, help='Sets BERT hidden_size parameter')
    parser.add_argument('-n', '--num_attention_heads', type=int, default=12, help='Sets BERT num_attention_heads parameter')
    parser.add_argument('-l', '--num_hidden_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-i', '--intermediate_size', type=int, default=1536, help='Sets BERT intermediate_size parameter')
    parser.add_argument('-b', '--bitwidth', type=int, default=8, help='The quantisation bitwidth (either 4 or 8)')
    parser.add_argument('-f', '--fps', type=int, default=3000, help='The target fps for auto folding')
    parser.add_argument('-c', '--clk', type=float, default=3.33, help='The target clock rate for the hardware')
    parser.add_argument('-s', '--stop_step', type=str, default=None, help='Step to stop at in the build flow')
    parser.add_argument('-p', '--param', type=str, default=None, help='Use a preconfigured file for the folding parameters')
    parser.add_argument('-x', '--fifodepth', type=bool, default=True, help='Skip the FIFO depth stage')
    parser.add_argument('-q', '--seqlen', type=int, default=128, help='Sets the sequence length parameter')
    parser.add_argument('-d', '--dcp', type=bool, default=True, help='Generate a DCP')

    args = parser.parse_args()
    main(args)
