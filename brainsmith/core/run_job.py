############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import onnx
import datetime
import json
import os
import shutil
import uuid
from onnxsim import simplify
from qonnx.util.cleanup import cleanup
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from brainsmith.jobs import JOB_REGISTRY


def run_job(job_name, model, args):
    # Find job steps
    job_name = args.job
    # Check if the job name is registered
    if job_name in JOB_REGISTRY.keys():
        job_steps = JOB_REGISTRY[job_name]
    # TODO: Add functionality to handle custom jobs

    # Create readable, unique build directory
    date = datetime.datetime.now().strftime("%b%d_%H%M%S")
    rand = str(uuid.uuid4())[:4]
    dir_name = f"{args.output}_{date}_{rand}"
    build_dir = os.environ.get("BSMITH_BUILD_DIR")
    job_dir = os.path.join(build_dir, dir_name)
    model_dir = os.path.join(job_dir, "intermediate_models")
    os.makedirs(model_dir)

    # Perform model preprocessing
    model, check = simplify(model)
    if not check:
        raise RuntimeError("Unable to simplify the Brevitas bert model")
    if args.save_intermediate:
        onnx.save(model, f"{model_dir}/simp.onnx")
    # TODO: Make model saving optional for cleanup
    cleanup(in_file=model_dir+"/simp.onnx", out_file=job_dir+"/df_input.onnx")

    # TODO: Add general way to generte numpy input/expected output

    # Build dataflow
    df_cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=args.standalone_thresholds,
        steps=job_steps,
        target_fps=args.fps,
        output_dir=job_dir,
        synth_clk_period_ns=args.clk,
        folding_config_file=args.param,
        stop_step=args.stop_step,
        auto_fifo_depths=args.fifodepth,
        fifosim_n_inferences=args.fifosim_n_inferences,
        verification_atol=args.verification_atol,
        split_large_fifos=args.split_large_fifos,
        stitched_ip_gen_dcp=args.dcp,
        board=args.board,
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            ],
        verify_input_npy=job_dir+"/input.npy",
        verify_expected_output_npy=job_dir+"/expected_output.npy",
        verify_save_full_context=args.save_intermediate,
        verify_steps=[
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
        ],
    )
    _ = build.build_dataflow_cfg(job_dir+"/df_input.onnx", df_cfg)

    # Export output model
    if args.stop_step is None:
        final_step = job_steps[-1].__name__
    else:
        final_step = args.stop_step
    shutil.copy2(f"{model_dir}/{final_step}.onnx", f"{job_dir}/output.onnx")

    # Extra metadata for handover
    handover_file = job_dir + "/stitched_ip/shell_handover.json"
    if os.path.exists(handover_file):
        with open(handover_file, "r") as fp:
            handover = json.load(fp)
        handover["num_layers"] = args.num_hidden_layers
        with open(handover_file, "w") as fp:
            json.dump(handover, fp, indent=4)
