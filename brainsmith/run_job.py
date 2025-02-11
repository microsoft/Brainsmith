
import os

import jobs.bert as bert


def load_custom_library(args, steps):
    # Load the custom library
    if args.custom_library is not None:
        custom_library = importlib.import_module(args.custom_library)
        custom_library_steps = custom_library.get_steps()
        steps.extend(custom_library_steps)
    return steps

def run_job(args, steps):
    tmp = "./intermediate_models"
    os.makedirs(tmp, exist_ok=True)

    # Initial model generation
    gen_initial_bert_model(
        outfile=f"{tmp}/initial.onnx",
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        bitwidth=args.bitwidth,
        seqlen=args.seqlen
    )

    # Initial model cleanup
    model = onnx.load(f"{tmp}/initial.onnx")  
    model_simp, check = simplify(model)  
    if check:  
        onnx.save(model_simp, f"{tmp}/simp.onnx")  
    else:  
        raise RuntimeError(f"Unable to simplify the Brevitas bert model")
    cleanup(in_file=f"{tmp}/simp.onnx", out_file=f"{tmp}/qonnx_cleanup.onnx")
    
    steps = [
        # Cleanup and custom graph surgery
        custom_step_cleanup,
        custom_step_remove_head,
        custom_step_remove_tail,
        custom_step_qonnx2finn,

        custom_step_generate_reference_io,
        custom_streamlining_step,
        custom_step_infer_hardware,
        step_create_dataflow_partition,
        step_specialize_layers,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_hw_codegen,
        step_hw_ipgen,
        step_measure_rtlsim_performance,
        step_set_fifo_depths,
        step_create_stitched_ip,
    ]

    cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=steps,
        target_fps=args.fps,
        output_dir=tmp,
        synth_clk_period_ns=args.clk,
        folding_config_file=args.param,
        stop_step=args.stop_step,
        auto_fifo_depths=args.fifodepth,
        split_large_fifos=True,
        stitched_ip_gen_dcp=args.dcp,
        board="V80",
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            ],
        verify_input_npy="input.npy",
        verify_expected_output_npy="expected_output.npy",
        verify_save_full_context=True,
        verify_steps=[
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
        ],
    )
    
    _ = build.build_dataflow_cfg(f"{tmp}/qonnx_cleanup.onnx", cfg)
    if args.stop_step is None:
        shutil.copy2(f"{tmp}/intermediate_models/{steps[-1].__name__}.onnx", args.output)
    else:
        shutil.copy2(f"{tmp}/intermediate_models/{args.stop_step}.onnx", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBERT FINN demo script')
    parser.add_argument('-o', '--output', help='Output ONNX file path', required=True)
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
