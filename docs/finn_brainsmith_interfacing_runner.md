## Brainsmith-FINN Entrypoint Interfacing

| FINN build phase | FINN build step | Brainsmith entrypoint |
|------------------|-----------------|----------------------|
| Conversion to FINN opset | Default: step_qonnx_to_finn, step_tidy_up | 1: Register canonical ops |
| Network optimizations | Default: step_streamline | 2: Topology transformations |
| Convert general ops to hardware kernels | Default: step_convert_to_hw | 3: Register hardware abstraction kernels |
| Separate consecutive groups of HWCustomOp nodes | step_create_dataflow_partition | - |
| Specialize kernels to RTL/HLS variants | Default: step_specialize_layers | 4: Register hardware specializations |
| Kernel optimizations | Default: step_target_fps_parallelization, step_apply_folding_config, step_minimize_bit_width | 5: Hardware kernel transformations |
| Output product: estimate reports | step_generate_estimate_reports | - |
| IP generation per layer | step_hw_codegen, step_hw_ipgen | - |
| Design optimizations | Default: step_set_fifo_depths | 6: Hardware graph transformations |
| Output product: stitched IP | step_create_stitched_ip | - |
| Output product: simulated stitched IP | step_measure_rtlsim_performance | - |
| Output product: out-of-context synthesis | step_out_of_context_synthesis | - |
| Output product: bitfile + driver | step_synthesize_bitfile, step_make_driver, step_deployment_package | - |

### *Brainsmith Entrypoints Description*

In the upcoming FINN refactor, the primary design points are sorted into six main entrypoints of different types (each of which are order-dependent).

**1. Register canonical ops**: Registers standard neural network operations that can be recognized and processed by the system during the initial conversion phase. This can either be with Brainsmith custom_ops or from external ONNX opset libraries, but must also point to a *transform* that replaces a standard ONNX operation or subgraph with the custom op. 

**2. Topology transformations**: *Transforms* that manipulate model topology before lowering to hardware. Examples: streamlining, constant folding, transpose merging, etc.

**3/4. Register hardware abstraction kernels/specializations**: Registers the HW *Kernels* at an abstraction level (HWCustomOp), and some or all of their hardware implementations (RTL/HLS Backends)

**5. Hardware kernel transformations**: *Transforms* that optimize the parameters of the hardware kernels, such as parallelization, folding configurations, and bit-width optimizations.

**6. Hardware graph transformations**: *Transforms* that optimize the hardware graph structure, such as setting FIFO depths, adjusting dataflow partitions, and other design-specific optimizations.

## Brainsmith-FINN Legacy Interfacing

As the planned entrypoints don't exist yet, we need a legacy interface to construct a DataflowBuildConfig and execute the FINN Builder. We do this by combining the steps that would be executed in the entrypoints above into a single list of steps.

```python
BUILD_STEPS = [
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
        custom_step_shell_metadata_handover,
    ]
```

This is then used to build a DataflowBuildConfig, which is then passed to the FINN Builder to execute the build process.

## Additional Flags

API definition: https://finn.readthedocs.io/en/latest/source_code/finn.builder.html#finn.builder.build_dataflow_config.DataflowBuildConfig

There are a lot of other flags for the DataflowBuildConfig. For legacy interfacing, they should be set to default values or overwritten by parameters of the same name in the blueprint. However, for the Entrypoint approach, we need to determine which flags we can determine from Brainsmith's flags (objectives, constraints, target_device, etc.), which will be replaced with transforms, and which require something else.

Example code to build and use a DataflowBuildConfig:

```python
# Build dataflow
df_cfg = build_cfg.DataflowBuildConfig(
    standalone_thresholds=args.standalone_thresholds,
    steps=BUILD_STEPS,
    target_fps=target_fps,
    output_dir=build_dir,
    synth_clk_period_ns=args.clk,
    folding_config_file=args.param,
    stop_step=args.stop_step,
    auto_fifo_depths=args.run_fifo_sizing,
    fifosim_n_inferences=args.fifosim_n_inferences,
    verification_atol=args.verification_atol,
    split_large_fifos=args.split_large_fifos,
    stitched_ip_gen_dcp=args.dcp,
    board=args.board,
    generate_outputs=[
        build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    verify_input_npy=build_dir+"/input.npy",
    verify_expected_output_npy=build_dir+"/expected_output.npy",
    verify_save_full_context=args.save_intermediate,
    verify_steps=[
        build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
        build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
    ],
)
_ = build.build_dataflow_cfg(build_dir+"/df_input.onnx", df_cfg)
```
