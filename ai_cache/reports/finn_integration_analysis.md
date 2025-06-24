# Current FINN Integration Flow Analysis

## Overview

The current FINN integration in `brainsmith/core/finn/` provides a bridge between the Blueprint V2 design space exploration (DSE) and FINN execution. It supports both a modern 6-entrypoint interface and a legacy interface.

## Key Components

### 1. FINNEvaluationBridge (`evaluation_bridge.py`)
The main interface that DSE calls to evaluate component combinations.

**Flow:**
1. Receives `ComponentCombination` from DSE
2. Converts to 6-entrypoint configuration
3. Routes to either legacy or modern interface based on blueprint config
4. Executes FINN build
5. Returns standardized metrics

**Key Methods:**
- `evaluate_combination()` - Main entry point
- `_combination_to_entrypoint_config()` - Maps DSE combination to 6 entrypoints
- `_execute_finn_run()` - Actual FINN execution via `build_dataflow_cfg()`

### 2. LegacyConversionLayer (`legacy_conversion.py`)
Converts 6-entrypoint config to FINN's DataflowBuildConfig format.

**Features:**
- Maps entrypoint components to FINN build steps
- Uses proven step functions from `brainsmith.libraries.transforms.steps`
- Blueprint-driven step ordering
- Handles all FINN configuration parameters

**Key Steps Used:**
- `onnx_preprocessing_step`
- `qonnx_to_finn_step`
- `streamlining_step`
- `infer_hardware_step`
- Various cleanup and optimization steps

### 3. MetricsExtractor (`metrics_extractor.py`)
Extracts standardized metrics from FINN build results.

**Metrics Extracted:**
- **Performance**: throughput (fps), latency (ms), clock frequency
- **Resources**: LUT/DSP/BRAM utilization, power consumption
- **Quality**: build success, warnings, build time
- **Composite**: resource efficiency

**Sources:**
- RTL simulation reports
- Synthesis reports
- Resource estimation files

### 4. ConfigBuilder (`config_builder.py`)
Utility for building FINN configurations from Blueprint V2.

**Functions:**
- Extracts constraints from blueprint
- Sets default parameters
- Validates configuration
- Maps blueprint objectives to FINN parameters

## Current Execution Flow

```
DSE Explorer
    ↓
FINNEvaluationBridge.evaluate_combination()
    ↓
Convert ComponentCombination → 6-entrypoint config
    ↓
Check if legacy interface requested
    ↓
┌─────────────────────────┬──────────────────────────┐
│ Legacy Path (Current)   │ Modern Path (Future)     │
├─────────────────────────┼──────────────────────────┤
│ LegacyConversionLayer   │ Direct 6-entrypoint      │
│         ↓               │         ↓                │
│ DataflowBuildConfig     │ 6-entrypoint config      │
│         ↓               │         ↓                │
│ build_dataflow_cfg()    │ build_dataflow_v2()      │
└─────────────────────────┴──────────────────────────┘
                    ↓
              FINN Result
                    ↓
            MetricsExtractor
                    ↓
         Standardized Metrics
```

## Key Observations

1. **Subprocess Isolation**: Current implementation imports FINN directly - no subprocess isolation
2. **Error Handling**: Falls back to RuntimeError on FINN failures - no mock results anymore
3. **Step Functions**: Uses proven step functions from libraries, not dynamic generation
4. **Configuration**: Extensive parameter mapping from blueprint to FINN config
5. **Metrics**: Comprehensive metric extraction from multiple FINN output files

## For New Backend Implementation

The new backends should:

1. **Maintain the same evaluation interface** - Accept model path and combination/config
2. **Use subprocess for FINN execution** - Isolate FINN execution for better error handling
3. **Extract same standardized metrics** - Ensure compatibility with DSE optimization
4. **Handle both workflow types** - 6-entrypoint vs legacy (build_steps)
5. **Reuse existing components** - MetricsExtractor can be reused
6. **Provide proper error handling** - No mock results, fail honestly

## Configuration Parameters

Key FINN parameters extracted from blueprints:
- `output_dir` - Where to save results
- `synth_clk_period_ns` - Target clock period
- `target_fps` - Performance target
- `fpga_part` - Target FPGA device
- `folding_config_file` - Parallelization config
- `auto_fifo_sizing` - FIFO optimization
- `verify_steps` - Verification options