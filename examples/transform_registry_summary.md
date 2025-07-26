# Transform Registry Summary

## Overview

The Brainsmith plugin system has a comprehensive transform registry that supports transforms from multiple frameworks. Transforms are registered automatically when their modules are imported.

## Transform Sources

### 1. **Brainsmith Native Transforms**
Located in `/brainsmith/transforms/` and organized by stage:

- **cleanup stage:**
  - `RemoveIdentityOps` - Remove identity operations from computation graph

- **topology_opt stage:**
  - `ExpandNorms` - Expand LayerNorms/RMSNorms into functional components

- **dataflow_opt stage:**
  - `InferFinnLoopOp` - Infer FINN loop operations (delegates to FINN)

- **kernel_opt stage:**
  - `SetPumpedCompute` - (in set_pumped_compute.py)
  - `TempShuffleFixer` - (in temp_shuffle_fixer.py)

- **metadata stage:**
  - `ExtractShellIntegrationMetadata` - (in extract_shell_integration_metadata.py)

### 2. **QONNX Transforms** (99 transforms)
Registered via `framework_adapters.py`, including:
- Batch/Tensor operations (BatchNormToAffine, ChangeBatchSize, etc.)
- Quantization operations (QCDQToQuant, QuantizeGraph, etc.)
- Channel operations (ConvertToChannelsLast, MoveChanLastUpstream, etc.)
- Graph transformations (GemmToMatMul, LowerConvsToMatMul, etc.)
- Utility operations (FoldConstants, InferShapes, GiveUniqueNodeNames, etc.)

### 3. **FINN Transforms** (79 transforms)
Also registered via `framework_adapters.py`, including:
- Basic transforms (ConvertQONNXtoFINN, FoldQuantWeights, etc.)
- Streamline absorb transforms (AbsorbAddIntoMultiThreshold, etc.)
- Streamline reorder transforms (MoveAddPastMul, MoveScalarMulPastMatMul, etc.)
- FPGA dataflow transforms (MinimizeAccumulatorWidth, MinimizeWeightBitWidth, etc.)

### 4. **Kernel Inference Transforms**
Special transforms that infer hardware kernels:
- Located in `/brainsmith/kernels/*/infer_*.py`
- Examples:
  - `InferLayerNorm` - Infer LayerNorm hardware kernels
  - `InferHWSoftmax` - Infer hardware softmax kernels
  - `InferShuffle` - Infer shuffle operations
  - `InferCropFromGather` - Infer crop operations from gather

## Registration Process

1. **Decorator-based Registration:**
   - Transforms use the `@transform` decorator
   - Registration happens at module import time
   - Example:
   ```python
   @transform(
       name="RemoveIdentityOps",
       stage="cleanup",
       description="Remove identity operations"
   )
   class RemoveIdentityOps(Transformation):
       ...
   ```

2. **Framework Adapters:**
   - External transforms (QONNX/FINN) are registered via `framework_adapters.py`
   - Dynamic import and registration when the module loads
   - Graceful handling of missing dependencies

3. **Registry Structure:**
   - Main registry: `transforms` dict (name → class)
   - Stage index: `transforms_by_stage` (stage → {name → class})
   - Framework index: `framework_transforms` (framework → {name → class})

## Transform Stages

Valid stages for transforms:
- `pre_proc` - Pre-processing transforms
- `cleanup` - Graph cleanup and simplification
- `topology_opt` - Topology optimization
- `kernel_opt` - Kernel-specific optimizations  
- `dataflow_opt` - Dataflow optimizations
- `post_proc` - Post-processing transforms
- `metadata` - Metadata extraction/annotation

## Usage in Blueprints

Transforms are referenced in blueprints by name:
```yaml
design_space:
  transforms:
    import:
      - ConvertQONNXtoFINN
      - GiveUniqueNodeNames
      - GiveReadableTensorNames
    
    optimize:
      - ["~", MinimizeAccumulatorWidth, MinimizeWeightBitWidth]
```

## Notes

- The actual availability of transforms depends on installed dependencies (qonnx, finn)
- Transforms must inherit from a base Transformation class
- The registry supports efficient lookup by name, stage, or framework
- Transform registration is lazy - only happens when modules are imported