# BrainSmith BERT New Demo - Key Insights and Notes

## Table of Contents
1. [BERT Demo FIFO Shape Fix](#bert-demo-fifo-shape-fix)
2. [Critical File Locations](#critical-file-locations)
3. [Key Functions and Roles](#key-functions-and-roles)
4. [Integration Points](#integration-points)
5. [Previous Issues and Solutions](#previous-issues-and-solutions)

## BERT Demo FIFO Shape Fix

### The Problem
The bert_new demo was failing with FIFO shape mismatches during FINN compilation, while the bert_direct demo (bypassing the 6-entrypoint architecture) worked correctly.

### Root Cause
The raw ONNX model exported from Brevitas contained graph structures that FINN couldn't properly analyze for dataflow compilation. The old demo had preprocessing built into its pipeline, but the new demo was missing this critical step.

### The Solution
Created a standard transform step `onnx_preprocessing_step` that automatically handles:
```python
# Step 1: Simplify the model (removes redundant ops)
model, check = simplify(model)

# Step 2: Run QONNX cleanup (standardizes structure)
cleanup(in_file=simp_path, out_file=df_input_path)
```

This preprocessing is now automatically applied as the first step in the blueprint's `legacy_preproc` sequence, eliminating the need for manual preprocessing.

## Critical File Locations

### Main Demo Files
- **Entry Point**: `demos/bert_new/end2end_bert.py`
  - Main orchestration and model generation
  - Contains the critical preprocessing function
  
- **Blueprint Adapter**: `demos/bert_new/blueprint_adapter.py`
  - Runtime blueprint configuration updates
  - Handles model-specific adaptations

### Core BrainSmith Files
- **Unified Blueprint**: `brainsmith/libraries/blueprints_v2/transformers/bert_demo.yaml`
  - Defines compilation flow and parameters
  - Contains critical `legacy_finn: true` flag
  
- **Legacy Conversion**: `brainsmith/core/finn/legacy_conversion.py`
  - Converts Blueprint V2 to FINN DataflowBuildConfig
  - Handles step ordering and parameter mapping

- **Transform Steps**: `brainsmith/libraries/transforms/steps/`
  - `preprocessing.py`: ONNX preprocessing (simplify + cleanup) ⭐ NEW
  - `bert.py`: BERT-specific transformations (head/tail removal)
  - `cleanup.py`: Model cleanup and dimension fixing
  - `validation.py`: Reference IO generation

### API Files
- **Core API**: `brainsmith/core/api.py`
  - Contains main `forge()` function
  - Routes to legacy FINN when needed

- **Evaluation Bridge**: `brainsmith/core/finn/evaluation_bridge.py`
  - Bridges DSE combinations to FINN execution
  - Handles metrics extraction

## Key Functions and Roles

### Model Generation Pipeline
1. **`generate_bert_model()`** (end2end_bert.py:38)
   - Creates PyTorch BERT with Brevitas quantization
   - Exports to ONNX format
   - Auto-adjusts incompatible configurations

2. **`onnx_preprocessing_step()`** (preprocessing.py) ⭐ CRITICAL
   - Automatically simplifies ONNX model structure
   - Runs QONNX cleanup
   - Returns FINN-compatible model
   - Now integrated as first transform step

3. **`create_adaptive_blueprint()`** (end2end_bert.py:182)
   - Loads base blueprint
   - Updates with runtime parameters
   - Saves adapted blueprint file

### Blueprint Processing
1. **`BlueprintAdapter.adapt_for_model_config()`** (blueprint_adapter.py:27)
   - Updates model dimensions
   - Sets FINN configuration
   - Handles folding config logic

2. **`LegacyConversionLayer.convert_to_dataflow_config()`** (legacy_conversion.py:50)
   - Builds step sequence from blueprint
   - Extracts FINN parameters
   - Creates DataflowBuildConfig

3. **`_build_step_sequence()`** (legacy_conversion.py:236)
   - Constructs ordered transformation list
   - Combines preproc + FINN + postproc steps
   - Validates step availability

### Transform Steps
1. **`remove_head_step()`** (bert.py:6)
   - Removes embedding layers up to first LayerNorm
   - Fixes dynamic dimensions
   - Rewires graph connections

2. **`remove_tail_step()`** (bert.py:69)
   - Removes pooler output branch
   - Cleans up unused nodes

3. **`cleanup_step()`** (cleanup.py:44)
   - Sorts commutative inputs
   - Removes identity operations

## Integration Points

### 1. Blueprint → FINN Integration
```
Blueprint YAML 
  → LegacyConversionLayer 
  → DataflowBuildConfig 
  → FINN execution
```

### 2. Model → Automatic Preprocessing → Forge
```
generate_bert_model() 
  → ONNX export
  → forge() 
    → onnx_preprocessing_step (automatic via blueprint)
    → remaining transform steps
```

### 3. CLI → Runtime Configuration
```
CLI arguments 
  → BlueprintAdapter 
  → Adapted blueprint 
  → forge parameters
```

### 4. Step Function Integration
- BrainSmith steps from `transforms/steps/`
- FINN steps from `finn.builder.build_dataflow_steps`
- Combined in blueprint-defined order

## Previous Issues and Solutions

### BERT Demo Step Ordering Issue

#### Problem
The `generate_reference_io_step` was failing because it was trying to execute a model that had already had its head/tail removed but hadn't been properly cleaned up yet.

#### Error Details
- Error occurred in multithreshold operation trying to reshape array of size 0
- Input shape showed dimension 0, meaning empty tensor
- This happened in step 5/20 after head/tail removal

#### Old Demo Step Order (from bert.py line 380):
1. custom_step_cleanup
2. custom_step_remove_head
3. custom_step_remove_tail
4. custom_step_qonnx2finn
5. custom_step_generate_reference_io  # <-- This works in old demo
6. custom_streamlining_step
7. custom_step_infer_hardware

#### Solution
The old demo's `custom_step_qonnx2finn` does important transformations (ExpandNorms, FoldConstants, ConvertDivToMul, ConvertQONNXtoFINN) that prepare the model for execution. The fix was ensuring proper preprocessing before forge.

### Critical Configuration Parameters

#### Must-Have Parameters
```yaml
finn_config:
  split_large_fifos: true       # Prevents FIFO shape errors
  fifosim_n_inferences: 2       # Matches old demo
  verification_atol: 0.1        # Appropriate for quantized models
```

#### Folding Configuration Logic
- If folding_config_file provided: Set target_fps=None
- Otherwise: Use conservative target_fps=10
- This prevents conflicts between manual and auto folding

### Key Discoveries

1. **Model Preprocessing is Essential**
   - Never skip simplify() and cleanup()
   - Now handled automatically by `onnx_preprocessing_step`
   - Must be first step in the transform sequence

2. **Blueprint Step Order Matters**
   - Steps must be in specific order for BERT
   - `onnx_preprocessing_step` must come first
   - Some steps depend on previous transformations

3. **Conservative Defaults Prevent Issues**
   - Low target_fps avoids aggressive parallelization
   - Split large FIFOs handles tensor size variations

4. **Legacy Interface Works Well**
   - `legacy_finn: true` provides stable path
   - Proven step sequences from existing demos

5. **Transform Steps are the Right Abstraction**
   - Preprocessing as a transform step ensures consistency
   - Automatic application prevents user errors
   - Proper integration with intermediate model saving