# BERT New Demo Technical Analysis Report

## Overview

This report provides a detailed technical analysis of the `bert_new` demo implementation, examining each component, function, and integration point to understand the complete system behavior.

## Function-by-Function Analysis

### 1. end2end_bert.py

#### `generate_bert_model()`
**Location**: end2end_bert.py:38-179
**Purpose**: Generate quantized BERT model and export to ONNX
**Critical Logic**:
```python
# Validation and auto-adjustment of attention heads
if hidden_size % num_attention_heads != 0:
    valid_heads = [h for h in [8, 12, 16, 20, 24] if hidden_size % h == 0]
    if valid_heads:
        num_attention_heads = max(valid_heads)
```
**Key Operations**:
1. Creates BertConfig with specified dimensions
2. Applies symbolic tracing for graph capture
3. Replaces SDPA with quantizable layers
4. Performs layerwise quantization with Brevitas
5. Exports to ONNX with explicit input names

#### `preprocess_model_for_finn()`
**Location**: end2end_bert.py:222-256
**Purpose**: Critical preprocessing step that fixes FIFO shape mismatches
**Implementation**:
```python
def preprocess_model_for_finn(model_path: str, output_dir: str) -> str:
    # Step 1: Simplify (removes redundant operations)
    model, check = simplify(model)
    if not check:
        raise RuntimeError("Unable to simplify the BERT model")
    
    # Step 2: QONNX cleanup (standardizes graph structure)
    cleanup(in_file=simp_path, out_file=df_input_path)
    
    return df_input_path
```
**Why It's Critical**: Without this preprocessing, FINN encounters shape mismatches during dataflow compilation because the raw ONNX export contains structures that FINN cannot properly analyze.

#### `create_adaptive_blueprint()`
**Location**: end2end_bert.py:182-219
**Purpose**: Create runtime-adapted blueprint configuration
**Key Logic**:
- Determines base blueprint path
- Extracts build directory from environment
- Calls blueprint adapter with all runtime parameters
- Returns path to generated blueprint file

#### `main()`
**Location**: end2end_bert.py:259-305
**Purpose**: Main orchestration function
**Execution Sequence**:
1. Generate BERT model → `bert_model.onnx`
2. Preprocess model → `df_input.onnx`
3. Create adaptive blueprint → `bert_demo_standard_384D_3L.yaml`
4. Execute forge with preprocessed model
5. Handle and display results

#### `handle_forge_results()`
**Location**: end2end_bert.py:308-356
**Purpose**: Process and display forge execution results
**Features**:
- Extracts performance metrics (throughput, resource usage)
- Generates metadata JSON files
- Provides user-friendly success/failure messages

### 2. blueprint_adapter.py

#### `BlueprintAdapter.adapt_for_model_config()`
**Location**: blueprint_adapter.py:27-108
**Purpose**: Dynamically update blueprint parameters
**Critical Updates**:
```python
# Model configuration updates
model_config['hidden_size'] = hidden_size
model_config['num_hidden_layers'] = num_hidden_layers

# FINN configuration updates
if folding_config_file:
    adapted['finn_config']['folding_config_file'] = folding_config_file
    adapted['finn_config']['target_fps'] = None  # Disable auto-folding
else:
    adapted['finn_config']['target_fps'] = target_fps or 10  # Conservative default
```
**Design Decision**: When a folding config is provided, auto-folding is disabled to prevent conflicts.

#### `_estimate_parameters()`
**Location**: blueprint_adapter.py:131-148
**Purpose**: Estimate BERT model parameter count
**Formula**:
```python
embedding_params = vocab_size * hidden_size + 512 * hidden_size + 2 * hidden_size
attention_params = 4 * hidden_size * hidden_size + 4 * hidden_size
ffn_params = 2 * hidden_size * hidden_size * 4 + hidden_size * 4 + hidden_size
layer_params = attention_params + ffn_params + layer_norm_params
total_params = embedding_params + num_layers * layer_params
```

### 3. LegacyConversionLayer

#### `convert_to_dataflow_config()`
**Location**: legacy_conversion.py:50-159
**Purpose**: Convert Blueprint V2 to FINN DataflowBuildConfig
**Critical Steps**:
1. Build step sequence from blueprint configuration
2. Extract FINN parameters with type conversions
3. Handle enum conversions for FINN types
4. Create DataflowBuildConfig with all parameters

#### `_build_step_sequence()`
**Location**: legacy_conversion.py:236-314
**Purpose**: Construct ordered list of transformation steps
**Three-Phase Structure**:
```python
# Phase 1: Preprocessing (from legacy_preproc)
steps.extend([cleanup_step, remove_head_step, ...])

# Phase 2: Standard FINN pipeline (always included)
steps.extend([step_create_dataflow_partition, step_specialize_layers, ...])

# Phase 3: Postprocessing (from legacy_postproc)
steps.extend([step_measure_rtlsim_performance, step_set_fifo_depths, ...])
```

#### `_build_finn_config_params()`
**Location**: legacy_conversion.py:344-440
**Purpose**: Extract FINN configuration from blueprint
**Parameter Sources**:
1. Platform settings → board, clock period
2. Build configuration → auto_fifo_depths, save_intermediate_models
3. finn_config section → Direct parameter overrides

### 4. Transform Steps

#### `remove_head_step()`
**Location**: bert.py:6-55
**Purpose**: Remove BERT embedding layers up to first LayerNorm
**Algorithm**:
1. Traverse from input to find first LayerNormalization
2. Remove all nodes in path
3. Rewire input directly to LayerNorm output consumers
4. Fix dynamic batch dimension to 1

#### `remove_tail_step()`
**Location**: bert.py:69-87
**Purpose**: Remove BERT pooler output branch
**Algorithm**:
1. Find 'global_out_1' output
2. Recursively traverse backwards to LayerNorm
3. Remove all nodes in this branch
4. Delete the output from graph

#### `cleanup_step()`
**Location**: cleanup.py:44-71
**Purpose**: Basic ONNX cleanup operations
**Operations**:
- Sort commutative inputs with initializers last
- Remove identity operations

#### `fix_dynamic_dimensions_step()`
**Location**: cleanup.py:107-151
**Purpose**: Convert dynamic dimensions to concrete values
**Critical for**: Hardware inference which requires fixed dimensions

## Critical Paths and Dependencies

### 1. Model Preprocessing Path (CRITICAL)
```
generate_bert_model() 
    → bert_model.onnx
    → preprocess_model_for_finn()
        → simplify()
        → cleanup()
    → df_input.onnx (FINN-compatible)
```
**Dependency**: Must use preprocessed model for forge, not raw ONNX

### 2. Blueprint Configuration Path
```
CLI arguments
    → create_adaptive_blueprint()
        → BlueprintAdapter.adapt_for_model_config()
        → Runtime-adapted blueprint YAML
    → forge(blueprint_path=...)
```
**Dependency**: Blueprint must be adapted before forge execution

### 3. Step Execution Path
```
Blueprint (legacy_preproc + legacy_postproc)
    → LegacyConversionLayer._build_step_sequence()
    → DataflowBuildConfig(steps=[...])
    → FINN execution
```
**Dependency**: Step order must match blueprint specification

### 4. Parameter Flow Path
```
Blueprint finn_config section
    → _build_finn_config_params()
    → DataflowBuildConfig parameters
    → FINN compilation behavior
```
**Dependency**: Critical parameters like `split_large_fifos` must be set

## Configuration Management Strategy

### 1. Hierarchical Configuration
```
Base Blueprint (bert_demo.yaml)
    ↓ (runtime adaptation)
Adapted Blueprint (model-specific)
    ↓ (conversion)
DataflowBuildConfig (FINN-specific)
```

### 2. Parameter Override Priority
1. CLI arguments (highest priority)
2. Folding config file (if provided)
3. Blueprint finn_config section
4. Blueprint defaults
5. Hardcoded defaults (lowest priority)

### 3. Critical Configuration Parameters
```yaml
finn_config:
  split_large_fifos: true       # Prevents FIFO errors
  fifosim_n_inferences: 2       # Simulation depth
  verification_atol: 0.1        # Numerical tolerance
  standalone_thresholds: true   # Optimization flag
```

## Error Handling Analysis

### 1. Model Generation Errors
```python
# Auto-adjustment for invalid configurations
if hidden_size % num_attention_heads != 0:
    # Find valid divisor and adjust
    num_attention_heads = max(valid_heads)
```

### 2. Preprocessing Errors
```python
# Explicit check for simplification success
model, check = simplify(model)
if not check:
    raise RuntimeError("Unable to simplify the BERT model")
```

### 3. Forge Execution Errors
```python
try:
    result = forge(...)
except Exception as e:
    # Detailed error reporting
    print(f"❌ forge failed with error: {e}")
    traceback.print_exc()
    result = {'status': 'failed', 'error': str(e)}
```

### 4. Missing Step Functions
```python
# In _build_step_sequence()
step_func = self._resolve_step_function(step_name)
if step_func:
    steps.append(step_func)
else:
    logger.warning(f"Step not found: {step_name}")
```

## Performance Considerations

### 1. Build Time Optimizations
- `stitched_ip_gen_dcp: false` - Skip synthesis by default
- `save_intermediate_models: true` - Enable debugging but increases I/O
- `stop_step` support - Early termination for testing

### 2. Memory Usage
- Model preprocessing creates temporary files (simp.onnx)
- Intermediate models saved at each step (configurable)
- FIFO depth auto-sizing prevents buffer overflow

### 3. Parallelization Control
- `target_fps` controls parallelization aggressiveness
- Conservative default (10 fps) prevents shape mismatches
- Folding config file provides explicit control

## Comparison with Legacy System

### 1. API Differences
| Aspect | Legacy (bert) | Modern (bert_new) |
|--------|---------------|-------------------|
| Entry Point | `hw_compiler.forge()` | `brainsmith.forge()` |
| Config Format | JSON + args | YAML blueprint |
| Model Path | Processed inline | Explicit preprocessing |
| Step Definition | Hardcoded list | Blueprint-driven |

### 2. Configuration Management
| Aspect | Legacy | Modern |
|--------|--------|--------|
| Folding Config | Separate JSON files | Integrated in blueprint |
| Build Settings | Scattered arguments | Unified finn_config |
| Platform Config | Environment variables | Blueprint platform section |

### 3. Error Handling
| Aspect | Legacy | Modern |
|--------|--------|--------|
| Model Validation | Implicit | Explicit with auto-adjust |
| Preprocessing | Hidden in forge | Explicit function |
| Step Failures | Continues silently | Logged warnings |

## Key Technical Insights

### 1. FIFO Shape Mismatch Root Cause
The FIFO shape mismatches occurred because:
- Raw ONNX models contain graph structures FINN cannot analyze
- The `simplify()` operation removes these problematic structures
- The `cleanup()` operation ensures consistent tensor naming

### 2. Blueprint Step Ordering
The specific order in `legacy_preproc` is critical:
1. `cleanup_step` - Basic graph cleanup
2. `remove_head_step` - BERT-specific transformation
3. `remove_tail_step` - BERT-specific transformation
4. `qonnx_to_finn_step` - Format conversion
5. `streamlining_step` - Optimizations
6. `infer_hardware_step` - Hardware mapping

### 3. Folding Configuration Interaction
When `folding_config_file` is provided:
- `target_fps` is set to None to disable auto-folding
- This prevents conflicts between manual and automatic folding
- The folding config must match the model dimensions

### 4. Legacy Conversion Layer Design
The conversion layer serves as a bridge:
- Maintains compatibility with existing FINN infrastructure
- Allows gradual migration to new architecture
- Preserves proven transformation sequences

## Recommendations

### 1. Always Preprocess Models
Never skip the `preprocess_model_for_finn()` step - it's essential for FINN compatibility.

### 2. Use Conservative Defaults
The default `target_fps=10` prevents aggressive optimizations that can cause issues.

### 3. Enable Intermediate Models
Keep `save_intermediate_models: true` for debugging capability.

### 4. Validate Configurations Early
Check model dimension compatibility before starting compilation.

### 5. Monitor Step Execution
Use verbose logging to track transformation progress and catch issues early.

## Future Improvements

### 1. Enhanced Error Messages
Provide more specific guidance when shape mismatches occur.

### 2. Automatic Preprocessing
Integrate preprocessing into the forge pipeline automatically.

### 3. Step Dependency Validation
Ensure required steps are present and in correct order.

### 4. Performance Profiling
Add timing information for each transformation step.

### 5. Configuration Validation
Comprehensive validation of blueprint parameters before execution.