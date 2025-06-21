# BERT New Demo Architecture Design Document

## Executive Summary

The `bert_new` demo represents the modern, blueprint-driven approach to generating BERT transformer accelerators using the BrainSmith platform. It showcases the transition from legacy hardcoded compilation flows to a flexible, configuration-driven architecture that integrates seamlessly with AMD's FINN compiler for FPGA deployment.

### Key Innovations

1. **Blueprint-Driven Configuration**: Uses YAML blueprints to define the entire compilation flow, replacing scattered JSON configs
2. **Runtime Adaptation**: Dynamically adjusts blueprint parameters based on model dimensions
3. **Clean API Design**: Simplified `forge()` API that abstracts complex compilation details
4. **Automatic Preprocessing**: ONNX model preprocessing (simplify + cleanup) is now a standard transform step
5. **Legacy Bridge**: Maintains compatibility with existing FINN infrastructure through a conversion layer

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Invocation                                │
│  python end2end_bert.py --output-dir ./bert_output --num-layers 3      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLI Argument Processing                           │
│  - Parse model configuration (hidden_size, num_layers, etc.)           │
│  - Parse optimization settings (target_fps, clock_period)              │
│  - Convert to internal format for compatibility                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BERT Model Generation                               │
│  - Create PyTorch BERT model with specified dimensions                 │
│  - Apply Brevitas quantization (Int8/Uint8)                            │
│  - Export to ONNX format                                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Blueprint Runtime Adaptation                         │
│  - Load base blueprint (bert_demo.yaml)                                │
│  - Update model configuration parameters                               │
│  - Set FINN compilation options                                        │
│  - Generate runtime-adapted blueprint file                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BrainSmith Forge API                              │
│  - Invoke forge() with model, blueprint, and objectives                │
│  - Route through legacy conversion layer (legacy_finn: true)           │
│  - Convert to FINN DataflowBuildConfig                                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FINN Compilation Pipeline                           │
│  - Execute blueprint-defined step sequence                             │
│  - First step: onnx_preprocessing_step (automatic simplify + cleanup)  │
│  - Apply transformations and optimizations                             │
│  - Generate RTL and create accelerator IP                              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Output Generation                              │
│  - Save accelerator.zip with RTL implementation                        │
│  - Generate performance metrics JSON                                    │
│  - Create metadata files for deployment                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Detailed Execution Flow

### 1. Entry Point and Initialization

The demo starts in `end2end_bert.py:main()`:

```python
def main(args):
    # Phase 1: Model Generation
    model_path = generate_bert_model(
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        ...
    )
    
    # Phase 2: Blueprint Adaptation
    blueprint_path = create_adaptive_blueprint(args)
    
    # Phase 3: Forge Execution (preprocessing now automatic!)
    result = forge(
        model_path=model_path,  # Raw model - preprocessing handled by blueprint
        blueprint_path=blueprint_path,
        target_device=args.board,
        output_dir=args.output_dir
    )
```

### 2. Model Generation Pipeline

The `generate_bert_model()` function creates a quantized BERT model:

1. **Configuration Validation**: Ensures `hidden_size % num_attention_heads == 0`
2. **Model Creation**: Uses transformers library to create BERT architecture
3. **Quantization**: Applies Brevitas Int8/Uint8 quantization
4. **ONNX Export**: Exports quantized model to ONNX format

### 3. Automatic Preprocessing via Transform Step

The preprocessing that resolved FIFO shape mismatches is now handled automatically by the `onnx_preprocessing_step` transform:

```python
# In brainsmith/libraries/transforms/steps/preprocessing.py
def onnx_preprocessing_step(model: Any, cfg: Any) -> Any:
    # Step 1: Simplify the model (critical for proper structure)
    simplified_model, check = simplify(onnx_model)
    
    # Step 2: Run QONNX cleanup
    qonnx_cleanup(in_file=simp_path, out_file=cleaned_path)
    
    return cleaned_model
```

This preprocessing is now the first step in the blueprint's `legacy_preproc` sequence.

### 4. Blueprint Runtime Adaptation

The `BlueprintAdapter` class dynamically updates the blueprint:

```python
class BlueprintAdapter:
    def adapt_for_model_config(self, hidden_size, num_hidden_layers, ...):
        # Update model configuration
        adapted['model_configuration']['hidden_size'] = hidden_size
        
        # Update FINN configuration
        if folding_config_file:
            adapted['finn_config']['folding_config_file'] = folding_config_file
        else:
            adapted['finn_config']['target_fps'] = target_fps or 10
        
        # Update platform settings
        adapted['platform']['board'] = target_device
```

### 5. Blueprint Structure

The `bert_demo.yaml` blueprint defines:

```yaml
# Legacy FINN interface configuration
legacy_finn: true

# Step ordering (critical for FIFO compatibility)
legacy_preproc:
  - "onnx_preprocessing_step"  # NEW: Automatic simplify + cleanup
  - "cleanup_step"
  - "remove_head_step"
  - "remove_tail_step"
  - "qonnx_to_finn_step"
  - "generate_reference_io_step"
  - "streamlining_step"
  - "infer_hardware_step"

# FINN configuration parameters
finn_config:
  split_large_fifos: true       # CRITICAL: Required for FIFO compatibility
  fifosim_n_inferences: 2       # Match old demo
  verification_atol: 0.1        # Match old demo
```

### 6. Forge API and Legacy Conversion

The forge API (`brainsmith/core/api.py`) routes through the legacy conversion layer:

```python
def forge(model_path, blueprint_path, ...):
    # Load blueprint with validation
    design_space = _load_blueprint_strict(blueprint_path)
    
    # Check if legacy FINN interface is requested
    if uses_legacy_finn(blueprint_config):
        # Route through LegacyConversionLayer
        return _execute_legacy_finn(model_path, entrypoint_config)
```

### 7. Legacy Conversion Layer

The `LegacyConversionLayer` converts blueprint configuration to FINN's `DataflowBuildConfig`:

```python
class LegacyConversionLayer:
    def convert_to_dataflow_config(self, entrypoint_config, blueprint_config):
        # Build step sequence from blueprint
        step_functions = self._build_step_sequence(blueprint_config)
        
        # Extract FINN parameters
        finn_params = self._build_finn_config_params(blueprint_config)
        
        # Create DataflowBuildConfig
        config = DataflowBuildConfig(
            steps=step_functions,
            output_dir=finn_params.get('output_dir'),
            split_large_fifos=finn_params.get('split_large_fifos', True),
            ...
        )
```

## Component Breakdown

### 1. end2end_bert.py
- **Purpose**: Main entry point and orchestration
- **Key Functions**:
  - `main()`: Orchestrates the entire flow
  - `generate_bert_model()`: Creates quantized BERT model
  - `create_adaptive_blueprint()`: Adapts blueprint for runtime config
  - `handle_forge_results()`: Processes and displays results

### 2. blueprint_adapter.py
- **Purpose**: Runtime blueprint configuration adaptation
- **Key Functions**:
  - `adapt_for_model_config()`: Updates blueprint parameters
  - `_apply_standard_optimizations()`: Sets optimization flags
  - `save_adapted_blueprint()`: Persists adapted configuration

### 3. bert_demo.yaml
- **Purpose**: Unified blueprint configuration
- **Key Sections**:
  - `legacy_finn`: Enables legacy FINN interface
  - `legacy_preproc/postproc`: Defines step ordering
  - `finn_config`: Direct FINN parameter control
  - `model_configuration`: Default model parameters

### 4. LegacyConversionLayer
- **Purpose**: Bridge between Blueprint V2 and FINN
- **Key Functions**:
  - `convert_to_dataflow_config()`: Main conversion function
  - `_build_step_sequence()`: Constructs ordered step list
  - `_build_finn_config_params()`: Extracts FINN parameters

### 5. Transform Steps
- **Location**: `brainsmith/libraries/transforms/steps/`
- **Key Steps**:
  - `onnx_preprocessing_step`: ONNX simplify + cleanup (NEW!)
  - `cleanup_step`: Basic ONNX cleanup
  - `remove_head_step`: BERT-specific head removal
  - `qonnx_to_finn_step`: QONNX to FINN conversion
  - `streamlining_step`: Graph optimizations
  - `infer_hardware_step`: Hardware mapping

## Data Flow Through the System

### 1. Input Data
```
CLI Arguments → ArgParser → Internal Format
    ↓
Model Config (hidden_size, num_layers, etc.)
    ↓
Optimization Config (target_fps, clock_period)
```

### 2. Model Transformation
```
PyTorch BERT → Brevitas Quantization → ONNX Export
    ↓
ONNX Model → Blueprint Processing → [onnx_preprocessing_step] → Cleaned Model
```

### 3. Configuration Flow
```
Base Blueprint → Runtime Adapter → Adapted Blueprint
    ↓
Blueprint Config → Legacy Converter → DataflowBuildConfig
```

### 4. Compilation Flow
```
DataflowBuildConfig + Model → FINN Pipeline → RTL Generation
    ↓
RTL + Synthesis → Accelerator IP → Output Files
```

## Key Technical Decisions

### 1. Automatic Model Preprocessing
**Decision**: Make preprocessing a standard transform step (`onnx_preprocessing_step`)
**Rationale**: Ensures consistent ONNX structure automatically without requiring manual preprocessing

### 2. Legacy FINN Interface
**Decision**: Use `legacy_finn: true` in blueprint
**Rationale**: Maintains compatibility with existing FINN infrastructure while preparing for future 6-entrypoint architecture

### 3. Blueprint Runtime Adaptation
**Decision**: Dynamically update blueprint parameters instead of having multiple static blueprints
**Rationale**: Reduces configuration proliferation and enables flexible model configurations

### 4. Conservative Folding Defaults
**Decision**: Default `target_fps=10` when no folding config provided
**Rationale**: Avoids aggressive parallelization that can cause shape mismatches

### 5. Step Ordering
**Decision**: Explicit step ordering with `onnx_preprocessing_step` first in `legacy_preproc`
**Rationale**: Ensures models are properly preprocessed before any other transformations

## Critical Configuration Parameters

### Essential FINN Parameters
- `split_large_fifos: true` - Required for handling large tensors
- `fifosim_n_inferences: 2` - Matches proven configuration
- `verification_atol: 0.1` - Appropriate tolerance for quantized models
- `standalone_thresholds: true` - Enables threshold optimization

### Model Configuration Validation
- Ensures `hidden_size % num_attention_heads == 0`
- Auto-adjusts attention heads if needed
- Validates intermediate size compatibility

### Platform Settings
- Target device mapping (V80, VCK190, etc.)
- Clock period configuration (default 5.0ns = 200MHz)
- Resource utilization constraints

## Error Handling and Edge Cases

### 1. Model Configuration Errors
- Auto-adjusts incompatible attention head counts
- Validates dimension compatibility
- Provides clear error messages

### 2. FINN Compilation Failures
- Captures and logs FINN errors
- Falls back to mock results if configured
- Preserves intermediate models for debugging

### 3. Blueprint Validation
- Strict validation of blueprint structure
- Checks for required sections
- Validates step function availability

## Integration Points

### 1. FINN Integration
- Through `LegacyConversionLayer`
- Maps blueprint steps to FINN step functions
- Handles parameter type conversions

### 2. Blueprint System
- Loads from `brainsmith/libraries/blueprints_v2/`
- Supports inheritance and overrides
- Enables DSE strategies

### 3. Transform Pipeline
- Modular step functions in `transforms/steps/`
- Category-based organization
- Dependency tracking

## Performance Considerations

### 1. Build Time Optimization
- Optional DCP generation (`stitched_ip_gen_dcp: false`)
- Configurable verification steps
- Intermediate model caching

### 2. Resource Utilization
- Configurable folding factors
- Target FPS-based parallelization
- Platform-specific optimizations

### 3. Memory Management
- FIFO depth auto-sizing
- Large FIFO splitting
- Buffer optimization strategies

## Future Enhancements

### 1. Native 6-Entrypoint Support
- Direct execution without legacy conversion
- Modern component-based architecture
- Enhanced DSE capabilities

### 2. Advanced Blueprint Features
- Multi-objective optimization
- Adaptive exploration strategies
- Performance prediction models

### 3. Extended Model Support
- Beyond BERT to other transformers
- Custom attention mechanisms
- Dynamic model architectures