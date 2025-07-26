# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Core FINN-compatible Build Steps

Brainsmith implementations of core FINN dataflow compiler steps.
These steps use the comprehensive plugin registration system to access
transforms from QONNX, FINN, and Brainsmith.

Key Features:
- Reimplements core FINN steps with enhanced functionality
- Uses plugin registry for all transform access
- Maintains compatibility with FINN dataflow compilation
- Adds debug model saving capabilities

Transform Usage Summary:
- QONNX: Core transforms for cleanup, quantization, graph manipulation
- FINN: Hardware-specific transforms for dataflow compilation
- Brainsmith: Custom transforms for extended functionality
"""

import os
import logging
from typing import Any

from brainsmith.core.plugins import step, get_transform
from brainsmith.core import apply_transforms

logger = logging.getLogger(__name__)


def save_debug_model(model, cfg, step_name):
    """Save model for debugging if preserve_intermediate_models is enabled."""
    if getattr(cfg, 'preserve_intermediate_models', False):
        debug_dir = os.path.join(cfg.output_dir, "debug_models")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save ONNX model
        model_path = os.path.join(debug_dir, f"{step_name}.onnx")
        model.save(model_path)
        
        # Log model structure info
        logger.info(f"Saved debug model: {step_name}")
        logger.info(f"  - Inputs: {[i.name for i in model.graph.input]}")
        logger.info(f"  - Outputs: {[o.name for o in model.graph.output]}")
        logger.info(f"  - Nodes: {len(model.graph.node)}")


# === Cleanup Steps ===

@step(
    name="fix_dynamic_dimensions",
    category="cleanup",
    dependencies=[],
    description="Fix all dynamic dimensions in the model to concrete values"
)
def fix_dynamic_dimensions_step(model, cfg):
    """
    Fix all dynamic dimensions in the model to concrete values.
    
    This step is crucial for hardware inference which requires concrete dimensions.
    It converts any remaining dynamic dimensions (like 'unk__0') to the value 1.
    """
    changes_made = 0
    
    # Fix graph inputs
    for inp in model.graph.input:
        for i, dim in enumerate(inp.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in input {inp.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    # Fix value_info tensors (intermediate tensors)
    for vi in model.graph.value_info:
        for i, dim in enumerate(vi.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in tensor {vi.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    # Fix graph outputs
    for out in model.graph.output:
        for i, dim in enumerate(out.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in output {out.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    logger.info(f"Fixed {changes_made} dynamic dimensions")
    
    # Save debug model
    save_debug_model(model, cfg, "04_after_fix_dynamic_dimensions")
    
    return model


# === Preprocessing Steps ===

@step(
    name="onnx_preprocessing",
    category="preprocessing",
    dependencies=[],
    description="Simplifies and cleans ONNX model for FINN dataflow compiler"
)
def onnx_preprocessing_step(model, cfg):
    """
    Standard ONNX preprocessing with simplify and cleanup for FINN compatibility.
    
    This preprocessing step is CRITICAL for avoiding FIFO shape mismatches in FINN.
    It ensures the ONNX model has the correct structure for dataflow compilation.
    """
    try:
        from onnxsim import simplify
        from qonnx.util.cleanup import cleanup as qonnx_cleanup
        import onnx
        from pathlib import Path
    except ImportError as e:
        raise RuntimeError(
            f"onnx_preprocessing_step requires onnxsim and qonnx packages: {e}. "
            "Install with: pip install onnxsim qonnx"
        )
    
    logger.info("Applying ONNX preprocessing for FINN compatibility")
    
    # Get output directory for intermediate files
    output_dir = getattr(cfg, 'output_dir', './intermediate_models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle both raw ONNX models and ModelWrapper objects
    if hasattr(model, 'model'):
        # This is a ModelWrapper, extract the ONNX model
        onnx_model = model.model
        is_wrapper = True
    else:
        # This is already an ONNX model
        onnx_model = model
        is_wrapper = False
    
    # Step 1: Simplify the model
    logger.info("  Step 1/2: Running ONNX simplification...")
    simplified_model, check = simplify(onnx_model)
    
    if not check:
        raise RuntimeError(
            "ONNX simplification failed. The model may have unsupported operations "
            "or invalid structure for FINN compilation."
        )
    
    # Save simplified model for qonnx cleanup
    simp_path = os.path.join(output_dir, "onnx_preprocessing_simp.onnx")
    onnx.save(simplified_model, simp_path)
    logger.debug(f"  Saved simplified model to: {simp_path}")
    
    # Step 2: Run QONNX cleanup
    logger.info("  Step 2/2: Running QONNX cleanup...")
    cleaned_path = os.path.join(output_dir, "onnx_preprocessing_cleaned.onnx")
    
    # qonnx cleanup works with file paths
    qonnx_cleanup(in_file=simp_path, out_file=cleaned_path)
    logger.debug(f"  Saved cleaned model to: {cleaned_path}")
    
    # Load the cleaned model back
    if is_wrapper:
        # Return as ModelWrapper and apply additional QONNX transforms
        from qonnx.core.modelwrapper import ModelWrapper
        cleaned_model = ModelWrapper(cleaned_path)
        
        # Apply additional QONNX transforms for better FINN compatibility
        logger.info("  Step 3/3: Applying additional QONNX transforms...")
        cleaned_model = apply_transforms(cleaned_model, [
            'RemoveIdentityOps',
            'RemoveUnusedTensors',
            'SortGraph'
        ])
        
        logger.info("✅ ONNX preprocessing completed successfully (ModelWrapper)")
        return cleaned_model
    else:
        # Return as raw ONNX model
        cleaned_model = onnx.load(cleaned_path)
        logger.info("✅ ONNX preprocessing completed successfully (ONNX model)")
        return cleaned_model


# === Quantization Steps ===

@step(
    name="quantization_preprocessing",
    category="quantization",
    dependencies=["cleanup_advanced"],
    description="QONNX quantization preprocessing using commonly useful transforms"
)
def quantization_preprocessing_step(model: Any, cfg: Any) -> Any:
    """
    QONNX quantization preprocessing using validated transforms.
    
    Uses commonly useful QONNX quantization transforms (100% invokable):
    - QCDQToQuant: Convert QuantizeLinear+DequantizeLinear to QONNX Quant nodes
    - QuantToQCDQ: Convert QONNX Quant nodes to standard ONNX format if needed
    
    Commonly used transforms:
    - GemmToMatMul: Convert Gemm to MatMul for easier quantization
    - BatchNormToAffine: Simplify BatchNorm for quantization
    """
    logger.info("Applying QONNX quantization preprocessing...")
    
    # Commonly useful quantization transforms (100% invokable)
    model = apply_transforms(model, [
        'QCDQToQuant',           # Standardize quantization
        'GemmToMatMul',          # Simplify Gemm operations  
        'BatchNormToAffine'      # Simplify BatchNorm
    ])
    
    logger.info("✅ Quantization preprocessing completed")
    return model


# === Conversion Steps ===

@step(
    name="qonnx_to_finn",
    category="conversion",
    dependencies=["quantization_preprocessing"],
    description="Convert QONNX to FINN with comprehensive preprocessing and SoftMax handling"
)
def qonnx_to_finn_step(model: Any, cfg: Any) -> Any:
    """
    Convert QONNX to FINN with comprehensive preprocessing and SoftMax handling.
    
    Phase 1: BrainSmith native transforms for expansion and folding  
    Phase 2: Final QONNX normalization and FINN conversion
    """
    logger.info("Phase 1: BrainSmith native expansion and folding...")
    
    # Get transforms when needed
    ExpandNorms = get_transform('ExpandNorms')
    model = model.transform(ExpandNorms())      # Expand normalization layers

    
    logger.info("Phase 2: Final QONNX normalization and FINN conversion...")
    model = apply_transforms(model, [
        'ExpandNorms',
        'FoldConstants',
        'ConvertDivToMul',
        'ConvertQONNXtoFINN'
    ])
    
    return model


# === Hardware Steps ===

@step(
    name="specialize_layers",
    category="hardware",
    description="Specialize layers with proper opset handling"
)
def specialize_layers_step(model, cfg):
    """
    Custom specialize layers step that ensures opset imports are handled correctly.
    """
    # Get transforms when needed
    GiveUniqueNodeNames = get_transform('GiveUniqueNodeNames')
    ApplyConfig = get_transform('ApplyConfig')
    SpecializeLayers = get_transform('SpecializeLayers')
    
    if cfg.specialize_layers_config_file is not None:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.specialize_layers_config_file))
    
    # Run the specialization
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
    
    # Ensure custom opset imports before shape inference and apply final transforms
    model = apply_transforms(model, [
        'EnsureCustomOpsetImports',
        'GiveUniqueNodeNames',
        'InferShapes',
        'InferDataTypes'
    ])
    
    return model


@step(
    name="infer_hardware",
    category="hardware",
    description="Infer hardware layers for custom operations"
)
def infer_hardware_step(model, cfg):
    """
    Infer hardware layers for operations.

    Custom step for infer hardware because we have some custom operations 
    in this plugin module we need a custom step for infering the hardware 
    for those operations.
    """
    # BrainSmith native hardware inference transforms
    model = apply_transforms(model, [
        'InferLayerNorm',
        'InferDuplicateStreamsLayer'
    ])
    
    # Debug: Check domains of all custom nodes
    logger.info("=== Node domains after hardware inference ===")
    for node in model.graph.node:
        if node.domain and node.domain != "":
            logger.info(f"Node '{node.name}' (op_type={node.op_type}) has domain: '{node.domain}'")
    
    model = apply_transforms(model, [
        'InferElementwiseBinaryOperation',
        'InferShuffle',
        'InferHWSoftmax',
        'InferThresholdingLayer',
        'InferQuantizedMatrixVectorActivation',
        'EnsureCustomOpsetImports'  # Ensure all custom domains have opset imports
    ])
    
    return model


# === Optimization Steps ===

@step(
    name="constrain_folding_and_set_pumped_compute",
    category="optimization", 
    dependencies=["streamlining"],
    description="Apply optimizations including folding constraints and pumped compute (MUST run before infer_hardware)"
)
def constrain_folding_and_set_pumped_compute_step(model, cfg):
    """Apply optimizations including folding constraints and pumped compute."""
    # BrainSmith native transforms
    model = apply_transforms(model, [
        'TempShuffleFixer',
        'SetPumpedCompute'
    ])
    return model