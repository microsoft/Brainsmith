"""ONNX model preprocessing operations for FINN compatibility."""

import logging
import os
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Import preprocessing tools with proper error handling
try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"onnxsim not available: {e}")
    ONNXSIM_AVAILABLE = False

try:
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    QONNX_CLEANUP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"qonnx cleanup not available: {e}")
    QONNX_CLEANUP_AVAILABLE = False


def onnx_preprocessing_step(model: Any, cfg: Any) -> Any:
    """
    Standard ONNX preprocessing with simplify and cleanup for FINN compatibility.
    
    This preprocessing step is CRITICAL for avoiding FIFO shape mismatches in FINN.
    It ensures the ONNX model has the correct structure for dataflow compilation.
    
    Category: preprocessing
    Dependencies: [onnxsim, qonnx]
    Description: Simplifies and cleans ONNX model for FINN dataflow compiler
    
    Args:
        model: ModelWrapper or ONNX model object
        cfg: Configuration object with output_dir
        
    Returns:
        Preprocessed model ready for FINN compilation
        
    Raises:
        RuntimeError: If preprocessing dependencies not available or simplification fails
    """
    if not ONNXSIM_AVAILABLE:
        raise RuntimeError(
            "onnx_preprocessing_step requires onnxsim package. "
            "Install with: pip install onnxsim"
        )
    
    if not QONNX_CLEANUP_AVAILABLE:
        raise RuntimeError(
            "onnx_preprocessing_step requires qonnx package. "
            "Install with: pip install qonnx"
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
    import onnx
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
        # Return as ModelWrapper if input was ModelWrapper
        from qonnx.core.modelwrapper import ModelWrapper
        cleaned_model = ModelWrapper(cleaned_path)
        logger.info("✅ ONNX preprocessing completed successfully (ModelWrapper)")
        return cleaned_model
    else:
        # Return as raw ONNX model
        cleaned_model = onnx.load(cleaned_path)
        logger.info("✅ ONNX preprocessing completed successfully (ONNX model)")
        return cleaned_model