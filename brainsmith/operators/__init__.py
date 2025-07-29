"""
BrainSmith Pure QONNX Operators Library

Pure QONNX-compatible custom operators (not hardware-specific).
Hardware-specific operators (HWCustomOps) are now in brainsmith.libraries.kernels.

This module serves as the main registry that QONNX discovers when looking for
pure QONNX custom operators with domain 'brainsmith.libraries.operators'.
"""

# Import pure QONNX operators directly (no subfolders)
from .norms import FuncLayerNorm

# Create registry for pure QONNX operators only
# (HWCustomOps are now in brainsmith.libraries.kernels with QONNX decorators)
custom_op = {
    'FuncLayerNorm': FuncLayerNorm,
}

# Log operator discovery for debugging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BrainSmith QONNX operators registry initialized with {len(custom_op)} operators:")
for op_name in sorted(custom_op.keys()):
    logger.info(f"  - {op_name}")