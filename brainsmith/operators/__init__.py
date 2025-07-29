"""
BrainSmith Pure QONNX Operators Library

Pure QONNX-compatible custom operators (not hardware-specific).
Hardware-specific operators (HWCustomOps) are now in brainsmith.libraries.kernels.

This module serves as the main registry that QONNX discovers when looking for
pure QONNX custom operators with domain 'brainsmith.libraries.operators'.
"""

# Import pure QONNX operators directly
from .norms import FuncLayerNorm
