"""
Compatibility shim for legacy brainsmith.libraries.operators.general domain.

This module provides backward compatibility for any existing ONNX models
that may have been exported with the old domain structure.
"""

# Import from the new location
from .norms import FuncLayerNorm

# Create custom_op dictionary for legacy compatibility
custom_op = {
    'FuncLayerNorm': FuncLayerNorm,
}

# Legacy domain discovery uses the custom_op dict above