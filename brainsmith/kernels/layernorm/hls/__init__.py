"""
Mapping module for FINN dynamic import compatibility.

This module provides a FINN-compatible interface to LayerNorm HLS implementations.
"""

# Import the HLS implementation from parent directory
from ..layernorm_hls import LayerNorm_hls

# Export for FINN's dynamic import mechanism
custom_op = {"LayerNorm_hls": LayerNorm_hls}