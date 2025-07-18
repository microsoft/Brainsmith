"""
BrainSmith Kernels

Plugin-based hardware kernel implementations.
"""

# Register the brainsmith.kernels domain with QONNX
from qonnx.custom_op.registry import register_domain

# Register the main kernels domain
register_domain("brainsmith.kernels", "brainsmith.kernels")

# Also register HLS and RTL subdomains for specialized variants
register_domain("brainsmith.kernels.hls", "brainsmith.kernels")
register_domain("brainsmith.kernels.rtl", "brainsmith.kernels")