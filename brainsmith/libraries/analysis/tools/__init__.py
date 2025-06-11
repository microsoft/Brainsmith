"""
BrainSmith Supplementary Tools

Tools that are not part of the core toolflow but provide additional
analysis and profiling capabilities. These tools complement the core
`forge` function but are separate from the main design space exploration
pipeline.
"""

try:
    from .profiling import roofline_analysis, RooflineProfiler
except ImportError:
    roofline_analysis = None
    RooflineProfiler = None

try:
    from .hw_kernel_gen import generate_hw_kernel
except ImportError:
    generate_hw_kernel = None

__all__ = [
    'roofline_analysis',
    'RooflineProfiler',
    'generate_hw_kernel'
]