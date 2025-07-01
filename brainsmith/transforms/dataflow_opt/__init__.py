"""
Dataflow optimization transforms for hardware-specific graph optimizations.

These transforms prepare the model for dataflow execution on FPGA hardware.
"""

# Import stub transforms for BERT compatibility (only for transforms missing from FINN)
from .infer_finn_loop_op import InferFinnLoopOp

__all__ = [
    'InferFinnLoopOp'
]