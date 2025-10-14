"""Parity testing framework for manual vs auto HWCustomOp implementations.

This package provides infrastructure for testing equivalence between:
- Manual FINN HWCustomOp implementations
- Brainsmith AutoHWCustomOp implementations

The ParityTestBase class provides generic test methods for validating:
- Shape methods (normal, folded)
- Stream widths
- Datatypes
- Expected cycles
- Python execution parity

Usage:
    from tests.parity import ParityTestBase

    class TestMyKernelParity(ParityTestBase):
        manual_op_class = ManualKernel
        auto_op_class = AutoKernel

        def make_test_model(self):
            # Create ONNX model...
            return model, node_name

        def get_shared_nodeattrs(self):
            return {"PE": 8, "SIMD": 16}
"""

from .base_parity_test import ParityTestBase

__all__ = ["ParityTestBase"]
