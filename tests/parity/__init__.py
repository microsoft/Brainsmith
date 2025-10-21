"""Parity testing framework for manual vs auto HWCustomOp implementations.

This package provides infrastructure for testing equivalence between:
- Manual FINN HWCustomOp implementations
- Brainsmith KernelOp implementations

Classes:
- ParityTestBase: 25 generic tests for all kernels
- ComputationalParityMixin: 7 additional tests for computational kernels (MVAU, VVAU)

Usage (Data Movement Kernel):
    from tests.parity import ParityTestBase

    class TestShuffleParity(ParityTestBase):
        # Gets 25 base tests
        pass

Usage (Computational Kernel):
    from tests.parity import ParityTestBase, ComputationalParityMixin

    class TestVVAUParity(ParityTestBase, ComputationalParityMixin):
        # Gets 25 base + 7 computational = 32 tests
        pass
"""

from .base_parity_test import ParityTestBase
from .computational_parity_test import ComputationalParityMixin

__all__ = ["ParityTestBase", "ComputationalParityMixin"]
