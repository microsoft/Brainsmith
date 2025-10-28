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
from tests.common.constants import (
    PARITY_DEFAULT_FPGA_PART_HLS as DEFAULT_FPGA_PART_HLS,
    PARITY_DEFAULT_FPGA_PART_RTL as DEFAULT_FPGA_PART_RTL,
    PARITY_DEFAULT_CLOCK_PERIOD_NS as DEFAULT_CLOCK_PERIOD_NS,
    UNSIGNED_TEST_DATA_CAP,
    SIGNED_TEST_DATA_MIN,
    SIGNED_TEST_DATA_MAX,
)

__all__ = [
    "ParityTestBase",
    "ComputationalParityMixin",
    "DEFAULT_FPGA_PART_HLS",
    "DEFAULT_FPGA_PART_RTL",
    "DEFAULT_CLOCK_PERIOD_NS",
    "UNSIGNED_TEST_DATA_CAP",
    "SIGNED_TEST_DATA_MIN",
    "SIGNED_TEST_DATA_MAX",
]
