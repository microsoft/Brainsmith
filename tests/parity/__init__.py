"""Parity testing utilities - OLD frameworks have been replaced.

NOTE: Old parity frameworks (ParityTestBase, ComputationalParityMixin) have been
replaced by new composition-based frameworks in tests/frameworks/.

What Remains Here:
------------------
- assertions.py: Parity-specific assertion helpers (still used by new frameworks)
- test_fixtures.py: make_execution_context() utility (still used by new frameworks)

Migration Guide:
----------------
Old:
    from tests.parity import ParityTestBase
    class TestMyKernel(ParityTestBase):
        ...

New (for single kernel tests):
    from tests.frameworks.single_kernel_test import SingleKernelTest
    class TestMyKernel(SingleKernelTest):
        ...

New (for dual kernel parity tests):
    from tests.frameworks.dual_kernel_test import DualKernelTest
    class TestMyKernel(DualKernelTest):
        ...

Benefits of New Architecture:
-----------------------------
- Composition over inheritance (no complex inheritance chains)
- Single Responsibility Principle (each utility does one thing)
- Reusable components (PipelineRunner, GoldenValidator, Executors)
- More tests: SingleKernelTest (6 tests), DualKernelTest (20 tests)
- Better error messages and debugging

See Also:
---------
- tests/frameworks/ - New test frameworks
- tests/IMPLEMENTATION_STATUS.md - Migration status
- tests/TEST_SUITE_ARCHITECTURE_MAP.md - Full architecture overview
"""

# Export constants (still useful)
from tests.common.constants import (
    PARITY_DEFAULT_FPGA_PART_HLS as DEFAULT_FPGA_PART_HLS,
    PARITY_DEFAULT_FPGA_PART_RTL as DEFAULT_FPGA_PART_RTL,
    PARITY_DEFAULT_CLOCK_PERIOD_NS as DEFAULT_CLOCK_PERIOD_NS,
    UNSIGNED_TEST_DATA_CAP,
    SIGNED_TEST_DATA_MIN,
    SIGNED_TEST_DATA_MAX,
)

# Export utilities that are still used
from .assertions import ParityAssertion, assert_shapes_match, assert_datatypes_match
from .test_fixtures import make_execution_context

__all__ = [
    # Constants
    "DEFAULT_FPGA_PART_HLS",
    "DEFAULT_FPGA_PART_RTL",
    "DEFAULT_CLOCK_PERIOD_NS",
    "UNSIGNED_TEST_DATA_CAP",
    "SIGNED_TEST_DATA_MIN",
    "SIGNED_TEST_DATA_MAX",
    # Utilities (still used)
    "ParityAssertion",
    "assert_shapes_match",
    "assert_datatypes_match",
    "make_execution_context",
]
