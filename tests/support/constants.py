"""Unified test constants for all test suites.

Consolidates constants from tests/utils/constants.py and tests/parity/constants.py
into a single source of truth with clear domain separation.

Sections:
- FPGA Configuration (DSE vs Parity testing)
- Test Data Generation
- Numerical Tolerances
- Tree Structure (DSE-specific)
- Determinism & Execution
"""

from typing import Final

# =============================================================================
# FPGA Configuration - Design Space Exploration (DSE)
# =============================================================================

# Clock period for DSE tests (conservative timing, 200 MHz)
# Used in: design_spaces.py for blueprint generation and tree building
DSE_DEFAULT_CLOCK_PERIOD_NS: Final[float] = 5.0

# Parallel build configuration for DSE
DSE_DEFAULT_PARALLEL_BUILDS: Final[int] = 4
DSE_DEFAULT_MAX_COMBINATIONS: Final[int] = 100000

# =============================================================================
# FPGA Configuration - Parity Testing (High-Performance)
# =============================================================================

# Clock period for parity tests (aggressive timing, ~333 MHz)
# Used in: executors.py for HLS/RTL backend execution
# Rationale: Parity tests target high-performance implementations
PARITY_DEFAULT_CLOCK_PERIOD_NS: Final[float] = 3.0

# FPGA part for HLS backend testing (UltraScale+)
# Used in: Vitis HLS synthesis for cppsim execution
PARITY_DEFAULT_FPGA_PART_HLS: Final[str] = "xcvu9p-flgb2104-2-i"

# FPGA part for RTL backend testing (Versal with DSP58)
# Used in: Advanced RTL synthesis for rtlsim execution
PARITY_DEFAULT_FPGA_PART_RTL: Final[str] = "xcvc1902-vsvd1760-2MP-e-S"

# =============================================================================
# Test Data Generation Parameters
# =============================================================================

# Maximum value for unsigned random test data (0-255 range)
# Caps extreme values from wide datatypes (e.g., UINT32 max = 4,294,967,295)
# Benefits:
# - Numerical stability: Prevents overflow in fixed-point arithmetic
# - Test practicality: Human-readable values aid debugging
# - Representative sampling: Real-world data rarely uses full datatype range
UNSIGNED_TEST_DATA_CAP: Final[int] = 256

# Signed test data range (INT8 equivalent: -128 to 128)
# Consistent with unsigned cap for test data generation
SIGNED_TEST_DATA_MIN: Final[int] = -128
SIGNED_TEST_DATA_MAX: Final[int] = 128

# =============================================================================
# Numerical Tolerances for Floating-Point Comparisons
# =============================================================================

# Relative tolerance for np.allclose() in parity tests
DEFAULT_RTOL: Final[float] = 1e-5

# Absolute tolerance for np.allclose() in parity tests
DEFAULT_ATOL: Final[float] = 1e-6

# =============================================================================
# Tree Structure Constants (DSE-specific)
# =============================================================================

# Tree efficiency calculations
SINGLE_BRANCH_EFFICIENCY_WITH_SEGMENTS: Final[int] = 5
SINGLE_BRANCH_EFFICIENCY_WITHOUT_SEGMENTS: Final[int] = 6

# Multi-level tree structure
MULTI_LEVEL_TOTAL_NODES: Final[int] = 7  # root + 2 first + 4 second level
MULTI_LEVEL_TOTAL_LEAVES: Final[int] = 4  # 2×2 second level branches
MULTI_LEVEL_TOTAL_PATHS: Final[int] = 4  # 2 branches × 2 sub-branches
MULTI_LEVEL_LEVEL_2_START_INDEX: Final[int] = 3  # For execution_order[3:]

# Tree efficiency metrics
NO_EFFICIENCY: Final[float] = 0.0  # Linear trees have no sharing
EFFICIENCY_DECIMAL_PLACES: Final[int] = 1
EFFICIENCY_PERCENTAGE_MULTIPLIER: Final[int] = 100

# Branch point identification
MIN_CHILDREN_FOR_BRANCH: Final[int] = 1  # More than 1 child = branch

# =============================================================================
# Determinism & Execution Constants
# =============================================================================

# Repetition count for determinism validation
DETERMINISM_TEST_ITERATIONS: Final[int] = 3

# Mock execution time for test artifacts (seconds)
MOCK_EXECUTION_TIME: Final[float] = 1.0

# Export all constants
__all__ = [
    # DSE Configuration
    "DSE_DEFAULT_CLOCK_PERIOD_NS",
    "DSE_DEFAULT_PARALLEL_BUILDS",
    "DSE_DEFAULT_MAX_COMBINATIONS",
    # Parity Configuration
    "PARITY_DEFAULT_CLOCK_PERIOD_NS",
    "PARITY_DEFAULT_FPGA_PART_HLS",
    "PARITY_DEFAULT_FPGA_PART_RTL",
    # Test Data Generation
    "UNSIGNED_TEST_DATA_CAP",
    "SIGNED_TEST_DATA_MIN",
    "SIGNED_TEST_DATA_MAX",
    # Numerical Tolerances
    "DEFAULT_RTOL",
    "DEFAULT_ATOL",
    # Tree Structure
    "SINGLE_BRANCH_EFFICIENCY_WITH_SEGMENTS",
    "SINGLE_BRANCH_EFFICIENCY_WITHOUT_SEGMENTS",
    "MULTI_LEVEL_TOTAL_NODES",
    "MULTI_LEVEL_TOTAL_LEAVES",
    "MULTI_LEVEL_TOTAL_PATHS",
    "MULTI_LEVEL_LEVEL_2_START_INDEX",
    "NO_EFFICIENCY",
    "EFFICIENCY_DECIMAL_PLACES",
    "EFFICIENCY_PERCENTAGE_MULTIPLIER",
    "MIN_CHILDREN_FOR_BRANCH",
    # Determinism & Execution
    "DETERMINISM_TEST_ITERATIONS",
    "MOCK_EXECUTION_TIME",
]
