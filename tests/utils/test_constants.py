"""Test constants for semantic values used across test files."""

# FPGA Configuration Constants
DEFAULT_CLOCK_PERIOD_NS = 5.0
DEFAULT_PARALLEL_BUILDS = 4
DEFAULT_MAX_COMBINATIONS = 100000

# Tree Structure Constants for calculated efficiency tests
SINGLE_BRANCH_EFFICIENCY_WITH_SEGMENTS = 5  # Total transforms when using segmentation
SINGLE_BRANCH_EFFICIENCY_WITHOUT_SEGMENTS = 6  # Total transforms without segmentation

# Execution Constants

# Multi-level Tree Structure Constants
MULTI_LEVEL_TOTAL_NODES = 7  # root + 2 first level + 4 second level branches
MULTI_LEVEL_TOTAL_LEAVES = 4  # 2×2 second level branches
MULTI_LEVEL_TOTAL_PATHS = 4  # 2 branches * 2 sub-branches each
MULTI_LEVEL_LEVEL_2_START_INDEX = 3  # For execution_order[3:]

# Repetition Constants for determinism tests
DETERMINISM_TEST_ITERATIONS = 3

# Tree Efficiency Constants
NO_EFFICIENCY = 0.0  # Linear trees and empty trees have no sharing efficiency
EFFICIENCY_DECIMAL_PLACES = 1  # Rounding precision for efficiency calculations
EFFICIENCY_PERCENTAGE_MULTIPLIER = 100

# Branch Point Constants
MIN_CHILDREN_FOR_BRANCH = 1  # More than 1 child makes a branch point

# Artifact Management Constants
MOCK_EXECUTION_TIME = 1.0  # Seconds for test SegmentResult