"""Test support utilities - assertions, executors, validators, context.

This module consolidates test utilities from tests/common/, tests/parity/,
and tests/utils/ into a single organized location matching industry standards.

Usage:
    # Convenient imports (recommended)
    from tests.support import (
        ParityAssertion,
        TreeAssertions,
        CppSimExecutor,
        GoldenValidator,
        make_execution_context
    )

    # Or explicit imports
    from tests.support.assertions import ParityAssertion
    from tests.support.executors import CppSimExecutor
    from tests.support.validator import GoldenValidator
    from tests.support.context import make_execution_context
"""

# Assertion classes and helpers
from .assertions import (
    # Base
    AssertionHelper,
    BlueprintAssertions,
    ExecutionAssertions,
    ExpectedExecutionLevel,
    ExpectedExecutionStats,
    ExpectedTreeStructure,
    # Kernel testing
    ParityAssertion,
    # DSE testing
    TreeAssertions,
    assert_arrays_close,
    assert_datatypes_match,
    assert_model_tensors_match,
    assert_shapes_match,
    assert_values_match,
    assert_widths_match,
    calculate_segment_efficiency,
)

# Constants (export all)
from .constants import *

# Test data generation
from .context import make_execution_context

# Executors
from .executors import (
    CppSimExecutor,
    PythonExecutor,
    RTLSimExecutor,
)

# Pipeline execution
from .pipeline import PipelineRunner

# Validation
from .validator import (
    GoldenValidator,
    TolerancePresets,
)

__all__ = [
    # Assertions - Base
    "AssertionHelper",

    # Assertions - Kernel Testing
    "ParityAssertion",
    "assert_shapes_match",
    "assert_datatypes_match",
    "assert_widths_match",
    "assert_values_match",
    "assert_arrays_close",
    "assert_model_tensors_match",

    # Assertions - DSE Testing
    "TreeAssertions",
    "ExecutionAssertions",
    "BlueprintAssertions",
    "ExpectedTreeStructure",
    "ExpectedExecutionLevel",
    "ExpectedExecutionStats",
    "calculate_segment_efficiency",

    # Execution
    "PythonExecutor",
    "CppSimExecutor",
    "RTLSimExecutor",
    "PipelineRunner",
    "make_execution_context",

    # Validation
    "GoldenValidator",
    "TolerancePresets",
]
