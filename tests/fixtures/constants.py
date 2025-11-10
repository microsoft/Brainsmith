"""Shared constants for test fixtures.

This module consolidates magic numbers and configuration values used across
multiple fixture modules, making them explicit and maintainable.
"""

# ============================================================================
# Random Seeds
# ============================================================================

FIXTURE_RANDOM_SEED = 42
"""Fixed random seed for deterministic fixture generation.

Used across all fixture generation to ensure reproducible test data.
All fixtures using random generation (models.py, test_data.py) should
use this seed for session-scoped consistency.
"""


# ============================================================================
# Quantization Parameters
# ============================================================================

QUANT_FULL_RANGE = 0
"""Quantization uses full signed range: [-2^(n-1), 2^(n-1)-1]

Example: INT8 uses full range [-128, 127]
"""

QUANT_NARROW_RANGE = 1
"""Quantization uses narrow signed range: [-(2^(n-1)-1), 2^(n-1)-1]

Example: INT8 uses narrow range [-127, 127]
Used in some quantization-aware training frameworks.
"""

QUANT_ROUNDING_MODE = "ROUND"
"""Default rounding mode for quantization operations.

Standard rounding to nearest integer (round-half-to-even).
"""


# ============================================================================
# Warning Configuration
# ============================================================================

ANNOTATION_WARNING_STACKLEVEL = 3
"""Stack level for annotation warnings to point to test method.

Stacklevel 3 points to the caller's caller:
1. _check_datatype_support (warning site)
2. annotate_model_datatypes (annotation function)
3. test_my_kernel (test method) <- warning points here

This ensures warnings appear at the test method location for easy debugging.
"""


__all__ = [
    "FIXTURE_RANDOM_SEED",
    "QUANT_FULL_RANGE",
    "QUANT_NARROW_RANGE",
    "QUANT_ROUNDING_MODE",
    "ANNOTATION_WARNING_STACKLEVEL",
]
