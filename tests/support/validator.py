"""Golden reference validation utilities for test frameworks.

This module provides GoldenValidator, a pure composition-based utility for
validating actual execution outputs against golden reference values.

Replaces the GoldenReferenceMixin inheritance pattern with cleaner composition:
- No abstract methods to implement
- No inheritance required
- Single responsibility: validate outputs
- Reusable across all test frameworks

Design Philosophy:
- Tests own the golden reference (not kernels, not validators)
- Validator is a pure utility - no state, no coupling
- Clear separation: tests define correctness, validator checks it
- Composable: works with any test framework

Usage:
    validator = GoldenValidator()

Verbose Mode (show passing assertions):
    Set environment variable BRAINSMITH_VERBOSE_TESTS=1 or use pytest option:
    pytest -v -s --log-cli-level=INFO

    # In your test
    golden_outputs = compute_golden_reference(inputs)  # Test-defined
    actual_outputs = op.execute_node(...)

    validator.validate(
        actual_outputs=actual_outputs,
        golden_outputs=golden_outputs,
        backend_name="Python execution",
        rtol=1e-7,
        atol=1e-9
    )
"""

import logging
import os

import numpy as np

from tests.support.assertions import assert_arrays_close

# Logger for verbose test output
logger = logging.getLogger(__name__)


class GoldenValidator:
    """Pure utility for validating outputs against golden reference.

    This class provides a single method: validate(). It compares actual
    execution outputs against expected golden reference values.

    Key Features:
    - Name-agnostic: compares by index, handles ONNX vs golden naming
    - Configurable tolerances: rtol/atol for np.allclose()
    - Clear error messages: shows which output failed and why
    - Stateless: no instance state, purely functional

    Unlike GoldenReferenceMixin, this class:
    - Does NOT force abstract methods on test classes
    - Does NOT require inheritance
    - Does NOT contain test logic (tests own golden reference)
    - ONLY validates outputs

    Example:
        validator = GoldenValidator()

        # Test defines golden reference
        def compute_golden_reference(inputs):
            return {"output": inputs["input0"] + inputs["input1"]}

        # Test runs execution
        inputs = {"inp0": arr0, "inp1": arr1}
        actual_outputs = op.execute_node(...)
        golden_outputs = compute_golden_reference(inputs)

        # Validator checks correctness
        validator.validate(
            actual_outputs=actual_outputs,
            golden_outputs=golden_outputs,
            backend_name="Python execution",
            rtol=1e-7,
            atol=1e-9
        )
    """

    def validate(
        self,
        actual_outputs: dict[str, np.ndarray],
        golden_outputs: dict[str, np.ndarray],
        backend_name: str,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> None:
        """Validate actual outputs match golden reference.

        Compares each output tensor against expected golden reference values
        using numpy.allclose with configurable tolerances.

        Handles name mismatches between ONNX tensor names (e.g., "outp") and
        golden reference standard names (e.g., "output") by comparing by index.

        Args:
            actual_outputs: Outputs from backend execution
                           Dict mapping tensor names → numpy arrays
                           Example: {"outp": np.array([11.0, 12.0, 13.0])}
            golden_outputs: Expected outputs from golden reference
                           Dict mapping tensor names → numpy arrays
                           Example: {"output": np.array([11.0, 12.0, 13.0])}
            backend_name: Name of backend for error messages
                         Examples: "Python execution", "HLS cppsim", "RTL rtlsim"
            rtol: Relative tolerance for np.allclose() (default: 1e-5)
                  Typical values:
                  - 1e-7: Python execution (very accurate)
                  - 1e-5: cppsim/rtlsim (fixed-point rounding)
                  - 1e-3: Approximate operations (softmax, layernorm)
            atol: Absolute tolerance for np.allclose() (default: 1e-6)
                  Typical values:
                  - 1e-9: Python execution
                  - 1e-6: cppsim/rtlsim
                  - 1e-4: Approximate operations

        Raises:
            AssertionError: If outputs don't match within tolerance

        Example:
            >>> validator = GoldenValidator()
            >>> actual = {"outp": np.array([11.0, 12.0, 13.0])}
            >>> golden = {"output": np.array([11.0, 12.0, 13.0])}
            >>> validator.validate(actual, golden, "Python", rtol=1e-7, atol=1e-9)
            # Passes silently (compares by index despite name mismatch)

            >>> actual = {"outp": np.array([11.0, 99.0, 13.0])}  # Wrong!
            >>> validator.validate(actual, golden, "Python", rtol=1e-7, atol=1e-9)
            AssertionError: Python output 0 (outp vs golden output) differs...

        Note on name handling:
            The validator compares outputs by index position, not by name.
            This handles ONNX naming conventions (e.g., "outp", "inp1") that
            differ from golden reference standard names (e.g., "output", "input0").

            Example:
                actual = {"outp": arr1, "outp2": arr2}
                golden = {"output": arr1, "output2": arr2}
                # Compares outp vs output (index 0), outp2 vs output2 (index 1)
        """
        # Convert to lists for index-based comparison
        # This handles ONNX name ("outp") vs golden name ("output") mismatches
        actual_list = list(actual_outputs.items())
        golden_list = list(golden_outputs.items())

        # Check count matches
        if len(actual_list) != len(golden_list):
            raise AssertionError(
                f"{backend_name} output count mismatch.\n"
                f"Expected: {len(golden_list)} outputs {list(golden_outputs.keys())}\n"
                f"Actual: {len(actual_list)} outputs {list(actual_outputs.keys())}"
            )

        # Check if verbose mode is enabled
        verbose = os.environ.get("BRAINSMITH_VERBOSE_TESTS", "0") == "1"

        # Compare by index (handles name mismatches)
        for i, ((actual_name, actual_array), (golden_name, golden_array)) in enumerate(
            zip(actual_list, golden_list)
        ):
            # Compare arrays with tolerance
            assert_arrays_close(
                actual_array,
                golden_array,
                f"{backend_name} output {i} ({actual_name} vs golden {golden_name})",
                rtol=rtol,
                atol=atol,
            )

            # Log successful comparison in verbose mode
            if verbose:
                logger.info(
                    f"✓ {backend_name} output {i} ({actual_name}) PASSED validation\n"
                    f"  Shape: {actual_array.shape}\n"
                    f"  Golden range: [{golden_array.min():.6f}, {golden_array.max():.6f}]\n"
                    f"  Actual range: [{actual_array.min():.6f}, {actual_array.max():.6f}]\n"
                    f"  Max abs diff: {np.abs(actual_array - golden_array).max():.2e}\n"
                    f"  Tolerance: rtol={rtol:.2e}, atol={atol:.2e}"
                )


# Convenience tolerance presets for common backends

class TolerancePresets:
    """Common tolerance presets for different backends.

    These are starting points - tests should adjust based on their specific
    numerical requirements.

    Usage:
        validator.validate(
            actual, golden, "Python",
            **TolerancePresets.PYTHON  # Unpacks rtol and atol
        )
    """

    # Python execution: very accurate, tight tolerances
    PYTHON = {"rtol": 1e-7, "atol": 1e-9}

    # HLS C++ simulation: fixed-point rounding, moderate tolerances
    CPPSIM = {"rtol": 1e-5, "atol": 1e-6}

    # RTL simulation: same precision as cppsim
    RTLSIM = {"rtol": 1e-5, "atol": 1e-6}

    # Approximate operations (softmax, layernorm, etc.): loose tolerances
    APPROXIMATE = {"rtol": 1e-3, "atol": 1e-4}

    # Bit-exact operations (integer arithmetic): zero tolerance
    EXACT = {"rtol": 0.0, "atol": 0.0}


# Convenience factory for test frameworks

def make_validator_with_presets() -> tuple[GoldenValidator, type[TolerancePresets]]:
    """Create validator and tolerance presets for convenient test setup.

    Returns:
        (validator, presets): GoldenValidator instance and TolerancePresets class

    Example:
        validator, tol = make_validator_with_presets()

        # Use preset tolerances
        validator.validate(actual, golden, "Python", **tol.PYTHON)
        validator.validate(actual, golden, "cppsim", **tol.CPPSIM)

        # Or customize
        validator.validate(actual, golden, "custom", rtol=1e-4, atol=1e-5)
    """
    return GoldenValidator(), TolerancePresets
