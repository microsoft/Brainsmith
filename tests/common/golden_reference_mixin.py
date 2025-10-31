"""Golden reference validation mixin for test frameworks.

Provides validate_against_golden() helper and tolerance configuration hooks
for test frameworks that need golden reference validation.

This module consolidates validation logic previously duplicated in:
- tests/pipeline/base_integration_test.py (IntegratedPipelineTest)
- tests/dual_pipeline/base_dual_pipeline_test.py (DualPipelineParityTest)

Key Design Principle:
- **Tests own the golden reference**, not kernels
- Golden reference is test logic, not production code
- Each test defines what "correct" means for its specific test case

Key Features:
- Configurable tolerances for different backends (Python, cppsim, rtlsim)
- Name-agnostic output comparison (handles ONNX vs golden naming)
- Clear error messages when outputs don't match
- Reusable across all test frameworks

Usage Pattern:
    from tests.common.golden_reference_mixin import GoldenReferenceMixin

    class MyTestFramework(GoldenReferenceMixin):
        def get_num_outputs(self):
            return 1

        def compute_golden_reference(self, inputs):
            \"\"\"Define golden reference IN THE TEST.\"\"\"
            # Test-specific logic - can be simple or complex
            return {"output": inputs["input0"] + inputs["input1"]}

        def test_something(self):
            golden = self.compute_golden_reference(inputs)
            self.validate_against_golden(actual, golden, "Python",
                                         self.get_golden_tolerance_python())

Design Philosophy:
- Separation of concerns: kernels contain production code, tests contain test logic
- Flexibility: different tests can have different golden references for same kernel
- No coupling: kernels don't depend on test infrastructure
- Single source of truth for validation logic (but test-defined correctness)
"""

from abc import abstractmethod
from typing import Dict
import numpy as np

# Import from existing utilities
from tests.parity.assertions import assert_arrays_close


class GoldenReferenceMixin:
    """Mixin providing golden reference validation capabilities.

    This mixin adds golden reference testing functionality to any test framework.
    It handles:
    1. Validating outputs against test-defined golden reference
    2. Configurable tolerances for different backends
    3. Name-agnostic comparison (ONNX names vs golden names)

    IMPORTANT: Tests define the golden reference, not kernels!
    - Golden reference is TEST LOGIC, not production code
    - Each test decides what "correct" means for its test case
    - No coupling between kernel implementation and test expectations

    Subclasses must implement:
    - compute_golden_reference(inputs) → Dict: Test-defined golden reference
    - get_num_outputs() → int: Number of output tensors

    Subclasses may optionally override:
    - get_golden_tolerance_python() → Dict[str, float]: Python execution tolerance
    - get_golden_tolerance_cppsim() → Dict[str, float]: HLS C++ simulation tolerance
    - get_golden_tolerance_rtlsim() → Dict[str, float]: RTL simulation tolerance

    Example:
        class TestMyKernel(GoldenReferenceMixin):
            def get_num_outputs(self):
                return 1

            def compute_golden_reference(self, inputs):
                \"\"\"Test-owned golden reference for MyKernel.\"\"\"
                # Simple case: element-wise addition
                return {"output": inputs["input0"] + inputs["input1"]}

            def test_python_execution(self):
                inputs = {"inp0": arr0, "inp1": arr1}
                actual_outputs = op.execute_node(...)

                # Golden reference defined by test, not kernel
                golden_outputs = self.compute_golden_reference(inputs)

                self.validate_against_golden(
                    actual_outputs,
                    golden_outputs,
                    "Python execution",
                    self.get_golden_tolerance_python()
                )
    """

    # =========================================================================
    # Abstract Methods - Subclasses MUST Implement
    # =========================================================================

    @abstractmethod
    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference for this test.

        **This is test logic, not production code!**

        Each test defines what "correct" means for its specific test case.
        This can be:
        - Simple: direct NumPy operations (inputs["input0"] + inputs["input1"])
        - Complex: PyTorch model inference, mathematical formulas, lookup tables
        - Approximate: for operations without exact reference (softmax, layernorm)

        Args:
            inputs: Dict mapping input names → numpy arrays
                   Example: {"input0": arr1, "input1": arr2}

        Returns:
            Dict mapping output names → expected numpy arrays
            Example: {"output": expected_arr}

        Example (simple):
            def compute_golden_reference(self, inputs):
                \"\"\"Element-wise addition.\"\"\"
                return {"output": inputs["input0"] + inputs["input1"]}

        Example (complex):
            def compute_golden_reference(self, inputs):
                \"\"\"BERT LayerNorm with epsilon.\"\"\"
                x = inputs["input0"]
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                normalized = (x - mean) / np.sqrt(var + 1e-5)
                return {"output": normalized}
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Return number of output tensors.

        Used for output validation count checking.

        Returns:
            Number of outputs

        Example:
            def get_num_outputs(self):
                return 1  # Most operations have single output
        """
        pass

    # =========================================================================
    # Optional Configuration Hooks - Override to Customize
    # =========================================================================

    def get_golden_tolerance_python(self) -> Dict[str, float]:
        """Tolerance for Python execution vs golden reference.

        Python execution (execute_node) should be very accurate, so we use
        tight tolerances by default.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()

        Override for kernels with lower numerical precision:
            def get_golden_tolerance_python(self):
                return {"rtol": 1e-4, "atol": 1e-5}  # Looser for approximate ops
        """
        return {"rtol": 1e-7, "atol": 1e-9}

    def get_golden_tolerance_cppsim(self) -> Dict[str, float]:
        """Tolerance for C++ simulation vs golden reference.

        HLS code generation may introduce rounding differences due to
        fixed-point arithmetic, so we use looser tolerances.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()

        Override for kernels requiring tighter validation:
            def get_golden_tolerance_cppsim(self):
                return {"rtol": 1e-7, "atol": 1e-8}  # Tighter for critical ops
        """
        return {"rtol": 1e-5, "atol": 1e-6}

    def get_golden_tolerance_rtlsim(self) -> Dict[str, float]:
        """Tolerance for RTL simulation vs golden reference.

        RTL simulation typically has same precision as cppsim (both use
        fixed-point arithmetic), so we use the same default tolerance.

        Returns:
            Dict with 'rtol' and 'atol' keys for np.allclose()
        """
        return self.get_golden_tolerance_cppsim()

    # =========================================================================
    # Validation Methods - Use These in Tests
    # =========================================================================

    # NOTE: compute_golden_reference() is now abstract and must be implemented
    # by each test class. Tests own the golden reference!

    def validate_against_golden(
        self,
        actual_outputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
        backend_name: str,
        tolerance: Dict[str, float],
    ) -> None:
        """Validate actual outputs match golden reference.

        Compares each output tensor against expected golden reference values
        using numpy.allclose with configurable tolerances.

        Handles name mismatches between ONNX tensor names (e.g., "outp") and
        golden reference standard names (e.g., "output") by comparing by index.

        Args:
            actual_outputs: Outputs from backend execution
                           Example: {"outp": actual_arr}
            golden_outputs: Expected outputs from golden reference
                           Example: {"output": expected_arr}
            backend_name: Name of backend for error messages
                         Examples: "Python execution", "HLS cppsim", "RTL rtlsim"
            tolerance: Dict with 'rtol' and 'atol' keys for np.allclose()
                      Example: {"rtol": 1e-5, "atol": 1e-6}

        Raises:
            AssertionError: If outputs don't match within tolerance

        Example:
            >>> actual = {"outp": np.array([11.0, 12.0, 13.0])}
            >>> golden = {"output": np.array([11.0, 12.0, 13.0])}
            >>> self.validate_against_golden(
            ...     actual, golden, "Python", {"rtol": 1e-7, "atol": 1e-9}
            ... )
            # Passes silently (compares by index despite name mismatch)

            >>> actual = {"outp": np.array([11.0, 99.0, 13.0])}  # Wrong!
            >>> self.validate_against_golden(
            ...     actual, golden, "Python", {"rtol": 1e-7, "atol": 1e-9}
            ... )
            AssertionError: Python vs golden reference for 'outp' differs...
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

        # Compare by index (handles name mismatches)
        for i, ((actual_name, actual_array), (golden_name, golden_array)) in enumerate(
            zip(actual_list, golden_list)
        ):
            # Compare arrays with tolerance
            assert_arrays_close(
                actual_array,
                golden_array,
                f"{backend_name} output {i} ({actual_name} vs golden {golden_name})",
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
            )
