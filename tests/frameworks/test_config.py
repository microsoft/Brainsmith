"""Unified test configuration for kernel tests (v2.5).

This module provides a type-safe, declarative configuration interface for kernel
tests, consolidating all test parameters into a single KernelTestConfig dataclass.

Usage:
    @pytest.fixture(
        params=[
            KernelTestConfig(
                test_id="add_int8_small_pe8",
                operation="Add",
                input_shapes={"input": (1, 64), "param": (1, 64)},
                input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
                input_streams={0: 8},  # PE=8 for first input (Stage 2+)
                fpgapart="xc7z020clg400-1",
                tolerance_python={"rtol": 1e-7, "atol": 1e-9},
            ),
        ],
        ids=lambda cfg: cfg.test_id
    )
    def kernel_test_config(request):
        return request.param

    class TestMyKernel(SingleKernelTest):
        def make_test_model(self, kernel_test_config):
            # Framework auto-applies configuration!
            pass

v2.5 Features:
- Single source of truth for all test parameters
- Type safety with full IDE autocomplete
- Semantic DSE API (with_input_stream/with_output_stream)
- Stage 2 stream dimension configuration
- Backward compatible with v2.4 tests
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qonnx.core.datatype import DataType


@dataclass
class KernelTestConfig:
    """Unified kernel test configuration (v2.5).

    This dataclass consolidates all test parameters into a single, type-safe
    configuration object. It replaces the scattered fixtures and methods from
    v2.4 with a declarative interface.

    Attributes:
        operation: ONNX operation name (e.g., "Add", "MatMul", "Conv")
        input_shapes: Dict mapping input names to shapes, e.g.,
                     {"input": (1, 64), "param": (1, 64)}
        input_dtypes: Dict mapping input names to DataTypes, e.g.,
                     {"input": DataType["INT8"], "param": DataType["INT8"]}

        input_streams: Optional dict mapping input indices to stream parallelism
                      values, e.g., {0: 8} sets PE=8 for first input.
                      Uses semantic with_input_stream() API (kernel-agnostic).
                      Available at Stage 2+.
        output_streams: Optional dict mapping output indices to stream parallelism
                       values, e.g., {0: 16}. Uses with_output_stream() API.
                       Available at Stage 2+.
        dse_dimensions: Optional dict of other DSE dimensions, e.g.,
                       {"mem_mode": "internal", "SIMD": 32}.
                       Applied via with_dimension() API.

        fpgapart: Optional FPGA part number for backend testing, e.g.,
                 "xc7z020clg400-1". If None, cppsim/rtlsim tests are skipped.
        backend_variants: Optional list of backend variants to test, e.g.,
                         ["hls", "rtl"]. Defaults to ["hls"].

        tolerance_python: Optional tolerance dict for Python execution tests,
                         e.g., {"rtol": 1e-7, "atol": 1e-9}
        tolerance_cppsim: Optional tolerance dict for C++ simulation tests
        tolerance_rtlsim: Optional tolerance dict for RTL simulation tests

        test_id: Optional test identifier for pytest. If not provided,
                auto-generated from operation, dtypes, and shapes.
        marks: Optional list of pytest marks to apply to this configuration

    Stage Availability (DSE parameters):
        - Stage 2 (Kernel, backend="fpgadataflow"):
          * input_streams, output_streams, dse_dimensions all available
          * Base kernel has full design_space
        - Stage 3 (Backend, backend="hls"/"rtl"):
          * Backend-specific parameters available (mem_mode, ram_style, resType)
          * These should be in dse_dimensions

    Example:
        # Minimal configuration (required fields only)
        config = KernelTestConfig(
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        )

        # Full configuration with stream dimensions and backend testing
        config = KernelTestConfig(
            test_id="add_int8_small_pe8_simd16",
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
            input_streams={0: 8},           # PE=8 (semantic API)
            dse_dimensions={"SIMD": 16},    # Additional DSE param
            fpgapart="xc7z020clg400-1",
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
            tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
        )

        # Backend-specific configuration (Stage 3 only)
        config = KernelTestConfig(
            operation="MatMul",
            input_shapes={"input": (1, 64), "weight": (64, 128)},
            input_dtypes={"input": DataType["INT8"], "weight": DataType["INT8"]},
            input_streams={0: 8},
            dse_dimensions={
                "mem_mode": "internal_decoupled",  # Stage 3 (backend)
                "ram_style": "ultra",              # Stage 3 (backend)
            },
            fpgapart="xc7z020clg400-1",
        )
    """

    # ========================================================================
    # Model Structure (Required)
    # ========================================================================

    operation: str
    """ONNX operation name (e.g., 'Add', 'MatMul', 'Conv')"""

    input_shapes: Dict[str, tuple]
    """Input tensor shapes, e.g., {'input': (1, 64), 'param': (1, 64)}"""

    input_dtypes: Dict[str, DataType]
    """Input tensor datatypes, e.g., {'input': DataType['INT8'], 'param': DataType['INT8']}"""

    # ========================================================================
    # DSE Configuration (Optional, auto-applied by framework)
    # ========================================================================

    input_streams: Optional[Dict[int, int]] = None
    """Stream parallelism for inputs, e.g., {0: 8} sets PE=8 for first input.
    Uses semantic with_input_stream(index, value) API - kernel-agnostic!
    Available at Stage 2 (kernel) and Stage 3 (backend)."""

    output_streams: Optional[Dict[int, int]] = None
    """Stream parallelism for outputs, e.g., {0: 16}.
    Uses with_output_stream(index, value) API.
    Available at Stage 2 (kernel) and Stage 3 (backend)."""

    dse_dimensions: Optional[Dict[str, Any]] = None
    """Other DSE dimensions, e.g., {'SIMD': 32, 'mem_mode': 'internal'}.
    Applied via with_dimension(name, value) API.
    Stage 2: SIMD, other kernel params
    Stage 3: mem_mode, ram_style, resType (backend-specific)"""

    # ========================================================================
    # Build Configuration (Optional)
    # ========================================================================

    fpgapart: Optional[str] = None
    """FPGA part number for backend testing, e.g., 'xc7z020clg400-1'.
    If None, cppsim/rtlsim tests are gracefully skipped."""

    backend_variants: Optional[List[str]] = None
    """Backend variants to test, e.g., ['hls', 'rtl']. Defaults to ['hls']."""

    # ========================================================================
    # Test Tolerances (Optional)
    # ========================================================================

    tolerance_python: Optional[Dict[str, float]] = None
    """Tolerance for Python execution tests, e.g., {'rtol': 1e-7, 'atol': 1e-9}"""

    tolerance_cppsim: Optional[Dict[str, float]] = None
    """Tolerance for C++ simulation tests, e.g., {'rtol': 1e-5, 'atol': 1e-6}"""

    tolerance_rtlsim: Optional[Dict[str, float]] = None
    """Tolerance for RTL simulation tests"""

    # ========================================================================
    # Test Metadata (Optional)
    # ========================================================================

    test_id: Optional[str] = None
    """Test identifier for pytest. Auto-generated if not provided."""

    marks: List[Any] = field(default_factory=list)
    """Pytest marks to apply, e.g., [pytest.mark.slow, pytest.mark.cppsim]"""

    def __post_init__(self):
        """Validate configuration and auto-generate test_id if needed."""
        # Validate required fields
        if not self.operation:
            raise ValueError("operation is required")
        if not self.input_shapes:
            raise ValueError("input_shapes is required")
        if not self.input_dtypes:
            raise ValueError("input_dtypes is required")

        # Validate input_shapes and input_dtypes have same keys
        if set(self.input_shapes.keys()) != set(self.input_dtypes.keys()):
            raise ValueError(
                f"input_shapes keys {set(self.input_shapes.keys())} "
                f"do not match input_dtypes keys {set(self.input_dtypes.keys())}"
            )

        # Validate stream indices
        if self.input_streams:
            max_input_idx = len(self.input_shapes) - 1
            for idx in self.input_streams.keys():
                if idx < 0 or idx > max_input_idx:
                    raise ValueError(
                        f"input_streams index {idx} out of range [0, {max_input_idx}]"
                    )

        # Set default backend_variants
        if self.backend_variants is None:
            self.backend_variants = ["hls"]

        # Auto-generate test_id if not provided
        if self.test_id is None:
            self.test_id = self._generate_test_id()

    def _generate_test_id(self) -> str:
        """Generate a test identifier from configuration.

        Format: {operation}_{dtype}_{shape}[_{stream_config}]

        Examples:
            - "Add_INT8_1-64"
            - "Add_INT8_1-64_pe8"
            - "MatMul_INT8_1-64-128_pe8_simd16"
        """
        # Get primary dtype (first input)
        first_input = list(self.input_dtypes.keys())[0]
        dtype_name = self.input_dtypes[first_input].name

        # Get primary shape (first input)
        shape = self.input_shapes[first_input]
        shape_str = "-".join(str(d) for d in shape)

        # Build base ID
        parts = [self.operation.lower(), dtype_name.lower(), shape_str]

        # Add stream configuration if present
        if self.input_streams:
            for idx, value in sorted(self.input_streams.items()):
                parts.append(f"pe{value}")

        if self.output_streams:
            for idx, value in sorted(self.output_streams.items()):
                parts.append(f"ope{value}")

        # Add key DSE dimensions
        if self.dse_dimensions:
            for name, value in sorted(self.dse_dimensions.items()):
                if name.lower() in ["simd", "mem_mode", "ram_style"]:
                    parts.append(f"{name.lower()}{value}")

        return "_".join(parts)

    def get_tolerance(self, execution_mode: str) -> Optional[Dict[str, float]]:
        """Get tolerance for specified execution mode.

        Args:
            execution_mode: One of "python", "cppsim", "rtlsim"

        Returns:
            Tolerance dict or None if not configured

        Raises:
            ValueError: If execution_mode is invalid
        """
        mode_map = {
            "python": self.tolerance_python,
            "cppsim": self.tolerance_cppsim,
            "rtlsim": self.tolerance_rtlsim,
        }

        if execution_mode not in mode_map:
            raise ValueError(
                f"Invalid execution_mode '{execution_mode}', "
                f"must be one of {list(mode_map.keys())}"
            )

        return mode_map[execution_mode]

    def has_backend_testing(self) -> bool:
        """Check if backend testing is enabled (fpgapart configured).

        Returns:
            True if cppsim/rtlsim tests should run, False otherwise
        """
        return self.fpgapart is not None

    def clone(self, **changes) -> "KernelTestConfig":
        """Create a modified copy of this configuration.

        Args:
            **changes: Fields to override in the new configuration

        Returns:
            New KernelTestConfig instance with specified changes

        Example:
            # Create variant with different stream parallelism
            config2 = config1.clone(input_streams={0: 16}, test_id="add_pe16")
        """
        from dataclasses import replace

        return replace(self, **changes)

    # ========================================================================
    # Extraction Methods with Defaults (v3.0)
    # ========================================================================

    def get_tolerance_python(self) -> Dict[str, float]:
        """Get Python execution tolerance with sensible defaults.

        Returns:
            Tolerance dict for numpy.allclose() comparison.
            Default: {"rtol": 1e-7, "atol": 1e-9} (tight for exact arithmetic)

        Example:
            tolerance = config.get_tolerance_python()
            np.allclose(actual, expected, **tolerance)
        """
        return self.tolerance_python or {"rtol": 1e-7, "atol": 1e-9}

    def get_tolerance_cppsim(self) -> Dict[str, float]:
        """Get C++ simulation tolerance with sensible defaults.

        Returns:
            Tolerance dict for numpy.allclose() comparison.
            Default: {"rtol": 1e-5, "atol": 1e-6} (moderate for fixed-point)

        Example:
            tolerance = config.get_tolerance_cppsim()
            np.allclose(actual, expected, **tolerance)
        """
        return self.tolerance_cppsim or {"rtol": 1e-5, "atol": 1e-6}

    def get_tolerance_rtlsim(self) -> Dict[str, float]:
        """Get RTL simulation tolerance with sensible defaults.

        Returns:
            Tolerance dict for numpy.allclose() comparison.
            Default: Delegates to cppsim tolerance (same precision expected)

        Example:
            tolerance = config.get_tolerance_rtlsim()
            np.allclose(actual, expected, **tolerance)
        """
        return self.tolerance_rtlsim or self.get_tolerance_cppsim()

    def get_fpgapart(self) -> Optional[str]:
        """Get FPGA part string for backend testing.

        Returns:
            FPGA part number (e.g., "xc7z020clg400-1") or None.
            If None, cppsim/rtlsim tests should be skipped.

        Example:
            fpgapart = config.get_fpgapart()
            if fpgapart:
                # Run backend tests
                op.set_nodeattr("fpgapart", fpgapart)
        """
        return self.fpgapart
