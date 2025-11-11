"""Clean executor protocol for backend execution.

This module provides a clean separation of concerns for backend execution:
- Executors ONLY execute (single responsibility)
- Tests handle comparison using GoldenValidator
- No mixed responsibilities, no private methods called publicly

Replaces the BackendExecutor pattern from tests/parity/executors.py which
violated SRP by mixing execution and comparison.

Design Philosophy:
- Single Responsibility: Executors execute, validators validate
- Protocol pattern (PEP 544): No inheritance required
- Composable: Works with any test framework
- Clear errors: Detailed failure messages for each step

Usage:
    from tests.support.executors import CppSimExecutor, RTLSimExecutor
    from tests.support.validator import GoldenValidator

    # Execute
    executor = CppSimExecutor()
    outputs = executor.execute(op, model, inputs)

    # Validate (separate responsibility)
    validator = GoldenValidator()
    validator.validate(outputs, golden_outputs, "cppsim", rtol=1e-5, atol=1e-6)
"""

import os
import tempfile
from typing import Protocol, runtime_checkable

import numpy as np
import pytest
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.modelwrapper import ModelWrapper

# Import test constants
from tests.support.constants import PARITY_DEFAULT_CLOCK_PERIOD_NS, PARITY_DEFAULT_FPGA_PART_HLS


@runtime_checkable
class Executor(Protocol):
    """Protocol for backend executors.

    Executors have ONE job: execute a backend and return outputs.
    They do NOT compare, validate, or make assertions.

    Comparison/validation is the job of test frameworks and validators.
    """

    def execute(
        self, op: HWCustomOp, model: ModelWrapper, inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Execute backend and return outputs.

        Args:
            op: Operator to execute
            model: Model containing operator
            inputs: Input tensors (execution context)

        Returns:
            Dict mapping output names → output arrays

        Raises:
            pytest.skip: If requirements not met or backend incompatible
            pytest.fail: If execution fails (with detailed error message)
        """
        ...


class PythonExecutor:
    """Python execution via execute_node().

    Executes ONNX operators using Python/NumPy implementation.
    This is the reference implementation for correctness.
    """

    def execute(
        self, op: HWCustomOp, model: ModelWrapper, inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Execute operator via QONNX execution (includes Quant nodes).

        Args:
            op: Operator to execute (target kernel node)
            model: Model containing operator (may include Quant nodes)
            inputs: Input tensors (with raw_* names if Quant nodes present)

        Returns:
            Dict with output tensors

        Raises:
            pytest.fail: If execution fails

        Note:
            Uses QONNX execution to run the entire model, including any Quant nodes
            that precede the target kernel. This ensures correct quantized execution
            and matches the golden reference computation.
        """
        try:
            # Set Python execution mode for target operator
            # Note: KernelOp uses empty string for Python execution, not "python"
            allowed_exec_modes = op.get_nodeattr_allowed_values("exec_mode")

            if "" in allowed_exec_modes:
                # KernelOp uses "" for Python execution
                op.set_nodeattr("exec_mode", "")
            elif "python" in allowed_exec_modes:
                # Standard FINN HWCustomOp uses "python"
                op.set_nodeattr("exec_mode", "python")
            else:
                pytest.fail(
                    f"Operator {op.__class__.__name__} has unknown exec_mode values: {allowed_exec_modes}"
                )

            # Execute full model with QONNX (includes Quant nodes)
            from qonnx.core.onnx_exec import execute_onnx

            # Execute up to and including the target operator
            return execute_onnx(
                model, inputs, return_full_exec_context=False, end_node=op.onnx_node
            )

        except Exception as e:
            pytest.fail(
                f"Python execution failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates a bug in execute_node() implementation or Quant node execution.\n"
            )


class CppSimExecutor:
    """HLS C++ simulation executor.

    Executes operators via Vivado HLS C++ simulation (cppsim).
    Validates code generation, compilation, and execution.

    Requires:
    - Environment sourced before running tests (source .brainsmith/env.sh)
    - VITIS_PATH environment variable set
    - HLSBackend inheritance on operator
    """

    def execute(
        self, op: HWCustomOp, model: ModelWrapper, inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Execute operator via cppsim.

        Args:
            op: Operator to execute (must be HLSBackend)
            model: Model containing operator
            inputs: Input tensors

        Returns:
            Dict with output tensors

        Raises:
            pytest.skip: If Vitis not available or operator not HLS backend
            pytest.fail: If code generation, compilation, or execution fails
        """
        # Check requirements
        if not os.environ.get("VITIS_PATH"):
            pytest.skip("Vitis required for C++ compilation (set VITIS_PATH)")

        # Verify backend type
        try:
            from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

            if not isinstance(op, HLSBackend):
                pytest.skip(
                    f"{op.__class__.__name__} is not an HLS backend. "
                    f"cppsim execution requires HLSBackend inheritance."
                )
        except ImportError:
            pytest.skip("HLSBackend not available")

        try:
            # Create temp directory for code generation
            tmpdir = tempfile.mkdtemp(prefix=f"cppsim_{op.__class__.__name__}_")
            op.set_nodeattr("code_gen_dir_cppsim", tmpdir)

            # Save model to code_gen_dir
            model.save(os.path.join(tmpdir, "node_model.onnx"))

            # Ensure kernel instance is available
            self._ensure_kernel_ready(op, model)

            # Generate C++ code
            op.code_generation_cppsim(model)

            # Compile C++ code
            op.compile_singlenode_code()

            # Set execution mode
            op.set_nodeattr("exec_mode", "cppsim")

        except Exception as e:
            pytest.fail(
                f"cppsim preparation failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates a code generation or compilation bug.\n"
                f"\n"
                f"Debug steps:\n"
                f"1. Check {tmpdir} for generated C++ files\n"
                f"2. Look for compilation errors in build logs\n"
                f"3. Compare with working backend implementation"
            )

        # Execute full model with QONNX (includes Quant nodes + cppsim backend)
        try:
            from qonnx.core.onnx_exec import execute_onnx

            # Execute up to and including the target operator
            return execute_onnx(
                model, inputs, return_full_exec_context=False, end_node=op.onnx_node
            )

        except Exception as e:
            pytest.fail(
                f"cppsim execution failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"Code compiled successfully but execution failed.\n"
                f"Check execute_node() implementation or Quant node execution."
            )

    def _ensure_kernel_ready(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Ensure kernel instance is available for code generation."""
        if hasattr(op, "build_design_space"):
            # New KernelOp API (Brainsmith dataflow system)
            op.build_design_space(model)
        elif hasattr(op, "get_kernel_instance"):
            # Legacy FINN API
            op.get_kernel_instance(model)
        elif hasattr(op, "get_kernel_model"):
            # Alternative legacy API
            op.get_kernel_model(model)


class RTLSimExecutor:
    """RTL simulation executor using Xilinx Simulator (xsim).

    Executes operators via XSI (Xilinx Simulator Interface) RTL simulation.
    Validates HDL generation and execution.

    Supports both:
    - HLS backends: Synthesizes HLS → RTL using Vitis HLS first
    - RTL backends: Uses HDL directly

    Requires:
    - Environment sourced before running tests (source .brainsmith/env.sh)
    - XSI (Xilinx Simulator) available via finn.xsi
    - RTLBackend or HLSBackend inheritance on operator
    """

    def __init__(
        self,
        fpgapart: str = PARITY_DEFAULT_FPGA_PART_HLS,
        clk_ns: float = PARITY_DEFAULT_CLOCK_PERIOD_NS,
    ):
        """Initialize RTL simulator with FPGA part and clock period.

        Args:
            fpgapart: FPGA part string (default: from test constants)
            clk_ns: Clock period in nanoseconds (default: from test constants)
        """
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns

    def execute(
        self, op: HWCustomOp, model: ModelWrapper, inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Execute operator via rtlsim.

        Args:
            op: Operator to execute (must be RTLBackend or HLSBackend)
            model: Model containing operator
            inputs: Input tensors

        Returns:
            Dict with output tensors

        Raises:
            pytest.skip: If XSI not available or operator not RTL/HLS backend
            pytest.fail: If HDL generation, compilation, or execution fails
        """
        # Check requirements
        try:
            from finn import xsi

            if not xsi.is_available():
                pytest.skip(
                    "XSI (Xilinx Simulator) not available. " "Run: python -m finn.xsi.setup"
                )
        except ImportError:
            pytest.skip("finn.xsi module not available")

        # Verify backend type
        try:
            from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
            from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

            if not isinstance(op, RTLBackend | HLSBackend):
                pytest.skip(
                    f"{op.__class__.__name__} is neither RTL nor HLS backend. "
                    f"rtlsim execution requires RTLBackend or HLSBackend inheritance."
                )

            is_hls_backend = isinstance(op, HLSBackend)
            is_rtl_backend = isinstance(op, RTLBackend)

        except ImportError:
            pytest.skip("RTLBackend/HLSBackend not available")

        # Track code generation directory for error messages
        code_gen_dir = None

        try:
            # Ensure node has a name (required for HLS synthesis)
            from qonnx.transformation.general import GiveUniqueNodeNames

            model = model.transform(GiveUniqueNodeNames())

            # Refresh op reference after transformation
            from finn.util.basic import getHWCustomOp

            hw_node = model.graph.node[0]
            op = getHWCustomOp(hw_node, model)

            # Ensure kernel instance is available
            self._ensure_kernel_ready(op, model)

            # Use FINN's transformation-based approach (like FINN's own tests)
            # This is the recommended way instead of calling methods directly
            # Note: PrepareIP will create code_gen_dir_ipgen automatically
            from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
            from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
            from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
            from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

            if is_hls_backend:
                # HLS backend: Use FINN's transformation pipeline
                model = model.transform(PrepareIP(self.fpgapart, self.clk_ns))
                model = model.transform(HLSSynthIP())
                # Get code_gen_dir for error messages
                hw_node = model.graph.node[0]
                op = getHWCustomOp(hw_node, model)
                code_gen_dir = op.get_nodeattr("code_gen_dir_ipgen")
            elif is_rtl_backend:
                # RTL backend: Generate HDL directly (no HLS synthesis needed)
                # For RTL, we still need to set code_gen_dir_ipgen
                code_gen_dir = tempfile.mkdtemp(prefix=f"rtlsim_{op.__class__.__name__}_")
                op.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
                op.generate_hdl(model, fpgapart=self.fpgapart)
            else:
                raise RuntimeError(f"Unknown backend type for {op.__class__.__name__}")

            # Prepare RTL simulation and set exec mode (same for both HLS/RTL)
            model = model.transform(PrepareRTLSim())
            model = model.transform(SetExecMode("rtlsim"))

            # Refresh op reference after transformations
            hw_node = model.graph.node[0]
            op = getHWCustomOp(hw_node, model)

        except Exception as e:
            debug_dir_msg = (
                f"1. Check {code_gen_dir} for generated HDL/HLS files\n" if code_gen_dir else ""
            )
            pytest.fail(
                f"rtlsim preparation failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates an HDL generation or simulation setup bug.\n"
                f"\n"
                f"Debug steps:\n"
                f"{debug_dir_msg}"
                f"2. Look for Vitis HLS synthesis errors (HLS) or xsim compilation errors (RTL)\n"
                f"3. Compare with working backend implementation"
            )

        # Execute full model with QONNX (includes Quant nodes + rtlsim backend)
        try:
            from qonnx.core.onnx_exec import execute_onnx

            # Execute up to and including the target operator
            return execute_onnx(
                model, inputs, return_full_exec_context=False, end_node=op.onnx_node
            )

        except Exception as e:
            pytest.fail(
                f"rtlsim execution failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"HDL generated successfully but execution failed.\n"
                f"Check execute_node() implementation or Quant node execution."
            )

    def _ensure_kernel_ready(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Ensure kernel instance is available for HDL generation."""
        if hasattr(op, "build_design_space"):
            # New KernelOp API (Brainsmith dataflow system)
            op.build_design_space(model)
        elif hasattr(op, "get_kernel_instance"):
            # Legacy FINN API
            op.get_kernel_instance(model)
        elif hasattr(op, "get_kernel_model"):
            # Alternative legacy API
            op.get_kernel_model(model)


# Convenience factory


def make_executors() -> dict[str, Executor]:
    """Create standard set of executors for common test patterns.

    Returns:
        Dict mapping backend names → executor instances

    Example:
        executors = make_executors()

        # Execute across all backends
        for backend_name, executor in executors.items():
            outputs = executor.execute(op, model, inputs)
            validator.validate(outputs, golden, backend_name, **tol)
    """
    return {"python": PythonExecutor(), "cppsim": CppSimExecutor(), "rtlsim": RTLSimExecutor()}
