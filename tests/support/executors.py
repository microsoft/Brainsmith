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
    from tests.common.executors import CppSimExecutor, RTLSimExecutor
    from tests.common.validator import GoldenValidator

    # Execute
    executor = CppSimExecutor()
    outputs = executor.execute(op, model, inputs)

    # Validate (separate responsibility)
    validator = GoldenValidator()
    validator.validate(outputs, golden_outputs, "cppsim", rtol=1e-5, atol=1e-6)
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
from typing import Dict, Protocol, runtime_checkable

from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from brainsmith.settings import load_config
from brainsmith.dataflow.kernel_op import KernelOp

# Import test constants
from tests.common.constants import (
    PARITY_DEFAULT_FPGA_PART_HLS,
    PARITY_DEFAULT_CLOCK_PERIOD_NS
)


@runtime_checkable
class Executor(Protocol):
    """Protocol for backend executors.

    Executors have ONE job: execute a backend and return outputs.
    They do NOT compare, validate, or make assertions.

    Comparison/validation is the job of test frameworks and validators.
    """

    def execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
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
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute operator via Python execute_node().

        Args:
            op: Operator to execute
            model: Model containing operator
            inputs: Input tensors

        Returns:
            Dict with output tensors

        Raises:
            pytest.fail: If execution fails
        """
        try:
            # Set Python execution mode
            # Note: KernelOp uses empty string for Python execution, not "python"
            # Check what exec_mode values are allowed by inspecting the node attributes
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

            # Execute via execute_node
            context = dict(inputs)  # Copy to avoid mutation
            op.execute_node(context, model.graph)

            # Extract outputs
            outputs = {}
            for output_name in op.onnx_node.output:
                if output_name in context:
                    outputs[output_name] = context[output_name]

            return outputs

        except Exception as e:
            pytest.fail(
                f"Python execution failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates a bug in execute_node() implementation.\n"
            )


class CppSimExecutor:
    """HLS C++ simulation executor.

    Executes operators via Vivado HLS C++ simulation (cppsim).
    Validates code generation, compilation, and execution.

    Requires:
    - VITIS_PATH environment variable set
    - HLSBackend inheritance on operator
    """

    def execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
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

        # Export Brainsmith configuration to environment
        settings = load_config()
        settings.export_to_environment()

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

        # Execute
        try:
            context = dict(inputs)  # Copy to avoid mutation
            op.execute_node(context, model.graph)

            # Extract outputs
            outputs = {}
            for output_name in op.onnx_node.output:
                if output_name in context:
                    outputs[output_name] = context[output_name]

            return outputs

        except Exception as e:
            pytest.fail(
                f"cppsim execution failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"Code compiled successfully but execution failed.\n"
                f"Check execute_node() implementation."
            )

    def _ensure_kernel_ready(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Ensure kernel instance is available for code generation."""
        if hasattr(op, 'build_design_space'):
            # New KernelOp API (Brainsmith dataflow system)
            op.build_design_space(model)
        elif hasattr(op, 'get_kernel_instance'):
            # Legacy FINN API
            op.get_kernel_instance(model)
        elif hasattr(op, 'get_kernel_model'):
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
    - XSI (Xilinx Simulator) available via finn.xsi
    - RTLBackend or HLSBackend inheritance on operator
    """

    def __init__(
        self,
        fpgapart: str = PARITY_DEFAULT_FPGA_PART_HLS,
        clk_ns: float = PARITY_DEFAULT_CLOCK_PERIOD_NS
    ):
        """Initialize RTL simulator with FPGA part and clock period.

        Args:
            fpgapart: FPGA part string (default: from test constants)
            clk_ns: Clock period in nanoseconds (default: from test constants)
        """
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns

    def execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
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
                    "XSI (Xilinx Simulator) not available. "
                    "Run: python -m finn.xsi.setup"
                )
        except ImportError:
            pytest.skip("finn.xsi module not available")

        # Verify backend type
        try:
            from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
            from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

            if not isinstance(op, (RTLBackend, HLSBackend)):
                pytest.skip(
                    f"{op.__class__.__name__} is neither RTL nor HLS backend. "
                    f"rtlsim execution requires RTLBackend or HLSBackend inheritance."
                )

            is_hls_backend = isinstance(op, HLSBackend)
            is_rtl_backend = isinstance(op, RTLBackend)

        except ImportError:
            pytest.skip("RTLBackend/HLSBackend not available")

        # Export Brainsmith configuration to environment
        settings = load_config()
        settings.export_to_environment()

        try:
            # Ensure node has a name (required for HLS synthesis)
            from qonnx.transformation.general import GiveUniqueNodeNames
            model = model.transform(GiveUniqueNodeNames())

            # Refresh op reference after transformation
            from finn.util.basic import getHWCustomOp
            hw_node = model.graph.node[0]
            op = getHWCustomOp(hw_node, model)

            # Create temp directory
            tmpdir = tempfile.mkdtemp(prefix=f"rtlsim_{op.__class__.__name__}_")
            op.set_nodeattr("code_gen_dir_ipgen", tmpdir)

            # Save model
            model.save(os.path.join(tmpdir, "node_model.onnx"))

            # Ensure kernel instance is available
            self._ensure_kernel_ready(op, model)

            # Generate RTL (either directly or via HLS synthesis)
            if is_hls_backend:
                # HLS backend: Synthesize HLS → RTL using Vitis HLS
                op.code_generation_ipgen(model, self.fpgapart, self.clk_ns)
                op.ipgen_singlenode_code()  # Runs Vitis HLS synthesis
            elif is_rtl_backend:
                # RTL backend: Generate HDL directly
                op.generate_hdl(model, fpgapart=self.fpgapart)
            else:
                raise RuntimeError(f"Unknown backend type for {op.__class__.__name__}")

            # Prepare RTL simulation (compile Verilog with XSI/xsim)
            op.prepare_rtlsim()

            # Set execution mode
            op.set_nodeattr("exec_mode", "rtlsim")

        except Exception as e:
            pytest.fail(
                f"rtlsim preparation failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"This indicates an HDL generation or simulation setup bug.\n"
                f"\n"
                f"Debug steps:\n"
                f"1. Check {tmpdir} for generated HDL/HLS files\n"
                f"2. Look for Vitis HLS synthesis errors (HLS) or xsim compilation errors (RTL)\n"
                f"3. Compare with working backend implementation"
            )

        # Execute
        try:
            context = dict(inputs)  # Copy to avoid mutation
            op.execute_node(context, model.graph)

            # Extract outputs
            outputs = {}
            for output_name in op.onnx_node.output:
                if output_name in context:
                    outputs[output_name] = context[output_name]

            return outputs

        except Exception as e:
            pytest.fail(
                f"rtlsim execution failed for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"HDL generated successfully but execution failed.\n"
                f"Check execute_node() implementation."
            )

    def _ensure_kernel_ready(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Ensure kernel instance is available for HDL generation."""
        if hasattr(op, 'build_design_space'):
            # New KernelOp API (Brainsmith dataflow system)
            op.build_design_space(model)
        elif hasattr(op, 'get_kernel_instance'):
            # Legacy FINN API
            op.get_kernel_instance(model)
        elif hasattr(op, 'get_kernel_model'):
            # Alternative legacy API
            op.get_kernel_model(model)


# Convenience factory

def make_executors() -> Dict[str, Executor]:
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
    return {
        "python": PythonExecutor(),
        "cppsim": CppSimExecutor(),
        "rtlsim": RTLSimExecutor()
    }
