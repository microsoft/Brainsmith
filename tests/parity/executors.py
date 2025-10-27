"""Backend execution helpers for parity testing.

This module provides backend executor classes for cppsim and rtlsim execution,
eliminating duplication between test_cppsim_execution_parity and
test_rtlsim_execution_parity.

Key Features:
- Template method pattern for execution pipeline
- Consistent error handling and cleanup
- Detailed failure messages
- Reusable execution logic

Usage:
    from tests.parity.executors import CppSimExecutor, RTLSimExecutor

    # In test methods:
    executor = CppSimExecutor()
    executor.execute_and_compare(
        manual_op, manual_model,
        auto_op, auto_model,
        test_context
    )
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Use absolute imports for compatibility with sys.path-based imports
try:
    from assertions import assert_arrays_close
    from test_fixtures import make_execution_context
except ImportError:
    from .assertions import assert_arrays_close
    from .test_fixtures import make_execution_context


class BackendExecutor(ABC):
    """Abstract base class for backend execution (cppsim, rtlsim).

    Provides template method pattern for execution pipeline:
    1. Check requirements (tools available)
    2. Verify backend compatibility
    3. Prepare and execute manual backend
    4. Prepare and execute auto backend
    5. Compare outputs

    Subclasses implement backend-specific execution logic.
    """

    def execute_and_compare(
        self,
        manual_op: HWCustomOp,
        manual_model: ModelWrapper,
        auto_op: HWCustomOp,
        auto_model: ModelWrapper,
        test_context: Dict[str, np.ndarray]
    ) -> None:
        """Execute both backends and compare outputs.

        Args:
            manual_op: Manual implementation operator
            manual_model: Model containing manual operator
            auto_op: Auto implementation operator
            auto_model: Model containing auto operator
            test_context: Execution context with input tensors

        Raises:
            pytest.skip: If requirements not met or backend incompatible
            AssertionError: If outputs differ
            Exception: If execution fails
        """
        # Check requirements (Vitis, Verilator, etc.)
        self._check_requirements()

        # Verify backend compatibility
        self._verify_backend_type(manual_op)
        self._verify_backend_type(auto_op)

        # Extract input tensors (exclude initializers)
        input_dict = self._extract_input_dict(
            manual_op, manual_model, test_context
        )

        # Ensure we have streaming inputs
        if not input_dict:
            pytest.skip(
                f"No streaming inputs found for {manual_op.__class__.__name__}. "
                f"All inputs are initializers (weights/parameters)."
            )

        # Execute manual backend
        manual_outputs = self._prepare_and_execute(
            manual_op, manual_model, test_context, is_manual=True
        )

        # Recreate context with same seed for auto
        auto_context = make_execution_context(auto_model, auto_op, seed=42)

        # Execute auto backend
        auto_outputs = self._prepare_and_execute(
            auto_op, auto_model, auto_context, is_manual=False
        )

        # Compare outputs
        self._compare_outputs(
            manual_outputs, auto_outputs,
            manual_op, auto_op
        )

    @abstractmethod
    def _check_requirements(self) -> None:
        """Check if required tools are available.

        Raises:
            pytest.skip: If requirements not met
        """
        pass

    @abstractmethod
    def _verify_backend_type(self, op: HWCustomOp) -> None:
        """Verify operator has correct backend type.

        Args:
            op: Operator to check

        Raises:
            pytest.skip: If backend type incorrect
        """
        pass

    @abstractmethod
    def _prepare_and_execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        context: Dict[str, np.ndarray],
        is_manual: bool
    ) -> Dict[str, np.ndarray]:
        """Prepare backend and execute.

        Args:
            op: Operator to execute
            model: Model containing operator
            context: Execution context
            is_manual: True if manual backend, False if auto

        Returns:
            Dict mapping output names to arrays

        Raises:
            Exception: If preparation or execution fails
        """
        pass

    def _extract_input_dict(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        context: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Extract streaming inputs (exclude initializers).

        Args:
            op: Operator
            model: Model containing operator
            context: Full execution context

        Returns:
            Dict with streaming inputs only
        """
        input_dict = {}
        for inp_name in op.onnx_node.input:
            if inp_name and inp_name in context:
                # Only include if not an initializer
                if model.get_initializer(inp_name) is None:
                    input_dict[inp_name] = context[inp_name]
        return input_dict

    def _compare_outputs(
        self,
        manual_outputs: Dict[str, np.ndarray],
        auto_outputs: Dict[str, np.ndarray],
        manual_op: HWCustomOp,
        auto_op: HWCustomOp
    ) -> None:
        """Compare output tensors.

        Args:
            manual_outputs: Outputs from manual backend
            auto_outputs: Outputs from auto backend
            manual_op: Manual operator (for output count)
            auto_op: Auto operator

        Raises:
            AssertionError: If outputs differ
        """
        # Determine number of outputs
        num_outputs = len(manual_op.onnx_node.output)

        # Compare each output
        for ind in range(num_outputs):
            output_name = manual_op.onnx_node.output[ind]

            # Check both backends produced the output
            assert output_name in manual_outputs, \
                f"Manual backend didn't produce output: {output_name}"
            assert output_name in auto_outputs, \
                f"Auto backend didn't produce output: {output_name}"

            manual_array = manual_outputs[output_name]
            auto_array = auto_outputs[output_name]

            # Compare arrays
            assert_arrays_close(
                manual_array,
                auto_array,
                f"Output {ind} ({output_name})",
                rtol=1e-5,
                atol=1e-6
            )


class CppSimExecutor(BackendExecutor):
    """HLS C++ simulation executor.

    Executes operators via Vivado HLS C++ simulation (cppsim).
    Validates code generation, compilation, and execution.
    """

    def _check_requirements(self) -> None:
        """Check for Vitis availability."""
        if not os.environ.get("VITIS_PATH"):
            pytest.skip("Vitis required for C++ compilation (set VITIS_PATH)")

    def _verify_backend_type(self, op: HWCustomOp) -> None:
        """Verify operator is HLS backend."""
        try:
            from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
            if not isinstance(op, HLSBackend):
                pytest.skip(
                    f"{op.__class__.__name__} is not an HLS backend. "
                    f"cppsim execution requires HLSBackend inheritance."
                )
        except ImportError:
            pytest.skip("HLSBackend not available")

    def _prepare_and_execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        context: Dict[str, np.ndarray],
        is_manual: bool
    ) -> Dict[str, np.ndarray]:
        """Prepare HLS backend and execute via cppsim.

        Args:
            op: Operator to execute
            model: Model containing operator
            context: Execution context
            is_manual: True if manual backend

        Returns:
            Dict with output tensors

        Raises:
            pytest.fail: If code generation, compilation, or execution fails
        """
        backend_name = "manual" if is_manual else "auto"

        # Set BSMITH_DIR for compilation (needed by Brainsmith kernels)
        os.environ["BSMITH_DIR"] = "/home/tafk/dev/brainsmith-1"

        try:
            # Create temp directory for code generation
            tmpdir = tempfile.mkdtemp(prefix=f"cppsim_{backend_name}_")
            op.set_nodeattr("code_gen_dir_cppsim", tmpdir)

            # Save model to code_gen_dir (needed by exec_precompiled_singlenode_model)
            model.save(os.path.join(tmpdir, "node_model.onnx"))

            # Ensure kernel instance is available for KernelOp-based backends
            if hasattr(op, 'build_design_space'):
                # New KernelOp API (Brainsmith dataflow system)
                op.build_design_space(model)
            elif hasattr(op, 'get_kernel_instance'):
                # Legacy FINN API
                op.get_kernel_instance(model)
            elif hasattr(op, 'get_kernel_model'):
                # Alternative legacy API
                op.get_kernel_model(model)

            # Generate C++ code
            op.code_generation_cppsim(model)

            # Compile C++ code
            op.compile_singlenode_code()

            # Set execution mode
            op.set_nodeattr("exec_mode", "cppsim")

        except Exception as e:
            pytest.fail(
                f"{backend_name.capitalize()} backend cppsim pipeline failed "
                f"for {op.__class__.__name__}:\n"
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

        # Execute backend via cppsim
        try:
            op.execute_node(context, model.graph)
            result = {op.onnx_node.output[0]: context[op.onnx_node.output[0]]}
            return result
        except Exception as e:
            pytest.fail(
                f"{backend_name.capitalize()} backend execution failed "
                f"for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"Code compiled successfully but execution failed.\n"
                f"Check execute_node() implementation."
            )


class RTLSimExecutor(BackendExecutor):
    """RTL simulation executor using Xilinx Simulator (xsim).

    Executes operators via XSI (Xilinx Simulator Interface) RTL simulation.
    Validates HDL generation and execution.

    For HLS backends: Synthesizes HLS → RTL using Vitis HLS first.
    For RTL backends: Uses HDL directly.
    """

    def _check_requirements(self) -> None:
        """Check for XSI (Xilinx Simulator) availability."""
        try:
            from finn import xsi
            if not xsi.is_available():
                pytest.skip(
                    "XSI (Xilinx Simulator) not available. "
                    "Run: python -m finn.xsi.setup"
                )
        except ImportError:
            pytest.skip("finn.xsi module not available")

    def _verify_backend_type(self, op: HWCustomOp) -> None:
        """Verify operator supports RTL simulation (RTL or HLS backends)."""
        try:
            from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
            from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

            if not isinstance(op, (RTLBackend, HLSBackend)):
                pytest.skip(
                    f"{op.__class__.__name__} is neither RTL nor HLS backend. "
                    f"rtlsim execution requires RTLBackend or HLSBackend inheritance."
                )
        except ImportError:
            pytest.skip("RTLBackend/HLSBackend not available")

    def _prepare_and_execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        context: Dict[str, np.ndarray],
        is_manual: bool
    ) -> Dict[str, np.ndarray]:
        """Prepare RTL backend and execute via rtlsim.

        Supports both RTL and HLS backends:
        - RTL backends: Generate HDL directly
        - HLS backends: Synthesize HLS → RTL using Vitis HLS, then simulate

        Args:
            op: Operator to execute
            model: Model containing operator
            context: Execution context
            is_manual: True if manual backend

        Returns:
            Dict with output tensors

        Raises:
            pytest.fail: If HDL generation, compilation, or execution fails
        """
        backend_name = "manual" if is_manual else "auto"

        # Set BSMITH_DIR for compilation
        os.environ["BSMITH_DIR"] = "/home/tafk/dev/brainsmith-1"

        # Get FPGA part and clock period (could be parameters)
        fpgapart = "xcvu9p-flgb2104-2-i"
        clk_ns = 3.0

        # Detect backend type
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
        from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
        is_hls_backend = isinstance(op, HLSBackend)
        is_rtl_backend = isinstance(op, RTLBackend)

        try:
            # Ensure node has a name (required for HLS synthesis)
            from qonnx.transformation.general import GiveUniqueNodeNames
            model = model.transform(GiveUniqueNodeNames())
            # Refresh op reference after transformation
            op = model.graph.node[0]
            from finn.util.basic import getHWCustomOp
            op = getHWCustomOp(op, model)

            # Create temp directory
            tmpdir = tempfile.mkdtemp(prefix=f"rtlsim_{backend_name}_")
            op.set_nodeattr("code_gen_dir_ipgen", tmpdir)
            # Note: fpgapart and clk_ns are passed as method parameters, not stored as attributes

            # Save model
            model.save(os.path.join(tmpdir, "node_model.onnx"))

            # Ensure kernel instance is available for KernelOp-based backends
            if hasattr(op, 'build_design_space'):
                # New KernelOp API (Brainsmith dataflow system)
                op.build_design_space(model)
            elif hasattr(op, 'get_kernel_instance'):
                # Legacy FINN API
                op.get_kernel_instance(model)
            elif hasattr(op, 'get_kernel_model'):
                # Alternative legacy API
                op.get_kernel_model(model)

            # Generate RTL (either directly or via HLS synthesis)
            if is_hls_backend:
                # HLS backend: Synthesize HLS → RTL using Vitis HLS
                # Following FINN pattern from test_fpgadataflow_addstreams.py
                op.code_generation_ipgen(model, fpgapart, clk_ns)
                op.ipgen_singlenode_code()  # Runs Vitis HLS synthesis
            elif is_rtl_backend:
                # RTL backend: Generate HDL directly
                op.generate_hdl(model, fpgapart=fpgapart)
            else:
                raise RuntimeError(f"Unknown backend type for {op.__class__.__name__}")

            # Prepare RTL simulation (compile Verilog with XSI/xsim)
            op.prepare_rtlsim()

            # Set execution mode
            op.set_nodeattr("exec_mode", "rtlsim")

        except Exception as e:
            pytest.fail(
                f"{backend_name.capitalize()} backend rtlsim pipeline failed "
                f"for {op.__class__.__name__}:\n"
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

        # Execute backend via rtlsim
        try:
            op.execute_node(context, model.graph)
            result = {op.onnx_node.output[0]: context[op.onnx_node.output[0]]}
            return result
        except Exception as e:
            pytest.fail(
                f"{backend_name.capitalize()} backend execution failed "
                f"for {op.__class__.__name__}:\n"
                f"\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\n"
                f"HDL generated successfully but execution failed.\n"
                f"Check execute_node() implementation."
            )
