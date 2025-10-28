"""Base class for pipeline integration testing with golden reference validation.

This module provides IntegratedPipelineTest, a comprehensive framework for testing
the complete ONNX → Hardware → Execution pipeline with golden reference validation.

Design Philosophy:
- Pipeline-centric: Tests complete transformation flow, not isolated components
- Golden-referenced: All backends must match NumPy/PyTorch ground truth
- Multi-backend: Validates Python, HLS cppsim, RTL rtlsim consistency
- Explicit stages: Each pipeline stage is validated independently

Phase 1 Implementation:
- Transform-based pipeline testing
- Golden reference validation
- Python execution testing
- HLS cppsim execution testing
- Basic pipeline validation (shapes, datatypes, node creation)

Future Phases:
- Phase 2: Cross-backend parity, parametric testing
- Phase 3: Pipeline snapshots, property validation, stage-by-stage validators
"""

import logging
import numpy as np
import pytest
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Type, List

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.custom_op.registry import getCustomOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from brainsmith.dataflow.kernel_op import KernelOp

# Reuse parity test infrastructure
try:
    from tests.parity.assertions import assert_arrays_close
    from tests.parity.executors import CppSimExecutor, RTLSimExecutor
    from tests.parity.test_fixtures import make_execution_context
    from tests.parity.backend_helpers import setup_hls_backend_via_specialize
except ImportError:
    # Fallback to absolute imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'parity'))
    from assertions import assert_arrays_close
    from executors import CppSimExecutor, RTLSimExecutor
    from test_fixtures import make_execution_context
    from backend_helpers import setup_hls_backend_via_specialize

logger = logging.getLogger(__name__)


class IntegratedPipelineTest(ABC):
    """Base class for pipeline integration tests with golden reference validation.

    Validates complete ONNX → Hardware → Execution pipeline:
    1. Standard ONNX node creation
    2. Shape inference (InferShapes)
    3. Datatype inference (InferDataTypes)
    4. Kernel inference (converts ONNX → Hardware node)
    5. Backend specialization (Hardware → Backend-specific)
    6. Code generation and execution (Python, cppsim, rtlsim)
    7. Golden reference validation (all backends match NumPy/PyTorch)

    Subclass Pattern:
        class TestMyKernelIntegration(IntegratedPipelineTest):
            def make_test_model(self):
                # Create standard ONNX node (e.g., Add, not AddStreams)
                return model, node_name

            def get_kernel_inference_transform(self):
                # Return transform that creates hardware node
                return InferKernelList

            def get_kernel_class(self):
                # Return kernel class for golden reference
                return MyKernel

    Phase 1 Tests (Inherited):
    - test_pipeline_creates_hw_node()
    - test_shapes_preserved_through_pipeline()
    - test_datatypes_preserved_through_pipeline()
    - test_python_execution_vs_golden()
    - test_cppsim_execution_vs_golden()
    """

    # ================================================================
    # Abstract Methods - Subclasses MUST implement
    # ================================================================

    @abstractmethod
    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create standard ONNX model for testing.

        Create a standard ONNX node (NOT a hardware node). The pipeline
        will transform it to a hardware node via kernel inference.

        Returns:
            (model, node_name): ModelWrapper and name of ONNX node

        Example:
            def make_test_model(self):
                # Create ONNX Add node (not AddStreams)
                inp0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 64])
                inp1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 64])
                out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

                node = helper.make_node("Add", ["input0", "input1"], ["output"])
                graph = helper.make_graph([node], "test_add", [inp0, inp1], [out])
                model = helper.make_model(graph)

                return ModelWrapper(model), "Add_0"
        """
        pass

    @abstractmethod
    def get_kernel_inference_transform(self) -> Type[Transformation]:
        """Return transform that converts ONNX node to hardware kernel.

        Returns:
            Transformation class (e.g., InferKernelList, InferLayerNorm)

        Example:
            def get_kernel_inference_transform(self):
                from brainsmith.transforms.infer_kernel_list import InferKernelList
                return InferKernelList
        """
        pass

    @abstractmethod
    def get_kernel_class(self) -> Type[KernelOp]:
        """Return kernel class for golden reference access.

        Returns:
            KernelOp class (e.g., AddStreams, ElementwiseBinaryOp)

        Example:
            def get_kernel_class(self):
                from brainsmith.kernels.addstreams import AddStreams
                return AddStreams
        """
        pass

    # ================================================================
    # Optional Configuration Hooks
    # ================================================================

    def get_num_inputs(self) -> int:
        """Number of inputs to test. Override if > 1."""
        return 1

    def get_num_outputs(self) -> int:
        """Number of outputs to test. Override if > 1."""
        return 1

    def configure_kernel_node(self, op: HWCustomOp, model: ModelWrapper) -> None:
        """Configure kernel node after inference (e.g., override PE/SIMD).

        Called after kernel inference creates HW node, before backend specialization.
        Use to set non-default parameters for testing.

        Args:
            op: The hardware kernel op instance
            model: The ModelWrapper containing the op

        Example:
            def configure_kernel_node(self, op, model):
                op.set_nodeattr("PE", 8)
                op.set_nodeattr("SIMD", 16)
        """
        pass

    def get_golden_tolerance_python(self) -> Dict[str, float]:
        """Tolerance for Python execution vs golden reference.

        Returns:
            Dict with 'rtol' and 'atol' keys

        Override for kernels with lower numerical precision.
        """
        return {"rtol": 1e-7, "atol": 1e-9}

    def get_golden_tolerance_cppsim(self) -> Dict[str, float]:
        """Tolerance for C++ simulation vs golden reference.

        Returns:
            Dict with 'rtol' and 'atol' keys

        Override for kernels with lower numerical precision.
        """
        return {"rtol": 1e-5, "atol": 1e-6}

    # ================================================================
    # Pipeline Execution Helpers
    # ================================================================

    def run_inference_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run complete inference pipeline (ONNX → Hardware node).

        Pipeline stages:
        1. Create standard ONNX model
        2. InferShapes
        3. InferDataTypes
        4. Kernel-specific inference (creates HW node)
        5. Configure kernel node (if overridden)

        Returns:
            (op, model): Hardware op instance and model

        Raises:
            RuntimeError: If pipeline fails to create HW node
        """
        # Create base model
        model, expected_node_name = self.make_test_model()

        # Run standard ONNX inference
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Run kernel-specific inference
        kernel_transform_class = self.get_kernel_inference_transform()
        kernel_transform = kernel_transform_class()
        model = model.transform(kernel_transform)

        # Find the hardware node
        hw_node = self._find_hw_node(model)
        if hw_node is None:
            available_nodes = [(n.name, n.op_type) for n in model.graph.node]
            raise RuntimeError(
                f"Kernel inference failed to create hardware node.\\n"
                f"Transform: {kernel_transform_class.__name__}\\n"
                f"Available nodes: {available_nodes}"
            )

        # Get op instance
        op = getCustomOp(hw_node)

        # Initialize KernelOp design_point (if it's a KernelOp)
        if isinstance(op, KernelOp):
            op._ensure_ready(model)

        # Allow subclass configuration
        self.configure_kernel_node(op, model)

        # Re-initialize after configuration (in case PE/SIMD changed)
        if isinstance(op, KernelOp):
            op._ensure_ready(model)

        return op, model

    def get_target_backend(self) -> Optional[str]:
        """Specify preferred backend implementation.

        Returns:
            Backend identifier ("hls", "rtl", etc.) or None for auto-select

        This sets preferred_impl_style but doesn't guarantee the backend
        will be selected. Test authors should configure the kernel with
        parameters appropriate for their target backend.

        Example:
            def get_target_backend(self):
                return "rtl"  # Prefer RTL backend

        Note:
            Phase 2 feature - manual backend specification.
            Override in subclasses to test specific backends.
        """
        return None

    def run_hls_specialization(
        self, base_op: HWCustomOp, base_model: ModelWrapper
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize kernel node to hardware backend.

        NOTE: Despite the name, this may specialize to HLS, RTL, or other
        backends depending on get_target_backend() and kernel configuration.

        Args:
            base_op: Base hardware kernel op
            base_model: Model containing base kernel node

        Returns:
            (hw_op, hw_model): Hardware-specialized op and model
        """
        target_backend = self.get_target_backend()

        # Set preferred_impl_style if specified
        if target_backend:
            base_op.set_nodeattr("preferred_impl_style", target_backend)

        hls_op, hls_model = setup_hls_backend_via_specialize(base_op, base_model)

        # Initialize KernelOp design_point after specialization
        if isinstance(hls_op, KernelOp):
            hls_op._ensure_ready(hls_model)

        return hls_op, hls_model

    def _find_hw_node(self, model: ModelWrapper) -> Optional[Any]:
        """Find hardware node in graph (after kernel inference).

        Searches for nodes that are HWCustomOp instances.

        Returns:
            ONNX node if found, None otherwise
        """
        for node in model.graph.node:
            try:
                op = getCustomOp(node)
                if isinstance(op, HWCustomOp):
                    return node
            except:
                continue
        return None

    # ================================================================
    # Golden Reference Helpers
    # ================================================================

    def compute_golden_reference(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute golden reference using kernel's reference implementation.

        Delegates to kernel class's compute_golden_reference() method.

        Args:
            inputs: Dict mapping input names → numpy arrays

        Returns:
            Dict mapping output names → expected numpy arrays

        Example:
            inputs = {"input0": np.array([...]), "input1": np.array([...])}
            golden = self.compute_golden_reference(inputs)
            # → {"output": np.array([...])}
        """
        kernel_class = self.get_kernel_class()
        if not hasattr(kernel_class, "compute_golden_reference"):
            raise NotImplementedError(
                f"{kernel_class.__name__} does not implement compute_golden_reference(). "
                f"Add static method: "
                f"@staticmethod\\n"
                f"def compute_golden_reference(inputs: Dict) -> Dict:\\n"
                f"    ..."
            )

        return kernel_class.compute_golden_reference(inputs)

    def validate_against_golden(
        self,
        actual_outputs: Dict[str, np.ndarray],
        golden_outputs: Dict[str, np.ndarray],
        backend_name: str,
        tolerance: Dict[str, float],
    ) -> None:
        """Validate actual outputs match golden reference.

        Args:
            actual_outputs: Outputs from backend execution
            golden_outputs: Expected outputs from golden reference
            backend_name: Name of backend for error messages ("Python", "C++ simulation", etc.)
            tolerance: Dict with 'rtol' and 'atol' keys

        Raises:
            AssertionError: If outputs don't match within tolerance
        """
        for output_name, golden_array in golden_outputs.items():
            if output_name not in actual_outputs:
                raise AssertionError(
                    f"{backend_name} missing output '{output_name}'.\\n"
                    f"Expected outputs: {list(golden_outputs.keys())}\\n"
                    f"Actual outputs: {list(actual_outputs.keys())}"
                )

            actual_array = actual_outputs[output_name]

            assert_arrays_close(
                actual_array,
                golden_array,
                f"{backend_name} vs golden reference for '{output_name}'",
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
            )

    # ================================================================
    # Execution Helpers
    # ================================================================

    def execute_python(
        self, op: HWCustomOp, model: ModelWrapper, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute kernel using Python simulation (execute_node).

        Args:
            op: Hardware op instance
            model: Model containing the op
            inputs: Input tensors

        Returns:
            Dict mapping output names → output arrays
        """
        context = dict(inputs)
        op.execute_node(context, model.graph)

        # Extract outputs
        outputs = {}
        for i in range(self.get_num_outputs()):
            output_name = op.onnx_node.output[i]
            outputs[output_name] = context[output_name]

        return outputs

    def execute_cppsim(
        self, op: HWCustomOp, model: ModelWrapper, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute kernel using HLS C++ simulation.

        Args:
            op: HLS backend op instance
            model: Model containing the op
            inputs: Input tensors

        Returns:
            Dict mapping output names → output arrays

        Requires:
            - VITIS_PATH environment variable
            - HLS backend (op inherits from HLSBackend)
        """
        executor = CppSimExecutor()

        # Execute via cppsim
        context = dict(inputs)
        outputs = executor._prepare_and_execute(op, model, context, is_manual=False)

        return outputs

    def execute_rtlsim(
        self, op: HWCustomOp, model: ModelWrapper, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute kernel using RTL Verilog simulation.

        Args:
            op: RTL backend op instance
            model: Model containing the op
            inputs: Input tensors

        Returns:
            Dict mapping output names → output arrays

        Requires:
            - Vivado installation with XSim
            - RTL backend (op inherits from RTLBackend)

        Note:
            Phase 2 feature - RTL simulation support for multi-backend testing.
        """
        executor = RTLSimExecutor()

        # Execute via rtlsim
        context = dict(inputs)
        outputs = executor._prepare_and_execute(op, model, context, is_manual=False)

        return outputs

    # ================================================================
    # Phase 1 Integration Tests
    # ================================================================

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_pipeline_creates_hw_node(self):
        """Validate that kernel inference creates hardware node.

        Pipeline:
        1. Create standard ONNX node
        2. Run inference (shapes, datatypes, kernel)
        3. Verify HW node was created
        4. Verify it's a HWCustomOp instance
        """
        op, model = self.run_inference_pipeline()

        # Verify op is HWCustomOp
        assert isinstance(
            op, HWCustomOp
        ), f"Kernel inference created {type(op)}, expected HWCustomOp"

        # Verify node exists in graph
        node_found = False
        for node in model.graph.node:
            if node.name == op.onnx_node.name:
                node_found = True
                break

        assert node_found, f"Hardware node '{op.onnx_node.name}' not found in graph"

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_shapes_preserved_through_pipeline(self):
        """Validate tensor shapes remain correct through inference pipeline."""
        op, model = self.run_inference_pipeline()

        # Validate input shapes
        for i in range(self.get_num_inputs()):
            input_name = op.onnx_node.input[i]
            if not input_name:  # Optional input
                continue

            # Get shape from model
            model_shape = model.get_tensor_shape(input_name)
            op_shape = op.get_normal_input_shape(i)

            assert tuple(model_shape) == tuple(op_shape), (
                f"Input {i} shape mismatch:\\n"
                f"  Model: {model_shape}\\n"
                f"  Op:    {op_shape}"
            )

        # Validate output shapes
        for i in range(self.get_num_outputs()):
            output_name = op.onnx_node.output[i]

            # Get shape from model
            model_shape = model.get_tensor_shape(output_name)
            op_shape = op.get_normal_output_shape(i)

            assert tuple(model_shape) == tuple(op_shape), (
                f"Output {i} shape mismatch:\\n"
                f"  Model: {model_shape}\\n"
                f"  Op:    {op_shape}"
            )

    @pytest.mark.pipeline
    @pytest.mark.phase1
    def test_datatypes_preserved_through_pipeline(self):
        """Validate tensor datatypes remain correct through inference pipeline."""
        op, model = self.run_inference_pipeline()

        # Validate input datatypes
        for i in range(self.get_num_inputs()):
            input_name = op.onnx_node.input[i]
            if not input_name:  # Optional input
                continue

            # Get datatype from model
            model_dt = model.get_tensor_datatype(input_name)
            op_dt = op.get_input_datatype(i)

            assert model_dt == op_dt, (
                f"Input {i} datatype mismatch:\\n"
                f"  Model: {model_dt}\\n"
                f"  Op:    {op_dt}"
            )

        # Validate output datatypes
        for i in range(self.get_num_outputs()):
            output_name = op.onnx_node.output[i]

            # Get datatype from model
            model_dt = model.get_tensor_datatype(output_name)
            op_dt = op.get_output_datatype(i)

            assert model_dt == op_dt, (
                f"Output {i} datatype mismatch:\\n"
                f"  Model: {model_dt}\\n"
                f"  Op:    {op_dt}"
            )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.phase1
    def test_python_execution_vs_golden(self):
        """Test Python execution (execute_node) matches golden reference.

        Validates:
        1. Kernel inference creates correct HW node
        2. Python execution produces correct results
        3. Results match golden reference within tolerance
        """
        # Run pipeline
        op, model = self.run_inference_pipeline()

        # Generate test inputs (deterministic)
        np.random.seed(42)
        inputs = make_execution_context(model, op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Execute via Python
        actual_outputs = self.execute_python(op, model, inputs)

        # Validate against golden
        tolerance = self.get_golden_tolerance_python()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "Python execution", tolerance
        )

    @pytest.mark.pipeline
    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.phase1
    def test_cppsim_execution_vs_golden(self):
        """Test hardware backend execution matches golden reference.

        NOTE: Despite the name "cppsim", this test actually validates whichever
        backend was selected during specialization:
        - HLS backend → C++ simulation (cppsim)
        - RTL backend → Verilog simulation (rtlsim)
        - Other backends → Appropriate simulator

        The backend is selected based on get_target_backend() preference
        and kernel configuration.

        Validates complete code generation pipeline:
        1. Kernel inference creates HW node
        2. Backend specialization creates backend-specific node
        3. Code generation succeeds (C++ or Verilog)
        4. Compilation/elaboration succeeds
        5. Simulation produces correct results
        6. Results match golden reference within tolerance

        Requires:
            - For HLS: VITIS_PATH environment variable
            - For RTL: Vivado installation with XSim
        """
        # Run pipeline to get base kernel
        base_op, base_model = self.run_inference_pipeline()

        # Specialize to hardware backend (HLS, RTL, etc.)
        hw_op, hw_model = self.run_hls_specialization(base_op, base_model)

        # Generate test inputs (deterministic)
        np.random.seed(42)
        inputs = make_execution_context(hw_model, hw_op)

        # Compute golden reference
        golden_outputs = self.compute_golden_reference(inputs)

        # Detect backend type and execute appropriately
        # Check for RTL backend indicators
        if hasattr(hw_op, 'generate_hdl') or 'RTL' in hw_op.__class__.__name__:
            # RTL backend - use rtlsim
            actual_outputs = self.execute_rtlsim(hw_op, hw_model, inputs)
            backend_label = "RTL simulation (rtlsim)"
            tolerance = self.get_golden_tolerance_cppsim()  # Same tolerance for now
        else:
            # HLS or other backend - use cppsim
            actual_outputs = self.execute_cppsim(hw_op, hw_model, inputs)
            backend_label = "HLS simulation (cppsim)"
            tolerance = self.get_golden_tolerance_cppsim()

        # Validate against golden
        self.validate_against_golden(
            actual_outputs, golden_outputs, backend_label, tolerance
        )


# ================================================================
# Helper Functions (Module-Level)
# ================================================================


def create_test_model_from_node(
    inputs: List[Tuple[str, int, List[int]]],
    outputs: List[Tuple[str, int, List[int]]],
    node_op_type: str,
    node_inputs: List[str],
    node_outputs: List[str],
    node_name: str = "",
    node_attributes: Optional[Dict[str, Any]] = None,
) -> ModelWrapper:
    """Helper to create ONNX model with single node.

    Args:
        inputs: List of (name, dtype_onnx, shape) for model inputs
        outputs: List of (name, dtype_onnx, shape) for model outputs
        node_op_type: ONNX op type (e.g., "Add", "Softmax")
        node_inputs: List of input names for the node
        node_outputs: List of output names for the node
        node_name: Optional node name
        node_attributes: Optional dict of node attributes

    Returns:
        ModelWrapper with single-node graph

    Example:
        model = create_test_model_from_node(
            inputs=[("input0", TensorProto.FLOAT, [1, 64]),
                    ("input1", TensorProto.FLOAT, [1, 64])],
            outputs=[("output", TensorProto.FLOAT, [1, 64])],
            node_op_type="Add",
            node_inputs=["input0", "input1"],
            node_outputs=["output"]
        )
    """
    import onnx.helper as helper
    from onnx import TensorProto

    # Create input/output ValueInfoProto
    graph_inputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, dtype, shape in inputs
    ]

    graph_outputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, dtype, shape in outputs
    ]

    # Create node
    if node_name:
        node = helper.make_node(
            node_op_type,
            node_inputs,
            node_outputs,
            name=node_name,
            **(node_attributes or {}),
        )
    else:
        node = helper.make_node(
            node_op_type, node_inputs, node_outputs, **(node_attributes or {})
        )

    # Create graph
    graph = helper.make_graph(
        [node], f"test_{node_op_type.lower()}", graph_inputs, graph_outputs
    )

    # Create model
    model = helper.make_model(graph)

    return ModelWrapper(model)
