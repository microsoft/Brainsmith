# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for MVAU parity testing.

Compares legacy FINN MVAU implementation against modern Brainsmith MVAU.
"""

from typing import Dict, Tuple
import numpy as np
import pytest
from onnx import helper, TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.dual_kernel_test import DualKernelTest
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferQuantizedMatrixVectorActivation
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList


class MVAUParityBase(DualKernelTest):
    """Base class for MVAU parity testing.

    Subclasses override get_mvau_config() to test different modes and memory configurations.

    Inherits 20 tests from DualKernelTest:
    - 7 core parity tests (shapes, widths, datatypes)
    - 5 HW estimation tests (cycles, resources)
    - 8 golden execution tests (Python, cppsim, rtlsim)
    """

    # ========================================================================
    # Configuration Methods (override in subclasses)
    # ========================================================================

    def get_mvau_config(self) -> Dict:
        """Return MVAU configuration dict.

        Must be overridden by subclasses.

        Returns:
            {
                'mw': int,              # Matrix width (input dimension)
                'mh': int,              # Matrix height (output dimension)
                'pe': int,              # Processing elements (output parallelism)
                'simd': int,            # Input parallelism
                'idt': DataType,        # Input datatype
                'wdt': DataType,        # Weight datatype
                'odt': DataType,        # Output datatype
                'no_activation': int,   # 0=MVTU, 1=MV
                'mem_mode': str,        # Memory mode (internal_embedded, internal_decoupled, external)
                'act_val': int,         # Activation bits (MVTU only, 0 for MV)
            }
        """
        raise NotImplementedError("Subclasses must override get_mvau_config()")

    # ========================================================================
    # DualKernelTest Required Methods
    # ========================================================================

    def get_manual_transform(self):
        """FINN's MVAU inference transform."""
        return InferQuantizedMatrixVectorActivation

    def get_auto_transform(self):
        """Brainsmith's unified inference transform."""
        return InferKernelList

    def get_num_inputs(self) -> int:
        """Return 2 for MV, 3 for MVTU."""
        config = self.get_mvau_config()
        return 2 if config['no_activation'] == 1 else 3

    def get_num_outputs(self) -> int:
        """Always 1 output."""
        return 1

    def get_backend_fpgapart(self) -> str:
        """Enable HLS backend testing."""
        return "xc7z020clg400-1"

    def get_backend_type(self) -> str:
        """Use HLS backend."""
        return "hls"

    # ========================================================================
    # Model Creation
    # ========================================================================

    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create ONNX model with MatMul (+ MultiThreshold for MVTU).

        Returns:
            (model, node_name): ModelWrapper and target node name
        """
        config = self.get_mvau_config()
        mw = config['mw']
        mh = config['mh']
        idt = config['idt']
        wdt = config['wdt']
        odt = config['odt']
        no_activation = config['no_activation']

        # Create inputs
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])

        if no_activation == 1:
            # MV mode: MatMul only
            outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])

            matmul_node = helper.make_node(
                "MatMul",
                ["inp", "weights"],
                ["outp"],
                name="MatMul_0"
            )

            graph = helper.make_graph(
                nodes=[matmul_node],
                name="mvau_test_mv",
                inputs=[inp],
                outputs=[outp],
            )

            target_node = "MatMul_0"

        else:
            # MVTU mode: MatMul + MultiThreshold
            matmul_out = helper.make_tensor_value_info("matmul_out", TensorProto.FLOAT, [1, mh])
            outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])

            matmul_node = helper.make_node(
                "MatMul",
                ["inp", "weights"],
                ["matmul_out"],
                name="MatMul_0"
            )

            mt_node = helper.make_node(
                "MultiThreshold",
                ["matmul_out", "thresholds"],
                ["outp"],
                domain="qonnx.custom_op.general",
                out_dtype=odt.name,
                name="MultiThreshold_0"
            )

            graph = helper.make_graph(
                nodes=[matmul_node, mt_node],
                name="mvau_test_mvtu",
                inputs=[inp],
                outputs=[outp],
            )

            target_node = "MatMul_0"  # Transform targets MatMul

        model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-parity-test"))

        # Set input datatypes (let transforms infer output datatype)
        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("weights", wdt)
        # Note: NOT setting output datatype - let FINN and Brainsmith infer it
        # This allows proper accumulator minimization comparison

        # Generate and set weights
        W = self.generate_weights()
        model.set_initializer("weights", W)

        if no_activation == 0:
            # Set threshold datatypes and values
            tdt = DataType["INT32"]
            model.set_tensor_datatype("thresholds", tdt)
            # Note: NOT setting matmul_out datatype - let transforms infer accumulator type

            T = self.generate_thresholds()
            model.set_initializer("thresholds", T)

        return model, target_node

    def generate_weights(self) -> np.ndarray:
        """Generate random weight matrix."""
        config = self.get_mvau_config()
        mw = config['mw']
        mh = config['mh']
        wdt = config['wdt']

        np.random.seed(42)
        return self.gen_finn_dt_tensor(wdt, (mw, mh))

    def generate_thresholds(self) -> np.ndarray:
        """Generate threshold matrix (MVTU only)."""
        config = self.get_mvau_config()
        if config['no_activation'] == 1:
            return None

        mh = config['mh']
        act_val = config['act_val']
        n_steps = (1 << act_val) - 1

        # Generate sorted thresholds
        np.random.seed(43)
        T = np.random.randint(-100, 100, (mh, n_steps)).astype(np.float32)
        T = np.sort(T, axis=1)
        return T

    @staticmethod
    def gen_finn_dt_tensor(dtype: DataType, shape: Tuple) -> np.ndarray:
        """Generate random tensor within datatype range."""
        if dtype.min() < 0:
            # Signed
            return np.random.randint(
                dtype.min(), dtype.max() + 1, shape
            ).astype(np.float32)
        else:
            # Unsigned
            return np.random.randint(
                0, dtype.max() + 1, shape
            ).astype(np.float32)

    # ========================================================================
    # Golden Reference
    # ========================================================================

    def compute_golden_reference(self, inputs: Dict) -> Dict:
        """Compute golden reference: MatMul (+ multithreshold if MVTU)."""
        config = self.get_mvau_config()
        no_activation = config['no_activation']

        # Get weights
        W = self.generate_weights()

        # MatMul
        result = np.matmul(inputs['inp'], W)

        if no_activation == 0:
            # Apply multi-threshold
            from qonnx.custom_op.general.multithreshold import multithreshold
            T = self.generate_thresholds()
            result = multithreshold(result, T)

        return {"outp": result}

    # ========================================================================
    # Kernel Configuration
    # ========================================================================

    def configure_kernel_node(self, op, model):
        """Configure MVAU node with PE, SIMD, mem_mode, etc.

        This is called by PipelineRunner after kernel creation.
        """
        config = self.get_mvau_config()

        # Set folding (PE, SIMD)
        op.set_nodeattr("PE", config['pe'])
        op.set_nodeattr("SIMD", config['simd'])

        # Set memory mode
        op.set_nodeattr("mem_mode", config['mem_mode'])

        # Set activation parameters (MVTU only)
        if config['no_activation'] == 0:
            op.set_nodeattr("ActVal", config['act_val'])

    # ========================================================================
    # Test Overrides (FINN Compatibility Exceptions)
    # ========================================================================

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_folded_shapes_parity(self):
        """Test folded input/output shapes match between implementations.

        FINN Limitation: FINN's MVAU only supports folded weight shapes for:
        - mem_mode="external" (weights streamed from external memory)
        - dynamic_input=1 (runtime-variable weights)

        For internal_embedded and internal_decoupled modes, FINN raises:
        "Undefined input shape for requested input"

        Brainsmith's design_point system handles all modes generically:
        - Weights with stream_shape=(1,1) return valid folded shape (MW, MH, 1)
        - This is architecturally cleaner (no special cases)

        This override skips weight folded shape checks for non-external modes
        while still validating activations and outputs.
        """
        from tests.support.assertions import assert_shapes_match

        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()
        config = self.get_mvau_config()

        # Input 0 (activations): Always test
        manual_shape = manual_op.get_folded_input_shape(0)
        auto_shape = auto_op.get_folded_input_shape(0)
        assert_shapes_match(manual_shape, auto_shape, 0, "folded input")

        # Input 1 (weights): Only test for external mode
        # FINN raises exception for internal_embedded/internal_decoupled
        if config['mem_mode'] == 'external':
            manual_shape = manual_op.get_folded_input_shape(1)
            auto_shape = auto_op.get_folded_input_shape(1)
            assert_shapes_match(manual_shape, auto_shape, 1, "folded input")
        # else: Skip weight folded shape for non-external modes (FINN limitation)

        # Input 2 (thresholds): Only present in MVTU mode
        if config['no_activation'] == 0:
            manual_shape = manual_op.get_folded_input_shape(2)
            auto_shape = auto_op.get_folded_input_shape(2)
            assert_shapes_match(manual_shape, auto_shape, 2, "folded input")

        # Output shapes: Always test
        for i in range(self.get_num_outputs()):
            manual_shape = manual_op.get_folded_output_shape(i)
            auto_shape = auto_op.get_folded_output_shape(i)
            assert_shapes_match(manual_shape, auto_shape, i, "folded output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_stream_widths_parity(self):
        """Test input/output stream widths match between implementations.

        FINN Limitation: FINN returns 0 for weight stream width in internal_embedded
        and internal_decoupled modes (weights not streamed).

        Brainsmith's design_point system returns actual width (datatype.bitwidth())
        even for static weights with stream_shape=(1,1).

        Skip weight stream width check for non-external modes.
        """
        from tests.support.assertions import assert_widths_match

        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()
        config = self.get_mvau_config()

        # Input 0 (activations): Always test
        manual_width = manual_op.get_instream_width(0)
        auto_width = auto_op.get_instream_width(0)
        assert_widths_match(manual_width, auto_width, 0, "Input")

        # Input 1 (weights): Only test for external mode
        if config['mem_mode'] == 'external':
            manual_width = manual_op.get_instream_width(1)
            auto_width = auto_op.get_instream_width(1)
            assert_widths_match(manual_width, auto_width, 1, "Input")
        # else: Skip weight stream width for non-external modes (FINN returns 0)

        # Input 2 (thresholds): Only present in MVTU mode
        if config['no_activation'] == 0:
            manual_width = manual_op.get_instream_width(2)
            auto_width = auto_op.get_instream_width(2)
            assert_widths_match(manual_width, auto_width, 2, "Input")

        # Output stream widths: Always test
        for i in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width(i)
            auto_width = auto_op.get_outstream_width(i)
            assert_widths_match(manual_width, auto_width, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.dual_kernel
    def test_stream_widths_padded_parity(self):
        """Test padded stream widths match between implementations.

        Same FINN limitation as test_stream_widths_parity.
        """
        from tests.support.assertions import assert_widths_match

        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()
        config = self.get_mvau_config()

        # Input 0 (activations): Always test
        manual_width = manual_op.get_instream_width_padded(0)
        auto_width = auto_op.get_instream_width_padded(0)
        assert_widths_match(manual_width, auto_width, 0, "Input (padded)")

        # Input 1 (weights): Only test for external mode
        if config['mem_mode'] == 'external':
            manual_width = manual_op.get_instream_width_padded(1)
            auto_width = auto_op.get_instream_width_padded(1)
            assert_widths_match(manual_width, auto_width, 1, "Input (padded)")
        # else: Skip weight stream width for non-external modes

        # Input 2 (thresholds): Only present in MVTU mode
        if config['no_activation'] == 0:
            manual_width = manual_op.get_instream_width_padded(2)
            auto_width = auto_op.get_instream_width_padded(2)
            assert_widths_match(manual_width, auto_width, 2, "Input (padded)")

        # Output stream widths: Always test
        for i in range(self.get_num_outputs()):
            manual_width = manual_op.get_outstream_width_padded(i)
            auto_width = auto_op.get_outstream_width_padded(i)
            assert_widths_match(manual_width, auto_width, i, "Output (padded)")
