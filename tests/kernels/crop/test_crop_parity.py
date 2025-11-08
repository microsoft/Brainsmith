"""Test parity between LegacyCrop and modern Crop implementations.

This test demonstrates the KernelParityTest framework by comparing two implementations:
- Primary (Modern): InferKernels → Crop (KernelOp) → Crop_hls
- Reference (Legacy): InferCropFromGather → LegacyCrop (HWCustomOp) → LegacyCrop_hls

Provides 18 inherited tests:
- 6 golden execution tests (primary/reference × python/cppsim/rtlsim)
- 7 core parity tests (shapes, widths, datatypes)
- 5 HW estimation tests (cycles, resources, efficiency)

Key Differences Being Tested:
- Legacy uses lowercase 'simd', modern uses uppercase 'SIMD'
- Legacy stores input_shape/output_shape, modern derives from ModelWrapper
- Legacy requires height/width/data_type attrs, modern uses schema

Test Coverage:
- 8 test configurations × 18 tests = 144 total test cases
- Height axis cropping (4 configs): INT8, INT16, INT4, large crops
- Width axis cropping (4 configs): INT8, INT16, INT4, large crops
"""

import pytest
import numpy as np
import onnx.helper as helper
from onnx import TensorProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.kernel_parity_test import KernelParityTest
from tests.frameworks.test_config import (
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
)


class TestCropParity(KernelParityTest):
    """Test parity between modern Crop (primary) and LegacyCrop (reference)."""

    # ========================================================================
    # Test Configuration Fixture - 8 Configurations
    # ========================================================================

    # ========================================================================
    # Test Configurations Mapping (internal)
    # ========================================================================

    # Maps test_id to (axis, crop_start, crop_end) for generating Gather indices
    CROP_CONFIGS = {
        # Height axis (axis=1)
        "crop_height_int8_symmetric": (1, 4, 4),  # axis, crop_north, crop_south
        "crop_height_int16_asymmetric": (1, 2, 6),
        "crop_height_int4_minimal": (1, 1, 1),
        "crop_height_int8_large": (1, 12, 12),
        # Width axis (axis=2)
        "crop_width_int8_symmetric": (2, 4, 4),  # axis, crop_west, crop_east
        "crop_width_int16_asymmetric": (2, 3, 5),
        "crop_width_int4_minimal": (2, 1, 1),
        "crop_width_int8_large": (2, 8, 8),
    }

    @pytest.fixture(
        params=[
            # ============================================================
            # Height Axis Cropping (axis=1) - 4 configurations
            # ============================================================

            KernelTestConfig(
                test_id="crop_height_int8_symmetric",
                model=ModelStructure(
                    operation="Gather",
                    input_shapes={"input": (1, 28, 28, 64)},  # NHWC
                    input_dtypes={"input": DataType["INT8"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),

            KernelTestConfig(
                test_id="crop_height_int16_asymmetric",
                model=ModelStructure(
                    operation="Gather",
                    input_shapes={"input": (1, 32, 32, 128)},  # NHWC
                    input_dtypes={"input": DataType["INT16"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),

            KernelTestConfig(
                test_id="crop_height_int4_minimal",
                model=ModelStructure(
                    operation="Gather",
                    input_shapes={"input": (1, 16, 16, 32)},  # NHWC
                    input_dtypes={"input": DataType["INT4"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),

            KernelTestConfig(
                test_id="crop_height_int8_large",
                model=ModelStructure(
                    operation="Gather",
                    input_shapes={"input": (1, 56, 56, 256)},  # NHWC
                    input_dtypes={"input": DataType["INT8"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),

            # ============================================================
            # Width Axis Cropping (axis=2) - 4 configurations
            # ============================================================

            KernelTestConfig(
                test_id="crop_width_int8_symmetric",
                model=ModelStructure(
                    operation="Gather",
                    input_shapes={"input": (1, 28, 28, 64)},  # NHWC
                    input_dtypes={"input": DataType["INT8"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),

            KernelTestConfig(
                test_id="crop_width_int16_asymmetric",
                model=ModelStructure(
                    operation="Gather",
                    input_shapes={"input": (1, 32, 32, 128)},  # NHWC
                    input_dtypes={"input": DataType["INT16"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),

            KernelTestConfig(
                test_id="crop_width_int4_minimal",
                model=ModelStructure(
                    operation="Gather",
                    input_shapes={"input": (1, 16, 16, 32)},  # NHWC
                    input_dtypes={"input": DataType["INT4"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),

            KernelTestConfig(
                test_id="crop_width_int8_large",
                model=ModelStructure(
                    operation="Gather",
                    input_shapes={"input": (1, 64, 64, 256)},  # NHWC
                    input_dtypes={"input": DataType["INT8"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
        ]
    )
    def kernel_test_config(self, request):
        """Provide test configurations via fixture.

        Test execution controlled by pytest marks:
        - Python only: pytest test_crop_parity.py -m "not slow" -v
        - Skip backend: pytest test_crop_parity.py -m "not cppsim and not rtlsim" -v
        - Only cppsim: pytest test_crop_parity.py -m "cppsim" -v
        - Only rtlsim: pytest test_crop_parity.py -m "rtlsim" -v
        - All tests: pytest test_crop_parity.py -v
        """
        return request.param

    # ========================================================================
    # Shared Model Creation
    # ========================================================================

    def make_test_model(self, kernel_test_config):
        """Create ONNX model with Gather node for crop conversion.

        The Gather node will be transformed into a Crop operation by both
        implementations. The indices must be consecutive for valid crop conversion.

        Args:
            kernel_test_config: Configuration containing input shapes and test_id

        Returns:
            Tuple of (model, input_names) - input names for DataType annotation
        """
        # Get input shape (NHWC format expected)
        input_shape = list(kernel_test_config.input_shapes["input"])

        # Get crop configuration from test_id
        test_id = kernel_test_config.test_id
        if test_id not in self.CROP_CONFIGS:
            raise ValueError(f"Unknown test_id '{test_id}' - not in CROP_CONFIGS")

        axis, crop_start, crop_end = self.CROP_CONFIGS[test_id]

        # Get dimension size for the specified axis
        dim_size = input_shape[axis]

        # Create consecutive indices for valid crop conversion
        # Example: crop_start=4, crop_end=4, dim_size=28 → indices=[4, 5, ..., 23]
        output_dim_size = dim_size - crop_start - crop_end
        indices = np.arange(crop_start, crop_start + output_dim_size, dtype=np.int64)

        # Calculate output shape
        output_shape = input_shape.copy()
        output_shape[axis] = output_dim_size

        # Create tensor infos with FLOAT container (QONNX convention)
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Create indices as initializer (constant tensor)
        indices_init = helper.make_tensor(
            name="indices",
            data_type=TensorProto.INT64,
            dims=[len(indices)],
            vals=indices.tolist(),
        )

        # Create Gather ONNX node
        node = helper.make_node(
            "Gather",
            ["input", "indices"],
            ["output"],
            name="Gather_0",
            axis=axis,
        )

        # Create graph and model
        graph = helper.make_graph(
            [node],
            "test_crop_parity",
            [inp],  # Only data input (indices is initializer)
            [out],
            initializer=[indices_init],
        )

        model = ModelWrapper(qonnx_make_model(graph))

        # Return model and input names for DataType annotation
        # Note: Only "input" needs annotation, indices is a constant
        return model, ["input"]

    # ========================================================================
    # Reference Implementation: LegacyCrop (HWCustomOp style)
    # ========================================================================

    def infer_kernel_reference(self, model, target_node):
        """Infer reference kernel using InferCropFromGather transform.

        This creates a LegacyCrop (HWCustomOp) node with legacy attributes:
        - Lowercase 'simd' parameter
        - Explicit input_shape/output_shape storage
        - height/width/data_type attributes

        Args:
            model: Stage 1 ONNX model with Gather node
            target_node: Name of the original Gather node (before transformation)

        Returns:
            Tuple of (op, model) where op is LegacyCrop instance
        """
        from brainsmith.kernels.crop import InferCropFromGather, LegacyCrop

        # Apply InferCropFromGather transform with SIMD=1 (default)
        # This transform creates a "Crop" node with domain="brainsmith.kernels"
        model = model.transform(InferCropFromGather(simd=1))

        # Find the Crop node by op_type
        # Note: InferCropFromGather doesn't preserve node name, search by op_type
        nodes_by_op_type = model.get_nodes_by_op_type("Crop")
        assert len(nodes_by_op_type) == 1, (
            f"Expected exactly 1 Crop node, found {len(nodes_by_op_type)}"
        )

        # Get the ONNX node and wrap with LegacyCrop explicitly
        # NOTE: We must instantiate LegacyCrop directly instead of using getCustomOp()
        # because getCustomOp() would return the modern Crop (KernelOp) which has
        # higher priority in the registry
        onnx_node = nodes_by_op_type[0]
        op = LegacyCrop(onnx_node)

        return op, model

    def get_backend_variants_reference(self):
        """Return legacy backend variants for LegacyCrop.

        Returns:
            List of backend classes [LegacyCrop_hls]
        """
        from brainsmith.kernels.crop import LegacyCrop_hls
        return [LegacyCrop_hls]

    def configure_kernel_reference(self, op, model, stage, config):
        """Configure reference kernel with SIMD normalization.

        Legacy uses lowercase 'simd', but we want to align with modern uppercase 'SIMD'.
        This method handles the normalization by setting simd=1 (default).

        Args:
            op: LegacyCrop operator
            model: Model wrapper
            stage: Pipeline stage (2=kernel, 3=backend)
            config: Test configuration
        """
        # For now, use default SIMD=1 for both implementations
        # Legacy stores as 'simd' (lowercase), which is already set by InferCropFromGather

        # Call parent auto_configure to apply any additional config from fixture
        self.auto_configure_from_fixture(op, model, stage, config)

    # ========================================================================
    # Primary Implementation: Crop (KernelOp style) - uses defaults
    # ========================================================================

    def get_kernel_op(self):
        """Return modern Crop KernelOp for primary implementation.

        Returns:
            Crop class (KernelOp-based)
        """
        from brainsmith.kernels.crop import Crop
        return Crop

    def configure_kernel(self, op, model, stage, config):
        """Configure primary kernel with SIMD normalization.

        Modern uses uppercase 'SIMD' via schema. Ensure consistency with legacy.

        Args:
            op: Crop operator
            model: Model wrapper
            stage: Pipeline stage (2=kernel, 3=backend)
            config: Test configuration
        """
        # For now, use default SIMD=1 for both implementations
        # Modern will have 'SIMD' (uppercase) via schema

        # Call parent auto_configure to apply any additional config from fixture
        self.auto_configure_from_fixture(op, model, stage, config)

    def infer_kernel(self, model, target_node):
        """Infer primary kernel using modern Crop.

        The Crop kernel's infer_from() changes the node name from "Gather_0"
        to "Crop_Gather_0", so we search by op_type instead of node name.

        Args:
            model: Stage 1 ONNX model with Gather node
            target_node: Name of the original Gather node (before transformation)

        Returns:
            Tuple of (op, model) where op is Crop instance
        """
        from brainsmith.primitives.transforms import InferKernels
        from qonnx.custom_op.registry import getCustomOp

        # Apply InferKernels transform with modern Crop
        kernel_op = self.get_kernel_op()
        model = model.transform(InferKernels([kernel_op]))

        # Find the Crop node by op_type (node name changed during transform)
        nodes_by_op_type = model.get_nodes_by_op_type("Crop")
        assert len(nodes_by_op_type) == 1, (
            f"Expected exactly 1 Crop node, found {len(nodes_by_op_type)}"
        )

        # Get the ONNX node and wrap with custom op
        onnx_node = nodes_by_op_type[0]
        op = getCustomOp(onnx_node, model)  # Pass model for KernelOp initialization

        return op, model

    # Primary implementation uses inherited defaults from KernelTestBase:
    # - get_backend_variants() (auto-detect from registry)

    # ========================================================================
    # Test Structure Information
    # ========================================================================

    def get_num_inputs(self):
        """Crop has 1 input (data tensor only, indices consumed during transform)."""
        return 1

    def get_num_outputs(self):
        """Crop has 1 output."""
        return 1

    # ========================================================================
    # Golden Reference
    # ========================================================================
    # No compute_golden_reference() override needed!
    # QONNX executes the Stage 1 Gather node automatically to produce golden reference.
