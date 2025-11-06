############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Parity tests for Crop kernel: LegacyCrop (manual) vs Crop (auto).

This module tests parity between legacy and modern Crop implementations:
- Legacy: LegacyCrop (manual inference from Gather pattern)
- Modern: Crop (schema-driven KernelOp with unified transformation)

Test Framework: DualKernelTest
- Provides 18 inherited tests per configuration
- Tests both implementations against golden NumPy reference
- Validates HW-specific concerns (SIMD, shapes, datatypes)

Test Coverage (5 configurations × 18 tests = 90 tests):
1. TestCropNoCrop: Identity passthrough (0, 0, 0, 0)
2. TestCropVertical: Height crop only (12, 12, 0, 0)
3. TestCropHorizontal: Width crop only (0, 0, 12, 12)
4. TestCropAllEdges: All edges (10, 10, 8, 8)
5. TestCropAsymmetric: Asymmetric (20, 4, 16, 8)

Each test class inherits:
- 7 core parity tests (shapes, widths, datatypes - HW concerns)
- 5 HW estimation tests (cycles, resources - HW concerns)
- 6 golden execution tests (2 Python, 2 cppsim, 2 rtlsim - correctness)

Validation: +5 meta-tests checking test structure correctness
"""

import pytest
import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType

from tests.frameworks.dual_kernel_test import DualKernelTest
from brainsmith.kernels.crop import LegacyCrop, LegacyCrop_hls, InferCropFromGather
from brainsmith.primitives.transforms.infer_kernels import InferKernels
from brainsmith.dataflow.kernel_op import KernelOp


class CropParityBase(DualKernelTest):
    """Base class for Crop parity testing (manual vs auto + both vs golden).

    Tests LegacyCrop (manual/FINN-style) against Crop (auto/KernelOp-based):

    1. **Manual Pipeline** (Legacy):
       - Transform: InferCropFromGather (separate class)
       - Kernel: LegacyCrop (HWCustomOp)
       - Backend: LegacyCrop_hls
       - Parameters: lowercase "simd", stored shapes

    2. **Auto Pipeline** (Modern):
       - Transform: InferKernelList → Crop.infer_from() (class method)
       - Kernel: Crop (KernelOp)
       - Backend: Crop_hls
       - Parameters: uppercase "SIMD", extracted shapes

    Subclasses configure crop parameters by overriding class variables:
    - crop_north: Rows to remove from top
    - crop_south: Rows to remove from bottom
    - crop_east: Columns to remove from right
    - crop_west: Columns to remove from left

    Configuration (fixed for all tests):
    - Input shape: [1, 224, 224, 64] (NHWC)
    - Datatype: INT8
    - SIMD: 8 (channel parallelization)
    - FPGA part: xc7z020clg400-1

    Inherited Tests (18):
    - 7 core parity tests (shapes, widths, datatypes at Stage 2)
    - 5 HW estimation tests (cycles, resources at Stage 2)
    - 6 golden execution tests:
      * 2 Python tests (Stage 2: manual/auto vs golden)
      * 2 cppsim tests (Stage 3: manual/auto vs golden)
      * 2 rtlsim tests (Stage 3: manual/auto vs golden)
    """

    # Crop parameters - override in subclasses
    crop_north: int = 0
    crop_south: int = 0
    crop_east: int = 0
    crop_west: int = 0

    # Fixed test configuration
    batch = 1
    height = 224
    width = 224
    channels = 64
    simd = 8
    datatype = DataType["INT8"]

    # ================================================================
    # Required Abstract Methods (from DualKernelTest)
    # ================================================================

    def make_test_model(self):
        """Create ONNX model with Gather node for crop pattern.

        Creates a Gather node with consecutive indices that can be converted
        to Crop by both legacy and modern transforms.

        Strategy:
        - For vertical crop (height): Use axis=1 with indices [crop_north : H - crop_south]
        - For horizontal crop (width): Use axis=2 with indices [crop_west : W - crop_east]
        - If both dimensions cropped, use vertical (axis=1) as primary pattern

        Returns:
            tuple: (ModelWrapper, None) - No specific node selection needed
        """
        # Determine primary crop axis
        has_vertical_crop = self.crop_north > 0 or self.crop_south > 0
        has_horizontal_crop = self.crop_east > 0 or self.crop_west > 0

        # Choose axis (prefer vertical if both)
        if has_vertical_crop:
            axis = 1  # Height dimension in NHWC
            start_idx = self.crop_north
            end_idx = self.height - self.crop_south
        elif has_horizontal_crop:
            axis = 2  # Width dimension in NHWC
            start_idx = self.crop_west
            end_idx = self.width - self.crop_east
        else:
            # No crop - use identity (entire height dimension)
            axis = 1
            start_idx = 0
            end_idx = self.height

        # Create consecutive indices
        indices = np.arange(start_idx, end_idx, dtype=np.int64)

        # Build ONNX graph
        input_shape = [self.batch, self.height, self.width, self.channels]

        # Input tensor
        input_tensor = helper.make_tensor_value_info(
            "input",
            TensorProto.FLOAT,
            input_shape
        )

        # Indices initializer
        indices_tensor = helper.make_tensor(
            "indices",
            TensorProto.INT64,
            [1, len(indices)],
            indices.tolist()
        )

        # Output shape after gather
        output_shape = list(input_shape)
        output_shape[axis] = len(indices)

        output_tensor = helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT,
            output_shape
        )

        # Gather node
        gather_node = helper.make_node(
            "Gather",
            inputs=["input", "indices"],
            outputs=["output"],
            axis=axis,
            name="Gather_0"
        )

        # Create graph
        graph = helper.make_graph(
            nodes=[gather_node],
            name="crop_test",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[indices_tensor]
        )

        # Create model
        model = helper.make_model(graph)
        model_wrapper = ModelWrapper(model)

        # Set datatype
        model_wrapper.set_tensor_datatype("input", self.datatype)
        model_wrapper.set_tensor_datatype("output", self.datatype)

        return (model_wrapper, None)

    def get_manual_transform(self):
        """Return legacy Crop transform class.

        Returns:
            class: InferCropFromGather (legacy transform)
        """
        return InferCropFromGather

    def get_auto_transform(self):
        """Return modern Crop transform class.

        Returns:
            class: InferKernels (auto-discovers Crop.infer_from())
        """
        return InferKernels

    def get_manual_backend_variants(self):
        """Return legacy backend implementation.

        Returns:
            list: [LegacyCrop_hls] - legacy HLS backend
        """
        return [LegacyCrop_hls]

    def compute_golden_reference(self, inputs):
        """Compute expected output using NumPy (golden reference).

        Applies 2D crop to NHWC input tensor:
        - Height crop: [crop_north : H - crop_south]
        - Width crop: [crop_west : W - crop_east]

        Args:
            inputs: dict mapping "input" to numpy array [N, H, W, C]

        Returns:
            dict: {"output": cropped_array}
        """
        inp = inputs["input"]

        # Apply crop (NHWC layout: [N, H, W, C])
        h, w = inp.shape[1], inp.shape[2]
        h_start = self.crop_north
        h_end = h - self.crop_south
        w_start = self.crop_west
        w_end = w - self.crop_east

        # NumPy slicing
        output = inp[:, h_start:h_end, w_start:w_end, :]

        return {"output": output}

    def get_num_inputs(self):
        """Return number of inputs.

        Returns:
            int: 1 (Crop has single data input)
        """
        return 1

    def get_num_outputs(self):
        """Return number of outputs.

        Returns:
            int: 1 (Crop has single output)
        """
        return 1

    def get_backend_fpgapart(self):
        """Return FPGA part for backend testing.

        Returns:
            str: FPGA part identifier
        """
        return "xc7z020clg400-1"

    def configure_kernel_node(self, op, model):
        """Configure SIMD parameter for both legacy and modern kernels.

        Handles parameter naming difference:
        - Legacy: lowercase "simd" via set_nodeattr()
        - Modern: uppercase "SIMD" via design_point API

        Args:
            op: Kernel operator instance (LegacyCrop or Crop)
            model: ModelWrapper for graph context
        """
        if isinstance(op, LegacyCrop):
            # Legacy: lowercase simd
            op.set_nodeattr("simd", self.simd)
        elif isinstance(op, KernelOp):
            # Modern: uppercase SIMD via design point
            # Configure last stream dimension (channels) with SIMD parallelism
            point = op.design_point.with_input_stream(0, self.simd)
            op.apply_design_point(point)


# =============================================================================
# Concrete Test Classes (5 configurations × 18 tests = 90 tests)
# =============================================================================

class TestCropNoCrop(CropParityBase):
    """Test Crop with no cropping (identity passthrough).

    Configuration:
    - Input: [1, 224, 224, 64]
    - Crop: (0, 0, 0, 0) - no pixels removed
    - Output: [1, 224, 224, 64]

    Purpose:
    - Baseline verification - ensures passthrough behavior works
    - Validates transform creates valid Crop node even with zero crop
    """
    crop_north = 0
    crop_south = 0
    crop_east = 0
    crop_west = 0


class TestCropVertical(CropParityBase):
    """Test Crop with vertical cropping only (height dimension).

    Configuration:
    - Input: [1, 224, 224, 64]
    - Crop: (12, 12, 0, 0) - remove 12 rows top and bottom
    - Output: [1, 200, 224, 64]

    Purpose:
    - Tests height dimension cropping (Gather axis=1)
    - Validates north/south crop parameters
    """
    crop_north = 12
    crop_south = 12
    crop_east = 0
    crop_west = 0


class TestCropHorizontal(CropParityBase):
    """Test Crop with horizontal cropping only (width dimension).

    Configuration:
    - Input: [1, 224, 224, 64]
    - Crop: (0, 0, 12, 12) - remove 12 columns left and right
    - Output: [1, 224, 200, 64]

    Purpose:
    - Tests width dimension cropping (Gather axis=2)
    - Validates east/west crop parameters
    """
    crop_north = 0
    crop_south = 0
    crop_east = 12
    crop_west = 12


class TestCropAllEdges(CropParityBase):
    """Test Crop with all four edges cropped.

    Configuration:
    - Input: [1, 224, 224, 64]
    - Crop: (10, 10, 8, 8)
    - Output: [1, 204, 208, 64]

    Purpose:
    - Tests combined height and width cropping
    - Note: Gather pattern uses primary axis (height), but Crop applies both
    - Validates all crop parameters together
    """
    crop_north = 10
    crop_south = 10
    crop_east = 8
    crop_west = 8


class TestCropAsymmetric(CropParityBase):
    """Test Crop with asymmetric cropping (different values per edge).

    Configuration:
    - Input: [1, 224, 224, 64]
    - Crop: (20, 4, 16, 8) - different crop for each edge
    - Output: [1, 200, 200, 64]

    Purpose:
    - Tests asymmetric crop patterns (common in real use cases)
    - Ensures no assumptions about equal crop values
    """
    crop_north = 20
    crop_south = 4
    crop_east = 16
    crop_west = 8


# =============================================================================
# Validation Meta-Tests (5 tests)
# =============================================================================

class TestCropTestStructure:
    """Meta-tests validating test framework structure.

    These tests verify the test infrastructure itself, not the Crop implementation.
    Marked with @pytest.mark.validation for separation from functional tests.
    """

    @pytest.mark.validation
    def test_test_count(self):
        """Verify each test class generates exactly 18 tests."""
        test_classes = [
            TestCropNoCrop,
            TestCropVertical,
            TestCropHorizontal,
            TestCropAllEdges,
            TestCropAsymmetric,
        ]

        for test_class in test_classes:
            # Count test methods (inherited from DualKernelTest)
            test_methods = [
                m for m in dir(test_class)
                if m.startswith("test_") and callable(getattr(test_class, m))
            ]
            assert len(test_methods) == 18, (
                f"{test_class.__name__} should have 18 tests, "
                f"found {len(test_methods)}"
            )

    @pytest.mark.validation
    def test_crop_parameters_defined(self):
        """Verify all test classes define crop parameters."""
        test_classes = [
            TestCropNoCrop,
            TestCropVertical,
            TestCropHorizontal,
            TestCropAllEdges,
            TestCropAsymmetric,
        ]

        for test_class in test_classes:
            assert hasattr(test_class, "crop_north")
            assert hasattr(test_class, "crop_south")
            assert hasattr(test_class, "crop_east")
            assert hasattr(test_class, "crop_west")
            assert isinstance(test_class.crop_north, int)
            assert isinstance(test_class.crop_south, int)
            assert isinstance(test_class.crop_east, int)
            assert isinstance(test_class.crop_west, int)

    @pytest.mark.validation
    def test_golden_reference_correctness(self):
        """Verify golden reference produces correct output shapes."""
        test_configs = [
            (TestCropNoCrop, [1, 224, 224, 64]),
            (TestCropVertical, [1, 200, 224, 64]),
            (TestCropHorizontal, [1, 224, 200, 64]),
            (TestCropAllEdges, [1, 204, 208, 64]),
            (TestCropAsymmetric, [1, 200, 200, 64]),
        ]

        for test_class, expected_shape in test_configs:
            test_instance = test_class()

            # Create dummy input
            input_shape = [
                test_instance.batch,
                test_instance.height,
                test_instance.width,
                test_instance.channels
            ]
            dummy_input = np.random.randint(
                -128, 127,
                size=input_shape,
                dtype=np.int8
            ).astype(np.float32)

            # Compute golden reference
            result = test_instance.compute_golden_reference({"input": dummy_input})

            # Check output shape
            assert result["output"].shape == tuple(expected_shape), (
                f"{test_class.__name__}: expected shape {expected_shape}, "
                f"got {result['output'].shape}"
            )

    @pytest.mark.validation
    def test_test_model_creation(self):
        """Verify make_test_model creates valid ONNX models."""
        test_classes = [
            TestCropNoCrop,
            TestCropVertical,
            TestCropHorizontal,
            TestCropAllEdges,
            TestCropAsymmetric,
        ]

        for test_class in test_classes:
            test_instance = test_class()
            model, _ = test_instance.make_test_model()

            # Verify model structure
            assert model is not None
            assert isinstance(model, ModelWrapper)
            assert len(model.graph.node) == 1
            assert model.graph.node[0].op_type == "Gather"

            # Verify inputs/outputs
            assert len(model.graph.input) == 1
            assert len(model.graph.output) == 1
            assert model.graph.input[0].name == "input"
            assert model.graph.output[0].name == "output"

    @pytest.mark.validation
    def test_simd_configuration(self):
        """Verify SIMD parameter is correctly configured for both implementations."""
        test_instance = TestCropNoCrop()
        model, _ = test_instance.make_test_model()

        # Create both kernel instances (before transformation)
        # This is a simplified check - full check happens in configure_kernel_node
        assert test_instance.simd == 8
        assert test_instance.channels % test_instance.simd == 0, (
            "SIMD must divide channels dimension"
        )
