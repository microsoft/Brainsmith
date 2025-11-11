############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for union type system (DimSpec, DatatypeSpec)."""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow.builder import BuildContext, DesignSpaceBuilder
from brainsmith.dataflow.schemas import InputSchema, KernelSchema, OutputSchema
from brainsmith.dataflow.spec_helpers import (
    derive_datatype,
    derive_dim,
)
from brainsmith.dataflow.types import (
    FULL_DIM,
    FULL_SHAPE,
    VALUE_OPTIMIZED,
)


# Mock ONNX graph for testing
class MockModelWrapper:
    """Mock ModelWrapper for testing without ONNX."""

    def __init__(self):
        self.tensors = {
            "input0": {"shape": (1, 768), "datatype": DataType["INT8"]},
            "input1": {"shape": (1, 64), "datatype": DataType["INT16"]},
            "output0": {"shape": (1, 768), "datatype": DataType["INT8"]},
            "weight": {"shape": (768,), "datatype": DataType["INT8"]},
        }
        self.initializers = {}

    def get_tensor_shape(self, name):
        return self.tensors[name]["shape"]

    def get_tensor_datatype(self, name):
        return self.tensors[name]["datatype"]

    def get_initializer(self, name):
        return self.initializers.get(name)


# =============================================================================
# DimSpec Union Type Tests
# =============================================================================


def test_dimspec_tuple_shorthand():
    """Test tuple shorthand (interface, dim_idx) for dimension derivation."""
    from brainsmith.dataflow.spec_helpers import derive_dim
    from brainsmith.dataflow.types import ShapeHierarchy

    # Custom tuple resolver that specifies BLOCK hierarchy (available during build)
    def block_dim_tuple(interface_name, dim_idx):
        """Helper to create tuple shorthand with BLOCK hierarchy."""
        return derive_dim(interface_name, ShapeHierarchy.BLOCK, dim_idx)

    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=[1, "SIMD"],
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=[FULL_DIM, block_dim_tuple("input0", -1)],  # Derive from input's last block dim
                stream_tiling=[1, "SIMD"],
            )
        ],
    )

    nodeattrs = {"SIMD": 64}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify block_tiling resolved correctly (derives from input's block shape)
    assert design_space.outputs["output0"].block_shape == (1, 768)


def test_dimspec_callable():
    """Test callable dimension function."""

    def custom_dim(interfaces, param_getter, model, tensor_name):
        """Custom: sum of last dimensions from all inputs (unified 4-param signature)."""
        input0_last = interfaces["input0"].tensor_shape[-1]
        input1_last = interfaces["input1"].tensor_shape[-1]
        return input0_last + input1_last  # 768 + 64 = 832

    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            ),
            InputSchema(
                name="input1",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            ),
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=[FULL_DIM, custom_dim],  # Custom dimension
                stream_tiling=None,
            )
        ],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0", "input1"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify callable dimension resolved correctly
    assert design_space.outputs["output0"].block_shape == (1, 832)


def test_dimspec_derive_dim_helper():
    """Test derive_dim() helper function."""
    from brainsmith.dataflow.types import ShapeHierarchy

    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=[1, "SIMD"],
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=[FULL_DIM, derive_dim("input0", ShapeHierarchy.BLOCK, -1)],  # Helper with BLOCK hierarchy
                stream_tiling=[1, "SIMD"],
            )
        ],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify derive_dim() resolved correctly
    assert design_space.outputs["output0"].block_shape == (1, 768)


# =============================================================================
# DatatypeSpec Union Type Tests
# =============================================================================


def test_datatypespec_string_shorthand():
    """Test string shorthand for datatype derivation."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
                datatype="input0",  # String shorthand: copy from input0
            )
        ],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify string shorthand resolved correctly
    assert design_space.outputs["output0"].datatype == DataType["INT8"]


def test_datatypespec_value_optimized():
    """Test VALUE_OPTIMIZED sentinel for value-based optimization."""
    mock_model = MockModelWrapper()
    # Add initializer with values in range [-10, 10]
    mock_model.initializers["weight"] = np.array([-10, -5, 0, 5, 10], dtype=np.int32)

    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            ),
            InputSchema(
                name="weight",
                block_tiling=[FULL_DIM],
                stream_tiling=None,
                datatype=VALUE_OPTIMIZED,  # Optimize from values
            ),
        ],
        outputs=[],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=mock_model,
        node_inputs=["input0", "weight"],
        node_outputs=[],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify VALUE_OPTIMIZED found smallest fitting type
    # Range [-10, 10] fits in INT5 (signed, -16 to 15)
    assert design_space.inputs["weight"].datatype == DataType["INT5"]


def test_datatypespec_derive_datatype_helper():
    """Test derive_datatype() helper function."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
                datatype=derive_datatype("input0"),  # Helper function
            )
        ],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify derive_datatype() resolved correctly
    assert design_space.outputs["output0"].datatype == DataType["INT8"]


def test_datatypespec_callable():
    """Test callable datatype function."""

    def custom_datatype(interfaces, param_getter, model, tensor_name):
        """Custom: use widest datatype from all inputs."""
        input0_dt = interfaces["input0"].datatype
        input1_dt = interfaces["input1"].datatype
        # Return the wider type (INT16 > INT8)
        return max([input0_dt, input1_dt], key=lambda dt: dt.bitwidth())

    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            ),
            InputSchema(
                name="input1",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            ),
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
                datatype=custom_datatype,  # Custom datatype function
            )
        ],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0", "input1"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify callable datatype resolved correctly (INT16 is wider than INT8)
    assert design_space.outputs["output0"].datatype == DataType["INT16"]


def test_datatypespec_fixed_datatype():
    """Test fixed BaseDataType specification."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
                datatype=DataType["INT32"],  # Fixed datatype
            )
        ],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify fixed datatype used
    assert design_space.outputs["output0"].datatype == DataType["INT32"]


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_dimspec_tuple_invalid_interface():
    """Test tuple shorthand with invalid interface name."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=[1, ("nonexistent", -1)],  # Invalid interface
            )
        ],
        outputs=[],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0"],
        node_outputs=[],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Should error during configure when tuple is resolved
    with pytest.raises(ValueError, match="not found"):
        design_space.configure({})


def test_datatypespec_string_invalid_interface():
    """Test string shorthand with invalid interface name."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=[FULL_DIM, FULL_DIM],
                stream_tiling=None,
                datatype="nonexistent",  # Invalid interface
            )
        ],
    )

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=MockModelWrapper(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()

    # Should error during build
    with pytest.raises(ValueError, match="not found"):
        builder.build(ctx)


# =============================================================================
# FULL_SHAPE Sentinel Tests
# =============================================================================


def test_full_shape_block_tiling_2d():
    """Test FULL_SHAPE in block_tiling with 2D tensors."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=FULL_SHAPE,  # Rank-agnostic
                stream_tiling=["SIMD"],
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=FULL_SHAPE,  # Rank-agnostic
                stream_tiling=["SIMD"],
            )
        ],
    )

    class Mock2DModel:
        def __init__(self):
            self.tensors = {
                "input0": {"shape": (128, 768), "datatype": DataType["INT8"]},
                "output0": {"shape": (128, 768), "datatype": DataType["INT8"]},
            }
            self.initializers = {}

        def get_tensor_shape(self, name):
            return self.tensors[name]["shape"]

        def get_tensor_datatype(self, name):
            return self.tensors[name]["datatype"]

        def get_initializer(self, name):
            return self.initializers.get(name)

    nodeattrs = {"SIMD": 64}
    ctx = BuildContext(
        schema=schema,
        model_w=Mock2DModel(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify FULL_SHAPE expanded to [FULL_DIM, FULL_DIM] for 2D tensor
    assert design_space.inputs["input0"].block_shape == (128, 768)
    assert design_space.outputs["output0"].block_shape == (128, 768)


def test_full_shape_block_tiling_4d():
    """Test FULL_SHAPE in block_tiling with 4D tensors."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=FULL_SHAPE,  # Rank-agnostic
                stream_tiling=["PE"],
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=FULL_SHAPE,  # Rank-agnostic
                stream_tiling=["PE"],
            )
        ],
    )

    class Mock4DModel:
        def __init__(self):
            self.tensors = {
                "input0": {"shape": (1, 224, 224, 64), "datatype": DataType["INT8"]},
                "output0": {"shape": (1, 224, 224, 64), "datatype": DataType["INT8"]},
            }
            self.initializers = {}

        def get_tensor_shape(self, name):
            return self.tensors[name]["shape"]

        def get_tensor_datatype(self, name):
            return self.tensors[name]["datatype"]

        def get_initializer(self, name):
            return self.initializers.get(name)

    nodeattrs = {"PE": 8}
    ctx = BuildContext(
        schema=schema,
        model_w=Mock4DModel(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify FULL_SHAPE expanded to [FULL_DIM] * 4 for 4D tensor
    assert design_space.inputs["input0"].block_shape == (1, 224, 224, 64)
    assert design_space.outputs["output0"].block_shape == (1, 224, 224, 64)


def test_full_shape_stream_tiling():
    """Test FULL_SHAPE in stream_tiling copies resolved block_shape."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=FULL_SHAPE,
                stream_tiling=FULL_SHAPE,  # Copies block_shape
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=FULL_SHAPE,
                stream_tiling=FULL_SHAPE,  # Copies block_shape
            )
        ],
    )

    class Mock3DModel:
        def __init__(self):
            self.tensors = {
                "input0": {"shape": (1, 128, 768), "datatype": DataType["INT8"]},
                "output0": {"shape": (1, 128, 768), "datatype": DataType["INT8"]},
            }
            self.initializers = {}

        def get_tensor_shape(self, name):
            return self.tensors[name]["shape"]

        def get_tensor_datatype(self, name):
            return self.tensors[name]["datatype"]

        def get_initializer(self, name):
            return self.initializers.get(name)

    nodeattrs = {}
    ctx = BuildContext(
        schema=schema,
        model_w=Mock3DModel(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Configure to resolve stream_tiling
    design_point = design_space.configure({})

    # Verify stream_tiling=FULL_SHAPE copied block_shape
    assert design_point.inputs["input0"].stream_shape == (1, 128, 768)
    assert design_point.outputs["output0"].stream_shape == (1, 128, 768)


def test_full_shape_with_tuple_shorthand():
    """Test FULL_SHAPE works with tuple shorthand in stream_tiling."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=FULL_SHAPE,  # Rank-agnostic
                stream_tiling=["PE"],
            )
        ],
        outputs=[
            OutputSchema(
                name="output0",
                block_tiling=FULL_SHAPE,  # Rank-agnostic
                stream_tiling=[("input0", -1)],  # Auto-pads to match rank
            )
        ],
    )

    class Mock4DModel:
        def __init__(self):
            self.tensors = {
                "input0": {"shape": (1, 224, 224, 64), "datatype": DataType["INT8"]},
                "output0": {"shape": (1, 224, 224, 64), "datatype": DataType["INT8"]},
            }
            self.initializers = {}

        def get_tensor_shape(self, name):
            return self.tensors[name]["shape"]

        def get_tensor_datatype(self, name):
            return self.tensors[name]["datatype"]

        def get_initializer(self, name):
            return self.initializers.get(name)

    nodeattrs = {"PE": 8}
    ctx = BuildContext(
        schema=schema,
        model_w=Mock4DModel(),
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Configure to resolve stream shapes
    design_point = design_space.configure({"PE": 8})

    # Verify:
    # - input0 has stream_tiling=["PE"] → auto-pads to [1, 1, 1, "PE"]
    # - output0 has stream_tiling=[("input0", -1)] → auto-pads to [1, 1, 1, ("input0", -1)]
    # - output0 copies input0's last stream dim (8)
    assert design_point.inputs["input0"].stream_shape == (1, 1, 1, 8)
    assert design_point.outputs["output0"].stream_shape == (1, 1, 1, 8)


def test_full_shape_addstreams_pattern():
    """Test FULL_SHAPE with AddStreams-like pattern (2 inputs, rank-agnostic)."""
    schema = KernelSchema(
        name="AddStreams",
        inputs=[
            InputSchema(
                name="input0",
                block_tiling=FULL_SHAPE,
                stream_tiling=["PE"],
            ),
            InputSchema(
                name="input1",
                block_tiling=FULL_SHAPE,
                stream_tiling=["PE"],
            ),
        ],
        outputs=[
            OutputSchema(
                name="output",
                block_tiling=FULL_SHAPE,
                stream_tiling=[("input0", -1)],  # Match input0 stream rate
            )
        ],
    )

    # Test with 2D tensors
    class Mock2DModel:
        def __init__(self):
            self.tensors = {
                "input0": {"shape": (128, 768), "datatype": DataType["INT8"]},
                "input1": {"shape": (128, 768), "datatype": DataType["INT8"]},
                "output": {"shape": (128, 768), "datatype": DataType["INT8"]},
            }
            self.initializers = {}

        def get_tensor_shape(self, name):
            return self.tensors[name]["shape"]

        def get_tensor_datatype(self, name):
            return self.tensors[name]["datatype"]

        def get_initializer(self, name):
            return self.initializers.get(name)

    nodeattrs = {"PE": 64}
    ctx = BuildContext(
        schema=schema,
        model_w=Mock2DModel(),
        node_inputs=["input0", "input1"],
        node_outputs=["output"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = DesignSpaceBuilder()
    design_space = builder.build(ctx)

    # Verify block shapes for 2D tensors
    assert design_space.inputs["input0"].block_shape == (128, 768)
    assert design_space.inputs["input1"].block_shape == (128, 768)
    assert design_space.outputs["output"].block_shape == (128, 768)

    # Configure and verify stream shapes
    design_point = design_space.configure({"PE": 64})
    assert design_point.inputs["input0"].stream_shape == (1, 64)
    assert design_point.inputs["input1"].stream_shape == (1, 64)
    assert design_point.outputs["output"].stream_shape == (1, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
