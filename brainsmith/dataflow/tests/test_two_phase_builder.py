############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for two-phase builder (build_invariant() method)."""

import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow.builder import KernelModelBuilder, BuildContext
from brainsmith.dataflow.schemas import KernelSchema, InputSchema, OutputSchema
from brainsmith.dataflow.models import InvariantKernelModel
from brainsmith.dataflow.validation import InvariantValidationContext
from brainsmith.dataflow.constraints import DatatypeInteger, ShapesEqual
from brainsmith.dataflow.types import ShapeHierarchy, FULL_DIM

# Mock ONNX graph for testing
class MockModelWrapper:
    """Mock ModelWrapper for testing without ONNX."""

    def __init__(self):
        self.tensors = {
            "input0": {"shape": (768,), "datatype": DataType["INT8"]},
            "output0": {"shape": (768,), "datatype": DataType["INT8"]},
        }
        self.initializers = {}

    def get_tensor_shape(self, name):
        return self.tensors[name]["shape"]

    def get_tensor_datatype(self, name):
        return self.tensors[name]["datatype"]

    def get_initializer(self, name):
        return self.initializers.get(name)


# =============================================================================
# Test build_invariant() Method
# =============================================================================

def test_build_invariant_basic():
    """Test build_invariant() creates InvariantKernelModel."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[InputSchema(
            name="input0",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
        )],
        outputs=[OutputSchema(
            name="output0",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
        )],
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

    builder = KernelModelBuilder()
    invariant = builder.build_invariant(ctx)

    # Verify type
    assert isinstance(invariant, InvariantKernelModel)

    # Verify structure
    assert invariant.name == "TestKernel"
    assert len(invariant.inputs) == 1
    assert len(invariant.outputs) == 1

    # Verify input
    assert invariant.inputs[0].name == "input0"
    assert invariant.inputs[0].tensor_shape == (768,)
    assert invariant.inputs[0].block_shape == (768,)
    assert invariant.inputs[0].stream_tiling == ["SIMD"]  # Preserved, not resolved
    assert invariant.inputs[0].datatype == DataType["INT8"]

    # Verify output
    assert invariant.outputs[0].name == "output0"
    assert invariant.outputs[0].stream_tiling == ["SIMD"]  # Preserved, not resolved

    # Verify valid ranges computed
    assert "SIMD" in invariant.parallelization_params
    assert 64 in invariant.parallelization_params["SIMD"]
    assert len(invariant.parallelization_params["SIMD"]) == 18  # divisors of 768


def test_build_invariant_multi_parameter():
    """Test build_invariant() with multiple parallelization parameters."""
    mock_model = MockModelWrapper()
    mock_model.tensors["input0"] = {"shape": (768, 64), "datatype": DataType["INT8"]}
    mock_model.tensors["output0"] = {"shape": (768, 64), "datatype": DataType["INT8"]}

    schema = KernelSchema(
        name="TestKernel",
        inputs=[InputSchema(
            name="input0",
            block_tiling=[FULL_DIM, FULL_DIM],
            stream_tiling=["MW", "MH"],
        )],
        outputs=[OutputSchema(
            name="output0",
            block_tiling=[FULL_DIM, FULL_DIM],
            stream_tiling=["MW", "MH"],
        )],
    )

    nodeattrs = {}

    ctx = BuildContext(
        schema=schema,
        model_w=mock_model,
        node_inputs=["input0"],
        node_outputs=["output0"],
        param_getter=lambda name: nodeattrs.get(name, 1),
        param_setter=lambda name, value: nodeattrs.update({name: value}),
        node_name="test_node",
    )

    builder = KernelModelBuilder()
    invariant = builder.build_invariant(ctx)

    # Verify both parameters have valid ranges
    assert "MW" in invariant.parallelization_params
    assert "MH" in invariant.parallelization_params
    assert len(invariant.parallelization_params["MW"]) == 18  # divisors of 768
    assert len(invariant.parallelization_params["MH"]) == 7   # divisors of 64


def test_build_invariant_no_stream_tiling():
    """Test build_invariant() with no stream tiling."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[InputSchema(
            name="input0",
            block_tiling=[FULL_DIM],
            stream_tiling=None,  # No stream tiling
        )],
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

    builder = KernelModelBuilder()
    invariant = builder.build_invariant(ctx)

    # No parallelization params
    assert invariant.parallelization_params == {}
    assert invariant.inputs[0].stream_tiling is None


# =============================================================================
# Test Constraint Splitting
# =============================================================================

def test_constraint_splitting():
    """Test constraints are correctly split into invariant vs variant."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[InputSchema(
            name="input0",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
        )],
        outputs=[OutputSchema(
            name="output0",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
        )],
        constraints=[
            # Invariant: no hierarchy
            DatatypeInteger(("input0", "output0")),
            # Invariant: TENSOR hierarchy
            ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.TENSOR),
            # Invariant: BLOCK hierarchy
            ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.BLOCK),
            # Variant: STREAM hierarchy
            ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.STREAM),
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

    builder = KernelModelBuilder()
    invariant = builder.build_invariant(ctx)

    # Verify constraint splitting
    assert len(invariant.invariant_constraints) == 3  # DatatypeInteger + 2 ShapesEqual
    assert len(invariant.variant_constraints) == 1    # ShapesEqual(STREAM)

    # Verify variant constraint is the STREAM one
    assert invariant.variant_constraints[0].hierarchy == ShapeHierarchy.STREAM


def test_is_invariant_constraint():
    """Test _is_invariant_constraint() classification logic."""
    builder = KernelModelBuilder()

    # Constraint without hierarchy: invariant
    c1 = DatatypeInteger(("input0",))
    assert builder._is_invariant_constraint(c1) is True

    # Constraint with TENSOR hierarchy: invariant
    c2 = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.TENSOR)
    assert builder._is_invariant_constraint(c2) is True

    # Constraint with BLOCK hierarchy: invariant
    c3 = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.BLOCK)
    assert builder._is_invariant_constraint(c3) is True

    # Constraint with STREAM hierarchy: variant
    c4 = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.STREAM)
    assert builder._is_invariant_constraint(c4) is False


# =============================================================================
# Test InvariantValidationContext
# =============================================================================

def test_invariant_validation_context_datatype():
    """Test InvariantValidationContext get_datatype()."""
    from brainsmith.dataflow.models import InvariantInterfaceModel

    inv_input = InvariantInterfaceModel(
        name="input0",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    ctx = InvariantValidationContext(
        inputs=[inv_input],
        outputs=[],
        internal_datatypes={},
        param_getter=lambda name: 1,
    )

    # Should work
    dt = ctx.get_datatype("input0")
    assert dt == DataType["INT8"]

    # Should raise KeyError for unknown interface
    with pytest.raises(KeyError, match="not found"):
        ctx.get_datatype("unknown")


def test_invariant_validation_context_shapes():
    """Test InvariantValidationContext get_shape() for TENSOR and BLOCK."""
    from brainsmith.dataflow.models import InvariantInterfaceModel

    inv_input = InvariantInterfaceModel(
        name="input0",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=[1, "SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    ctx = InvariantValidationContext(
        inputs=[inv_input],
        outputs=[],
        internal_datatypes={},
        param_getter=lambda name: 1,
    )

    # TENSOR hierarchy should work
    tensor_shape = ctx.get_shape("input0", ShapeHierarchy.TENSOR)
    assert tensor_shape == (1, 768)

    # BLOCK hierarchy should work
    block_shape = ctx.get_shape("input0", ShapeHierarchy.BLOCK)
    assert block_shape == (1, 768)

    # STREAM hierarchy should raise RuntimeError
    with pytest.raises(RuntimeError, match="Stream shapes not available"):
        ctx.get_shape("input0", ShapeHierarchy.STREAM)


def test_invariant_validation_context_is_dynamic():
    """Test InvariantValidationContext is_dynamic()."""
    from brainsmith.dataflow.models import InvariantInterfaceModel

    inv_input_dynamic = InvariantInterfaceModel(
        name="input0",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,  # Dynamic
    )

    inv_input_weight = InvariantInterfaceModel(
        name="input1",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=True,  # Weight
    )

    ctx = InvariantValidationContext(
        inputs=[inv_input_dynamic, inv_input_weight],
        outputs=[],
        internal_datatypes={},
        param_getter=lambda name: 1,
    )

    # Dynamic input
    assert ctx.is_dynamic("input0") is True

    # Weight input
    assert ctx.is_dynamic("input1") is False


def test_invariant_validation_context_get_interfaces():
    """Test InvariantValidationContext get_interfaces()."""
    from brainsmith.dataflow.models import InvariantInterfaceModel

    inv_input = InvariantInterfaceModel(
        name="input0",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    inv_output = InvariantInterfaceModel(
        name="output0",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    ctx = InvariantValidationContext(
        inputs=[inv_input],
        outputs=[inv_output],
        internal_datatypes={},
        param_getter=lambda name: 1,
    )

    interfaces = ctx.get_interfaces()
    assert "input0" in interfaces
    assert "output0" in interfaces
    assert len(interfaces) == 2


# =============================================================================
# Test Backward Compatibility
# =============================================================================

def test_build_backward_compatible():
    """Test build() still works via two-phase construction."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[InputSchema(
            name="input0",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
        )],
        outputs=[OutputSchema(
            name="output0",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
        )],
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

    builder = KernelModelBuilder()
    kernel_model = builder.build(ctx)

    # Verify type (legacy KernelModel)
    from brainsmith.dataflow.models import KernelModel
    assert isinstance(kernel_model, KernelModel)

    # Verify structure
    assert kernel_model.name == "TestKernel"
    assert len(kernel_model.inputs) == 1
    assert len(kernel_model.outputs) == 1

    # Verify stream shapes are resolved
    assert kernel_model.inputs[0].stream_shape == (64,)
    assert kernel_model.outputs[0].stream_shape == (64,)


def test_build_produces_same_result():
    """Test build() produces same result as build_invariant() + configure()."""
    schema = KernelSchema(
        name="TestKernel",
        inputs=[InputSchema(
            name="input0",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
        )],
        outputs=[OutputSchema(
            name="output0",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
        )],
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

    builder = KernelModelBuilder()

    # Method 1: build() (legacy)
    kernel_model = builder.build(ctx)

    # Method 2: build_invariant() + configure() (new)
    invariant = builder.build_invariant(ctx)
    configured = invariant.configure({"SIMD": 64})

    # Compare results
    assert kernel_model.name == configured.name
    assert len(kernel_model.inputs) == len(configured.inputs)
    assert len(kernel_model.outputs) == len(configured.outputs)

    # Input comparison
    assert kernel_model.inputs[0].name == configured.inputs[0].name
    assert kernel_model.inputs[0].tensor_shape == configured.inputs[0].tensor_shape
    assert kernel_model.inputs[0].block_shape == configured.inputs[0].block_shape
    assert kernel_model.inputs[0].stream_shape == configured.inputs[0].stream_shape
    assert kernel_model.inputs[0].datatype == configured.inputs[0].datatype

    # Output comparison
    assert kernel_model.outputs[0].name == configured.outputs[0].name
    assert kernel_model.outputs[0].tensor_shape == configured.outputs[0].tensor_shape
    assert kernel_model.outputs[0].block_shape == configured.outputs[0].block_shape
    assert kernel_model.outputs[0].stream_shape == configured.outputs[0].stream_shape
    assert kernel_model.outputs[0].datatype == configured.outputs[0].datatype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
