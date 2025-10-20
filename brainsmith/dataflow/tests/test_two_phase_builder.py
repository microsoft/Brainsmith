############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for two-phase builder (build() method)."""

import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow.builder import KernelModelBuilder, BuildContext
from brainsmith.dataflow.schemas import KernelSchema, InputSchema, OutputSchema
from brainsmith.dataflow.models import KernelDesignSpace
from brainsmith.dataflow.validation import DesignSpaceValidationContext
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
# Test build() Method
# =============================================================================

def test_build_basic():
    """Test build() creates KernelDesignSpace."""
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
    design_space = builder.build(ctx)

    # Verify type
    assert isinstance(design_space, KernelDesignSpace)

    # Verify structure
    assert design_space.name == "TestKernel"
    assert len(design_space.inputs) == 1
    assert len(design_space.outputs) == 1

    # Verify input (dict-based access)
    assert "input0" in design_space.inputs
    input0 = design_space.inputs["input0"]
    assert input0.name == "input0"
    assert input0.tensor_shape == (768,)
    assert input0.block_shape == (768,)
    assert input0.stream_tiling == ["SIMD"]  # Preserved, not resolved
    assert input0.datatype == DataType["INT8"]

    # Verify output (dict-based access)
    assert "output0" in design_space.outputs
    output0 = design_space.outputs["output0"]
    assert output0.name == "output0"
    assert output0.stream_tiling == ["SIMD"]  # Preserved, not resolved

    # Verify valid ranges computed
    assert "SIMD" in design_space.parallelization_params
    assert 64 in design_space.parallelization_params["SIMD"]
    assert len(design_space.parallelization_params["SIMD"]) == 18  # divisors of 768


def test_build_multi_parameter():
    """Test build() with multiple parallelization parameters."""
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
    design_space = builder.build(ctx)

    # Verify both parameters have valid ranges
    assert "MW" in design_space.parallelization_params
    assert "MH" in design_space.parallelization_params
    assert len(design_space.parallelization_params["MW"]) == 18  # divisors of 768
    assert len(design_space.parallelization_params["MH"]) == 7   # divisors of 64


def test_build_no_stream_tiling():
    """Test build() with no stream tiling."""
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
    design_space = builder.build(ctx)

    # No parallelization params
    assert design_space.parallelization_params == {}
    assert design_space.inputs["input0"].stream_tiling is None


# =============================================================================
# Test Constraint Splitting
# =============================================================================

def test_constraint_splitting():
    """Test constraints are correctly split into structural vs parametric."""
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
            # Structural: no hierarchy
            DatatypeInteger(("input0", "output0")),
            # Structural: TENSOR hierarchy
            ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.TENSOR),
            # Structural: BLOCK hierarchy
            ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.BLOCK),
            # Parametric: STREAM hierarchy
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
    design_space = builder.build(ctx)

    # Verify constraint splitting
    # Note: Structural constraints (3 total: DatatypeInteger + 2 ShapesEqual)
    # are validated during build() but not stored in the model
    assert len(design_space.parametric_constraints) == 1    # ShapesEqual(STREAM)

    # Verify parametric constraint is the STREAM one
    assert design_space.parametric_constraints[0].hierarchy == ShapeHierarchy.STREAM


def test_evaluation_phase_classification():
    """Test constraint evaluation_phase property classification logic."""
    # Constraint without hierarchy: structural
    c1 = DatatypeInteger(("input0",))
    assert c1.evaluation_phase == 'structural'

    # Constraint with TENSOR hierarchy: structural
    c2 = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.TENSOR)
    assert c2.evaluation_phase == 'structural'

    # Constraint with BLOCK hierarchy: structural
    c3 = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.BLOCK)
    assert c3.evaluation_phase == 'structural'

    # Constraint with STREAM hierarchy: parametric
    c4 = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.STREAM)
    assert c4.evaluation_phase == 'parametric'


# =============================================================================
# Test DesignSpaceValidationContext
# =============================================================================

def test_design_space_validation_context_datatype():
    """Test DesignSpaceValidationContext get_datatype()."""
    from brainsmith.dataflow.models import InterfaceDesignSpace

    inv_input = InterfaceDesignSpace(
        name="input0",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    ctx = DesignSpaceValidationContext(
        inputs={"input0": inv_input},
        outputs={},
        internal_datatypes={},
        param_getter=lambda name: 1,
    )

    # Should work
    dt = ctx.get_datatype("input0")
    assert dt == DataType["INT8"]

    # Should raise KeyError for unknown interface
    with pytest.raises(KeyError, match="not found"):
        ctx.get_datatype("unknown")


def test_design_space_validation_context_shapes():
    """Test DesignSpaceValidationContext get_shape() for TENSOR and BLOCK."""
    from brainsmith.dataflow.models import InterfaceDesignSpace

    inv_input = InterfaceDesignSpace(
        name="input0",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=[1, "SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    ctx = DesignSpaceValidationContext(
        inputs={"input0": inv_input},
        outputs={},
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


def test_design_space_validation_context_is_dynamic():
    """Test DesignSpaceValidationContext is_dynamic()."""
    from brainsmith.dataflow.models import InterfaceDesignSpace

    inv_input_dynamic = InterfaceDesignSpace(
        name="input0",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,  # Dynamic
    )

    inv_input_weight = InterfaceDesignSpace(
        name="input1",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=True,  # Weight
    )

    ctx = DesignSpaceValidationContext(
        inputs={"input0": inv_input_dynamic, "input1": inv_input_weight},
        outputs={},
        internal_datatypes={},
        param_getter=lambda name: 1,
    )

    # Dynamic input
    assert ctx.is_dynamic("input0") is True

    # Weight input
    assert ctx.is_dynamic("input1") is False


def test_design_space_validation_context_get_interfaces():
    """Test DesignSpaceValidationContext get_interfaces()."""
    from brainsmith.dataflow.models import InterfaceDesignSpace

    inv_input = InterfaceDesignSpace(
        name="input0",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    inv_output = InterfaceDesignSpace(
        name="output0",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    ctx = DesignSpaceValidationContext(
        inputs={"input0": inv_input},
        outputs={"output0": inv_output},
        internal_datatypes={},
        param_getter=lambda name: 1,
    )

    interfaces = ctx.get_interfaces()
    assert "input0" in interfaces
    assert "output0" in interfaces
    assert len(interfaces) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
