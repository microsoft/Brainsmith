"""Tests for Quant node insertion utilities."""

import numpy as np
import onnx.helper as oh
import pytest
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from tests.fixtures.test_data import generate_test_data
from tests.fixtures.model_annotation import insert_input_quant_nodes


class TestQuantInsertion:
    """Test Quant node insertion for all types."""

    def _create_simple_model(self, input_name="input", output_name="output"):
        """Helper to create a simple identity model."""
        inp = oh.make_tensor_value_info(input_name, TensorProto.FLOAT, [16, 9])
        out = oh.make_tensor_value_info(output_name, TensorProto.FLOAT, [16, 9])
        node = oh.make_node("Identity", [input_name], [output_name])
        graph = oh.make_graph([node], "test", [inp], [out])
        return ModelWrapper(qonnx_make_model(graph))

    def test_int8_quant_insertion(self):
        """Test IntQuant node insertion for INT8."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["INT8"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check Quant node was inserted
        assert len(model.graph.node) == 2  # Quant + Identity
        assert model.graph.node[0].op_type == "Quant"
        assert model.graph.node[0].domain == "qonnx.custom_op.general"

        # Check node attributes (look up by name, not index)
        signed_attr = [a for a in model.graph.node[0].attribute if a.name == "signed"][0]
        assert signed_attr.i == 1  # INT8 is signed

        # Check input was renamed
        assert model.graph.input[0].name == "raw_input"

        # Check DataType annotation
        assert model.get_tensor_datatype("input") == DataType["INT8"]

    def test_int9_quant_insertion(self):
        """Test IntQuant node insertion for arbitrary bitwidth INT9."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["INT9"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check Quant node created
        assert model.graph.node[0].op_type == "Quant"

        # Check DataType annotation
        assert model.get_tensor_datatype("input") == DataType["INT9"]

        # Check bitwidth initializer
        bitwidth_init = model.get_initializer(model.graph.node[0].input[3])
        assert bitwidth_init == 9.0

    def test_uint4_quant_insertion(self):
        """Test IntQuant node insertion for UINT4."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["UINT4"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check Quant node created
        assert model.graph.node[0].op_type == "Quant"

        # Check signed=0 for unsigned type
        signed_attr = [a for a in model.graph.node[0].attribute if a.name == "signed"][
            0
        ]
        assert signed_attr.i == 0

        # Check DataType annotation
        assert model.get_tensor_datatype("input") == DataType["UINT4"]

    def test_float_quant_insertion_e5m2(self):
        """Test FloatQuant node insertion for E5M2 FP8 format."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["FLOAT<5,2,15>"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check FloatQuant node was inserted
        assert len(model.graph.node) == 2  # FloatQuant + Identity
        assert model.graph.node[0].op_type == "FloatQuant"
        assert model.graph.node[0].domain == "qonnx.custom_op.general"

        # Check has 6 inputs (input, scale, exp_bw, mant_bw, bias, max_val)
        assert len(model.graph.node[0].input) == 6

        # Check DataType annotation
        assert model.get_tensor_datatype("input") == DataType["FLOAT<5,2,15>"]

        # Verify initializer values
        exp_bw = model.get_initializer(model.graph.node[0].input[2])
        mant_bw = model.get_initializer(model.graph.node[0].input[3])
        assert exp_bw == 5.0
        assert mant_bw == 2.0

    def test_float_quant_insertion_custom(self):
        """Test FloatQuant node insertion for custom float format."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(
            model, {"input": DataType["FLOAT<5,10,15>"]}
        )
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check FloatQuant node created
        assert model.graph.node[0].op_type == "FloatQuant"

        # Check DataType annotation
        assert model.get_tensor_datatype("input") == DataType["FLOAT<5,10,15>"]

        # Verify initializer values
        exp_bw = model.get_initializer(model.graph.node[0].input[2])
        mant_bw = model.get_initializer(model.graph.node[0].input[3])
        bias = model.get_initializer(model.graph.node[0].input[4])
        assert exp_bw == 5.0
        assert mant_bw == 10.0
        assert bias == 15.0

    def test_bipolar_quant_insertion(self):
        """Test BipolarQuant node insertion."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["BIPOLAR"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check BipolarQuant node was inserted
        assert len(model.graph.node) == 2  # BipolarQuant + Identity
        assert model.graph.node[0].op_type == "BipolarQuant"
        assert model.graph.node[0].domain == "qonnx.custom_op.general"

        # Check has 2 inputs (input, scale)
        assert len(model.graph.node[0].input) == 2

        # Check DataType annotation
        assert model.get_tensor_datatype("input") == DataType["BIPOLAR"]

    def test_binary_quant_insertion(self):
        """Test IntQuant node insertion for BINARY type."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["BINARY"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check Quant node created (BINARY uses IntQuant with bitwidth=1)
        assert model.graph.node[0].op_type == "Quant"

        # Check DataType annotation
        assert model.get_tensor_datatype("input") == DataType["BINARY"]

        # Check bitwidth=1
        bitwidth_init = model.get_initializer(model.graph.node[0].input[3])
        assert bitwidth_init == 1.0

    def test_mixed_int_float_insertion(self):
        """Test mixed INT8 + FLOAT<5,10,15> inputs."""
        # Create model with two inputs
        inp1 = oh.make_tensor_value_info("int_input", TensorProto.FLOAT, [16, 9])
        inp2 = oh.make_tensor_value_info("float_input", TensorProto.FLOAT, [16, 9])
        out = oh.make_tensor_value_info("output", TensorProto.FLOAT, [16, 9])
        node = oh.make_node("Add", ["int_input", "float_input"], ["output"])
        graph = oh.make_graph([node], "test", [inp1, inp2], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        # Insert both Quant types
        model = insert_input_quant_nodes(
            model,
            {
                "int_input": DataType["INT8"],
                "float_input": DataType["FLOAT<5,10,15>"],
            },
        )
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check both Quant nodes inserted
        quant_nodes = [n for n in model.graph.node if n.op_type in ["Quant", "FloatQuant"]]
        assert len(quant_nodes) == 2

        # Identify which is which by checking inputs
        int_quant = [n for n in quant_nodes if "int_input" in n.output][0]
        float_quant = [n for n in quant_nodes if "float_input" in n.output][0]

        assert int_quant.op_type == "Quant"
        assert float_quant.op_type == "FloatQuant"

        # Check DataType annotations
        assert model.get_tensor_datatype("int_input") == DataType["INT8"]
        assert model.get_tensor_datatype("float_input") == DataType["FLOAT<5,10,15>"]

    def test_mixed_all_types_insertion(self):
        """Test mixed INT, FLOAT, and BIPOLAR inputs."""
        # Create model with three inputs
        inp1 = oh.make_tensor_value_info("int_input", TensorProto.FLOAT, [16])
        inp2 = oh.make_tensor_value_info("float_input", TensorProto.FLOAT, [16])
        inp3 = oh.make_tensor_value_info("bipolar_input", TensorProto.FLOAT, [16])
        out = oh.make_tensor_value_info("output", TensorProto.FLOAT, [16])

        # Simple concat node
        node = oh.make_node(
            "Concat", ["int_input", "float_input", "bipolar_input"], ["output"], axis=0
        )
        graph = oh.make_graph([node], "test", [inp1, inp2, inp3], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        # Insert all three Quant types
        model = insert_input_quant_nodes(
            model,
            {
                "int_input": DataType["INT9"],
                "float_input": DataType["FLOAT<5,2,15>"],
                "bipolar_input": DataType["BIPOLAR"],
            },
        )
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check all three Quant nodes inserted
        quant_types = [n.op_type for n in model.graph.node if "Quant" in n.op_type]
        assert len(quant_types) == 3
        assert "Quant" in quant_types  # IntQuant
        assert "FloatQuant" in quant_types
        assert "BipolarQuant" in quant_types

        # Check DataType annotations
        assert model.get_tensor_datatype("int_input") == DataType["INT9"]
        assert model.get_tensor_datatype("float_input") == DataType["FLOAT<5,2,15>"]
        assert model.get_tensor_datatype("bipolar_input") == DataType["BIPOLAR"]

    def test_float32_no_quant_insertion(self):
        """Test that FLOAT32 inputs don't get Quant nodes."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["FLOAT32"]})
        model = model.transform(InferShapes())

        # Should only have Identity node, no Quant
        assert len(model.graph.node) == 1
        assert model.graph.node[0].op_type == "Identity"

        # Input name should not be renamed
        assert model.graph.input[0].name == "input"

    def test_end_to_end_int_execution(self):
        """Test that IntQuant nodes execute correctly with generated data."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["INT9"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Generate test data
        input_data = generate_test_data(DataType["INT9"], (16, 9))

        # Execute model
        output = execute_onnx(model, {"raw_input": input_data})

        # Check output is quantized to INT9 range
        assert np.all(output["output"] >= DataType["INT9"].min())
        assert np.all(output["output"] <= DataType["INT9"].max())
        # Values should be integers (in float32 container)
        assert np.all(output["output"] == np.round(output["output"]))

    def test_end_to_end_float_execution(self):
        """Test that FloatQuant nodes execute correctly with generated data."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["FLOAT<5,10,15>"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Generate test data
        input_data = generate_test_data(DataType["FLOAT<5,10,15>"], (16, 9))

        # Execute model
        output = execute_onnx(model, {"raw_input": input_data})

        # Check output is within FP8 range
        assert np.all(np.abs(output["output"]) <= DataType["FLOAT<5,10,15>"].max())

    def test_end_to_end_bipolar_execution(self):
        """Test that BipolarQuant nodes execute correctly with generated data."""
        model = self._create_simple_model()
        model = insert_input_quant_nodes(model, {"input": DataType["BIPOLAR"]})
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Generate test data
        input_data = generate_test_data(DataType["BIPOLAR"], (16, 9))

        # Execute model
        output = execute_onnx(model, {"raw_input": input_data})

        # Check output is bipolar {-1, +1}
        unique_values = set(output["output"].flatten())
        assert unique_values.issubset({-1.0, 1.0})

    def test_multiple_inputs_same_type(self):
        """Test inserting Quant nodes for multiple inputs of the same type."""
        # Create model with two INT8 inputs
        inp1 = oh.make_tensor_value_info("input1", TensorProto.FLOAT, [16])
        inp2 = oh.make_tensor_value_info("input2", TensorProto.FLOAT, [16])
        out = oh.make_tensor_value_info("output", TensorProto.FLOAT, [16])
        node = oh.make_node("Add", ["input1", "input2"], ["output"])
        graph = oh.make_graph([node], "test", [inp1, inp2], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        # Insert Quant for both
        model = insert_input_quant_nodes(
            model, {"input1": DataType["INT8"], "input2": DataType["INT8"]}
        )
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Check both Quant nodes inserted
        quant_nodes = [n for n in model.graph.node if n.op_type == "Quant"]
        assert len(quant_nodes) == 2

        # Check both have correct DataType
        assert model.get_tensor_datatype("input1") == DataType["INT8"]
        assert model.get_tensor_datatype("input2") == DataType["INT8"]

    def test_invalid_input_name(self):
        """Test that invalid input name raises error."""
        model = self._create_simple_model()

        with pytest.raises(ValueError, match="not found"):
            insert_input_quant_nodes(
                model, {"nonexistent_input": DataType["INT8"]}
            )
