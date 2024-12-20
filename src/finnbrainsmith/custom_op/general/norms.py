import numpy as np
from onnx import helper
from qonnx.custom_op.base import CustomOp
from qonnx.core.datatype import DataType


class FuncLayerNorm(CustomOp):
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "axis": ("i", True, -1),
            "epsilon": ("f", True, 1e-5),
            # FINN DataTypes for inputs, weight, bias, outputs
            "InputDataType": ("s", True, ""),
            "OutputDataType": ("s", True, ""),
            "backend": ("s", True, "general"),
        }
        return my_attrs

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node("Relu", [node.input[0]], [node.output[0]])

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("InputDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("OutputDataType")]

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dtype = DataType[self.get_nodeattr("OutputDataType")]
        model.set_tensor_datatype(node.output[0], dtype)
    
    def execute_node(self, context, graph):
        node = self.onnx_node
        # Get tensor values
        in_act = context[node.input[0]]
        out_act = context[node.output[0]]
        # Get any shape info that needs reuse
        ishape = in_act.shape
        assert ishape == out_act.shape, "In/out shapes don't match"
        # Get attributes
        norm_shape = ishape[self.get_nodeattr("axis"):]
        epsilon = self.get_nodeattr("epsilon")
        # Compute functional LayerNorm (no learned params)    
        mean = np.mean(in_act, axis=-1)
        variance = np.var(in_act, axis=-1)
        mean = np.expand_dims(mean, axis=-1)
        variance = np.expand_dims(variance, axis=-1)
        std_dev = np.sqrt(variance + epsilon)
        context[node.output[0]] = (in_act - mean)/std_dev
        # return context[node.output[0]]
    
    def verify_node(self):
        """Verifies that all attributes the node needs are there and
        that particular attributes are set correctly. Also checks if
        the number of inputs is equal to the expected number."""
        pass