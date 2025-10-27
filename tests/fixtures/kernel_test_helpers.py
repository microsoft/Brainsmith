# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable test helpers for kernel unit testing.

This module provides builders for common ONNX test models used in
kernel schema validation, inference testing, and transformation testing.

For parity testing (comparing manual vs auto implementations), use
tests/parity/test_fixtures.py and ParityTestBase instead.

Key Features:
- Fluent OnnxModelBuilder API for constructing test models
- Convenience functions for common patterns (binary, unary, parametric ops)
- Eliminates 100+ lines of boilerplate per kernel test file
- Single source of truth for test model construction

Usage:
    from tests.fixtures.kernel_test_helpers import make_parametric_op_model

    # Simple parametric operation (one dynamic, one static input)
    model, node = make_parametric_op_model("Add", param_input="bias")
    assert ChannelwiseOp.can_infer_from(node, model)

    # Binary operation (two dynamic inputs)
    model, node = make_binary_op_model("Add")
    assert AddStreams.can_infer_from(node, model)

    # Advanced: Custom builder with fluent API
    model, node = (OnnxModelBuilder()
        .op_type("Add")
        .inputs(["data", "bias"])
        .shape([1, 8, 8, 64])
        .static_input("bias", shape=[64])
        .datatype(DataType["INT8"])
        .build())
"""

from typing import List, Union, Optional, Tuple
import numpy as np
from onnx import helper, TensorProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model


class OnnxModelBuilder:
    """Fluent builder for ONNX test models with sane defaults.

    Eliminates boilerplate for common kernel testing patterns:
    - Binary ops (Add, Mul) with two dynamic inputs
    - Unary ops (Softmax, LayerNorm) with one dynamic input
    - Parametric ops (Channelwise) with dynamic + static inputs
    - Comparison ops (LessOrEqual, GreaterOrEqual)

    Example:
        # Simple Add with two inputs
        model, node = (OnnxModelBuilder()
            .op_type("Add")
            .inputs(["in0", "in1"])
            .shape([1, 224, 224, 64])
            .datatype(DataType["INT8"])
            .build())

        # Parametric op with static parameter
        model, node = (OnnxModelBuilder()
            .op_type("Add")
            .inputs(["data", "bias"])
            .shape([1, 8, 8, 64])
            .static_input("bias", shape=[64])
            .datatype(DataType["INT8"])
            .build())

        # Swapped input order (for testing canonical ordering)
        model, node = (OnnxModelBuilder()
            .op_type("Add")
            .inputs(["bias", "data"])  # Reversed!
            .shape([1, 8, 8, 64])
            .static_input("bias", shape=[64])
            .build())
    """

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._op_type: str = "Add"
        self._inputs: List[str] = ["in0", "in1"]
        self._outputs: List[str] = ["output"]
        self._shape: List[int] = [1, 8, 8, 64]
        self._datatype: DataType = DataType["INT8"]
        self._static_inputs: dict = {}  # {name: shape}
        self._initializer_values: dict = {}  # {name: numpy_array}
        self._input_shapes: dict = {}  # Override per-input shapes
        self._domain: str = ""  # Standard ONNX domain
        self._node_name: str = None

    def op_type(self, op_type: str) -> 'OnnxModelBuilder':
        """Set operation type (e.g., 'Add', 'Softmax').

        Args:
            op_type: ONNX operation type string

        Returns:
            Self for method chaining
        """
        self._op_type = op_type
        return self

    def inputs(self, inputs: List[str]) -> 'OnnxModelBuilder':
        """Set input tensor names.

        Args:
            inputs: List of input tensor names

        Returns:
            Self for method chaining
        """
        self._inputs = inputs
        return self

    def outputs(self, outputs: List[str]) -> 'OnnxModelBuilder':
        """Set output tensor names.

        Args:
            outputs: List of output tensor names

        Returns:
            Self for method chaining
        """
        self._outputs = outputs
        return self

    def shape(self, shape: List[int]) -> 'OnnxModelBuilder':
        """Set default shape for all inputs/outputs.

        Args:
            shape: Tensor shape (e.g., [1, 8, 8, 64])

        Returns:
            Self for method chaining
        """
        self._shape = shape
        return self

    def input_shape(self, name: str, shape: List[int]) -> 'OnnxModelBuilder':
        """Override shape for specific input.

        Args:
            name: Input tensor name
            shape: Shape for this specific input

        Returns:
            Self for method chaining
        """
        self._input_shapes[name] = shape
        return self

    def datatype(self, datatype: DataType) -> 'OnnxModelBuilder':
        """Set datatype for all tensors.

        Args:
            datatype: QONNX DataType (e.g., DataType["INT8"])

        Returns:
            Self for method chaining
        """
        self._datatype = datatype
        return self

    def static_input(
        self,
        name: str,
        shape: Union[List[int], int] = None,
        values: np.ndarray = None
    ) -> 'OnnxModelBuilder':
        """Mark input as static (initializer) with optional shape/values.

        Args:
            name: Input tensor name to make static
            shape: Shape for the static input (int or list). If int, treated as [int].
                  If None, uses last dimension of default shape (for per-channel params)
            values: Explicit numpy array. If None, uses zeros with appropriate shape.

        Returns:
            Self for method chaining

        Example:
            # Per-channel parameter (inferred from default shape)
            builder.static_input("bias")

            # Scalar parameter
            builder.static_input("scalar", shape=1)

            # Custom shape
            builder.static_input("weight", shape=[64, 32])

            # Custom values
            builder.static_input("threshold", values=np.array([0.5]))
        """
        if values is not None:
            self._initializer_values[name] = values
            self._static_inputs[name] = list(values.shape)
        else:
            if shape is None:
                # Default: per-channel parameter (last dim of default shape)
                shape = [self._shape[-1]]
            elif isinstance(shape, int):
                shape = [shape]

            self._static_inputs[name] = shape
            self._initializer_values[name] = np.zeros(shape)

        return self

    def domain(self, domain: str) -> 'OnnxModelBuilder':
        """Set custom domain (default is standard ONNX).

        Args:
            domain: ONNX domain string (e.g., "brainsmith.kernels")

        Returns:
            Self for method chaining
        """
        self._domain = domain
        return self

    def name(self, name: str) -> 'OnnxModelBuilder':
        """Set node name.

        Args:
            name: Node name string

        Returns:
            Self for method chaining
        """
        self._node_name = name
        return self

    def build(self) -> Tuple[ModelWrapper, NodeProto]:
        """Build the ONNX model and return (model, node).

        Returns:
            (ModelWrapper, NodeProto): The model and the target operation node

        Raises:
            ValueError: If configuration is invalid
        """
        # Create input value infos (non-static only)
        dynamic_inputs = []
        for inp_name in self._inputs:
            if inp_name not in self._static_inputs:
                shape = self._input_shapes.get(inp_name, self._shape)
                dynamic_inputs.append(
                    helper.make_tensor_value_info(inp_name, TensorProto.FLOAT, shape)
                )

        # Create initializers for static inputs
        initializers = []
        for name, shape in self._static_inputs.items():
            values = self._initializer_values.get(name, np.zeros(shape))
            init = helper.make_tensor(
                name,
                TensorProto.FLOAT,
                shape,
                values.flatten().tolist()
            )
            initializers.append(init)

        # Create output value info
        output_shape = self._shape  # Could be customized if needed
        outputs_info = [
            helper.make_tensor_value_info(out, TensorProto.FLOAT, output_shape)
            for out in self._outputs
        ]

        # Create node
        node_args = {
            "inputs": self._inputs,
            "outputs": self._outputs,
        }
        if self._node_name:
            node_args["name"] = self._node_name
        if self._domain:
            node_args["domain"] = self._domain

        node = helper.make_node(self._op_type, **node_args)

        # Create graph
        graph = helper.make_graph(
            [node],
            "test_graph",
            dynamic_inputs,
            outputs_info,
            initializers
        )

        # Create model
        model = qonnx_make_model(graph)
        model_w = ModelWrapper(model)

        # Set datatypes for all tensors
        for inp_name in self._inputs:
            model_w.set_tensor_datatype(inp_name, self._datatype)
        for out in self._outputs:
            model_w.set_tensor_datatype(out, self._datatype)

        return model_w, node


# =============================================================================
# Convenience Functions for Common Patterns
# =============================================================================

def make_binary_op_model(
    op_type: str,
    shape: List[int] = None,
    input0: str = "in0",
    input1: str = "in1",
    output: str = "output",
    datatype: DataType = DataType["INT8"]
) -> Tuple[ModelWrapper, NodeProto]:
    """Create binary operation model (Add, Mul, etc.) with two dynamic inputs.

    Use this for kernels that operate on two streaming inputs, such as:
    - AddStreams (element-wise add of two streams)
    - MulStreams (element-wise multiply of two streams)

    Args:
        op_type: Operation type (e.g., "Add", "Mul")
        shape: Input/output shape (default: [1, 8, 8, 64])
        input0: First input name (default: "in0")
        input1: Second input name (default: "in1")
        output: Output name (default: "output")
        datatype: Tensor datatype (default: DataType["INT8"])

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> model, node = make_binary_op_model("Add")
        >>> assert AddStreams.can_infer_from(node, model)

        >>> # Custom shape and names
        >>> model, node = make_binary_op_model(
        ...     "Add",
        ...     shape=[1, 224, 224, 64],
        ...     input0="stream0",
        ...     input1="stream1"
        ... )
    """
    builder = (OnnxModelBuilder()
        .op_type(op_type)
        .inputs([input0, input1])
        .outputs([output])
        .datatype(datatype))

    if shape:
        builder.shape(shape)

    return builder.build()


def make_parametric_op_model(
    op_type: str,
    dynamic_input: str = "data",
    param_input: str = "param",
    shape: List[int] = None,
    param_shape: Union[List[int], int] = None,
    datatype: DataType = DataType["INT8"],
    input_order: str = "dynamic_first"
) -> Tuple[ModelWrapper, NodeProto]:
    """Create parametric operation model (one dynamic, one static input).

    Use this for kernels that have one streaming input and one static parameter:
    - Channelwise operations (Add with bias, Mul with scale, comparison with threshold)
    - Thresholding operations
    - Any op where one input is weights/parameters

    Args:
        op_type: Operation type (e.g., "Add", "Mul", "LessOrEqual")
        dynamic_input: Name for dynamic (streaming) input (default: "data")
        param_input: Name for static (parameter) input (default: "param")
        shape: Shape for dynamic input (default: [1, 8, 8, 64])
        param_shape: Shape for parameter. If None, uses per-channel (last dim of shape).
                    If int, converts to [int]. If list, uses as-is.
        datatype: Tensor datatype (default: DataType["INT8"])
        input_order: "dynamic_first" or "static_first" (for testing canonical ordering)

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> # Canonical order (dynamic, static)
        >>> model, node = make_parametric_op_model("Add", param_input="bias")
        >>> assert ChannelwiseOp.can_infer_from(node, model)

        >>> # Swapped order (tests that inference handles this)
        >>> model, node = make_parametric_op_model(
        ...     "Add",
        ...     param_input="bias",
        ...     input_order="static_first"
        ... )

        >>> # Scalar parameter
        >>> model, node = make_parametric_op_model(
        ...     "Add",
        ...     param_input="scalar",
        ...     param_shape=1
        ... )

        >>> # Comparison operation
        >>> model, node = make_parametric_op_model(
        ...     "LessOrEqual",
        ...     param_input="threshold"
        ... )
    """
    if input_order == "dynamic_first":
        inputs = [dynamic_input, param_input]
    else:
        inputs = [param_input, dynamic_input]

    builder = (OnnxModelBuilder()
        .op_type(op_type)
        .inputs(inputs)
        .static_input(param_input, shape=param_shape)
        .datatype(datatype))

    if shape:
        builder.shape(shape)

    return builder.build()


def make_unary_op_model(
    op_type: str,
    input_name: str = "input",
    output: str = "output",
    shape: List[int] = None,
    datatype: DataType = DataType["INT8"]
) -> Tuple[ModelWrapper, NodeProto]:
    """Create unary operation model (Softmax, LayerNorm, etc.).

    Use this for kernels that have a single streaming input:
    - Softmax
    - LayerNorm
    - Activation functions (ReLU, Sigmoid, etc.)

    Args:
        op_type: Operation type (e.g., "Softmax", "LayerNorm")
        input_name: Input tensor name (default: "input")
        output: Output tensor name (default: "output")
        shape: Input/output shape (default: [1, 8, 8, 64])
        datatype: Tensor datatype (default: DataType["INT8"])

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> model, node = make_unary_op_model("Softmax", shape=[1, 10, 768])
        >>> assert SoftmaxOp.can_infer_from(node, model)

        >>> # LayerNorm with custom name
        >>> model, node = make_unary_op_model(
        ...     "LayerNorm",
        ...     input_name="activations",
        ...     shape=[1, 12, 768]
        ... )
    """
    builder = (OnnxModelBuilder()
        .op_type(op_type)
        .inputs([input_name])
        .outputs([output])
        .datatype(datatype))

    if shape:
        builder.shape(shape)

    return builder.build()
