# Portions derived from FINN project
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ElementwiseBinaryOp hardware kernel using modern KernelOp system.

This module implements a polymorphic kernel that handles all elementwise binary
operations (Add, Sub, Mul, Div, logical, comparison, bitwise) with a single
kernel class. Operations are distinguished by the 'func' parameter.

Supported Input Patterns:
- Dynamic + Static (Phase 1): One streaming input, one static parameter
- Dynamic + Dynamic (Phase 2): Both inputs streaming with broadcasting support

Broadcasting Support (Phase 2):
- ONNX multi-directional broadcasting semantics
- Handles rank mismatches and dimension broadcasts
- Optimized HLS code generation for broadcast patterns
"""

import logging
import numpy as np
from onnx import NodeProto, helper
from typing import Optional

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.dataflow import KernelOp, FULL_DIM
from brainsmith.dataflow.types import VALUE_OPTIMIZED
from brainsmith.dataflow.transformation import TransformationResult
from brainsmith.dataflow.spec_helpers import (
    add_datatype, sub_datatype, mul_datatype, smallest_datatype_for_range
)
import brainsmith.dataflow as df
from brainsmith.core.plugins import kernel

logger = logging.getLogger(__name__)


# =============================================================================
# Datatype Derivation Helpers
# =============================================================================

def _derive_div_datatype(interfaces, param_getter, model, tensor_name):
    """Compute division output datatype according to FINN/UG1399 rules.

    Division: width = lhs_width if rhs unsigned, else lhs_width + 1
    Signedness: signed if either operand is signed

    Args:
        interfaces: Dict of interface design points
        param_getter: Callable to get kernel parameters
        model: ModelWrapper (unused but required by signature)
        tensor_name: Tensor name (unused but required by signature)

    Returns:
        DataType for division output
    """
    lhs_dt = interfaces["lhs"].datatype
    rhs_dt = interfaces["rhs"].datatype

    lhs_width = lhs_dt.bitwidth()
    signed = any([lhs_dt.signed(), rhs_dt.signed()])

    # Width: lhs_width if rhs unsigned, else lhs_width + 1
    out_width = lhs_width if not rhs_dt.signed() else lhs_width + 1

    return DataType[f"INT{out_width}" if signed else f"UINT{out_width}"]


def _derive_bitwise_datatype(interfaces, param_getter, model, tensor_name):
    """Compute bitwise operation output datatype according to FINN/UG1399 rules.

    Bitwise (AND, OR, XOR): max(lhs_width, rhs_width), preserve signedness

    Args:
        interfaces: Dict of interface design points
        param_getter: Callable to get kernel parameters
        model: ModelWrapper (unused but required by signature)
        tensor_name: Tensor name (unused but required by signature)

    Returns:
        DataType for bitwise operation output
    """
    lhs_dt = interfaces["lhs"].datatype
    rhs_dt = interfaces["rhs"].datatype

    lhs_width = lhs_dt.bitwidth()
    rhs_width = rhs_dt.bitwidth()
    max_width = max(lhs_width, rhs_width)

    signed = any([lhs_dt.signed(), rhs_dt.signed()])

    return DataType[f"INT{max_width}" if signed else f"UINT{max_width}"]


def _elementwise_binary_output_datatype():
    """Polymorphic datatype resolver for ElementwiseBinaryOp.

    Dispatches to appropriate builder based on 'func' nodeattr.
    Supports all 15+ binary operations with operation-specific datatype rules.

    Returns:
        Callable resolver function with unified signature
    """
    def resolver(interfaces, param_getter, model, tensor_name):
        func = param_getter("func")

        # Arithmetic operations (context-aware builders from spec_helpers)
        if func == "Add":
            return add_datatype("lhs", "rhs")(interfaces, param_getter, model, tensor_name)
        elif func == "Sub":
            return sub_datatype("lhs", "rhs")(interfaces, param_getter, model, tensor_name)
        elif func == "Mul":
            return mul_datatype("lhs", "rhs")(interfaces, param_getter, model, tensor_name)
        elif func == "Div":
            return _derive_div_datatype(interfaces, param_getter, model, tensor_name)

        # Logical operations → BINARY (0 or 1)
        elif func in ("And", "Or", "Xor"):
            return smallest_datatype_for_range(0, 1)

        # Comparison operations → BINARY (0 or 1)
        elif func in ("Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"):
            return smallest_datatype_for_range(0, 1)

        # Bitwise operations → max width, preserve signedness
        elif func in ("BitwiseAnd", "BitwiseOr", "BitwiseXor"):
            return _derive_bitwise_datatype(interfaces, param_getter, model, tensor_name)

        # BitShift operations → use LHS datatype (shift doesn't change type)
        elif func in ("BitShiftLeft", "BitShiftRight"):
            return interfaces["lhs"].datatype

        else:
            raise ValueError(
                f"Unsupported func '{func}' in ElementwiseBinaryOp. "
                f"This should have been caught by schema validation."
            )

    return resolver


# =============================================================================
# Schema Definition
# =============================================================================

def _validate_input_pattern(ctx):
    """Validate input pattern and broadcasting compatibility.

    Supports two patterns:
    - "dynamic_static": LHS dynamic, RHS static (Phase 1)
    - "dynamic_dynamic": Both dynamic with broadcasting (Phase 2)

    Returns:
        None if valid, error message string if invalid
    """
    input_pattern = ctx.param_getter("input_pattern")

    # Validate pattern-specific constraints
    if input_pattern == "dynamic_static":
        # Phase 1: LHS dynamic, RHS static
        if "lhs" not in ctx.inputs or "rhs" not in ctx.inputs:
            return "Missing required inputs 'lhs' or 'rhs'"

        lhs = ctx.inputs["lhs"]
        rhs = ctx.inputs["rhs"]

        # RHS must be static (weight)
        if not rhs.is_weight:
            return "RHS must be static (initializer) for dynamic_static pattern"

    elif input_pattern == "dynamic_dynamic":
        # Phase 2: Both dynamic, must be broadcastable
        if "lhs" not in ctx.inputs or "rhs" not in ctx.inputs:
            return "Missing required inputs 'lhs' or 'rhs'"

        lhs = ctx.inputs["lhs"]
        rhs = ctx.inputs["rhs"]

        # Both must be dynamic (not weights)
        if lhs.is_weight or rhs.is_weight:
            return "Both inputs must be dynamic (not initializers) for dynamic_dynamic pattern"

        # Shapes must be broadcastable (checked at design space build time)
        # Note: BroadcastInfo.compute() will be called during HLS code generation
        # to get detailed broadcasting metadata

    else:
        return f"Unknown input_pattern '{input_pattern}'. Expected 'dynamic_static' or 'dynamic_dynamic'"

    return None


ELEMENTWISE_BINARY_SCHEMA = df.KernelSchema(
    name="ElementwiseBinaryOp",

    inputs=[
        df.InputSchema(
            name="lhs",
            block_tiling=[FULL_DIM],       # Process full tensor (no blocking)
            stream_tiling=["PE"],          # PE parallelism on last dimension
            required_layout=None,
        ),
        df.InputSchema(
            name="rhs",
            # Note: Tiling is minimal for backward compatibility with Phase 1 (static)
            # For Phase 2 dynamic+dynamic, HLS backend will create streaming interface
            # based on input_pattern parameter
            block_tiling=[FULL_DIM],       # Full tensor (needed for shape inference)
            stream_tiling=["PE"],          # PE parallelism (used only if dynamic)
            datatype=VALUE_OPTIMIZED,      # Optimize from actual values
            required_layout=None,
        ),
    ],

    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],                      # Full tensor
            stream_tiling=[("lhs", -1)],                  # Match LHS PE
            datatype=_elementwise_binary_output_datatype(),  # Polymorphic dispatch
            required_layout=None,
        )
    ],

    # STRUCTURAL (fixed at inference)
    kernel_params={
        # Operation type: matches ONNX op_type
        "func": (
            "s", True, "Add",
            {
                # Arithmetic
                "Add", "Sub", "Mul", "Div",
                # Logical
                "And", "Or", "Xor",
                # Comparison
                "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual",
                # Bitwise
                "BitwiseAnd", "BitwiseOr", "BitwiseXor",
                # BitShift (direction specified separately)
                "BitShiftLeft", "BitShiftRight",
            }
        ),
        # Input pattern: determines which inputs are streaming
        "input_pattern": (
            "s", True, "dynamic_static",
            {"dynamic_static", "dynamic_dynamic"}
        ),
    },

    # DSE DIMENSIONS (explorable resource parameters)
    dse_dimensions={
        # RAM style for parameter storage (HLS-specific, only for static inputs)
        "ram_style": df.DSEDimension(
            name="ram_style",
            values={"auto", "distributed", "block", "ultra"},
            default="auto"
        ),
        # Memory mode for constant parameters
        "mem_mode": df.DSEDimension(
            name="mem_mode",
            values={"internal_embedded", "internal_decoupled"},
            default="internal_embedded"
        ),
    },

    constraints=[
        # Both inputs must be integer types
        df.DatatypeInteger(("lhs", "rhs")),

        # Pattern-specific validation (dynamic vs static, broadcasting)
        df.CustomConstraint(
            check_fn=_validate_input_pattern,
            description="Validate input pattern (dynamic_static or dynamic_dynamic)"
        ),

        # Shapes must be broadcastable (ONNX semantics)
        # Note: This is implicitly validated by BroadcastInfo.compute() during HLS codegen
    ],
)


# =============================================================================
# ElementwiseBinaryOp Kernel Implementation
# =============================================================================

@kernel(
    description="Elementwise binary operations (arithmetic, logical, comparison, bitwise) with PE parallelism and broadcasting",
    author="Migrated from AMD FINN by Thomas Keller"
)
class ElementwiseBinaryOp(KernelOp):
    """Hardware kernel for elementwise binary operations.

    Implements a polymorphic kernel supporting 15+ binary operations:
    - Arithmetic: Add, Sub, Mul, Div
    - Logical: And, Or, Xor
    - Comparison: Equal, Less, LessOrEqual, Greater, GreaterOrEqual
    - Bitwise: BitwiseAnd, BitwiseOr, BitwiseXor
    - BitShift: BitShiftLeft, BitShiftRight

    Supported Input Patterns:
    - dynamic_static: One streaming input (LHS), one static parameter (RHS)
    - dynamic_dynamic: Both inputs streaming with ONNX broadcasting support

    Broadcasting Features (Phase 2):
    - ONNX multi-directional broadcasting semantics
    - Handles rank mismatches (e.g., [1,64,64,128] + [128])
    - Dimension broadcasts (e.g., [1,64,64,128] + [1,1,1,128])
    - Optimized HLS code generation with conditional reads

    The operation type is determined by the 'func' parameter, and the input
    pattern is determined by the 'input_pattern' parameter, both set during
    inference from the ONNX graph.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> df.KernelSchema:
        return ELEMENTWISE_BINARY_SCHEMA

    # ================================================================
    # ONNX → KernelOp Inference (Unified System)
    # ================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if node can be inferred as ElementwiseBinaryOp.

        Supported Patterns:
        - Phase 1 (dynamic_static): One dynamic, one static input
        - Phase 2 (dynamic_dynamic): Both dynamic inputs with broadcasting

        Args:
            node: ONNX node to check
            model: ModelWrapper for accessing graph info

        Returns:
            True if node can be inferred as this kernel
        """
        from brainsmith.dataflow.inference_helpers import (
            find_static_dynamic_pair,
            find_dynamic_dynamic_pair,
            check_shapes_broadcastable
        )

        # Must be supported operation
        if node.op_type not in ELEMENTWISE_BINARY_SCHEMA.kernel_params["func"][3]:
            return False

        # Must have exactly 2 inputs
        if len(node.input) != 2:
            return False

        # Check for dynamic+static pattern (Phase 1)
        static_dynamic_pair = find_static_dynamic_pair(node.input, model)
        if static_dynamic_pair is not None:
            # Reject output dequantization (Mul with float output, no successors)
            if node.op_type == "Mul":
                if not model.find_direct_successors(node):
                    out_dtype = model.get_tensor_datatype(node.output[0])
                    if out_dtype.name == "FLOAT32":
                        return False
            return True

        # Check for dynamic+dynamic pattern (Phase 2)
        dynamic_dynamic_pair = find_dynamic_dynamic_pair(node.input, model)
        if dynamic_dynamic_pair is not None:
            # Shapes must be broadcastable
            if not check_shapes_broadcastable(node.input[0], node.input[1], model):
                return False
            return True

        # Neither pattern matched (both static, or other issue)
        return False

    @classmethod
    def infer_from(cls, node: NodeProto, model: ModelWrapper, insert_index: int) -> TransformationResult:
        """Infer ElementwiseBinaryOp from ONNX binary operation.

        Detects input pattern and creates appropriate HW node:
        - Phase 1 (dynamic_static): Reorders to (dynamic, static)
        - Phase 2 (dynamic_dynamic): Preserves order, validates broadcasting

        Args:
            node: ONNX node to transform
            model: ModelWrapper for accessing graph info
            insert_index: Index where to insert new node

        Returns:
            TransformationResult with new HW node
        """
        from brainsmith.dataflow.inference_helpers import (
            find_static_dynamic_pair,
            find_dynamic_dynamic_pair,
            get_broadcast_info
        )

        # Try dynamic+static pattern first (Phase 1)
        static_dynamic_pair = find_static_dynamic_pair(node.input, model)
        if static_dynamic_pair is not None:
            lhs_input, rhs_input = static_dynamic_pair  # (dynamic, static)
            input_pattern = "dynamic_static"

        else:
            # Try dynamic+dynamic pattern (Phase 2)
            dynamic_dynamic_pair = find_dynamic_dynamic_pair(node.input, model)
            if dynamic_dynamic_pair is not None:
                lhs_input, rhs_input = dynamic_dynamic_pair  # Both dynamic
                input_pattern = "dynamic_dynamic"

                # Log broadcasting information for debugging
                broadcast_info = get_broadcast_info(lhs_input, rhs_input, model)
                if broadcast_info and broadcast_info.has_broadcast:
                    logger.info(
                        f"ElementwiseBinaryOp {node.name}: Broadcasting detected - "
                        f"LHS{broadcast_info.lhs_shape} + RHS{broadcast_info.rhs_shape} "
                        f"→ {broadcast_info.output_shape}"
                    )
            else:
                raise ValueError(
                    f"Node {node.name} doesn't match any ElementwiseBinaryOp pattern. "
                    f"Expected either (dynamic, static) or (dynamic, dynamic) inputs."
                )

        # Create ElementwiseBinaryOp node with detected pattern
        hw_node = helper.make_node(
            "ElementwiseBinaryOp",
            inputs=[lhs_input, rhs_input],
            outputs=node.output,
            name=node.name,
            domain="brainsmith.kernels",
            # Kernel parameters
            func=node.op_type,
            input_pattern=input_pattern,  # NEW: Track which pattern is active
        )

        # Return transformation result
        return TransformationResult(
            nodes_to_remove=[node],
            nodes_to_insert=[hw_node],
            actual_layouts={
                "lhs": None,    # No layout requirement
                "rhs": None,    # No layout requirement
                "output": None, # No layout requirement
            },
        )

    # ================================================================
    # Execution Support (For cppsim compatibility)
    # ================================================================

    @property
    def npy_op(self):
        """NumPy operation for execute_node() simulation.

        Maps 'func' parameter to corresponding NumPy ufunc.

        Returns:
            NumPy ufunc for the operation
        """
        op_map = {
            # Arithmetic
            "Add": np.add,
            "Sub": np.subtract,
            "Mul": np.multiply,
            "Div": np.divide,
            # Logical
            "And": np.logical_and,
            "Or": np.logical_or,
            "Xor": np.logical_xor,
            # Comparison
            "Equal": np.equal,
            "Less": np.less,
            "LessOrEqual": np.less_equal,
            "Greater": np.greater,
            "GreaterOrEqual": np.greater_equal,
            # Bitwise
            "BitwiseAnd": np.bitwise_and,
            "BitwiseOr": np.bitwise_or,
            "BitwiseXor": np.bitwise_xor,
            # BitShift
            "BitShiftLeft": np.left_shift,
            "BitShiftRight": np.right_shift,
        }
        func = self.get_nodeattr("func")
        return op_map[func]

    @property
    def cpp_op(self):
        """C++ operation template for HLS code generation.

        Maps 'func' parameter to C++ expression template.
        Templates use {0} for LHS and {1} for RHS.

        Returns:
            C++ expression template string
        """
        op_map = {
            # Arithmetic
            "Add": "({0} + {1})",
            "Sub": "({0} - {1})",
            "Mul": "({0} * {1})",
            "Div": "({0} / {1})",
            # Logical
            "And": "({0} && {1})",
            "Or": "({0} || {1})",
            "Xor": "(bool({0}) != bool({1}))",
            # Comparison
            "Equal": "({0} == {1})",
            "Less": "({0} < {1})",
            "LessOrEqual": "({0} <= {1})",
            "Greater": "({0} > {1})",
            "GreaterOrEqual": "({0} >= {1})",
            # Bitwise
            "BitwiseAnd": "({0} & {1})",
            "BitwiseOr": "({0} | {1})",
            "BitwiseXor": "({0} ^ {1})",
            # BitShift
            "BitShiftLeft": "({0} << {1})",
            "BitShiftRight": "({0} >> {1})",
        }
        func = self.get_nodeattr("func")
        return op_map[func]

    def execute_node(self, context, graph):
        """Execute node in Python simulation.

        Applies the elementwise operation with broadcasting semantics.

        Args:
            context: Execution context dict (tensor_name -> numpy array)
            graph: ONNX graph (for metadata)
        """
        node = self.onnx_node

        # Get inputs from context
        lhs = context[node.input[0]]
        rhs = context[node.input[1]]

        # Get datatypes
        lhs_dtype = self.design_point.inputs["lhs"].datatype
        rhs_dtype = self.design_point.inputs["rhs"].datatype

        # Convert to appropriate types for NumPy
        # Use int64 for integer types to avoid overflow in intermediate computation
        lhs = lhs.astype(np.int64) if lhs_dtype.is_integer() else lhs
        rhs = rhs.astype(np.int64) if rhs_dtype.is_integer() else rhs

        # Apply operation with broadcasting
        out = self.npy_op(lhs, rhs)

        # Store result (as float32 container, QONNX convention)
        context[node.output[0]] = out.astype(np.float32)

    # ================================================================
    # ONNX Shape Compatibility
    # ================================================================

    def make_shape_compatible_op(self, model):
        """Make ElementwiseBinaryOp compatible with ONNX shape inference.

        Returns equivalent ONNX op that has standard shape inference.
        For binary operations, use Add which supports broadcasting.

        Args:
            model: ModelWrapper (unused but required by signature)

        Returns:
            ONNX NodeProto compatible with shape inference
        """
        # Use Add for shape inference (supports broadcasting)
        return helper.make_node(
            "Add",
            self.onnx_node.input,
            self.onnx_node.output
        )
