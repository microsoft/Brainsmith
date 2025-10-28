############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com> (AutoShuffle migration)
############################################################################

"""Shuffle hardware kernel using the modern KernelOp system.

This module implements tensor rearrangement (transpose + reshape) using the
unified dataflow system with schema-driven design, declarative constraints,
and two-phase construction for efficient Design Space Exploration.

Key Features:
- **Schema-driven**: All structure defined in SHUFFLE_SCHEMA
- **No shape storage**: Extracts shapes from ModelWrapper (never stores)
- **Declarative constraints**: Permutation validation, divisibility
- **Two-phase construction**: DesignSpace (once) → Configuration (many)
- **Unified transformation**: Can infer from Transpose nodes (with optional Reshape)

Example ONNX Pattern:
    # Pattern 1: Transpose alone
    Transpose(input: INT8[1,56,56,128], perm=[0,2,1,3])
    -> output: INT8[1,56,56,128]

    # Pattern 2: Reshape → Transpose → Reshape
    Reshape([1,56,56,128] → [1,56,56,4,32])
    Transpose(perm=[0,2,1,3,4])
    Reshape([1,56,56,4,32] → [1,56,56,128])

Hardware Mapping:
    Perfect loop nest generation via input_gen.hpp template
    SIMD parallelism for last dimension processing
"""

import numpy as np
from onnx import NodeProto
from typing import Optional, Dict, Any, Callable, List, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import get_by_name

from brainsmith.dataflow import KernelOp, FULL_DIM
import brainsmith.dataflow as df
from brainsmith.registry import kernel


# =============================================================================
# Helper Functions for Dimension Computation and Custom Constraints
# =============================================================================

def _compute_output_dim_factory(dim_index: int) -> Callable:
    """Factory for dimension functions that apply permutation to input shape.

    Creates a function that computes output dimension by applying the
    permutation to the input shape at the specified dimension index.

    Args:
        dim_index: Which output dimension to compute (0-indexed)

    Returns:
        Function compatible with unified 4-param callback signature

    Example:
        # For perm=[0, 2, 1, 3] on input [1, 56, 56, 128]:
        # dim 0: input[perm[0]] = input[0] = 1
        # dim 1: input[perm[1]] = input[2] = 56 (width)
        # dim 2: input[perm[2]] = input[1] = 56 (height)
        # dim 3: input[perm[3]] = input[3] = 128
    """
    def _compute_output_dim(
        interfaces: Dict[str, Any],
        param_getter: Callable,
        model: Any,
        tensor_name: Optional[str]
    ) -> int:
        """Compute output dimension by applying permutation.

        Args:
            interfaces: Dict mapping interface names to InterfaceDesignSpace/Configuration
            param_getter: Function to retrieve nodeattr values
            model: ModelWrapper (unused but part of unified signature)
            tensor_name: Output tensor name (unused but part of unified signature)

        Returns:
            Output dimension value at dim_index
        """
        perm = param_getter("perm")
        input_shape = interfaces["input"].tensor_shape

        # Validate dimension index
        if dim_index >= len(perm):
            raise ValueError(
                f"dim_index {dim_index} >= permutation length {len(perm)}"
            )

        # Apply permutation: output[i] = input[perm[i]]
        source_dim = perm[dim_index]
        return input_shape[source_dim]

    return _compute_output_dim


def _shuffle_perfect_loopnest_coeffs(shape: Tuple[int, ...], perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute perfect loop nest coefficients for HLS input_gen template.

    Given an input shape and permutation, calculate the coefficients for
    the perfect loop nest that will be generated in HLS code.

    This is the mathematical transformation that maps the logical transpose
    operation into hardware loop structure for efficient streaming.

    Args:
        shape: Input tensor shape (e.g., [1, 56, 56, 128])
        perm: Permutation array (e.g., [0, 2, 1, 3])

    Returns:
        Tuple of loop coefficients for input_gen template

    Example:
        shape = [1, 56, 56, 128], perm = [0, 2, 1, 3]
        adjusted_shape = [1, 56, 56, 128, 1]
        input_coeffs = [401408, 7168, 128, 1]
        out_coeffs = [401408, 128, 7168, 1]  # Reordered by perm
    """
    # Add trailing 1 for coefficient computation
    adjusted_shape = list(shape) + [1]

    # Compute strides: product of all dimensions after current
    input_coeffs = [np.prod(adjusted_shape[i+1:]) for i in range(len(shape))]

    # Reorder coefficients by permutation
    out_coeffs = [input_coeffs[i] for i in perm]

    return tuple(out_coeffs)


def _innerloop_moves(shape: Tuple[int, ...], perm: Tuple[int, ...]) -> int:
    """Check if the innermost dimension moves in the permutation.

    Returns 1 if the innermost dimension is permuted to a different position,
    0 if it stays in place. This affects HLS generation strategy.

    Args:
        shape: Input tensor shape
        perm: Permutation array

    Returns:
        1 if inner dimension moves, 0 if it stays in place

    Example:
        perm = [0, 2, 1, 3]: innermost (3) stays at position 3 → returns 0
        perm = [0, 1, 3, 2]: innermost (3) moves to position 2 → returns 1
    """
    innermost_original = len(shape) - 1
    new_position = list(perm).index(innermost_original)

    return 0 if new_position == len(perm) - 1 else 1


def _validate_permutation(ctx) -> Optional[str]:
    """Validate permutation is valid for input shape.

    Checks:
    - Permutation length matches input dimensions
    - Permutation contains each index exactly once (is a valid permutation)
    - Permutation values are in range [0, n-1]

    Args:
        ctx: Validation context (DesignSpaceValidationContext or ConfigurationValidationContext)

    Returns:
        Error message if invalid, None if valid
    """
    # Get parameters (always available in design space / configuration contexts)
    perm = ctx.get_param("perm")
    input_shape = ctx.get_shape("input", df.ShapeHierarchy.TENSOR)

    # Validate permutation length
    if len(perm) != len(input_shape):
        return (
            f"Permutation length {len(perm)} != input dimensions {len(input_shape)}. "
            f"Got perm={perm}, input_shape={input_shape}"
        )

    # Validate permutation is valid (contains each index 0..n-1 exactly once)
    expected_indices = set(range(len(perm)))
    actual_indices = set(perm)

    if actual_indices != expected_indices:
        return (
            f"Invalid permutation {perm}. "
            f"Must contain each index 0..{len(perm)-1} exactly once. "
            f"Missing: {expected_indices - actual_indices}, "
            f"Extra: {actual_indices - expected_indices}"
        )

    return None  # Valid


# =============================================================================
# Module-Level Schema (Structure + Validation + Transformation)
# =============================================================================

# Dynamic output shape - each dimension computed by applying permutation
# Note: We create 4 dimension functions for common 4D case (NHWC)
# For other dimensionalities, schema would need to be generalized
_OUTPUT_BLOCK_TILING = [
    _compute_output_dim_factory(0),  # Direct callable: permuted dim 0
    _compute_output_dim_factory(1),  # Direct callable: permuted dim 1
    _compute_output_dim_factory(2),  # Direct callable: permuted dim 2
    _compute_output_dim_factory(3),  # Direct callable: permuted dim 3
]

SHUFFLE_SCHEMA = df.KernelSchema(
    name="Shuffle",

    # ========== STRUCTURE ==========
    # Note: Shuffle supports transpose of 4D tensors (typical for vision models)
    # Input/output shapes are related by permutation
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM, FULL_DIM],
            stream_tiling=["SIMD"],  # Stream last dimension with SIMD
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            # Each output dimension is input[perm[i]]
            block_tiling=_OUTPUT_BLOCK_TILING,
            stream_tiling=[1, 1, 1, ("input", -1)],  # Match input SIMD
            datatype="input",  # Pass-through datatype
        )
    ],

    # ========== KERNEL PARAMETERS ==========
    kernel_params={
        "perm": ("ints", True, []),  # Permutation array (e.g., [0, 2, 1, 3])
        # Computed parameters (derived from perm and shape during transformation)
        "loop_coeffs": ("ints", False, []),  # Perfect loop nest coefficients
        "inner_moves": ("i", False, 0),  # 1 if innermost dim moves, 0 otherwise
    },

    # ========== VALIDATION ==========
    constraints=[
        # Data input must be dynamic (not an initializer)
        df.IsDynamic(("input",)),

        # SIMD must divide last dimension (parametric constraint)
        df.DimensionDivisible("input", -1, "SIMD", hierarchy=df.ShapeHierarchy.STREAM),

        # Custom permutation validation
        df.Custom(_validate_permutation, "Valid permutation of input dimensions"),
    ],

    # ========== TRANSFORMATION ==========
    attribute_mapping={
        "perm": "perm",  # Direct mapping from Transpose.perm attribute
    },
)


# =============================================================================
# Shuffle Kernel Implementation
# =============================================================================

@kernel(
    description="Hardware shuffle (rearrange and transpose) operation (KernelOp version)",
    author="Shane Fleming, migrated by Thomas Keller"
)
class Shuffle(KernelOp):
    """Hardware kernel for tensor rearrangement with schema-driven design.

    Performs transpose operations (with optional reshape before/after):
    - Permutes tensor dimensions according to perm parameter
    - Generates perfect loop nest for efficient hardware streaming
    - Supports SIMD parallelization on last dimension

    Schema Auto-Generates:
    - "SIMD" from stream_tiling=["SIMD"]
    - "input0Datatype" from input interface
    - "output0Datatype" from output interface (string shorthand: "input")
    - Permutation from kernel_params
    - Loop coefficients (computed during transformation)

    Output Shape Computation:
    - Each output dimension computed by factory functions applying perm
    - Uses direct callables (no wrapper classes)

    Design Space Exploration:
    - SIMD: divisors of input last dimension

    Example:
        # Create Shuffle node from ONNX Transpose
        node = ... # Transpose node with perm=[0, 2, 1, 3]
        model = ... # ModelWrapper
        result = Shuffle.infer_from(node, model, insert_index=0)

        # DSE: Explore SIMD values
        op = Shuffle(result.nodes_to_insert[0])
        for config in df.iter_valid_configurations(op, model):
            op.set_nodeattr("SIMD", config["SIMD"])
            ki = op.get_design_point(model)  # Returns KernelDesignPoint
            # Evaluate performance...
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> df.KernelSchema:
        """Build Shuffle schema (constant for all instances)."""
        return SHUFFLE_SCHEMA

    # ================================================================
    # Inference (Custom - needs Reshape detection and loop computation)
    # ================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if Transpose node can be converted to Shuffle.

        Validates:
        - Node is Transpose op
        - Has perm attribute
        - Schema constraints satisfied

        Note: Handles optional Reshape nodes during infer_from(), not here.

        Args:
            node: ONNX Transpose node to validate
            model: ModelWrapper for graph context

        Returns:
            True if this Transpose can be converted to Shuffle
        """
        # Check op type
        if node.op_type != "Transpose":
            return False

        # Check has perm attribute
        perm_attr = get_by_name(node.attribute, "perm")
        if perm_attr is None:
            return False

        # Check output count
        if len(node.output) != 1:
            return False

        # Note: Schema constraints (datatype, dynamic/static, etc.) will be validated
        # during build() after transformation. can_infer_from() only checks ONNX
        # pattern matching (op type, has perm attribute, output count).

        return True

    @classmethod
    def infer_from(
        cls,
        node: NodeProto,
        model: ModelWrapper,
        insert_index: int
    ) -> df.TransformationResult:
        """Create Shuffle HW node from ONNX Transpose node.

        Detects and handles four patterns:
        1. Transpose alone
        2. Reshape → Transpose
        3. Transpose → Reshape
        4. Reshape → Transpose → Reshape

        Computes loop_coeffs and inner_moves from the reshaped tensor shape
        and permutation (for HLS input_gen template).

        NOTE: Shuffle works with any tensor layout (it permutes dimensions).
        However, the global normalize_dataflow_layouts preprocessing pass ensures
        inputs are in NHWC layout for consistency with other dataflow kernels.

        Schema constraints already validated this node is compatible via can_infer_from().
        This method focuses purely on transformation.

        Args:
            node: ONNX Transpose node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes

        Returns:
            TransformationResult with Shuffle node and removed nodes

        Raises:
            ValueError: If Transpose pattern is invalid
        """
        from onnx import helper

        schema = cls.build_schema(node, model)

        # Track nodes to remove
        nodes_to_remove = [node]

        # ============================================================
        # Step 1: Detect preceding Reshape (pattern 2 or 4)
        # ============================================================
        new_in_tensor = node.input[0]
        in_shape = model.get_tensor_shape(node.input[0])
        in_reshaped = in_shape  # Shape after optional reshape

        producer = model.find_producer(node.input[0])
        if producer is not None and producer.op_type == "Reshape":
            # Pattern 2 or 4: Reshape before Transpose
            new_in_tensor = producer.input[0]
            in_shape = model.get_tensor_shape(new_in_tensor)
            in_reshaped = model.get_tensor_shape(node.input[0])  # Reshaped shape
            nodes_to_remove.append(producer)

        # ============================================================
        # Step 2: Detect following Reshape (pattern 3 or 4)
        # ============================================================
        new_out_tensor = node.output[0]
        out_shape = model.get_tensor_shape(new_out_tensor)
        out_reshaped = out_shape  # Shape before optional reshape

        consumer = model.find_consumer(node.output[0])
        if consumer is not None and consumer.op_type == "Reshape":
            # Pattern 3 or 4: Reshape after Transpose
            new_out_tensor = consumer.output[0]
            out_shape = model.get_tensor_shape(new_out_tensor)  # Final shape
            out_reshaped = model.get_tensor_shape(node.output[0])  # Transposed shape
            nodes_to_remove.append(consumer)

        # ============================================================
        # Step 3: Extract permutation and validate
        # ============================================================
        perm_attr = get_by_name(node.attribute, "perm")
        if perm_attr is None:
            raise ValueError(
                f"Transpose node '{node.name}' has no perm attribute"
            )

        perm = list(perm_attr.ints)

        # Validate permutation matches reshaped dimensions
        if len(perm) != len(in_reshaped):
            raise ValueError(
                f"Permutation length {len(perm)} != reshaped input dimensions {len(in_reshaped)}. "
                f"Got perm={perm}, in_reshaped={in_reshaped}"
            )

        if len(perm) != len(out_reshaped):
            raise ValueError(
                f"Permutation length {len(perm)} != reshaped output dimensions {len(out_reshaped)}. "
                f"Got perm={perm}, out_reshaped={out_reshaped}"
            )

        # Validate datatypes match (input and output must be same type)
        idt = model.get_tensor_datatype(new_in_tensor)
        odt = model.get_tensor_datatype(new_out_tensor)

        if idt != odt:
            raise ValueError(
                f"Input datatype {idt.name} != output datatype {odt.name}. "
                f"Shuffle requires identical input/output datatypes."
            )

        # ============================================================
        # Step 4: Compute derived parameters for HLS
        # ============================================================
        # Use reshaped shape for loop coefficient computation
        # (this is the shape the Transpose actually operates on)
        loop_coeffs = _shuffle_perfect_loopnest_coeffs(
            tuple(in_reshaped),
            tuple(perm)
        )
        inner_moves = _innerloop_moves(
            tuple(in_reshaped),
            tuple(perm)
        )

        # ============================================================
        # Step 5: Create HW node
        # ============================================================
        hw_node = helper.make_node(
            "Shuffle",
            inputs=[new_in_tensor],
            outputs=[new_out_tensor],
            domain="brainsmith.kernels",
            name=f"Shuffle_{node.name}",

            # Core parameters
            perm=perm,

            # Computed parameters for HLS
            loop_coeffs=list(loop_coeffs),
            inner_moves=inner_moves,

            # Initial parallelization
        )

        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=nodes_to_remove,
            actual_layouts={
                "input": None,  # Shuffle doesn't require specific layout
                "output": None,
            },
            metadata={
                "schema_name": schema.name,
                "source_pattern": "Transpose+Reshape",
                "perm": perm,
                "loop_coeffs": loop_coeffs,
                "inner_moves": inner_moves,
                "removed_reshapes": len(nodes_to_remove) - 1,
            }
        )

    # ================================================================
    # Execution (Reference Implementation)
    # ================================================================

    def execute_node(self, context, graph):
        """Reference numpy execution for testing.

        Applies transpose operation to input tensor:
        1. Reshape input to intermediate shape (if needed)
        2. Transpose with perm
        3. Reshape to final output shape (if needed)

        Note: We use the stored perm parameter directly. The in_shape/out_shape
        are extracted from the ModelWrapper context, not stored as nodeattrs.

        Args:
            context: Execution context (dict mapping tensor names to numpy arrays)
            graph: ONNX graph (not used)
        """
        node = self.onnx_node

        # Get input data
        input_data = context[node.input[0]]

        # Get permutation
        perm = self.get_nodeattr("perm")

        # For reference execution, we assume input is already in correct shape
        # Apply permutation directly
        transposed = np.transpose(input_data, axes=perm)

        # Store result
        context[node.output[0]] = transposed

    # ================================================================
    # FINN Compatibility
    # ================================================================

    def make_shape_compatible_op(self, model_w):
        """Create Shuffle-compatible op for shape inference.

        Returns a Shuffle node matching legacy behavior for FINN's shape inference.
        This dummy node is used during graph transformations to propagate shapes,
        not for actual execution.

        The base KernelOp class uses auto-detection which returns "RandomNormal",
        but legacy Shuffle returns "Shuffle". We override to match legacy behavior
        for parity testing.

        Args:
            model_w: ModelWrapper for ONNX graph access

        Returns:
            ONNX NodeProto with op_type="Shuffle" for shape inference
        """
        from onnx import helper

        in_shape = self.get_normal_input_shape(model_w=model_w)
        out_shape = self.get_normal_output_shape(model_w=model_w)

        return helper.make_node(
            "Shuffle",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            in_shape=list(in_shape),
            out_shape=list(out_shape)
        )
