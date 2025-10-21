############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Clean LayerNorm schema demonstrating Schema + Pattern separation.

This schema describes WHAT to make (product specification):
- Structure: inputs, outputs, parameters
- Constraints: validation rules for product values
- DSE: initial parallelization configuration

This schema does NOT describe HOW to find sources:
- No source_ops (Pattern's responsibility)
- No attribute_mapping (Pattern's responsibility)
- No ONNX source constraints (Pattern's responsibility)
"""

import brainsmith.dataflow as df
from brainsmith.dataflow import DerivedDatatype, DerivedDim, FULL_DIM
from brainsmith.dataflow.constraints import Custom


# =============================================================================
# Clean Product Schema
# =============================================================================

LAYERNORM_SCHEMA = df.KernelSchema(
    name="LayerNorm",
    domain="brainsmith.kernels",

    # =========================================================================
    # STRUCTURE: What the product looks like
    # =========================================================================

    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],         # (1, 1, channels) - process full spatial dims
            stream_tiling=["SIMD"],          # Stream channels with SIMD parallelism
            required_layout="NHWC",          # Hardware requires NHWC layout
        )
    ],

    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],                  # (1, 1, channels)
            stream_tiling=[DerivedDim("input", -1)],  # Output streams at same rate as input
            datatype=DerivedDatatype("input"),        # Output datatype same as input
            required_layout="NHWC",                   # Hardware produces NHWC layout
        )
    ],

    # =========================================================================
    # PARAMETERS: Kernel-specific configuration
    # =========================================================================

    kernel_params={
        "epsilon": ("f", True, 1e-5),  # Normalization epsilon for numerical stability
    },

    # =========================================================================
    # CONSTRAINTS: Product validation rules
    # =========================================================================

    constraints=[
        # Product constraint: epsilon must be positive for numerical stability
        Custom(
            lambda ctx: None if ctx.get_param("epsilon") > 0 else "epsilon must be positive",
            "epsilon must be positive for numerical stability"
        ),
    ],

    # =========================================================================
    # DSE: Initial design space exploration configuration
    # =========================================================================

    initial_parallelization={"SIMD": 1},
)


__all__ = ["LAYERNORM_SCHEMA"]
