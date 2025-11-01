# Portions derived from FINN project
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MVAU kernel schema definition with modern KernelOp patterns.

This module defines the schema for the MVAU kernel following modern Brainsmith patterns:
- VALUE_OPTIMIZED for automatic datatype optimization
- No explicit MW/MH nodeattrs (derived from weight tensor shape)
- Declarative constraints enforced by builder
- Polymorphic output datatype based on noActivation mode
- Optional thresholds input for multi-threshold activation (MVTU mode)
"""

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM, VALUE_OPTIMIZED
from brainsmith.dataflow.spec_helpers import smallest_datatype_for_range
from qonnx.core.datatype import DataType
from qonnx.util.basic import calculate_matvec_accumulator_range
import numpy as np


def _mvau_accumulator_datatype_resolver():
    """Compute accumulator datatype from matrix-vector multiply range.

    Analyzes actual weight values (if static) or uses worst-case type bounds
    to determine minimal accumulator width needed.
    """
    def resolver(interfaces, param_getter, model, tensor_name):
        # Get interfaces
        input_iface = interfaces["input"]
        weight_iface = interfaces["weights"]

        # Get input datatype
        idt = input_iface.datatype

        # Get weights (must be static)
        weights = model.get_initializer(weight_iface.tensor_name)
        if weights is None:
            raise ValueError("Accumulator datatype requires static weights")

        # Convert to bipolar if binaryXnorMode
        if param_getter("binaryXnorMode"):
            weights = 2 * weights - 1

        # Calculate accumulator range
        if not (param_getter("runtime_writeable_weights") or param_getter("dynamic_input")):
            # Use actual weight values (best case - tighter bounds)
            (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)
        else:
            # Use worst-case from datatypes (conservative)
            wdt = weight_iface.datatype
            mw, mh = weights.shape
            lower_worst = wdt.min() * np.ones((mw, mh))
            upper_worst = wdt.max() * np.ones((mw, mh))
            lower_range = calculate_matvec_accumulator_range(lower_worst, idt)
            upper_range = calculate_matvec_accumulator_range(upper_worst, idt)
            acc_min = min(min(lower_range), min(upper_range))
            acc_max = max(max(lower_range), max(upper_range))

        # Return minimal datatype for this range
        return smallest_datatype_for_range(acc_min, acc_max)

    return resolver


def _mvau_output_datatype_resolver():
    """Polymorphic output datatype resolver (depends on noActivation).

    - noActivation=1 (MV mode): output = accumulator type
    - noActivation=0 (MVTU mode): output determined by thresholds/ActVal
    """
    def resolver(interfaces, param_getter, model, tensor_name):
        no_act = param_getter("noActivation")

        if no_act == 1:
            # MV mode: output = accumulator type
            # Reuse accumulator resolver (same computation)
            return _mvau_accumulator_datatype_resolver()(interfaces, param_getter, model, tensor_name)
        else:
            # MVTU mode: output determined by ActVal
            # ActVal determines output range: [0, 2^ActVal - 1]
            act_val = param_getter("ActVal")
            if act_val > 0:
                return smallest_datatype_for_range(0, 2**act_val - 1)
            else:
                # ActVal=0 means output = accumulator (no quantization)
                return _mvau_accumulator_datatype_resolver()(interfaces, param_getter, model, tensor_name)

    return resolver


# Phase 1: Base schema (MV mode, internal_embedded only)
# Will be extended in later phases
MVAU_SCHEMA = df.KernelSchema(
    name="MVAU",

    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],    # Process full spatial dims
            stream_tiling=["SIMD"],      # Input parallelism
            required_layout="NHWC",
        ),
        df.InputSchema(
            name="weights",
            block_tiling=[],              # Static data
            stream_tiling=[],             # Not streamed (internal_embedded)
            datatype=VALUE_OPTIMIZED,     # Auto-optimize from weight values!
        ),
        # Note: thresholds input (input[2]) is added conditionally by infer_from()
        # when noActivation=0. It's not part of the static schema since MVAU
        # supports both 2-input (MV) and 3-input (MVTU) modes.
    ],

    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],
            stream_tiling=["PE"],                      # Output parallelism
            datatype=_mvau_output_datatype_resolver(), # Polymorphic based on noActivation
            required_layout="NHWC",
        )
    ],

    # STRUCTURAL PARAMETERS (fixed at inference, stored as nodeattrs)
    kernel_params={
        # Operation modes
        "noActivation": ("i", False, 1, {0, 1}),        # 0=MVTU (with thresholds), 1=MV (matmul only)
        "binaryXnorMode": ("i", False, 0, {0, 1}),      # Phase 1: only 0 supported
        "mem_mode": ("s", False, "internal_embedded",
                     {"internal_embedded", "internal_decoupled", "external"}),
        "dynamic_input": ("i", False, 0, {0, 1}),       # Phase 1: only 0 supported

        # Spatial/batch config
        "numInputVectors": ("ints", False, [1]),

        # Activation config (for MVTU mode - Phase 2)
        "ActVal": ("i", False, 0),

        # Runtime config
        "runtime_writeable_weights": ("i", False, 0, {0, 1}),
        "pumpedMemory": ("i", False, 0, {0, 1}),
    },

    # INTERNAL DATATYPES (kernel-specific computed types)
    internal_datatypes={
        "accDataType": _mvau_accumulator_datatype_resolver(),  # Computed from weight value analysis
    },

    # DSE DIMENSIONS (explorable resource parameters)
    # NOTE: PE and SIMD are auto-extracted from stream_tiling specs, not defined here
    dse_dimensions={
        "resType": df.DSEDimension(
            name="resType",
            values={"auto", "lut", "dsp"},
            default="auto",
        ),
        "ram_style": df.DSEDimension(
            name="ram_style",
            values={"auto", "block", "distributed", "ultra"},
            default="auto",
        ),
        "ram_style_thresholds": df.DSEDimension(
            name="ram_style_thresholds",
            values={"auto", "block", "distributed"},
            default="auto",
        ),
    },

    constraints=[
        df.IsDynamic(("input",)),
        df.IsStatic(("weights",)),
        df.DatatypeInteger(("input", "weights")),

        # Matrix dimensions: input[-1] must match weights[0] (MW)
        df.TensorDimMatches("input", -1, [("weights", 0)]),

        # Note: Threshold constraints are not included here since thresholds
        # input is conditional (only present when noActivation=0).
        # Validation for thresholds happens at runtime when they're present.

        # Divisibility constraints enforced by builder:
        # - PE divides MH (output channels = weights[1])
        # - SIMD divides MW (input channels = weights[0])
    ]
)
