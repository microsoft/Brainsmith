# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Core FINN-compatible Build Steps

Brainsmith implementations of core FINN dataflow compiler steps.
These steps use the comprehensive component registry to access
transforms from QONNX, FINN, and Brainsmith.
"""

import os
import logging
from typing import Any

from brainsmith.primitives.transforms.expand_norms import ExpandNorms
from brainsmith.primitives.transforms.temp_shuffle_fixer import TempShuffleFixer
from brainsmith.primitives.transforms.set_pumped_compute import SetPumpedCompute
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ConvertDivToMul,
    GiveUniqueNodeNames,
    ApplyConfig,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.builder.build_dataflow_steps import step_specialize_layers
from brainsmith.primitives.transforms.specialize_kernels import SpecializeKernels

logger = logging.getLogger(__name__)

# Import decorator for registration
from brainsmith.registry import step

# === Conversion Steps ===

@step(name='qonnx_to_finn')
def qonnx_to_finn_step(model: Any, cfg: Any) -> Any:
    """Convert QONNX to FINN opset."""

    for transform in [
        ExpandNorms(),
        FoldConstants(),
        ConvertDivToMul(),
        ConvertQONNXtoFINN()
    ]:
        model = model.transform(transform)

    return model


# === Hardware Steps ===

@step(name='specialize_layers')
def specialize_layers_step(model: Any, cfg: Any) -> Any:
    """Specialize hardware layers using registry-based backend discovery."""

    if cfg.specialize_layers_config_file is not None:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.specialize_layers_config_file))

    # Run Brainsmith registry-based specialization first
    model = model.transform(SpecializeKernels(cfg))

    # Run FINN's step_specialize_layers as catch-all for any remaining ops
    # model = step_specialize_layers(model, cfg)

    for transform in [
        GiveUniqueNodeNames(),
        InferShapes(),
        InferDataTypes()
    ]:
        model = model.transform(transform)

    return model


# === Optimization Steps ===

@step(name='constrain_folding_and_set_pumped_compute')
def constrain_folding_and_set_pumped_compute_step(model, cfg):
    """Apply optimizations including folding constraints and pumped compute."""
    for transform in [TempShuffleFixer(), SetPumpedCompute()]:
        model = model.transform(transform)
    return model
