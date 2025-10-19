# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Core FINN-compatible Build Steps

Brainsmith implementations of core FINN dataflow compiler steps.
These steps use the comprehensive plugin registration system to access
transforms from QONNX, FINN, and Brainsmith.
"""

import os
import logging
from typing import Any

from brainsmith.transforms import import_transform
from brainsmith._internal.io.transform_utils import apply_transforms

logger = logging.getLogger(__name__)

# === Conversion Steps ===
# Note: Steps are now registered in brainsmith/plugins.py using the registry system

def qonnx_to_finn_step(model: Any, cfg: Any) -> Any:
    """Convert QONNX to FINN opset."""
    
    model = apply_transforms(model, [
        'ExpandNorms',
        'FoldConstants',
        'ConvertDivToMul',
        'ConvertQONNXtoFINN'
    ])
    
    return model


# === Hardware Steps ===

def specialize_layers_step(model: Any, cfg: Any) -> Any:
    """Custom specialize layers step that ensures opset imports are handled correctly."""
    # Load transforms individually when parameters are needed
    # (use apply_transforms for parameter-free bulk transforms)
    GiveUniqueNodeNames = import_transform('GiveUniqueNodeNames')
    ApplyConfig = import_transform('ApplyConfig')
    SpecializeLayers = import_transform('SpecializeLayers')
    
    if cfg.specialize_layers_config_file is not None:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.specialize_layers_config_file))

    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))

    model = apply_transforms(model, [
        'GiveUniqueNodeNames',
        'InferShapes',
        'InferDataTypes'
    ])
    
    return model


# === Optimization Steps ===

def constrain_folding_and_set_pumped_compute_step(model, cfg):
    """Apply optimizations including folding constraints and pumped compute."""
    model = apply_transforms(model, [
        'TempShuffleFixer',
        'SetPumpedCompute'
    ])
    return model
