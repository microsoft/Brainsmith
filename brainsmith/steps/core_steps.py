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

from brainsmith.core.plugins import step, get_transform

logger = logging.getLogger(__name__)

# === Conversion Steps ===

@step(
    name="qonnx_to_finn",
    category="cleanup",
    dependencies=["quantization_preprocessing"],
    description="Convert from QONNX to FINN opset"
)
def qonnx_to_finn_step(model: Any, cfg: Any) -> Any:
    """
    Convert QONNX to FINN opset.
    """

    ExpandNorms = get_transform('ExpandNorms')
    model = model.transform(ExpandNorms())

    FoldConstants = get_transform('FoldConstants')
    model = model.transform(FoldConstants())

    ConvertDivToMul = get_transform('ConvertDivToMul')
    model = model.transform(ConvertDivToMul())

    ConvertQONNXtoFINN = get_transform('ConvertQONNXtoFINN')
    model = model.transform(ConvertQONNXtoFINN())

    return model


# === Hardware Steps ===

@step(
    name="specialize_layers",
    category="hardware",
    description="Specialize layers with optional config override"
)
def specialize_layers_step(model, cfg):
    """
    Custom specialize layers step that ensures opset imports are handled correctly.
    """
    # Get transforms when needed
    GiveUniqueNodeNames = get_transform('GiveUniqueNodeNames')
    ApplyConfig = get_transform('ApplyConfig')
    SpecializeLayers = get_transform('SpecializeLayers')
    
    if cfg.specialize_layers_config_file is not None:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.specialize_layers_config_file))
    
    # Run the specialization
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))

    # Ensure custom opset imports before shape inference and apply final transforms
    GiveUniqueNodeNames = get_transform('GiveUniqueNodeNames')
    model = model.transform(GiveUniqueNodeNames())

    InferShapes = get_transform('InferShapes')
    model = model.transform(InferShapes())

    InferDataTypes = get_transform('InferDataTypes')
    model = model.transform(InferDataTypes())

    return model


# === Optimization Steps ===

@step(
    name="constrain_folding_and_set_pumped_compute",
    category="optimization", 
    dependencies=["streamlining"],
    description="Apply optimizations including folding constraints and pumped compute (MUST run before infer_hardware)"
)
def constrain_folding_and_set_pumped_compute_step(model, cfg):
    """Apply optimizations including folding constraints and pumped compute."""
    # Brainsmith native transforms
    TempShuffleFixer = get_transform('TempShuffleFixer')
    model = model.transform(TempShuffleFixer())

    SetPumpedCompute = get_transform('SetPumpedCompute')
    model = model.transform(SetPumpedCompute())

    return model