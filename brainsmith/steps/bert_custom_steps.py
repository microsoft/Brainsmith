# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
BERT-Specific Custom Build Steps

Custom steps specifically for BERT model processing, including:
- Head and tail removal for model decomposition
- Metadata extraction for shell integration
- Reference I/O generation for validation

These steps are highly specific to BERT model architecture and
are not general-purpose FINN dataflow compilation steps.
"""

import os
import shutil
import logging
from typing import Any
import numpy as np

from brainsmith.registry import step, get_transform
from brainsmith._internal.io.transform_utils import apply_transforms

logger = logging.getLogger(__name__)


def save_debug_model(model, cfg, step_name):
    """Save model for debugging if preserve_intermediate_models is enabled."""
    if getattr(cfg, 'preserve_intermediate_models', False):
        debug_dir = os.path.join(cfg.output_dir, "debug_models")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save ONNX model
        model_path = os.path.join(debug_dir, f"{step_name}.onnx")
        model.save(model_path)
        
        # Log model structure info
        logger.info(f"Saved debug model: {step_name}")
        logger.info(f"  - Inputs: {[i.name for i in model.graph.input]}")
        logger.info(f"  - Outputs: {[o.name for o in model.graph.output]}")
        logger.info(f"  - Nodes: {len(model.graph.node)}")
        if model.graph.node:
            logger.info(f"  - First node: {model.graph.node[0].name} ({model.graph.node[0].op_type})")
            # Check for LayerNormalization nodes
            ln_nodes = [n for n in model.graph.node if n.op_type == "LayerNormalization"]
            if ln_nodes:
                logger.info(f"  - Found {len(ln_nodes)} LayerNormalization nodes")


# === Metadata Steps ===

@step(
    name="shell_metadata_handover",
    category="metadata",
    dependencies=[],
    description="Extract metadata for shell integration process"
)
def shell_metadata_handover_step(model, cfg):
    """
    Extract metadata for shell integration process.
    
    This information is stored in a json file that is passed to the build process.
    It adds this to the stitched_ip output directory and checks it exists ahead of time.
    """
    from finn.builder.build_dataflow_config import DataflowOutputType
    
    if DataflowOutputType.STITCHED_IP in cfg.generate_outputs:
        if os.path.isdir(cfg.output_dir + '/stitched_ip'):
            # Brainsmith native transform - load when needed
            ExtractShellIntegrationMetadata = get_transform('ExtractShellIntegrationMetadata')
            model = model.transform(ExtractShellIntegrationMetadata(
                cfg.output_dir + "/stitched_ip/shell_handover.json"
            ))
            # copy over the ref IO *.npy files into the stitched_ip for handover
            shutil.copy(cfg.verify_input_npy, cfg.output_dir + '/stitched_ip')
            shutil.copy(cfg.verify_expected_output_npy, cfg.output_dir + '/stitched_ip')
            return model
        else:
            raise RuntimeError(f"Error: could not find stitched IP directory so unable to create metadata. Please ensure this is called after the create_stitched_ip step")
    return model


# === Pre-Processing ===

@step(
    name="bert_cleanup",
    category="cleanup",
    dependencies=[],
    description="Graph cleanup/preparation step for BERT models",
)
def bert_cleanup_step(model: Any, cfg: Any) -> Any:
    """Basic cleanup with identity removal and input sorting."""
    
    model = apply_transforms(model, [
        'SortCommutativeInputsInitializerLast',
        'RemoveIdentityOps'
    ])

    return model


# === Streamlining Steps ===

@step(
    name="bert_streamlining",
    category="topology_opt",
    dependencies=["qonnx_to_finn"],
    description="Comprehensive streamlining with QONNX preprocessing and FINN absorption"
)
def bert_streamlining_step(model, cfg):
    """
    BERT custom step for streamlining

    Some additional streamlining steps are required here
    to handle the Mul nodes leftover from the SoftMax
    transformations done in custom_step_qonnx2finn.

    In particular, we need to move the Mul operation
    at the output of the QuantSoftMax lower in the graph
    so that it has the option to be merged into a MultiThreshold 
    node. In particular:

        * MoveScalarMulPastMatMul : moves the Mul past the DynMatMul
        * ModeScalarLinearPartInvariants : moves the Mul over the
          reshape and transpose
        * AbsorbMulIntoMultiThreshold : absorbs the Mul into the MT
    """
    
    model = apply_transforms(model, [
        'AbsorbSignBiasIntoMultiThreshold',
        'AbsorbAddIntoMultiThreshold',
        'AbsorbMulIntoMultiThreshold',
        'RoundAndClipThresholds'
    ])
    
    # Apply transform with parameter
    MoveOpPastFork = get_transform('MoveOpPastFork')
    model = model.transform(MoveOpPastFork(["Mul"]))
    
    model = apply_transforms(model, [
        'MoveScalarMulPastMatMul',
        'MoveScalarLinearPastInvariants',
        'AbsorbMulIntoMultiThreshold',
        'AbsorbAddIntoMultiThreshold'
    ])
    
    # Final cleanup
    InferDataTypes = get_transform('InferDataTypes')
    GiveUniqueNodeNames = get_transform('GiveUniqueNodeNames')
    model = model.transform(InferDataTypes(allow_scaledint_dtypes=False))
    model = model.transform(GiveUniqueNodeNames())
    
    return model
