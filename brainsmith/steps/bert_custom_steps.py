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

from brainsmith.registry import step, get_transform
from brainsmith._internal.io.transform_utils import apply_transforms

logger = logging.getLogger(__name__)


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
            ExtractShellIntegrationMetadata = get_transform('ExtractShellIntegrationMetadata')
            model = model.transform(ExtractShellIntegrationMetadata(
                cfg.output_dir + "/stitched_ip/shell_handover.json"
            ))
            # copy over the ref IO *.npy files into the stitched_ip for handover
            shutil.copy(cfg.verify_input_npy, cfg.output_dir + '/stitched_ip')
            shutil.copy(cfg.verify_expected_output_npy, cfg.output_dir + '/stitched_ip')
            return model
        else:
            raise RuntimeError(
                "Stitched IP directory not found. "
                "Ensure shell_metadata_handover runs after create_stitched_ip step."
            )
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
def bert_streamlining_step(model: Any, cfg: Any) -> Any:
    """BERT-specific streamlining with SoftMax Mul node handling.

    Problem:
        SoftMax transformations in qonnx_to_finn leave Mul nodes that must
        be moved lower in the graph to merge with MultiThreshold nodes.

    Solution:
        1. MoveScalarMulPastMatMul - move Mul past DynMatMul
        2. MoveScalarLinearPastInvariants - move over reshape/transpose
        3. AbsorbMulIntoMultiThreshold - merge into MultiThreshold

    Dependencies:
        Requires qonnx_to_finn step.
    """
    # Apply bulk transforms without parameters
    model = apply_transforms(model, [
        'AbsorbSignBiasIntoMultiThreshold',
        'AbsorbAddIntoMultiThreshold',
        'AbsorbMulIntoMultiThreshold',
        'RoundAndClipThresholds'
    ])

    # Load transform individually to pass parameters
    MoveOpPastFork = get_transform('MoveOpPastFork')
    model = model.transform(MoveOpPastFork(["Mul"]))
    
    model = apply_transforms(model, [
        'MoveScalarMulPastMatMul',
        'MoveScalarLinearPastInvariants',
        'AbsorbMulIntoMultiThreshold',
        'AbsorbAddIntoMultiThreshold'
    ])

    # Final cleanup with parameterized transforms
    InferDataTypes = get_transform('InferDataTypes')
    GiveUniqueNodeNames = get_transform('GiveUniqueNodeNames')
    model = model.transform(InferDataTypes(allow_scaledint_dtypes=False))
    model = model.transform(GiveUniqueNodeNames())
    
    return model
