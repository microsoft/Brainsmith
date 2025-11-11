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

import logging
import os
import shutil
from typing import Any

from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
)
from finn.transformation.streamline.reorder import (
    MoveOpPastFork,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.transformation.general import GiveUniqueNodeNames, SortCommutativeInputsInitializerLast
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps

from brainsmith.primitives.transforms.extract_shell_integration_metadata import (
    ExtractShellIntegrationMetadata,
)

logger = logging.getLogger(__name__)

# Import decorator for registration
from brainsmith.registry import step  # noqa: E402

# === Pre-Processing ===

@step(name='bert_cleanup')
def bert_cleanup_step(model: Any, cfg: Any) -> Any:
    """Basic cleanup with identity removal and input sorting."""

    for transform in [
        SortCommutativeInputsInitializerLast(),
        RemoveIdentityOps()
    ]:
        model = model.transform(transform)

    return model


# === Streamlining Steps ===

@step(name='bert_streamlining')
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
    for transform in [
        AbsorbSignBiasIntoMultiThreshold(),
        AbsorbAddIntoMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        RoundAndClipThresholds()
    ]:
        model = model.transform(transform)

    # Transform with parameters
    model = model.transform(MoveOpPastFork(["Mul"]))

    for transform in [
        MoveScalarMulPastMatMul(),
        MoveScalarLinearPastInvariants(),
        AbsorbMulIntoMultiThreshold(),
        AbsorbAddIntoMultiThreshold()
    ]:
        model = model.transform(transform)

    # Final cleanup with parameterized transforms
    model = model.transform(InferDataTypes(allow_scaledint_dtypes=False))
    model = model.transform(GiveUniqueNodeNames())

    return model


# === Metadata Steps ===

@step(name='shell_metadata_handover')
def shell_metadata_handover_step(model, cfg):
    """
    Extract metadata for shell integration process.

    This information is stored in a json file that is passed to the build process.
    It adds this to the stitched_ip output directory and checks it exists ahead of time.
    """
    from finn.builder.build_dataflow_config import DataflowOutputType

    if DataflowOutputType.STITCHED_IP in cfg.generate_outputs:
        if os.path.isdir(cfg.output_dir + '/stitched_ip'):
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
