# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith transformation steps.

Step functions for model transformation pipelines. All steps are eagerly
imported for simplicity. Manifest caching provides performance.

All step functions are available through the registry:

    from brainsmith.registry import get_step

    step_fn = get_step("shell_metadata_handover")
    model = step_fn(model, cfg)
"""

# Core FINN-compatible steps
from brainsmith.steps.core_steps import (
    qonnx_to_finn_step,
    specialize_layers_step,
    constrain_folding_and_set_pumped_compute_step,
)

# BERT-specific steps
from brainsmith.steps.bert_custom_steps import (
    shell_metadata_handover_step,
    bert_cleanup_step,
    bert_streamlining_step,
)

# Kernel inference
from brainsmith.steps.kernel_inference import infer_kernels_step

__all__ = [
    'qonnx_to_finn_step',
    'specialize_layers_step',
    'constrain_folding_and_set_pumped_compute_step',
    'shell_metadata_handover_step',
    'bert_cleanup_step',
    'bert_streamlining_step',
    'infer_kernels_step',
]
