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

# Dataflow graph construction
from brainsmith.steps.build_dataflow_graph import (
    build_dataflow_graph,
    insert_infrastructure_kernels_step,
    infer_computational_kernels_step,
)
# Specialization to HW backends
from brainsmith.steps.specialize_kernel_backends import (
    specialize_kernel_backends,
    build_hw_graph,  # Legacy alias
)

# Layout normalization
from brainsmith.steps.normalize_layouts import normalize_dataflow_layouts_step

# Parameter exploration
from brainsmith.steps.parameter_exploration import explore_kernel_params_step

# Parallelization
from brainsmith.steps.parallelization import (
    apply_parallelization_config_step,
    target_fps_parallelization_step,
)

__all__ = [
    'qonnx_to_finn_step',
    'specialize_layers_step',
    'constrain_folding_and_set_pumped_compute_step',
    'shell_metadata_handover_step',
    'bert_cleanup_step',
    'bert_streamlining_step',
    'build_dataflow_graph',
    'insert_infrastructure_kernels_step',
    'infer_computational_kernels_step',
    'specialize_kernel_backends',
    'build_hw_graph',  # Legacy alias
    'normalize_dataflow_layouts_step',
    'explore_kernel_params_step',
    'apply_parallelization_config_step',
    'target_fps_parallelization_step',
]
