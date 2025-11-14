# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith transformation steps.

Model transformation pipeline functions (ONNX/QONNX → dataflow). Loaded during
component discovery to trigger @step decorator registration.

Access via registry:
    from brainsmith import get_step
    step_fn = get_step("qonnx_to_finn_step")
"""

# Core FINN-compatible steps
# BERT-specific steps
from brainsmith.steps.bert_steps import (
    bert_cleanup_step,
    bert_streamlining_step,
    shell_metadata_handover_step,
)

# Dataflow graph construction
from brainsmith.steps.build_dataflow_graph import (
    build_dataflow_graph,
    infer_computational_kernels_step,
    insert_infrastructure_kernels_step,
)
from brainsmith.steps.core_steps import (
    constrain_folding_and_set_pumped_compute_step,
    qonnx_to_finn_step,
    specialize_layers_step,
)

# Layout normalization
from brainsmith.steps.normalize_layouts import normalize_dataflow_layouts_step

# Parallelization
from brainsmith.steps.parallelization import (
    apply_parallelization_config_step,
    target_fps_parallelization_step,
)

# Parameter exploration
from brainsmith.steps.parameter_exploration import explore_kernel_params_step

# Specialization to HW backends
from brainsmith.steps.specialize_kernel_backends import specialize_kernel_backends

__all__ = [
    'qonnx_to_finn_step',
    'specialize_layers_step',
    'constrain_folding_and_set_pumped_compute_step',
    'shell_metadata_handover_step',
    'bert_cleanup_step',
    'bert_streamlining_step',
    'build_dataflow_graph',
    'infer_computational_kernels_step',
    'insert_infrastructure_kernels_step',
    'specialize_kernel_backends',
    'normalize_dataflow_layouts_step',
    'explore_kernel_params_step',
    'apply_parallelization_config_step',
    'target_fps_parallelization_step'
]
