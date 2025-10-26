# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Steps Module - Lazy Loading

Steps use lazy loading pattern (like kernels) to defer expensive imports.
Components are imported only when accessed, avoiding upfront import costs.

All step functionality is available through the loader:

    from brainsmith.registry import get_step, list_steps

    # Get step function by name
    step_fn = get_step("shell_metadata_handover")
    model = step_fn(model, cfg)

    # List all available steps
    steps = list_steps()
"""

from brainsmith.registry import create_lazy_module

# ============================================================================
# Step Registry (Metadata Only - NO imports!)
# ============================================================================

COMPONENTS = {
    'steps': {
        # Core FINN-compatible steps
        # Map decorator names to module paths
        # When lazy loaded, the module will export the decorated function
        'qonnx_to_finn': '.core_steps',
        'specialize_layers': '.core_steps',
        'constrain_folding_and_set_pumped_compute': '.core_steps',

        # BERT-specific steps
        'shell_metadata_handover': '.bert_custom_steps',
        'bert_cleanup': '.bert_custom_steps',
        'bert_streamlining': '.bert_custom_steps',

        # Kernel inference
        'infer_kernels': '.kernel_inference',
    }
}

# ============================================================================
# Lazy Loading (PEP 562) - Unified Pattern
# ============================================================================

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
