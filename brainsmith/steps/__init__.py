# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Steps Module - Lazy Loading for Performance

Step modules are imported only when needed, avoiding expensive upfront imports.
Steps are registered via decorators when their module is loaded.

All step functionality is available through the lazy loader:

    from brainsmith.loader import get_step, list_steps

    # Get step function by name
    step_fn = get_step("shell_metadata_handover")
    model = step_fn(model, cfg)

    # List all available steps
    steps = list_steps()

Note: This module provides lazy loading of step *modules* (not individual steps).
Individual steps are discovered via decorators when modules are imported.
"""

from brainsmith.plugin_helpers import create_lazy_module

# ============================================================================
# Step Module Registry (Metadata Only - NO imports!)
# ============================================================================

COMPONENTS = {
    'modules': {
        'core_steps': '.core_steps',
        'bert_custom_steps': '.bert_custom_steps',
        'kernel_inference': '.kernel_inference',
    }
}

# ============================================================================
# Lazy Loading (PEP 562) - Unified Pattern
# ============================================================================

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
