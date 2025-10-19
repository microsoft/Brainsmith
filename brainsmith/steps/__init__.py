# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Steps Module

All step functionality is available through the lazy loader:

    from brainsmith.loader import get_step, list_steps

    # Get step function by name
    step_fn = get_step("shell_metadata_handover")
    model = step_fn(model, cfg)

    # List all available steps
    steps = list_steps()
"""

# Import all step modules (no decorator registration needed anymore)
from . import core_steps
from . import bert_custom_steps
from . import kernel_inference
