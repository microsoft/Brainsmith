# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from brainsmith.blueprints.manager import get_build_steps

# Legacy registry for backward compatibility
# Now uses the new YAML blueprint system under the hood
REGISTRY = {
    "bert": lambda: get_build_steps("bert"),
}

def get_blueprint_steps(blueprint_name: str):
    """Get build steps for a blueprint, supporting both legacy and new systems."""
    if blueprint_name in REGISTRY:
        steps_func = REGISTRY[blueprint_name]
        if callable(steps_func):
            return steps_func()
        else:
            return steps_func
    else:
        # Try to load from YAML blueprint system
        return get_build_steps(blueprint_name)
