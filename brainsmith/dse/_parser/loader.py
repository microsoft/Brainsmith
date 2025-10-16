# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint YAML loading and inheritance resolution.

Handles loading YAML blueprints and resolving inheritance chains.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from brainsmith._internal.io.yaml import load_yaml, expand_env_vars_with_context


def load_blueprint_with_inheritance(blueprint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    """Load blueprint YAML and resolve inheritance.

    Args:
        blueprint_path: Path to blueprint YAML file

    Returns:
        Tuple of (raw_data, merged_data, parent_path)
        - raw_data: Blueprint data without inheritance merging
        - merged_data: Blueprint data with inheritance merged
        - parent_path: Path to parent blueprint if extends is used, None otherwise
    """
    # Context vars for environment expansion
    context_vars = {
        'BLUEPRINT_DIR': str(Path(blueprint_path).parent.absolute()),
        'BSMITH_DIR': os.environ.get(
            'BSMITH_DIR',
            str(Path(__file__).parent.parent.parent.parent.absolute())
        )
    }

    # Load raw data to check inheritance chain
    raw_data = load_yaml(
        blueprint_path,
        expand_env_vars=True,
        support_inheritance=False,
        context_vars=context_vars
    )

    parent_path = None

    # Resolve parent path if this blueprint extends another
    if 'extends' in raw_data:
        parent_path = raw_data['extends']
        # Expand env vars in parent path
        parent_path = expand_env_vars_with_context(parent_path, context_vars)

        # Resolve parent path relative to current file
        if not Path(parent_path).is_absolute():
            parent_path = str(Path(blueprint_path).parent / parent_path)

    # Load the full merged data for config extraction
    merged_data = load_yaml(
        blueprint_path,
        expand_env_vars=True,
        support_inheritance=True,
        context_vars=context_vars
    )

    return raw_data, merged_data, parent_path

