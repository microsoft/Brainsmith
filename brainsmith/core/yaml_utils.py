# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
YAML utilities for blueprint loading and merging.
"""

import os
import yaml
from typing import Dict, Any, Tuple, Optional


def load_blueprint_with_inheritance(
    blueprint_path: str, 
    return_parent: bool = False
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load blueprint YAML with inheritance support.
    
    Args:
        blueprint_path: Path to blueprint YAML file
        return_parent: If True, also return parent data before merge
        
    Returns:
        Tuple of (merged data, parent data if requested)
    """
    with open(blueprint_path, 'r') as f:
        data = yaml.safe_load(f)
    
    parent_data = None
    
    if 'extends' in data:
        # Resolve parent path relative to child
        parent_path = os.path.join(
            os.path.dirname(blueprint_path), 
            data['extends']
        )
        parent_data, _ = load_blueprint_with_inheritance(parent_path, False)
        merged_data = deep_merge(parent_data, data)
        return (merged_data, parent_data if return_parent else None)
    
    return (data, None)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary (parent blueprint)
        override: Override dictionary (child blueprint)
        
    Returns:
        New merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result