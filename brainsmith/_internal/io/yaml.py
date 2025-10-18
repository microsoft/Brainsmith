# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unified YAML parsing utility with environment variable resolution.

This module provides thread-safe YAML loading with support for:
- Environment variable expansion using ${VAR} syntax
- YAML inheritance via 'extends' field
- Automatic path resolution for Path-typed fields
- Context variable injection

Note: Only ${VAR} syntax is supported for variable expansion (not $VAR).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_yaml(
    file_path: Union[str, Path],
    expand_env_vars: bool = True,
    context_vars: Optional[Dict[str, str]] = None,
    support_inheritance: bool = False,
    path_fields: Optional[List[str]] = None,
    schema_class: Optional[type] = None,
) -> Dict[str, Any]:
    """Load a YAML file with optional environment variable expansion and inheritance.

    Environment variables are expanded using ${VAR} syntax (not $VAR).
    This implementation is thread-safe and does not mutate os.environ.

    Args:
        file_path: Path to the YAML file
        expand_env_vars: Whether to expand environment variables (default: True)
        context_vars: Additional context variables for expansion
        support_inheritance: Whether to support YAML inheritance via 'extends' field
        path_fields: Explicit list of field names that contain paths (dot notation for nested)
        schema_class: Optional Pydantic model class to extract path fields from

    Returns:
        Parsed YAML data with environment variables expanded and paths resolved

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    # Prepare context variables for inheritance
    default_context = {
        'YAML_DIR': str(file_path.parent.absolute()),
        'BSMITH_DIR': os.environ.get('BSMITH_DIR', str(Path(__file__).parent.parent.parent.absolute()))
    }
    
    # Merge with user-provided context
    if context_vars:
        default_context.update(context_vars)
    
    # Load the YAML file
    if support_inheritance:
        data = _load_with_inheritance(file_path, default_context)
    else:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f) or {}
    
    # Expand environment variables if requested
    if expand_env_vars:
        data = expand_env_vars_with_context(data, default_context)
    
    # Resolve relative paths if path_fields are specified or schema is provided
    if path_fields is not None or schema_class is not None:
        # Extract path fields from schema if provided
        if schema_class is not None and path_fields is None:
            path_fields = extract_path_fields_from_schema(schema_class)
        
        if path_fields:
            data = resolve_relative_paths(data, file_path.parent, path_fields)
    
    return data


def _load_with_inheritance(file_path: Path, context_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Load a YAML file with inheritance support via 'extends' field.
    
    Args:
        file_path: Path to the YAML file
        context_vars: Context variables for environment expansion
        
    Returns:
        Merged YAML data
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    # Check for inheritance
    if 'extends' in data:
        parent_path = data.pop('extends')
        
        # Expand environment variables in parent path with context
        if context_vars:
            # Temporarily set context vars in environment
            parent_path = expand_env_vars_with_context(parent_path, context_vars)
        else:
            parent_path = os.path.expandvars(parent_path)
        
        # Resolve parent path relative to current file
        if not Path(parent_path).is_absolute():
            parent_path = file_path.parent / parent_path
        else:
            parent_path = Path(parent_path)
        
        # Update context for parent file
        parent_context = context_vars.copy() if context_vars else {}
        parent_context['YAML_DIR'] = str(parent_path.parent.absolute())
        
        # Load parent data
        parent_data = _load_with_inheritance(parent_path, parent_context)
        
        # Merge with child data (child overrides parent)
        return _deep_merge(parent_data, data)
    
    return data


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with overlay values taking precedence.
    
    Args:
        base: Base dictionary
        overlay: Dictionary to merge on top
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def expand_env_vars(data: Any) -> Any:
    """
    Recursively expand environment variables in a data structure.
    
    Supports both ${VAR} and $VAR syntax. Handles nested dicts and lists.
    
    Args:
        data: Data structure to process
        
    Returns:
        Data with environment variables expanded
    """
    if isinstance(data, str):
        # Use os.path.expandvars which handles both ${VAR} and $VAR
        return os.path.expandvars(data)
    elif isinstance(data, dict):
        return {k: expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_vars(item) for item in data]
    else:
        # Numbers, booleans, None, etc. - return as-is
        return data


def expand_env_vars_with_context(
    data: Any,
    context_vars: Dict[str, str]
) -> Any:
    """Expand environment variables with additional context variables.

    This implementation is thread-safe and does not mutate os.environ.
    Uses string.Template for variable expansion with ${VAR} syntax.

    Args:
        data: Data structure to process
        context_vars: Additional variables to make available during expansion

    Returns:
        Data with environment variables expanded

    Note:
        Only ${VAR} syntax is supported (not $VAR without braces).
        Context variables take precedence over environment variables.
    """
    from string import Template

    if isinstance(data, str):
        # Combine environment with context (context takes precedence)
        combined = {**os.environ, **context_vars}

        # Use Template.safe_substitute for ${VAR} syntax
        template = Template(data)
        return template.safe_substitute(combined)

    elif isinstance(data, dict):
        return {k: expand_env_vars_with_context(v, context_vars)
                for k, v in data.items()}

    elif isinstance(data, list):
        return [expand_env_vars_with_context(item, context_vars)
                for item in data]

    else:
        # Numbers, booleans, None, etc. - return as-is
        return data


def find_yaml_file(
    search_paths: List[Union[str, Path]],
    filenames: List[str]
) -> Optional[Path]:
    """
    Search for a YAML file in multiple paths with multiple possible names.
    
    Args:
        search_paths: List of directories to search in priority order
        filenames: List of filenames to look for in priority order
        
    Returns:
        Path to the first found file, or None if not found
    """
    for search_path in search_paths:
        search_path = Path(search_path)
        if search_path.exists() and search_path.is_dir():
            for filename in filenames:
                file_path = search_path / filename
                if file_path.exists() and file_path.is_file():
                    return file_path
    
    return None


def resolve_relative_paths(
    data: Any,
    base_path: Path,
    path_fields: List[str],
    current_path: str = ""
) -> Any:
    """
    Recursively resolve relative paths in a data structure.
    
    Only processes fields explicitly listed in path_fields.
    Supports dot notation for nested fields (e.g., 'finn.finn_root').
    
    Args:
        data: Data structure to process
        base_path: Base path to resolve relative paths against
        path_fields: List of field names to treat as paths (dot notation for nested)
        current_path: Current path in the data structure (for internal recursion)
        
    Returns:
        Data with relative paths resolved to absolute paths
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Build the full field path
            field_path = f"{current_path}.{key}" if current_path else key
            
            # Check if this exact field path is in our list
            if field_path in path_fields and isinstance(value, str):
                # This field should be treated as a path
                path = Path(value)
                if not path.is_absolute():
                    resolved = (base_path / path).resolve()
                    result[key] = str(resolved)
                else:
                    result[key] = value
            else:
                # Recursively process nested structures
                result[key] = resolve_relative_paths(value, base_path, path_fields, field_path)
        return result
    elif isinstance(data, list):
        return [resolve_relative_paths(item, base_path, path_fields, current_path) for item in data]
    else:
        # Numbers, booleans, None, strings (that aren't path fields) - return as-is
        return data


def extract_path_fields_from_schema(schema_class: type) -> List[str]:
    """
    Extract field names that are Path types from a Pydantic model.
    
    Args:
        schema_class: Pydantic model class
        
    Returns:
        List of field names that are Path types (dot notation for nested)
    """
    path_fields = []
    
    # Check if this is a Pydantic model
    try:
        from pydantic import BaseModel
        if not issubclass(schema_class, BaseModel):
            return path_fields
    except ImportError:
        return path_fields
    
    # Extract fields from model
    for field_name, field_info in schema_class.model_fields.items():
        # Get the annotation
        annotation = field_info.annotation
        
        # Handle Optional types
        if hasattr(annotation, '__origin__'):
            if annotation.__origin__ is Union:
                # Extract the non-None type from Optional[Path]
                args = [arg for arg in annotation.__args__ if arg is not type(None)]
                if args:
                    annotation = args[0]
        
        # Check if it's a Path type
        if annotation is Path or (hasattr(annotation, '__name__') and annotation.__name__ == 'Path'):
            path_fields.append(field_name)
        # Check for nested Pydantic models
        elif hasattr(annotation, '__base__') and issubclass(annotation, BaseModel):
            nested_fields = extract_path_fields_from_schema(annotation)
            path_fields.extend([f"{field_name}.{nested}" for nested in nested_fields])
    
    return path_fields


def dump_yaml(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    **kwargs
) -> None:
    """
    Write data to a YAML file.
    
    Args:
        data: Data to write
        file_path: Path to write to
        **kwargs: Additional arguments passed to yaml.dump()
    """
    file_path = Path(file_path)
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set default formatting options
    default_kwargs = {
        'default_flow_style': False,
        'sort_keys': False,
        'width': 80,
        'indent': 2
    }
    default_kwargs.update(kwargs)
    
    with open(file_path, 'w') as f:
        yaml.dump(data, f, **default_kwargs)