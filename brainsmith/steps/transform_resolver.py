"""
Transform Resolver for FINN Steps

Resolves transform names to actual transform classes using the plugin registry.
"""

import logging
from typing import Dict, List, Type, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class TransformResolutionError(Exception):
    """Raised when a transform cannot be resolved."""
    pass


class AmbiguousTransformError(Exception):
    """Raised when a transform name matches multiple frameworks without explicit prefix."""
    pass


@lru_cache(maxsize=128)
def _get_transform_class(transform_name: str) -> Type:
    """
    Get a transform class by name using the plugin registry with smart conflict detection.
    
    This is cached to avoid repeated lookups. Supports:
    - Prefixed names: "qonnx:RemoveIdentityOps", "finn:ConvertQONNXtoFINN"
    - Unprefixed unique names: "FoldConstants" (if unique across frameworks)
    - Automatic conflict detection: raises clear error for ambiguous names
    
    Args:
        transform_name: Name of the transform to resolve, optionally prefixed
        
    Returns:
        Transform class
        
    Raises:
        TransformResolutionError: If transform cannot be found
        AmbiguousTransformError: If unprefixed name matches multiple frameworks
    """
    from brainsmith.core.plugins import get_registry
    registry = get_registry()
    
    # Handle prefixed names (e.g., "qonnx:RemoveIdentityOps")
    if ":" in transform_name:
        framework, name = transform_name.split(":", 1)
        transform_cls = registry.get_transform(name, framework=framework)
        if transform_cls:
            logger.debug(f"Resolved prefixed transform '{transform_name}'")
            return transform_cls
    else:
        # Handle unprefixed names - check for conflicts
        name = transform_name
        
        # Check direct lookup first
        transform_cls = registry.get_transform(name)
        if transform_cls:
            logger.debug(f"Resolved transform '{name}' (direct lookup)")
            return transform_cls
        
        # Check across frameworks for conflicts
        found_frameworks = []
        found_class = None
        
        for framework in registry.framework_transforms.keys():
            framework_cls = registry.get_transform(name, framework=framework)
            if framework_cls:
                found_frameworks.append(framework)
                found_class = framework_cls
        
        if len(found_frameworks) == 1:
            logger.debug(f"Resolved unique transform '{name}' from framework '{found_frameworks[0]}'")
            return found_class
        elif len(found_frameworks) > 1:
            raise TransformResolutionError(
                f"Ambiguous transform '{name}' found in multiple frameworks: {found_frameworks}. "
                f"Use prefixed name like '{found_frameworks[0]}:{name}' to specify framework."
            )
    
    # Not found - provide helpful error with context
    all_plugins = registry.list_all_plugins()
    all_transforms = [p for p in all_plugins if p['metadata'].get('type') == 'transform']
    
    # Count by framework
    qonnx_count = len([t for t in all_transforms if t['metadata'].get("framework") == "qonnx"])
    finn_count = len([t for t in all_transforms if t['metadata'].get("framework") == "finn"])
    brainsmith_count = len([t for t in all_transforms if t['metadata'].get("framework") == "brainsmith"])
    
    # Suggest similar names
    all_names = [t["name"] for t in all_transforms]
    
    # Find potential matches (simple substring matching)
    search_name = transform_name.split(":")[-1] if ":" in transform_name else transform_name
    suggestions = [name for name in all_names if search_name.lower() in name.lower()][:3]
    
    error_msg = (
        f"Transform '{transform_name}' not found in plugin registry. "
        f"Available: {brainsmith_count} BrainSmith, {qonnx_count} QONNX, {finn_count} FINN transforms."
    )
    
    if suggestions:
        error_msg += f" Similar transforms: {', '.join(suggestions)}"
    
    error_msg += " Use prefixes like 'qonnx:TransformName' or 'finn:TransformName' for external transforms."
    
    raise TransformResolutionError(error_msg)


def resolve_transforms(transform_names: List[str]) -> Dict[str, Type]:
    """
    Resolve a list of transform names to their classes.
    
    Args:
        transform_names: List of transform names to resolve
        
    Returns:
        Dictionary mapping transform names to classes
        
    Raises:
        TransformResolutionError: If any transform cannot be resolved
    """
    if not transform_names:
        return {}
    
    resolved = {}
    errors = []
    
    for name in transform_names:
        try:
            resolved[name] = _get_transform_class(name)
            logger.debug(f"Resolved transform '{name}' to {resolved[name]}")
        except TransformResolutionError as e:
            errors.append(str(e))
    
    if errors:
        raise TransformResolutionError(
            "Failed to resolve transforms:\n" + "\n".join(errors)
        )
    
    return resolved


def validate_transform_dependencies(step_name: str, transform_names: List[str]) -> List[str]:
    """
    Validate that all required transforms are available.
    
    Args:
        step_name: Name of the step requiring transforms
        transform_names: List of required transform names
        
    Returns:
        List of validation errors (empty if all valid)
    """
    if not transform_names:
        return []
    
    errors = []
    
    for name in transform_names:
        try:
            _get_transform_class(name)
        except TransformResolutionError:
            errors.append(f"Step '{step_name}' requires transform '{name}' which is not available")
    
    return errors


# Plugin discovery removed - plugins register themselves on import


def clear_cache():
    """Clear the transform resolution cache. Useful for testing."""
    _get_transform_class.cache_clear()