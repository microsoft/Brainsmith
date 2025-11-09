# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Domain matching and derivation utilities for the component registry.

This module provides utilities for bidirectional domain resolution:
1. Module path → ONNX domain (derive_domain_from_module)
2. ONNX domain → Source (match_domain_to_source)
3. Short form expansion (expand_short_form)

These utilities enable hierarchical prefix matching, aligning Brainsmith's
component namespace system with ONNX domain conventions.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Component type to subdomain mapping
_SUBDOMAIN_MAP = {
    'kernel': 'kernels',
    'backend': 'kernels',  # Backends use same domain as kernels
    'step': 'steps',
}


def get_subdomain_for_type(component_type: str) -> Optional[str]:
    """Get subdomain for a component type.

    Args:
        component_type: One of 'kernel', 'backend', 'step'

    Returns:
        Subdomain string or None if type doesn't use subdomains

    Examples:
        >>> get_subdomain_for_type('kernel')
        'kernels'
        >>> get_subdomain_for_type('step')
        'steps'
    """
    return _SUBDOMAIN_MAP.get(component_type)


def match_domain_to_source(domain: str) -> str:
    """Match ONNX domain to source via longest prefix matching.

    Matches against discovered sources (from entrypoints and component_sources).
    Uses longest matching prefix to handle hierarchical domain structures.

    Args:
        domain: ONNX domain string (e.g., "brainsmith.kernels")

    Returns:
        Source name that best matches domain

    Examples:
        >>> match_domain_to_source("brainsmith.kernels")
        'brainsmith'
        >>> match_domain_to_source("finn.custom_op.fpgadataflow.hls")
        'finn'
        >>> match_domain_to_source("my_proj.experimental.kernels")
        'my_proj.experimental'  # If both registered, longer wins
    """
    from brainsmith.registry._state import _discovered_sources

    # Find all matching prefixes from discovered sources
    matches = []
    for source_prefix in _discovered_sources:
        # Exact match or prefix match with dot separator
        if domain == source_prefix or domain.startswith(source_prefix + '.'):
            matches.append((source_prefix, len(source_prefix)))

    if not matches:
        # No match - fall back to source_priority or 'custom'
        logger.debug(f"No source match for domain '{domain}', using fallback")
        try:
            from brainsmith.settings import get_config
            config = get_config()
            if config.source_priority:
                return config.source_priority[0]
        except (ImportError, AttributeError):
            pass
        return 'custom'

    # Return longest match (most specific)
    matches.sort(key=lambda x: x[1], reverse=True)
    matched_source = matches[0][0]

    logger.debug(f"Matched domain '{domain}' to source '{matched_source}'")
    return matched_source


def derive_domain_from_module(module_name: str) -> str:
    """Derive ONNX domain from Python module path.

    Uses pattern matching and component_sources configuration to determine
    the appropriate ONNX domain for a module. Follows conventions:
    - brainsmith.kernels.* → brainsmith.kernels
    - brainsmith.steps.* → brainsmith.steps
    - finn.custom_op.fpgadataflow.hls.* → finn.custom_op.fpgadataflow.hls
    - {source}.{category}.* → {source}.{category}

    Args:
        module_name: Python module path (e.g., "brainsmith.kernels.layernorm.layernorm_hls")

    Returns:
        ONNX domain string (e.g., "brainsmith.kernels")

    Examples:
        >>> derive_domain_from_module("brainsmith.kernels.layernorm.layernorm_hls")
        'brainsmith.kernels'
        >>> derive_domain_from_module("brainsmith.steps.build_dataflow_graph")
        'brainsmith.steps'
        >>> derive_domain_from_module("finn.custom_op.fpgadataflow.hls.mvau_hls")
        'finn.custom_op.fpgadataflow.hls'
    """
    # Pattern 1: brainsmith.kernels.* → brainsmith.kernels
    if module_name.startswith("brainsmith.kernels.") or module_name == "brainsmith.kernels":
        return "brainsmith.kernels"

    # Pattern 2: brainsmith.steps.* → brainsmith.steps
    if module_name.startswith("brainsmith.steps.") or module_name == "brainsmith.steps":
        return "brainsmith.steps"

    # Pattern 3: FINN special case (keep full path to language)
    if module_name.startswith("finn.custom_op.fpgadataflow."):
        parts = module_name.split('.')
        # finn.custom_op.fpgadataflow.hls.X → finn.custom_op.fpgadataflow.hls
        if len(parts) > 4 and parts[3] in ('hls', 'rtl'):
            return '.'.join(parts[:4])
        # finn.custom_op.fpgadataflow.X → finn.custom_op.fpgadataflow
        # Return base domain (first 3 segments)
        return '.'.join(parts[:3])

    # Pattern 4: Generic hierarchical - match against discovered sources
    # Check both component_sources (filesystem) and discovered sources (entrypoints)
    try:
        from brainsmith.registry._state import _discovered_sources

        # Find matching source prefix from all discovered sources
        for source_prefix in _discovered_sources:
            # Check for both standard and unique module name patterns
            # Standard: "acme.kernels.my_kernel"
            # Unique: "acme__custom_ops.my_kernel" (from discovery)
            if module_name.startswith(source_prefix + '.') or \
               module_name.startswith(source_prefix + '__'):

                # Extract category (first segment after prefix)
                if '.' in module_name:
                    # Handle unique module names: "acme__custom_ops" → "custom_ops"
                    if '__' in source_prefix:
                        # For unique names like "project__kernels", use the part after __
                        root_parts = source_prefix.split('__')
                        if len(root_parts) > 1:
                            return f"{source_prefix}.kernels"

                    # Standard case: extract category from path
                    after_prefix_idx = len(source_prefix) + 1
                    if after_prefix_idx < len(module_name):
                        remaining = module_name[after_prefix_idx:]
                        category = remaining.split('.')[0] if '.' in remaining else remaining

                        # Validate category is a known subdomain
                        if category in _SUBDOMAIN_MAP.values():
                            return f"{source_prefix}.{category}"

                # Fallback: source + .kernels
                return f"{source_prefix}.kernels"

    except ImportError:
        # Registry not available (shouldn't happen in normal use)
        logger.debug("Registry state unavailable during domain derivation")

    # Fallback: First two segments or root + .kernels
    parts = module_name.split('.')
    if len(parts) >= 2:
        # Check if second segment is a known category
        if parts[1] in _SUBDOMAIN_MAP.values():
            return '.'.join(parts[:2])

    # Last resort: root segment + .kernels
    return f"{parts[0]}.kernels"


def expand_short_form(name: str, component_type: str) -> str:
    """Expand short form component reference to full domain.

    Short forms like "brainsmith:LayerNorm" are expanded based on component
    type to their full domain form like "brainsmith.kernels:LayerNorm".

    If the name already contains a domain (has dot before colon), it is
    returned unchanged.

    Args:
        name: Component name (short or full form)
        component_type: Component type ('kernel', 'backend', 'step')

    Returns:
        Fully qualified name with domain

    Examples:
        >>> expand_short_form("brainsmith:LayerNorm", "kernel")
        'brainsmith.kernels:LayerNorm'
        >>> expand_short_form("brainsmith:build_dataflow_graph", "step")
        'brainsmith.steps:build_dataflow_graph'
        >>> expand_short_form("brainsmith.kernels:LayerNorm", "kernel")
        'brainsmith.kernels:LayerNorm'  # Already full, unchanged
        >>> expand_short_form("LayerNorm", "kernel")
        'LayerNorm'  # No colon, return as-is
    """
    # Not qualified - return as-is
    if ':' not in name:
        return name

    prefix, component = name.split(':', 1)

    # Already full domain (has dot in prefix)
    if '.' in prefix:
        return name

    # Expand short prefix based on component type
    subdomain = get_subdomain_for_type(component_type)
    if subdomain:
        full_domain = f"{prefix}.{subdomain}"
        expanded = f"{full_domain}:{component}"
        logger.debug(f"Expanded '{name}' to '{expanded}' (type: {component_type})")
        return expanded

    # No subdomain for this type, return unchanged
    return name
