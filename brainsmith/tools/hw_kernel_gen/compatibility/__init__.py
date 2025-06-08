"""
Compatibility layer for legacy generator integration.

This module provides adapters and utilities for integrating legacy generators
with the new Week 3 orchestration architecture while maintaining backward
compatibility.
"""

from .legacy_adapter import (
    LegacyGeneratorAdapter,
    HWCustomOpLegacyAdapter,
    RTLTemplateLegacyAdapter,
    LegacyGeneratorFactory,
    create_legacy_adapter,
    # Aliases for backward compatibility
    HWCustomOpAdapter,
    RTLTemplateAdapter
)

__all__ = [
    "LegacyGeneratorAdapter",
    "HWCustomOpLegacyAdapter", 
    "RTLTemplateLegacyAdapter",
    "LegacyGeneratorFactory",
    "create_legacy_adapter",
    "HWCustomOpAdapter",
    "RTLTemplateAdapter"
]