"""
Natural Plugin Access Collections - REMOVED

This module has been removed as part of the unified plugin system.
All collections functionality has been moved to the access layer.
"""

raise ImportError(
    "brainsmith.plugin.collections has been removed in favor of the unified plugin system.\n"
    "\n"
    "Use instead:\n"
    "  from brainsmith.plugin.access.transforms import TransformCollection\n"
    "  from brainsmith.plugin.access.kernels import KernelCollection\n"
    "  from brainsmith.plugin.access.wrappers import TransformWrapper, KernelWrapper\n"
    "\n"
    "Or use the global collections:\n"
    "  from brainsmith.plugins import transforms, kernels, steps\n"
    "\n"
    "Migration examples:\n"
    "  # Before:\n"
    "  from brainsmith.plugin.collections import TransformCollection\n"
    "  \n"
    "  # After:\n"
    "  from brainsmith.plugin.access.transforms import TransformCollection\n"
    "  \n"
    "  # Or better yet, use global collections:\n"
    "  from brainsmith.plugins import transforms\n"
    "  model = transforms.ExpandNorms()(model)\n"
    "\n"
    "See: brainsmith.plugin.access for the new access layer."
)