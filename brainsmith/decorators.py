"""
BrainSmith Decorators - REMOVED

This module has been removed as part of the unified plugin system.
All decorator functionality has been moved to the unified system.
"""

raise ImportError(
    "brainsmith.decorators has been removed in favor of the unified plugin system.\n"
    "\n"
    "Use instead:\n"
    "  from brainsmith.plugin.decorators import transform, kernel, backend\n"
    "  # or use the unified decorator:\n"
    "  from brainsmith.plugin.decorators import plugin\n"
    "\n"
    "Migration examples:\n"
    "  # Before:\n"
    "  from brainsmith.decorators import transform\n"
    "  @transform(name='MyTransform', stage='topology_opt')\n"
    "  \n"
    "  # After:\n"
    "  from brainsmith.plugin.decorators import transform\n"
    "  @transform(name='MyTransform', stage='topology_opt')\n"
    "  \n"
    "  # Or use unified decorator:\n"
    "  from brainsmith.plugin.decorators import plugin\n"
    "  @plugin(type='transform', name='MyTransform', stage='topology_opt')\n"
    "\n"
    "See: brainsmith.plugin.decorators for the new unified decorator system."
)