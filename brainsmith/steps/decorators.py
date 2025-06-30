"""
FINN Step Decorator - REMOVED

This module has been removed as part of the unified plugin system.
All step decorator functionality has been moved to the unified system.
"""

raise ImportError(
    "brainsmith.steps.decorators has been removed in favor of the unified plugin system.\n"
    "\n"
    "Use instead:\n"
    "  from brainsmith.plugin.decorators import step\n"
    "  # or use the unified decorator:\n"
    "  from brainsmith.plugin.decorators import plugin\n"
    "\n"
    "Migration examples:\n"
    "  # Before:\n"
    "  from brainsmith.steps.decorators import finn_step\n"
    "  @finn_step(name='my_step', category='metadata')\n"
    "  \n"
    "  # After:\n"
    "  from brainsmith.plugin.decorators import step\n"
    "  @step(name='my_step', category='metadata')\n"
    "  \n"
    "  # Or use unified decorator:\n"
    "  from brainsmith.plugin.decorators import plugin\n"
    "  @plugin(type='step', name='my_step', category='metadata')\n"
    "\n"
    "See: brainsmith.plugin.decorators for the new unified decorator system."
)