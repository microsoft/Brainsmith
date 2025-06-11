"""
Simplified FINN Integration Module

Clean, simple interface for FINN dataflow accelerator builds.
Wraps core functionality and prepares for future 4-hooks interface.

This module provides a dramatically simplified interface compared to the previous
enterprise-grade orchestration framework, focusing on essential functionality
while maintaining preparation for FINN's future 4-hooks interface evolution.
"""

from .interface import FINNInterface, build_accelerator, validate_finn_config, prepare_4hooks_config
from .types import FINNConfig, FINNResult, FINNHooksConfig

# Version information
__version__ = "2.0.0"  # Major version bump for clean refactor
__author__ = "BrainSmith Development Team"

# Clean exports - only essentials
__all__ = [
    # Main interface
    'FINNInterface',
    'build_accelerator', 
    'validate_finn_config',
    'prepare_4hooks_config',
    
    # Essential types
    'FINNConfig',
    'FINNResult', 
    'FINNHooksConfig'
]

# Module information
MODULE_INFO = {
    'name': 'Simplified FINN Integration',
    'version': __version__,
    'description': 'Clean FINN interface with 4-hooks preparation',
    'features': [
        'Simple function-based API',
        'Core FINN integration',
        '4-hooks preparation',
        'Clean configuration management'
    ],
    'files': 3,
    'lines_of_code': '~300',
    'complexity': 'Simple',
    'reduction_from_v1': {
        'files': '70% reduction (10 → 3)',
        'lines': '93% reduction (~4500 → ~300)',
        'exports': '70% reduction (20+ → 7)'
    }
}


# Convenience function for quick access to module info
def get_module_info() -> dict:
    """Get information about the simplified FINN module."""
    return MODULE_INFO.copy()


# Legacy compatibility note (no actual compatibility provided)
_MIGRATION_NOTE = """
This is a complete rewrite of the FINN integration module.
The previous enterprise orchestration framework has been replaced
with a simple, function-based interface.

Key changes:
- Use build_accelerator() instead of FINNIntegrationEngine
- Use FINNConfig instead of complex configuration managers
- Use FINNResult instead of EnhancedFINNResult
- 4-hooks preparation replaces complex orchestration

For migration assistance, see the simplified API documentation.
"""


def show_migration_note():
    """Display migration information for users of the previous interface."""
    print(_MIGRATION_NOTE)