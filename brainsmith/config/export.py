"""Export configuration to environment variables.

DEPRECATED: This module is maintained for backward compatibility only.
Please use config.export_to_environment() directly instead.
"""

from .schema import BrainsmithConfig


def export_to_environment(config: BrainsmithConfig, verbose: bool = False) -> None:
    """Export validated config to environment variables.
    
    DEPRECATED: Use config.export_to_environment() instead.
    This function now delegates to the unified method in BrainsmithConfig.
    
    Args:
        config: Validated configuration object
        verbose: Whether to print export information
    """
    # Delegate to the unified method
    config.export_to_environment(include_internal=False, verbose=verbose)


