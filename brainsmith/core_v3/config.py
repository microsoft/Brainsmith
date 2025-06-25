"""
Global configuration management for Brainsmith.

This module provides a hierarchical configuration system with the following
priority order (most specific to most general):
1. Blueprint/project configuration
2. User configuration (~/.brainsmith/config.yaml)
3. Environment variables
4. Default values (embedded in this module)
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


logger = logging.getLogger(__name__)


@dataclass
class BrainsmithConfig:
    """
    Global configuration settings for Brainsmith.
    
    Attributes:
        max_combinations: Maximum allowed design space combinations
        timeout_minutes: Default timeout for DSE jobs in minutes
    """
    max_combinations: int = 100_000  # Default: 100k combinations
    timeout_minutes: int = 60        # Default: 1 hour


def load_config() -> BrainsmithConfig:
    """
    Load configuration from all sources and merge according to priority.
    
    Priority (highest to lowest):
    1. Environment variables
    2. User config file (~/.brainsmith/config.yaml)
    3. Default values
    
    Note: Blueprint-level config is handled separately as it's
    specific to each DSE run.
    
    Returns:
        Merged configuration object
    """
    # Start with defaults
    config = BrainsmithConfig()
    
    # Load user config if it exists
    user_config_path = Path.home() / ".brainsmith" / "config.yaml"
    if user_config_path.exists():
        try:
            with open(user_config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            if user_config:
                # Update with user settings
                if "max_combinations" in user_config:
                    config.max_combinations = int(user_config["max_combinations"])
                    logger.debug(f"Loaded max_combinations from user config: {config.max_combinations}")
                    
                if "timeout_minutes" in user_config:
                    config.timeout_minutes = int(user_config["timeout_minutes"])
                    logger.debug(f"Loaded timeout_minutes from user config: {config.timeout_minutes}")
                    
        except Exception as e:
            logger.warning(f"Failed to load user config from {user_config_path}: {e}")
    
    # Override with environment variables if present
    env_max_combinations = os.environ.get("BRAINSMITH_MAX_COMBINATIONS")
    if env_max_combinations:
        try:
            config.max_combinations = int(env_max_combinations)
            logger.debug(f"Loaded max_combinations from environment: {config.max_combinations}")
        except ValueError:
            logger.warning(f"Invalid BRAINSMITH_MAX_COMBINATIONS value: {env_max_combinations}")
    
    env_timeout = os.environ.get("BRAINSMITH_TIMEOUT_MINUTES")
    if env_timeout:
        try:
            config.timeout_minutes = int(env_timeout)
            logger.debug(f"Loaded timeout_minutes from environment: {config.timeout_minutes}")
        except ValueError:
            logger.warning(f"Invalid BRAINSMITH_TIMEOUT_MINUTES value: {env_timeout}")
    
    logger.info(f"Loaded global config: max_combinations={config.max_combinations}, "
                f"timeout_minutes={config.timeout_minutes}")
    
    return config


# Singleton instance - loaded once on module import
_global_config: Optional[BrainsmithConfig] = None


def get_config() -> BrainsmithConfig:
    """
    Get the global configuration instance.
    
    This loads the configuration on first access and caches it
    for subsequent calls.
    
    Returns:
        Global configuration object
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def reset_config():
    """
    Reset the cached configuration.
    
    This is mainly useful for testing to force a reload of configuration.
    """
    global _global_config
    _global_config = None