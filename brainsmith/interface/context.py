"""Streamlined context management for Brainsmith dual CLI system."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import click

logger = logging.getLogger(__name__)

from brainsmith.config import BrainsmithConfig, load_config, get_default_config
from brainsmith.utils import load_yaml, dump_yaml


@dataclass
class ApplicationContext:
    """Manages CLI settings and configuration.
    
    Handles the settings hierarchy:
    1. Command-line arguments (highest priority)
    2. Environment variables
    3. Project configuration (./brainsmith_settings.yaml)
    4. User configuration (~/.brainsmith/config.yaml)
    5. Built-in defaults (lowest priority)
    """
    
    # Core settings
    verbose: bool = False
    debug: bool = False
    config_file: Optional[Path] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Loaded configuration
    config: Optional[BrainsmithConfig] = None
    
    # User config path
    user_config_path: Path = field(default_factory=lambda: Path.home() / ".brainsmith" / "config.yaml")
    
    def load_configuration(self) -> None:
        """Load configuration with overrides."""
        logger.debug(f"Loading configuration from project_file={self.config_file}")
        
        # Load with user config support
        self.config = load_config(
            project_file=self.config_file,
            user_file=self.user_config_path if self.user_config_path.exists() else None
        )
        
        # Apply CLI overrides
        if self.overrides:
            config_dict = self.config.model_dump()
            
            for key, value in self.overrides.items():
                # Handle nested keys (e.g., "finn.num_workers")
                parts = key.split('.')
                target = config_dict
                
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                
                target[parts[-1]] = value
            
            # Recreate config with overrides
            self.config = BrainsmithConfig(**config_dict)
        
        # Apply verbose/debug from context
        if self.verbose:
            self.config.verbose = True
        if self.debug:
            self.config.debug = True
    
    def get_effective_config(self) -> BrainsmithConfig:
        """Get configuration, loading if needed."""
        if not self.config:
            self.load_configuration()
        return self.config
    
    def set_user_config_value(self, key: str, value: Any) -> None:
        """Set a value in user configuration."""
        # Load existing user config
        data = {}
        if self.user_config_path.exists():
            data = load_yaml(self.user_config_path, expand_env_vars=False)
        
        # Handle nested keys
        parts = key.split('.')
        target = data
        
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        target[parts[-1]] = value
        
        # Save user config
        dump_yaml(data, self.user_config_path, sort_keys=True)
        
        # Reload configuration to reflect changes
        self.load_configuration()
    
    def export_environment(self, shell: str = "bash") -> str:
        """Export configuration as shell environment."""
        if not self.config:
            return ""
        
        # Get all environment variables
        env_vars = self.config.export_to_environment(verbose=False, export=False)
        
        lines = []
        if shell in ["bash", "zsh", "sh"]:
            for key, value in env_vars.items():
                # Properly escape values
                escaped_value = str(value).replace("'", "'\"'\"'")
                lines.append(f"export {key}='{escaped_value}'")
        elif shell == "fish":
            for key, value in env_vars.items():
                lines.append(f"set -x {key} '{value}'")
        elif shell == "powershell":
            for key, value in env_vars.items():
                lines.append(f"$env:{key} = '{value}'")
        else:
            raise ValueError(f"Unsupported shell: {shell}")
        
        return "\n".join(lines)


def get_context_from_parent(ctx: click.Context) -> Optional[ApplicationContext]:
    """Get ApplicationContext from parent command if available."""
    while ctx:
        if hasattr(ctx, 'obj') and isinstance(ctx.obj, ApplicationContext):
            return ctx.obj
        ctx = ctx.parent
    return None