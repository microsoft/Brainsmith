"""Shared context management for Brainsmith dual CLI system."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

import click
from pydantic import ValidationError

from brainsmith.config import BrainsmithConfig, load_config, get_default_config


@dataclass
class ApplicationContext:
    """Application context that manages settings across brainsmith and smith commands.
    
    This context handles the settings hierarchy:
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
    
    # Override settings from CLI
    overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Loaded configuration
    config: Optional[BrainsmithConfig] = None
    
    # User config path
    user_config_path: Path = field(default_factory=lambda: Path.home() / ".brainsmith" / "config.yaml")
    
    def __post_init__(self):
        """Initialize the context and load configuration."""
        if self.config is None:
            self.load_configuration()
    
    def load_configuration(self) -> None:
        """Load configuration with proper hierarchy."""
        try:
            # Load with user config support
            self.config = load_config(
                project_file=self.config_file,
                user_file=self.user_config_path if self.user_config_path.exists() else None
            )
            
            # Apply CLI overrides
            if self.overrides:
                self._apply_overrides()
            
            # Apply verbose/debug from context
            if self.verbose:
                self.config.verbose = True
            if self.debug:
                self.config.debug = True
                
        except Exception as e:
            # Fall back to defaults on error
            self.config = get_default_config()
            if self.verbose:
                click.echo(f"Warning: Failed to load configuration: {e}", err=True)
    
    def _apply_overrides(self) -> None:
        """Apply command-line overrides to configuration."""
        if not self.config or not self.overrides:
            return
            
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
        
        try:
            # Recreate config with overrides
            self.config = BrainsmithConfig(**config_dict)
        except ValidationError as e:
            if self.verbose:
                click.echo(f"Warning: Invalid override values: {e}", err=True)
    
    def get_user_config_data(self) -> Dict[str, Any]:
        """Get user configuration data if it exists."""
        if self.user_config_path.exists():
            import yaml
            try:
                with open(self.user_config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}
    
    def save_user_config(self, data: Dict[str, Any]) -> None:
        """Save user configuration data."""
        import yaml
        
        # Ensure directory exists
        self.user_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.user_config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=True)
    
    def set_user_config_value(self, key: str, value: Any) -> None:
        """Set a single value in user configuration."""
        data = self.get_user_config_data()
        
        # Handle nested keys
        parts = key.split('.')
        target = data
        
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        target[parts[-1]] = value
        
        self.save_user_config(data)
        # Reload configuration to reflect changes
        self.load_configuration()
    
    def export_environment(self, shell: str = "bash") -> str:
        """Export configuration as shell environment variables."""
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
    
    def get_effective_config(self) -> BrainsmithConfig:
        """Get the effective configuration after all overrides."""
        if not self.config:
            self.load_configuration()
        return self.config or get_default_config()


def pass_context(f):
    """Decorator to pass ApplicationContext to commands."""
    def new_func(*args, **kwargs):
        return f(ApplicationContext(), *args, **kwargs)
    return click.decorators.update_wrapper(new_func, f)


def get_context_from_parent(ctx: click.Context) -> Optional[ApplicationContext]:
    """Get ApplicationContext from parent command if available."""
    while ctx:
        if hasattr(ctx, 'obj') and isinstance(ctx.obj, ApplicationContext):
            return ctx.obj
        ctx = ctx.parent
    return None