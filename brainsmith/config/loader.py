"""Configuration loading and management for Brainsmith."""

import os
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from pydantic import ValidationError

from .schema import BrainsmithConfig

console = Console()


def load_config(
    project_file: Optional[Path] = None,
    **cli_overrides
) -> BrainsmithConfig:
    """Load configuration with pydantic-settings priority resolution.
    
    Priority order (highest to lowest):
    1. CLI arguments (passed as kwargs)
    2. Environment variables (BSMITH_* prefix)
    3. Project settings file (brainsmith_settings.yaml)
    4. Built-in defaults (from schema Field defaults)
    
    Args:
        project_file: Path to project settings file (for non-standard locations)
        **cli_overrides: CLI argument overrides
        
    Returns:
        Validated BrainsmithConfig object
    """
    try:
        # Pass yaml file path if provided
        if project_file:
            # This will be picked up by YamlSettingsSource
            os.environ['_BRAINSMITH_YAML_FILE'] = str(project_file)
        
        # Create config with CLI overrides
        # Pydantic-settings handles the rest automatically
        config = BrainsmithConfig(**cli_overrides)
        
        # Clean up temp env var
        os.environ.pop('_BRAINSMITH_YAML_FILE', None)
        
        return config
        
    except ValidationError as e:
        console.print("[bold red]Configuration validation failed:[/bold red]")
        for error in e.errors():
            field = " → ".join(str(x) for x in error["loc"])
            console.print(f"  [red]{field}: {error['msg']}[/red]")
        raise


def validate_and_report(config: BrainsmithConfig, raise_on_error: bool = True) -> bool:
    """Validate configuration and report issues based on priority.
    
    Args:
        config: Configuration to validate
        raise_on_error: Whether to raise exception on core errors
        
    Returns:
        True if validation passed, False if there were errors
    """
    results = config.validate_by_priority()
    
    # Report errors
    if results["errors"]:
        console.print("\n[bold red]Configuration Errors:[/bold red]")
        for error in results["errors"]:
            console.print(f"  ❌ {error}")
        if raise_on_error:
            raise ValueError("Configuration validation failed")
    
    # Report warnings
    if results["warnings"]:
        console.print("\n[bold yellow]Configuration Warnings:[/bold yellow]")
        for warning in results["warnings"]:
            console.print(f"  ⚠️  {warning}")
    
    # Report info (only if verbose)
    if results["info"] and os.environ.get("BSMITH_DEBUG") == "1":
        console.print("\n[bold blue]Configuration Info:[/bold blue]")
        for info in results["info"]:
            console.print(f"  ℹ️  {info}")
    
    return len(results["errors"]) == 0


# Singleton instance
_config: Optional[BrainsmithConfig] = None


def get_config() -> BrainsmithConfig:
    """Get singleton configuration instance.
    
    This loads the configuration once and caches it for the session.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    """Reset the singleton configuration (mainly for testing)."""
    global _config
    _config = None