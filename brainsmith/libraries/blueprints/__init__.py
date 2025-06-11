"""
BrainSmith Blueprints Library - Registry Dictionary Pattern

Simple, explicit blueprint discovery using registry dictionary.  
No magical filesystem scanning - blueprints explicitly registered.

Main Functions:
- get_blueprint(name): Get blueprint YAML path by name with fail-fast errors
- list_blueprints(): List all available blueprint names

Example Usage:
    from brainsmith.libraries.blueprints import get_blueprint, list_blueprints
    
    # List available blueprints
    blueprints = list_blueprints()  # ['cnn_accelerator', 'mobilenet_accelerator']
    
    # Get specific blueprint
    blueprint_path = get_blueprint('cnn_accelerator')
"""

from typing import List
from pathlib import Path

# Simple registry maps blueprint names to their YAML file paths
AVAILABLE_BLUEPRINTS = {
    "cnn_accelerator": "basic/cnn_accelerator.yaml",
    "mobilenet_accelerator": "advanced/mobilenet_accelerator.yaml",
}

def get_blueprint(name: str) -> str:
    """
    Get blueprint YAML file path by name. Fails fast if not found.
    
    Args:
        name: Blueprint name to retrieve
        
    Returns:
        Absolute path to blueprint YAML file
        
    Raises:
        KeyError: If blueprint not found (with available options)
    """
    if name not in AVAILABLE_BLUEPRINTS:
        available = ", ".join(AVAILABLE_BLUEPRINTS.keys())
        raise KeyError(f"Blueprint '{name}' not found. Available: {available}")
    
    relative_path = AVAILABLE_BLUEPRINTS[name]
    blueprint_path = Path(__file__).parent / relative_path
    
    if not blueprint_path.exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    return str(blueprint_path)

def list_blueprints() -> List[str]:
    """
    List all available blueprint names.
    
    Returns:
        List of blueprint names
    """
    return list(AVAILABLE_BLUEPRINTS.keys())

def load_blueprint_yaml(name: str) -> dict:
    """
    Load blueprint YAML content as dictionary.
    
    Args:
        name: Blueprint name to load
        
    Returns:
        Blueprint configuration dictionary
        
    Raises:
        KeyError: If blueprint not found
        FileNotFoundError: If blueprint file not found
    """
    import yaml
    
    blueprint_path = get_blueprint(name)
    
    with open(blueprint_path, 'r') as f:
        blueprint_data = yaml.safe_load(f)
    
    return blueprint_data

# Legacy compatibility functions - redirect to new implementation
def discover_all_blueprints(rescan=False):
    """Legacy function - returns blueprints as dict for compatibility"""
    blueprints = {}
    for name in list_blueprints():
        blueprints[name] = get_blueprint(name)
    return blueprints

def get_blueprint_by_name(blueprint_name: str):
    """Legacy function - redirect to get_blueprint"""
    try:
        return get_blueprint(blueprint_name)
    except KeyError:
        return None

def find_blueprints_by_category(category: str):
    """Legacy function - simple category filtering"""
    blueprints = []
    for name, relative_path in AVAILABLE_BLUEPRINTS.items():
        if category in relative_path:  # Simple category matching based on path
            blueprints.append(name)
    return blueprints

# Export all public functions and types
__all__ = [
    # New registry functions
    'get_blueprint',
    'list_blueprints',
    'load_blueprint_yaml',
    'AVAILABLE_BLUEPRINTS',
    
    # Legacy compatibility
    'discover_all_blueprints',
    'get_blueprint_by_name',
    'find_blueprints_by_category',
]

# Module metadata
__version__ = "2.0.0"  # Bumped for registry refactoring
__author__ = "BrainSmith Development Team"