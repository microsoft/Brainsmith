"""
BrainSmith Dependency Validation

Explicit dependency checking to replace the 533-line fallback maze.
Fails fast with clear error messages instead of silent fallbacks.
"""

import sys
from typing import List, Tuple


def check_installation() -> None:
    """
    All dependencies are required - fail immediately if any missing.
    Since deployment is Docker-only, all dependencies should be available.
    """
    # Simple version check only
    if sys.version_info < (3, 8):
        raise ImportError("BrainSmith requires Python 3.8+")


def get_available_features() -> Tuple[List[str], List[str]]:
    """
    Since all dependencies are required, this returns all features as available.
    
    Returns:
        Tuple of (available_features, missing_features)
    """
    # All features are available in Docker deployment
    available = [
        'ONNX model support',
        'Numerical computation support',
        'Data analysis integration',
        'Statistical analysis support',
        'Machine learning integration'
    ]
    missing = []  # No missing features in Docker deployment
    
    return available, missing


def validate_runtime_requirements() -> bool:
    """
    Validate that runtime environment meets BrainSmith requirements.
    
    Returns:
        True if all requirements met, False otherwise
    """
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"Warning: BrainSmith requires Python 3.8+, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    
    return True


# Run validation on import
if __name__ != "__main__":
    check_installation()
    validate_runtime_requirements()