"""
BrainSmith Dependency Validation

Explicit dependency checking to replace the 533-line fallback maze.
Fails fast with clear error messages instead of silent fallbacks.
"""

import sys
from typing import List, Tuple


def check_installation() -> None:
    """
    Fast dependency check on import - fail immediately if critical deps missing.
    
    Raises:
        ImportError: If any critical dependencies are missing with install instructions
    """
    missing_critical = []
    missing_optional = []
    
    # Critical dependencies - required for core functionality
    critical_deps = [
        ('yaml', 'pyyaml', 'YAML configuration parsing'),
        ('pathlib', None, 'Path handling (built-in Python 3.4+)')
    ]
    
    # Optional dependencies - enhance functionality but not required
    optional_deps = [
        ('onnx', 'onnx', 'ONNX model loading'),
        ('numpy', 'numpy', 'Numerical computations'),
        ('pandas', 'pandas', 'Data analysis integration'),
        ('scipy', 'scipy', 'Scientific computing integration')
    ]
    
    # Check critical dependencies
    for module_name, install_name, description in critical_deps:
        try:
            __import__(module_name)
        except ImportError:
            missing_critical.append((install_name or module_name, description))
    
    # Check optional dependencies
    for module_name, install_name, description in optional_deps:
        try:
            __import__(module_name)
        except ImportError:
            missing_optional.append((install_name or module_name, description))
    
    # Fail fast if critical dependencies missing
    if missing_critical:
        install_cmd = " ".join([dep[0] for dep in missing_critical])
        error_msg = f"""
BrainSmith Critical Dependencies Missing:

Missing packages:
{chr(10).join(f"  • {name}: {desc}" for name, desc in missing_critical)}

Install with:
  pip install {install_cmd}

Or install all dependencies:
  pip install brainsmith[all]
"""
        raise ImportError(error_msg.strip())
    
    # Log optional dependencies if missing (but don't fail)
    if missing_optional:
        print(f"BrainSmith: {len(missing_optional)} optional dependencies missing (enhanced features disabled)")


def get_available_features() -> Tuple[List[str], List[str]]:
    """
    Get lists of available and missing features.
    
    Returns:
        Tuple of (available_features, missing_features)
    """
    available = []
    missing = []
    
    features = [
        ('onnx', 'ONNX model support'),
        ('numpy', 'Numerical computation support'),
        ('pandas', 'Data analysis integration'),
        ('scipy', 'Statistical analysis support'),
        ('sklearn', 'Machine learning integration')
    ]
    
    for module_name, feature_desc in features:
        try:
            __import__(module_name)
            available.append(feature_desc)
        except ImportError:
            missing.append(feature_desc)
    
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