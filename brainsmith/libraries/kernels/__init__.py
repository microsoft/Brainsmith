"""
BrainSmith Kernels Library - Registry Dictionary Pattern

Simple, explicit kernel discovery using registry dictionary.
No magical filesystem scanning - components explicitly registered.

Main Functions:
- get_kernel(name): Get kernel package by name with fail-fast errors  
- list_kernels(): List all available kernel names
- get_kernel_files(name): Get file paths for kernel package

Example Usage:
    from brainsmith.libraries.kernels import get_kernel, list_kernels
    
    # List available kernels  
    kernels = list_kernels()  # ['conv2d_hls', 'matmul_rtl']
    
    # Get specific kernel
    conv_kernel = get_kernel('conv2d_hls')
    
    # Get kernel files
    files = get_kernel_files('conv2d_hls')
"""

from .types import KernelPackage
from .package_loader import load_kernel_package
import yaml
from pathlib import Path
from typing import List, Dict

# Simple registry maps kernel names to their package directories
AVAILABLE_KERNELS = {
    "conv2d_hls": "conv2d_hls",
    "matmul_rtl": "matmul_rtl",
}

def get_kernel(name: str) -> KernelPackage:
    """
    Get kernel package by name. Fails fast if not found.
    
    Args:
        name: Kernel name to retrieve
        
    Returns:
        KernelPackage object
        
    Raises:
        KeyError: If kernel not found (with available options)
    """
    if name not in AVAILABLE_KERNELS:
        available = ", ".join(AVAILABLE_KERNELS.keys())
        raise KeyError(f"Kernel '{name}' not found. Available: {available}")
    
    package_dir = AVAILABLE_KERNELS[name]
    return load_kernel_package(package_dir)

def list_kernels() -> List[str]:
    """
    List all available kernel names.
    
    Returns:
        List of kernel names
    """
    return list(AVAILABLE_KERNELS.keys())

def get_kernel_files(name: str) -> Dict[str, str]:
    """
    Get all file paths for a kernel package.
    
    Args:
        name: Kernel name
        
    Returns:
        Dictionary mapping logical file names to absolute paths
        
    Raises:
        KeyError: If kernel not found
    """
    if name not in AVAILABLE_KERNELS:
        available = ", ".join(AVAILABLE_KERNELS.keys())
        raise KeyError(f"Kernel '{name}' not found. Available: {available}")
    
    package_dir = AVAILABLE_KERNELS[name]
    package_path = Path(__file__).parent / package_dir
    manifest_path = package_path / "kernel.yaml"
    
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    
    # Resolve file paths relative to package directory
    files = {}
    for logical_name, filename in manifest.get("files", {}).items():
        files[logical_name] = str(package_path / filename)
    
    return files

# Legacy compatibility imports - keep existing function names working
from .functions import (
    find_compatible_kernels,
    optimize_kernel_parameters,
    select_optimal_kernel,
    validate_kernel_package,
    install_kernel_library,
    generate_finn_config
)

# Legacy compatibility functions - redirect to new implementation
def discover_all_kernels(additional_paths=None):
    """Legacy function - returns kernels as dict for compatibility"""
    kernels = {}
    for name in list_kernels():
        kernels[name] = get_kernel(name)
    return kernels

def get_kernel_by_name(kernel_name: str):
    """Legacy function - redirect to get_kernel"""
    try:
        return get_kernel(kernel_name)
    except KeyError:
        return None

# Import data types
from .types import (
    KernelPackage,
    ValidationResult,
    KernelRequirements,
    KernelSelection,
    OperatorType,
    BackendType
)

# Export all public functions and types
__all__ = [
    # New registry functions
    'get_kernel',
    'list_kernels',
    'get_kernel_files',
    'AVAILABLE_KERNELS',
    
    # Legacy compatibility
    'discover_all_kernels',
    'get_kernel_by_name',
    
    # Core functions
    'find_compatible_kernels',
    'optimize_kernel_parameters', 
    'select_optimal_kernel',
    'validate_kernel_package',
    'install_kernel_library',
    'generate_finn_config',
    
    # Data types
    'KernelPackage',
    'ValidationResult',
    'KernelRequirements',
    'KernelSelection',
    'OperatorType',
    'BackendType'
]

# Module metadata
__version__ = "2.0.0"  # Bumped for registry refactoring
__author__ = "BrainSmith Development Team"