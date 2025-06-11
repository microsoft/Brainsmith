"""
Simple kernel management system for BrainSmith.
North Star-aligned: Clean, simple functions for kernel discovery and optimization.

Main Functions:
- discover_all_kernels(): Find all available kernel packages
- find_compatible_kernels(): Find kernels matching requirements  
- select_optimal_kernel(): Select and configure best kernel
- generate_finn_config(): Generate FINN configuration from selections

Example Usage:
    from brainsmith.kernels import discover_all_kernels, select_optimal_kernel, KernelRequirements
    
    # Discover available kernels
    kernels = discover_all_kernels()
    
    # Select optimal kernel for convolution
    requirements = KernelRequirements(operator_type="Convolution", datatype="int8")
    selection = select_optimal_kernel(requirements)
    
    # Generate FINN config
    finn_config = generate_finn_config({"layer1": selection})
"""

# Import core functions
from .functions import (
    discover_all_kernels,
    load_kernel_package,
    find_compatible_kernels,
    get_kernel_files,
    optimize_kernel_parameters,
    select_optimal_kernel,
    validate_kernel_package,
    install_kernel_library,
    generate_finn_config
)

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
    # Core functions
    'discover_all_kernels',
    'load_kernel_package', 
    'find_compatible_kernels',
    'get_kernel_files',
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
__version__ = "1.0.0"
__author__ = "BrainSmith Development Team"