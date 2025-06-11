"""
Simple kernel discovery and management functions.
North Star-aligned: Pure functions for data transformation and discovery.
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .types import KernelPackage, ValidationResult, KernelRequirements, KernelSelection

logger = logging.getLogger(__name__)


def discover_all_kernels(additional_paths: Optional[List[str]] = None) -> Dict[str, KernelPackage]:
    """
    Discover all kernel packages in brainsmith/kernels/ + additional paths.
    
    Args:
        additional_paths: Additional directories to search for kernel packages
        
    Returns:
        Dictionary mapping kernel names to KernelPackage objects
    """
    kernels = {}
    search_paths = []
    
    # Add default kernels directory
    current_dir = Path(__file__).parent
    default_kernels_dir = current_dir
    if default_kernels_dir.exists():
        search_paths.append(str(default_kernels_dir))
    
    # Add additional paths
    if additional_paths:
        search_paths.extend(additional_paths)
    
    logger.info(f"Discovering kernels in {len(search_paths)} paths")
    
    for search_path in search_paths:
        path_kernels = _discover_kernels_in_directory(search_path)
        kernels.update(path_kernels)
        logger.debug(f"Found {len(path_kernels)} kernels in {search_path}")
    
    logger.info(f"Discovered {len(kernels)} total kernel packages")
    return kernels


def _discover_kernels_in_directory(directory: str) -> Dict[str, KernelPackage]:
    """Discover kernel packages in a specific directory"""
    kernels = {}
    
    if not os.path.exists(directory):
        return kernels
    
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            # Skip files, only look at directories
            if not os.path.isdir(item_path):
                continue
            
            # Skip special directories
            if item.startswith('.') or item in ['__pycache__', 'functions.py', 'types.py', '__init__.py']:
                continue
            
            # Look for kernel.yaml in the directory
            kernel_yaml_path = os.path.join(item_path, 'kernel.yaml')
            if os.path.exists(kernel_yaml_path):
                try:
                    kernel = load_kernel_package(item_path)
                    if kernel:
                        kernels[kernel.name] = kernel
                        logger.debug(f"Loaded kernel package: {kernel.name}")
                except Exception as e:
                    logger.warning(f"Failed to load kernel package at {item_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error discovering kernels in {directory}: {e}")
    
    return kernels


def load_kernel_package(package_path: str) -> Optional[KernelPackage]:
    """
    Load a specific kernel package from directory.
    
    Args:
        package_path: Path to kernel package directory
        
    Returns:
        KernelPackage object or None if loading fails
    """
    kernel_yaml_path = os.path.join(package_path, 'kernel.yaml')
    
    if not os.path.exists(kernel_yaml_path):
        logger.warning(f"No kernel.yaml found in {package_path}")
        return None
    
    try:
        with open(kernel_yaml_path, 'r') as f:
            kernel_data = yaml.safe_load(f)
        
        # Create KernelPackage from YAML data
        kernel = KernelPackage(
            name=kernel_data.get('name', os.path.basename(package_path)),
            operator_type=kernel_data.get('operator_type', 'Custom'),
            backend=kernel_data.get('backend', 'RTL'),
            version=kernel_data.get('version', '1.0.0'),
            author=kernel_data.get('author', ''),
            license=kernel_data.get('license', ''),
            description=kernel_data.get('description', ''),
            parameters=kernel_data.get('parameters', {}),
            files=kernel_data.get('files', {}),
            performance=kernel_data.get('performance', {}),
            validation=kernel_data.get('validation', {}),
            repository=kernel_data.get('repository', {}),
            package_path=package_path
        )
        
        return kernel
    
    except Exception as e:
        logger.error(f"Failed to load kernel package from {package_path}: {e}")
        return None


def find_compatible_kernels(requirements: Union[KernelRequirements, Dict[str, Any]], 
                          available_kernels: Optional[Dict[str, KernelPackage]] = None) -> List[str]:
    """
    Find kernels matching requirements.
    
    Args:
        requirements: Kernel requirements specification
        available_kernels: Available kernels (auto-discover if None)
        
    Returns:
        List of compatible kernel names
    """
    if available_kernels is None:
        available_kernels = discover_all_kernels()
    
    # Convert dict to KernelRequirements if needed
    if isinstance(requirements, dict):
        req_dict = requirements
    else:
        req_dict = {
            'operator_type': requirements.operator_type,
            'datatype': requirements.datatype,
            'min_pe': requirements.min_pe,
            'max_pe': requirements.max_pe,
            'min_simd': requirements.min_simd,
            'max_simd': requirements.max_simd,
            'backend_preference': requirements.backend_preference
        }
    
    compatible_kernels = []
    
    for kernel_name, kernel in available_kernels.items():
        if kernel.is_compatible_with(req_dict):
            compatible_kernels.append(kernel_name)
    
    # Sort by preference (verified kernels first, then by version)
    def sort_key(kernel_name: str) -> tuple:
        kernel = available_kernels[kernel_name]
        verified = kernel.validation.get('verified', False)
        version = kernel.version
        return (not verified, version)  # False sorts before True, so verified first
    
    compatible_kernels.sort(key=sort_key)
    
    logger.debug(f"Found {len(compatible_kernels)} compatible kernels for requirements")
    return compatible_kernels


def get_kernel_files(kernel_name: str, 
                    available_kernels: Optional[Dict[str, KernelPackage]] = None) -> Dict[str, str]:
    """
    Get file paths for all components of a kernel.
    
    Args:
        kernel_name: Name of kernel package
        available_kernels: Available kernels (auto-discover if None)
        
    Returns:
        Dictionary mapping logical file names to absolute paths
    """
    if available_kernels is None:
        available_kernels = discover_all_kernels()
    
    if kernel_name not in available_kernels:
        logger.warning(f"Kernel '{kernel_name}' not found")
        return {}
    
    kernel = available_kernels[kernel_name]
    file_paths = {}
    
    for logical_name, relative_path in kernel.files.items():
        absolute_path = kernel.get_file_path(logical_name)
        if absolute_path and os.path.exists(absolute_path):
            file_paths[logical_name] = absolute_path
        else:
            logger.warning(f"File not found: {logical_name} -> {absolute_path}")
    
    return file_paths


def optimize_kernel_parameters(kernel: KernelPackage, 
                             requirements: Dict[str, Any],
                             strategy: str = 'balanced') -> Dict[str, Any]:
    """
    Optimize kernel parameters for given requirements.
    
    Args:
        kernel: Kernel package to optimize
        requirements: Performance/resource requirements
        strategy: Optimization strategy ('throughput', 'latency', 'area', 'balanced')
        
    Returns:
        Optimized parameter configuration
    """
    pe_min, pe_max = kernel.get_pe_range()
    simd_min, simd_max = kernel.get_simd_range()
    
    # Simple parameter optimization based on strategy
    if strategy == 'throughput':
        # Maximize parallelism for throughput
        optimal_pe = pe_max
        optimal_simd = simd_max
        memory_mode = 'internal'  # Faster access
        
    elif strategy == 'latency':
        # High parallelism but focus on pipeline efficiency
        optimal_pe = min(pe_max, 32)  # Cap to avoid routing delays
        optimal_simd = min(simd_max, 16)
        memory_mode = 'internal'
        
    elif strategy == 'area':
        # Minimize resource usage
        optimal_pe = pe_min
        optimal_simd = simd_min
        memory_mode = 'external'  # Save on-chip memory
        
    else:  # balanced
        # Balanced approach
        optimal_pe = (pe_min + pe_max) // 2
        optimal_simd = (simd_min + simd_max) // 2
        memory_mode = 'internal'
    
    # Apply constraints from requirements
    if 'max_pe' in requirements and requirements['max_pe'] is not None:
        optimal_pe = min(optimal_pe, requirements['max_pe'])
    if 'max_simd' in requirements and requirements['max_simd'] is not None:
        optimal_simd = min(optimal_simd, requirements['max_simd'])
    
    return {
        'pe_parallelism': optimal_pe,
        'simd_width': optimal_simd,
        'memory_mode': memory_mode,
        'folding_factors': {},  # Could be enhanced with more sophisticated folding
        'custom_options': {}
    }


def select_optimal_kernel(requirements: Union[KernelRequirements, Dict[str, Any]],
                         strategy: str = 'balanced',
                         available_kernels: Optional[Dict[str, KernelPackage]] = None) -> Optional[KernelSelection]:
    """
    Select and configure optimal kernel for requirements.
    
    Args:
        requirements: Kernel requirements
        strategy: Optimization strategy
        available_kernels: Available kernels (auto-discover if None)
        
    Returns:
        KernelSelection with optimized parameters or None if no suitable kernel
    """
    compatible_kernels = find_compatible_kernels(requirements, available_kernels)
    
    if not compatible_kernels:
        logger.warning("No compatible kernels found for requirements")
        return None
    
    if available_kernels is None:
        available_kernels = discover_all_kernels()
    
    # Select best kernel (first in sorted list is best)
    best_kernel_name = compatible_kernels[0]
    best_kernel = available_kernels[best_kernel_name]
    
    # Convert requirements to dict if needed
    if isinstance(requirements, KernelRequirements):
        req_dict = {
            'max_pe': requirements.max_pe,
            'max_simd': requirements.max_simd,
            'performance_requirements': requirements.performance_requirements,
            'resource_constraints': requirements.resource_constraints
        }
    else:
        req_dict = requirements
    
    # Optimize parameters
    optimal_params = optimize_kernel_parameters(best_kernel, req_dict, strategy)
    
    selection = KernelSelection(
        kernel=best_kernel,
        pe_parallelism=optimal_params['pe_parallelism'],
        simd_width=optimal_params['simd_width'],
        memory_mode=optimal_params['memory_mode'],
        folding_factors=optimal_params['folding_factors'],
        custom_options=optimal_params['custom_options']
    )
    
    logger.info(f"Selected kernel '{best_kernel_name}' with PE={selection.pe_parallelism}, SIMD={selection.simd_width}")
    return selection


def validate_kernel_package(package_path: str) -> ValidationResult:
    """
    Validate kernel package for completeness and correctness.
    
    Args:
        package_path: Path to kernel package directory
        
    Returns:
        ValidationResult with validation status and issues
    """
    result = ValidationResult()
    
    # Check if directory exists
    if not os.path.exists(package_path):
        result.add_error(f"Package directory does not exist: {package_path}")
        return result
    
    if not os.path.isdir(package_path):
        result.add_error(f"Package path is not a directory: {package_path}")
        return result
    
    # Check for kernel.yaml
    kernel_yaml_path = os.path.join(package_path, 'kernel.yaml')
    if not os.path.exists(kernel_yaml_path):
        result.add_error("Missing kernel.yaml manifest file")
        return result
    
    try:
        # Load and validate YAML structure
        with open(kernel_yaml_path, 'r') as f:
            kernel_data = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['name', 'operator_type', 'backend', 'version']
        for field in required_fields:
            if field not in kernel_data:
                result.add_error(f"Missing required field in kernel.yaml: {field}")
        
        # Check files section
        if 'files' in kernel_data:
            for logical_name, relative_path in kernel_data['files'].items():
                file_path = os.path.join(package_path, relative_path)
                if not os.path.exists(file_path):
                    result.add_error(f"Referenced file does not exist: {relative_path}")
        else:
            result.add_warning("No files section in kernel.yaml")
        
        # Check parameters section
        if 'parameters' in kernel_data:
            params = kernel_data['parameters']
            if 'pe_range' in params:
                pe_range = params['pe_range']
                if not isinstance(pe_range, list) or len(pe_range) != 2:
                    result.add_error("pe_range must be a list of two integers")
                elif pe_range[0] > pe_range[1]:
                    result.add_error("pe_range minimum cannot be greater than maximum")
            
            if 'simd_range' in params:
                simd_range = params['simd_range']
                if not isinstance(simd_range, list) or len(simd_range) != 2:
                    result.add_error("simd_range must be a list of two integers")
                elif simd_range[0] > simd_range[1]:
                    result.add_error("simd_range minimum cannot be greater than maximum")
        
        # Success if no errors
        if not result.errors:
            result.is_valid = True
            logger.debug(f"Kernel package validation passed: {package_path}")
    
    except yaml.YAMLError as e:
        result.add_error(f"Invalid YAML syntax in kernel.yaml: {e}")
    except Exception as e:
        result.add_error(f"Error validating kernel package: {e}")
    
    return result


def install_kernel_library(source: str, target_path: Optional[str] = None) -> List[str]:
    """
    Install external kernel library.
    
    Args:
        source: Git URL or local path to kernel library
        target_path: Where to install (auto-determine if None)
        
    Returns:
        List of newly available kernel names
    """
    # Simple implementation - in production could use git clone, etc.
    logger.warning("install_kernel_library is a placeholder - implement git clone or copy logic")
    return []


def generate_finn_config(selections: Dict[str, KernelSelection]) -> Dict[str, Any]:
    """
    Generate FINN configuration from kernel selections.
    
    Args:
        selections: Dictionary mapping layer names to kernel selections
        
    Returns:
        FINN configuration dictionary
    """
    finn_config = {
        'folding_config': {},
        'kernels': {},
        'global_settings': {
            'target_platform': 'zynq',
            'optimization_level': 2
        }
    }
    
    for layer_name, selection in selections.items():
        # Add folding configuration for this layer
        finn_config['folding_config'][layer_name] = selection.to_finn_config()
        
        # Add kernel information
        finn_config['kernels'][layer_name] = {
            'kernel_name': selection.kernel.name,
            'operator_type': selection.kernel.operator_type,
            'backend': selection.kernel.backend,
            'files': selection.kernel.files
        }
    
    return finn_config