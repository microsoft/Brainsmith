"""
Kernel Registry System

Auto-discovery and management of kernel packages in the BrainSmith libraries.
Provides registration, caching, and lookup functionality for kernel collections.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
from .types import KernelPackage, OperatorType, BackendType

logger = logging.getLogger(__name__)


class KernelRegistry:
    """Registry for auto-discovery and management of kernel packages."""
    
    def __init__(self, kernel_dirs: Optional[List[str]] = None):
        """
        Initialize kernel registry.
        
        Args:
            kernel_dirs: List of directories to search for kernels.
                        Defaults to current kernels directory
        """
        if kernel_dirs is None:
            # Default to the current kernels directory
            current_dir = Path(__file__).parent
            kernel_dirs = [str(current_dir)]
        
        self.kernel_dirs = kernel_dirs
        self.kernel_cache = {}
        self.metadata_cache = {}
        self._last_scan_time = 0
        
        logger.info(f"Kernel registry initialized with dirs: {self.kernel_dirs}")
    
    def discover_kernels(self, rescan: bool = False) -> Dict[str, KernelPackage]:
        """
        Discover all available kernel packages.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping kernel names to KernelPackage objects
        """
        if self.kernel_cache and not rescan:
            return self.kernel_cache
        
        discovered = {}
        
        for kernel_dir in self.kernel_dirs:
            if not os.path.exists(kernel_dir):
                logger.warning(f"Kernel directory not found: {kernel_dir}")
                continue
            
            # Look for subdirectories containing kernel.yaml files
            for item in os.listdir(kernel_dir):
                item_path = os.path.join(kernel_dir, item)
                
                if os.path.isdir(item_path):
                    kernel_yaml_path = os.path.join(item_path, "kernel.yaml")
                    
                    if os.path.exists(kernel_yaml_path):
                        try:
                            kernel_package = self._load_kernel_package(kernel_yaml_path, item_path)
                            discovered[kernel_package.name] = kernel_package
                            logger.debug(f"Discovered kernel: {kernel_package.name}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to load kernel from {item_path}: {e}")
        
        # Cache the results
        self.kernel_cache = discovered
        self._last_scan_time = os.path.getmtime(self.kernel_dirs[0]) if self.kernel_dirs else 0
        
        logger.info(f"Discovered {len(discovered)} kernel packages")
        return discovered
    
    def get_kernel(self, kernel_name: str) -> Optional[KernelPackage]:
        """
        Get a specific kernel package by name.
        
        Args:
            kernel_name: Name of the kernel package
            
        Returns:
            KernelPackage object or None if not found
        """
        kernels = self.discover_kernels()
        return kernels.get(kernel_name)
    
    def find_kernels_by_operator(self, operator_type: OperatorType) -> List[KernelPackage]:
        """
        Find kernels that support a specific operator type.
        
        Args:
            operator_type: Type of operator to search for
            
        Returns:
            List of matching KernelPackage objects
        """
        kernels = self.discover_kernels()
        matches = []
        
        for kernel in kernels.values():
            if kernel.operator_type == operator_type:
                matches.append(kernel)
        
        return matches
    
    def find_kernels_by_backend(self, backend_type: BackendType) -> List[KernelPackage]:
        """
        Find kernels that use a specific backend.
        
        Args:
            backend_type: Type of backend to search for
            
        Returns:
            List of matching KernelPackage objects
        """
        kernels = self.discover_kernels()
        matches = []
        
        for kernel in kernels.values():
            if kernel.backend == backend_type:
                matches.append(kernel)
        
        return matches
    
    def list_kernel_names(self) -> List[str]:
        """Get list of all available kernel names."""
        kernels = self.discover_kernels()
        return list(kernels.keys())
    
    def list_operator_types(self) -> Set[OperatorType]:
        """Get set of all supported operator types."""
        kernels = self.discover_kernels()
        return {kernel.operator_type for kernel in kernels.values()}
    
    def list_backend_types(self) -> Set[BackendType]:
        """Get set of all available backend types."""
        kernels = self.discover_kernels()
        return {kernel.backend for kernel in kernels.values()}
    
    def get_kernel_info(self, kernel_name: str) -> Optional[Dict]:
        """
        Get summary information about a kernel.
        
        Args:
            kernel_name: Name of the kernel
            
        Returns:
            Dictionary with kernel summary or None if not found
        """
        kernel = self.get_kernel(kernel_name)
        if not kernel:
            return None
        
        return {
            'name': kernel.name,
            'operator_type': kernel.operator_type,
            'backend': kernel.backend,
            'version': kernel.version,
            'description': kernel.description,
            'path': kernel.path,
            'file_count': len(kernel.files),
            'verified': kernel.validation.get('verified', False),
            'last_tested': kernel.validation.get('last_tested', 'Unknown')
        }
    
    def refresh_cache(self):
        """Refresh the kernel cache by clearing it."""
        self.kernel_cache.clear()
        self.metadata_cache.clear()
        logger.info("Kernel registry cache refreshed")
    
    def validate_kernel(self, kernel_name: str) -> tuple[bool, List[str]]:
        """
        Validate a kernel package.
        
        Args:
            kernel_name: Name of the kernel to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        kernel = self.get_kernel(kernel_name)
        if not kernel:
            return False, [f"Kernel '{kernel_name}' not found"]
        
        errors = []
        
        # Check required files exist
        for file_type, file_path in kernel.files.items():
            full_path = os.path.join(kernel.path, file_path)
            if not os.path.exists(full_path):
                errors.append(f"Missing file: {file_type} -> {file_path}")
        
        # Validate metadata structure
        if not kernel.name:
            errors.append("Kernel name is required")
        
        if not kernel.operator_type:
            errors.append("Operator type is required")
        
        if not kernel.backend:
            errors.append("Backend type is required")
        
        return len(errors) == 0, errors
    
    def _load_kernel_package(self, yaml_path: str, kernel_dir: str) -> KernelPackage:
        """
        Load a kernel package from a YAML file.
        
        Args:
            yaml_path: Path to kernel.yaml file
            kernel_dir: Directory containing the kernel
            
        Returns:
            KernelPackage object
        """
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Convert string types to enums
        operator_type = OperatorType(metadata.get('operator_type', 'Unknown'))
        backend_type = BackendType(metadata.get('backend', 'Unknown'))
        
        return KernelPackage(
            name=metadata.get('name', ''),
            operator_type=operator_type,
            backend=backend_type,
            version=metadata.get('version', '1.0.0'),
            author=metadata.get('author', ''),
            license=metadata.get('license', ''),
            description=metadata.get('description', ''),
            parameters=metadata.get('parameters', {}),
            files=metadata.get('files', {}),
            performance=metadata.get('performance', {}),
            validation=metadata.get('validation', {}),
            repository=metadata.get('repository', {}),
            path=kernel_dir
        )


# Global registry instance
_global_registry = None


def get_kernel_registry() -> KernelRegistry:
    """Get the global kernel registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = KernelRegistry()
    return _global_registry


# Convenience functions for common operations
def discover_all_kernels(rescan: bool = False) -> Dict[str, KernelPackage]:
    """
    Discover all available kernel packages.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping kernel names to KernelPackage objects
    """
    registry = get_kernel_registry()
    return registry.discover_kernels(rescan)


def get_kernel_by_name(kernel_name: str) -> Optional[KernelPackage]:
    """
    Get a kernel package by name.
    
    Args:
        kernel_name: Name of the kernel
        
    Returns:
        KernelPackage object or None if not found
    """
    registry = get_kernel_registry()
    return registry.get_kernel(kernel_name)


def find_kernels_for_operator(operator_type: OperatorType) -> List[KernelPackage]:
    """
    Find all kernels that support a specific operator type.
    
    Args:
        operator_type: Type of operator
        
    Returns:
        List of matching KernelPackage objects
    """
    registry = get_kernel_registry()
    return registry.find_kernels_by_operator(operator_type)


def list_available_kernels() -> List[str]:
    """
    Get list of all available kernel names.
    
    Returns:
        List of kernel names
    """
    registry = get_kernel_registry()
    return registry.list_kernel_names()


def refresh_kernel_registry():
    """Refresh the kernel registry cache."""
    registry = get_kernel_registry()
    registry.refresh_cache()