"""
Kernel Registry System

Auto-discovery and management of kernel packages in the BrainSmith libraries.
Provides registration, caching, and lookup functionality for kernel collections.

BREAKING CHANGE: Now uses unified BaseRegistry interface with standardized method names.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from brainsmith.core.registry import BaseRegistry, ComponentInfo
from .types import KernelPackage, OperatorType, BackendType

logger = logging.getLogger(__name__)


class KernelRegistry(BaseRegistry[KernelPackage]):
    """Registry for auto-discovery and management of kernel packages."""
    
    def __init__(self, search_dirs: Optional[List[str]] = None, config_manager=None):
        """
        Initialize kernel registry.
        
        Args:
            search_dirs: List of directories to search for kernels.
                        If None, uses default kernel directories.
            config_manager: Optional configuration manager.
        """
        super().__init__(search_dirs, config_manager)
        # For backward compatibility, maintain kernel_cache reference
        self.kernel_cache = self._cache
        self.metadata_cache = self._metadata_cache
    
    def discover_components(self, rescan: bool = False) -> Dict[str, KernelPackage]:
        """
        Discover all available kernel packages.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping kernel names to KernelPackage objects
        """
        if not rescan and self._cache:
            return self._cache
        
        discovered = {}
        
        for kernel_dir in self.search_dirs:
            if not os.path.exists(kernel_dir):
                self._log_warning(f"Kernel directory not found: {kernel_dir}")
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
                            self._log_warning(f"Failed to load kernel from {item_path}: {e}")
        
        # Cache the results
        self._cache = discovered
        self.kernel_cache = self._cache  # Maintain backward compatibility reference
        self._last_scan_time = os.path.getmtime(self.search_dirs[0]) if self.search_dirs else 0
        
        self._log_info(f"Discovered {len(discovered)} kernel packages")
        return discovered
    
    def find_components_by_type(self, operator_type: OperatorType) -> List[KernelPackage]:
        """
        Find kernels that support a specific operator type.
        
        Args:
            operator_type: Type of operator to search for
            
        Returns:
            List of matching KernelPackage objects
        """
        kernels = self.discover_components()
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
        kernels = self.discover_components()
        matches = []
        
        for kernel in kernels.values():
            if kernel.backend == backend_type:
                matches.append(kernel)
        
        return matches
    
    def list_operator_types(self) -> Set[OperatorType]:
        """Get set of all supported operator types."""
        kernels = self.discover_components()
        return {OperatorType(kernel.operator_type) for kernel in kernels.values() if kernel.operator_type}
    
    def list_backend_types(self) -> Set[BackendType]:
        """Get set of all available backend types."""
        kernels = self.discover_components()
        return {BackendType(kernel.backend) for kernel in kernels.values() if kernel.backend}
    
    def _get_default_dirs(self) -> List[str]:
        """Get default search directories for kernel registry."""
        current_dir = Path(__file__).parent
        return [str(current_dir)]
    
    def _extract_info(self, component: KernelPackage) -> Dict[str, Any]:
        """Extract standardized info from kernel component."""
        return {
            'name': component.name,
            'type': 'kernel',
            'operator_type': component.operator_type.value if hasattr(component.operator_type, 'value') else str(component.operator_type),
            'backend': component.backend.value if hasattr(component.backend, 'value') else str(component.backend),
            'version': component.version,
            'description': component.description,
            'path': component.package_path,
            'file_count': len(component.files) if component.files else 0,
            'verified': component.validation.get('verified', False) if component.validation else False,
            'last_tested': component.validation.get('last_tested', 'Unknown') if component.validation else 'Unknown'
        }
    
    def _validate_component_implementation(self, component: KernelPackage) -> tuple[bool, List[str]]:
        """Kernel-specific validation logic."""
        errors = []
        
        # Check required files exist
        if component.files:
            for file_type, file_path in component.files.items():
                full_path = os.path.join(component.package_path, file_path)
                if not os.path.exists(full_path):
                    errors.append(f"Missing file: {file_type} -> {file_path}")
        
        # Validate metadata structure
        if not component.name:
            errors.append("Kernel name is required")
        
        if not component.operator_type:
            errors.append("Operator type is required")
        
        if not component.backend:
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
            package_path=kernel_dir
        )


# Global registry instance
_global_registry = None


def get_kernel_registry() -> KernelRegistry:
    """Get the global kernel registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = KernelRegistry()
    return _global_registry


# BREAKING CHANGE: Updated convenience functions to use new unified interface
def discover_all_kernels(rescan: bool = False) -> Dict[str, KernelPackage]:
    """
    Discover all available kernel packages.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping kernel names to KernelPackage objects
    """
    registry = get_kernel_registry()
    return registry.discover_components(rescan)


def get_kernel_by_name(kernel_name: str) -> Optional[KernelPackage]:
    """
    Get a kernel package by name.
    
    Args:
        kernel_name: Name of the kernel
        
    Returns:
        KernelPackage object or None if not found
    """
    registry = get_kernel_registry()
    return registry.get_component(kernel_name)


def find_kernels_for_operator(operator_type: OperatorType) -> List[KernelPackage]:
    """
    Find all kernels that support a specific operator type.
    
    Args:
        operator_type: Type of operator
        
    Returns:
        List of matching KernelPackage objects
    """
    registry = get_kernel_registry()
    return registry.find_components_by_type(operator_type)


def list_available_kernels() -> List[str]:
    """
    Get list of all available kernel names.
    
    Returns:
        List of kernel names
    """
    registry = get_kernel_registry()
    return registry.list_component_names()


def refresh_kernel_registry():
    """Refresh the kernel registry cache."""
    registry = get_kernel_registry()
    registry.refresh_cache()