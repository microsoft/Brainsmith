"""
Kernels Library Implementation

Provides organized access to existing custom_op/ functionality
through the standard library interface.
"""

import os
import glob
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from ..base import BaseLibrary, LibraryComponent
from ..base.exceptions import LibraryInitializationError, ComponentNotFoundError
from .registry import KernelRegistry, discover_kernels
from .mapping import ParameterMapper


class KernelComponent(LibraryComponent):
    """Represents a single kernel implementation."""
    
    def __init__(self, name: str, kernel_path: str, kernel_info: Dict[str, Any]):
        """
        Initialize kernel component.
        
        Args:
            name: Kernel name
            kernel_path: Path to kernel implementation
            kernel_info: Kernel metadata and parameters
        """
        super().__init__(name, "kernel")
        self.kernel_path = kernel_path
        self.kernel_info = kernel_info
        self.parameters = kernel_info.get('parameters', {})
        
        # Extract metadata from kernel info
        self.metadata = {
            'path': kernel_path,
            'type': kernel_info.get('type', 'unknown'),
            'description': kernel_info.get('description', ''),
            'supported_precisions': kernel_info.get('supported_precisions', ['int8']),
            'pe_parallelism': kernel_info.get('pe_parallelism', [1, 2, 4, 8, 16]),
            'simd_parallelism': kernel_info.get('simd_parallelism', [1, 2, 4, 8, 16])
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get kernel parameters."""
        return {
            'pe': {
                'type': 'integer',
                'values': self.metadata['pe_parallelism'],
                'default': 1,
                'description': 'Processing element parallelism'
            },
            'simd': {
                'type': 'integer', 
                'values': self.metadata['simd_parallelism'],
                'default': 1,
                'description': 'SIMD parallelism'
            },
            'precision': {
                'type': 'categorical',
                'values': self.metadata['supported_precisions'],
                'default': 'int8',
                'description': 'Data precision'
            }
        }
    
    def execute(self, inputs: Dict[str, Any], 
                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute kernel operation.
        
        For Week 2, this returns metadata about the kernel configuration.
        """
        return {
            'kernel_name': self.name,
            'kernel_path': self.kernel_path,
            'parameters': parameters,
            'estimated_resources': self._estimate_resources(parameters),
            'status': 'configured'
        }
    
    def _estimate_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource usage based on parameters."""
        pe = parameters.get('pe', 1)
        simd = parameters.get('simd', 1)
        
        # Simple resource estimation
        base_luts = 100
        base_ffs = 150
        base_brams = 1
        
        return {
            'luts': base_luts * pe * simd,
            'ffs': base_ffs * pe * simd,
            'brams': base_brams * max(1, pe // 4),
            'dsps': pe if 'int' in parameters.get('precision', 'int8') else pe * 2
        }


class KernelsLibrary(BaseLibrary):
    """
    Kernels Library Implementation
    
    Organizes and provides access to existing custom_op/ functionality.
    """
    
    def __init__(self, name: str = "kernels"):
        """Initialize kernels library."""
        super().__init__(name, "1.0.0")
        self.registry = KernelRegistry()
        self.parameter_mapper = ParameterMapper()
        self.kernels: Dict[str, KernelComponent] = {}
        
        # Default paths to search for kernels
        self.search_paths = [
            "custom_op/",
            "../finn/custom_op/",
            os.path.expanduser("~/finn/custom_op/")
        ]
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """
        Initialize kernels library.
        
        Args:
            config: Configuration with optional search_paths
            
        Returns:
            True if initialization successful
        """
        try:
            config = config or {}
            
            # Update search paths if provided
            if 'search_paths' in config:
                self.search_paths = config['search_paths']
            
            # Discover kernels from existing custom_op/ functionality
            discovered_kernels = discover_kernels(self.search_paths)
            
            # Register discovered kernels
            for kernel_name, kernel_info in discovered_kernels.items():
                kernel_component = KernelComponent(
                    kernel_name, 
                    kernel_info['path'],
                    kernel_info
                )
                self.kernels[kernel_name] = kernel_component
                self.registry.register_kernel(kernel_name, kernel_component)
            
            # Set capabilities based on discovered kernels
            self.capabilities = set(self.kernels.keys())
            self.capabilities.add('parameter_mapping')
            self.capabilities.add('resource_estimation')
            
            self.initialized = True
            self.logger.info(f"Kernels library initialized with {len(self.kernels)} kernels")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize kernels library: {e}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get list of kernel capabilities."""
        return list(self.capabilities)
    
    def get_design_space_parameters(self) -> Dict[str, Any]:
        """
        Get design space parameters for all kernels.
        
        Returns:
            Combined parameter space for all kernels
        """
        parameters = {}
        
        # Common kernel parameters
        parameters['kernels'] = {
            'pe_values': [1, 2, 4, 8, 16, 32],
            'simd_values': [1, 2, 4, 8, 16, 32],
            'precision_options': ['int8', 'int16', 'int32'],
            'available_kernels': list(self.kernels.keys())
        }
        
        # Individual kernel parameters
        for kernel_name, kernel in self.kernels.items():
            parameters[f'kernel_{kernel_name}'] = kernel.get_parameters()
        
        return parameters
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate kernel parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        kernel_params = parameters.get('kernels', {})
        
        # Validate PE values
        pe = kernel_params.get('pe', 1)
        if not isinstance(pe, int) or pe <= 0:
            errors.append("PE must be a positive integer")
        
        # Validate SIMD values
        simd = kernel_params.get('simd', 1) 
        if not isinstance(simd, int) or simd <= 0:
            errors.append("SIMD must be a positive integer")
        
        # Validate precision
        precision = kernel_params.get('precision', 'int8')
        valid_precisions = ['int8', 'int16', 'int32']
        if precision not in valid_precisions:
            errors.append(f"Precision must be one of {valid_precisions}")
        
        # Validate specific kernel requests
        if 'kernel_selection' in kernel_params:
            requested_kernels = kernel_params['kernel_selection']
            if isinstance(requested_kernels, list):
                for kernel_name in requested_kernels:
                    if kernel_name not in self.kernels:
                        errors.append(f"Kernel '{kernel_name}' not available")
        
        return len(errors) == 0, errors
    
    def execute(self, operation: str, parameters: Dict[str, Any], 
                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute kernel library operation.
        
        Args:
            operation: Operation to execute
            parameters: Operation parameters  
            context: Execution context
            
        Returns:
            Operation results
        """
        context = context or {}
        
        if operation == "get_design_space":
            return self._get_kernel_design_space(parameters)
        
        elif operation == "configure_kernels":
            return self._configure_kernels(parameters, context)
        
        elif operation == "estimate_resources":
            return self._estimate_total_resources(parameters)
        
        elif operation == "list_kernels":
            return self._list_available_kernels()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _get_kernel_design_space(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get design space for kernel exploration."""
        design_space = {
            'total_kernels': len(self.kernels),
            'available_kernels': list(self.kernels.keys()),
            'parameter_ranges': self.get_design_space_parameters(),
            'source': 'existing_custom_op_functionality'
        }
        
        return design_space
    
    def _configure_kernels(self, parameters: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Configure kernels with given parameters."""
        kernel_config = parameters.get('kernels', {})
        
        results = {
            'configured_kernels': [],
            'total_resources': {'luts': 0, 'ffs': 0, 'brams': 0, 'dsps': 0},
            'status': 'success'
        }
        
        # Configure each requested kernel or all kernels
        kernels_to_configure = kernel_config.get('kernel_selection', list(self.kernels.keys()))
        
        for kernel_name in kernels_to_configure:
            if kernel_name in self.kernels:
                kernel = self.kernels[kernel_name]
                kernel_result = kernel.execute({}, kernel_config)
                results['configured_kernels'].append(kernel_result)
                
                # Accumulate resources
                resources = kernel_result.get('estimated_resources', {})
                for resource, value in resources.items():
                    if resource in results['total_resources']:
                        results['total_resources'][resource] += value
        
        return results
    
    def _estimate_total_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate total resource usage."""
        kernel_config = parameters.get('kernels', {})
        
        total_resources = {'luts': 0, 'ffs': 0, 'brams': 0, 'dsps': 0}
        
        for kernel_name, kernel in self.kernels.items():
            resources = kernel._estimate_resources(kernel_config)
            for resource, value in resources.items():
                total_resources[resource] += value
        
        return {
            'total_resources': total_resources,
            'kernel_count': len(self.kernels),
            'estimation_method': 'linear_scaling'
        }
    
    def _list_available_kernels(self) -> Dict[str, Any]:
        """List all available kernels."""
        kernel_info = {}
        
        for kernel_name, kernel in self.kernels.items():
            kernel_info[kernel_name] = kernel.get_info()
        
        return {
            'kernels': kernel_info,
            'total_count': len(self.kernels),
            'source': 'existing_custom_op'
        }
    
    def get_kernel(self, name: str) -> Optional[KernelComponent]:
        """Get a specific kernel by name."""
        return self.kernels.get(name)
    
    def list_kernels(self) -> List[str]:
        """List all available kernel names."""
        return list(self.kernels.keys())