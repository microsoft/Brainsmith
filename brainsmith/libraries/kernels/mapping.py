"""
Parameter mapping utilities for kernels.

Provides mapping between design space parameters and kernel configurations.
"""

from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ParameterMapper:
    """
    Maps design space parameters to kernel-specific configurations.
    
    Handles the translation between high-level design space exploration
    parameters and the specific parameters needed by individual kernels.
    """
    
    def __init__(self):
        """Initialize parameter mapper."""
        self.logger = logging.getLogger("brainsmith.libraries.kernels.mapping")
        
        # Parameter mapping rules
        self.parameter_mappings = {
            'performance': {
                'throughput_target': 'pe_parallelism',
                'latency_target': 'simd_parallelism'
            },
            'resource': {
                'lut_budget': 'resource_constraint',
                'bram_budget': 'memory_constraint'
            },
            'precision': {
                'accuracy_target': 'supported_precisions'
            }
        }
    
    def map_design_space_to_kernel(self, design_params: Dict[str, Any], 
                                  kernel_name: str) -> Dict[str, Any]:
        """
        Map design space parameters to kernel-specific parameters.
        
        Args:
            design_params: High-level design space parameters
            kernel_name: Target kernel name
            
        Returns:
            Kernel-specific parameter configuration
        """
        kernel_params = {}
        
        # Extract kernel-specific parameters
        if 'kernels' in design_params:
            kernel_config = design_params['kernels']
            
            # Map PE parallelism
            if 'pe' in kernel_config:
                kernel_params['pe'] = kernel_config['pe']
            elif 'throughput_target' in design_params:
                kernel_params['pe'] = self._map_throughput_to_pe(
                    design_params['throughput_target'], kernel_name)
            else:
                kernel_params['pe'] = 1
            
            # Map SIMD parallelism
            if 'simd' in kernel_config:
                kernel_params['simd'] = kernel_config['simd']
            elif 'latency_target' in design_params:
                kernel_params['simd'] = self._map_latency_to_simd(
                    design_params['latency_target'], kernel_name)
            else:
                kernel_params['simd'] = 1
            
            # Map precision
            if 'precision' in kernel_config:
                kernel_params['precision'] = kernel_config['precision']
            elif 'accuracy_target' in design_params:
                kernel_params['precision'] = self._map_accuracy_to_precision(
                    design_params['accuracy_target'])
            else:
                kernel_params['precision'] = 'int8'
        
        # Apply resource constraints
        if 'resource_constraints' in design_params:
            kernel_params = self._apply_resource_constraints(
                kernel_params, design_params['resource_constraints'])
        
        self.logger.debug(f"Mapped parameters for {kernel_name}: {kernel_params}")
        return kernel_params
    
    def map_kernel_to_design_space(self, kernel_params: Dict[str, Any], 
                                  kernel_name: str) -> Dict[str, Any]:
        """
        Map kernel parameters back to design space representation.
        
        Args:
            kernel_params: Kernel-specific parameters
            kernel_name: Source kernel name
            
        Returns:
            Design space parameter representation
        """
        design_params = {
            'kernels': {},
            'performance': {},
            'resources': {}
        }
        
        # Map kernel parameters back
        if 'pe' in kernel_params:
            design_params['kernels']['pe'] = kernel_params['pe']
            design_params['performance']['parallelism_pe'] = kernel_params['pe']
        
        if 'simd' in kernel_params:
            design_params['kernels']['simd'] = kernel_params['simd']
            design_params['performance']['parallelism_simd'] = kernel_params['simd']
        
        if 'precision' in kernel_params:
            design_params['kernels']['precision'] = kernel_params['precision']
            design_params['performance']['data_precision'] = kernel_params['precision']
        
        # Estimate performance implications
        design_params['performance']['estimated_throughput'] = self._estimate_throughput(
            kernel_params, kernel_name)
        
        # Estimate resource implications
        design_params['resources']['estimated_resources'] = self._estimate_resources(
            kernel_params, kernel_name)
        
        return design_params
    
    def _map_throughput_to_pe(self, throughput_target: float, kernel_name: str) -> int:
        """Map throughput target to PE parallelism."""
        # Simple heuristic mapping
        if throughput_target < 100:
            return 1
        elif throughput_target < 500:
            return 2
        elif throughput_target < 1000:
            return 4
        elif throughput_target < 2000:
            return 8
        else:
            return 16
    
    def _map_latency_to_simd(self, latency_target: float, kernel_name: str) -> int:
        """Map latency target to SIMD parallelism."""
        # Simple heuristic mapping
        if latency_target > 1000:  # High latency tolerance
            return 1
        elif latency_target > 500:
            return 2
        elif latency_target > 100:
            return 4
        else:
            return 8
    
    def _map_accuracy_to_precision(self, accuracy_target: float) -> str:
        """Map accuracy target to precision."""
        if accuracy_target > 0.99:
            return 'int32'
        elif accuracy_target > 0.95:
            return 'int16'
        else:
            return 'int8'
    
    def _apply_resource_constraints(self, kernel_params: Dict[str, Any], 
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resource constraints to kernel parameters."""
        # Reduce parallelism if resource constraints are tight
        if 'lut_budget' in constraints:
            max_luts = constraints['lut_budget']
            estimated_luts = self._estimate_lut_usage(kernel_params)
            
            if estimated_luts > max_luts:
                # Reduce PE first, then SIMD
                while estimated_luts > max_luts and kernel_params.get('pe', 1) > 1:
                    kernel_params['pe'] = max(1, kernel_params['pe'] // 2)
                    estimated_luts = self._estimate_lut_usage(kernel_params)
                
                while estimated_luts > max_luts and kernel_params.get('simd', 1) > 1:
                    kernel_params['simd'] = max(1, kernel_params['simd'] // 2)
                    estimated_luts = self._estimate_lut_usage(kernel_params)
        
        return kernel_params
    
    def _estimate_throughput(self, kernel_params: Dict[str, Any], kernel_name: str) -> float:
        """Estimate throughput based on kernel parameters."""
        base_throughput = 100  # Base throughput
        pe_factor = kernel_params.get('pe', 1)
        simd_factor = kernel_params.get('simd', 1)
        
        # Simple linear scaling (in practice this would be more complex)
        estimated_throughput = base_throughput * pe_factor * simd_factor
        
        return estimated_throughput
    
    def _estimate_resources(self, kernel_params: Dict[str, Any], kernel_name: str) -> Dict[str, Any]:
        """Estimate resource usage based on kernel parameters."""
        base_luts = 100
        base_ffs = 150
        base_brams = 1
        base_dsps = 1
        
        pe = kernel_params.get('pe', 1)
        simd = kernel_params.get('simd', 1)
        
        return {
            'luts': base_luts * pe * simd,
            'ffs': base_ffs * pe * simd,
            'brams': base_brams * max(1, pe // 4),
            'dsps': base_dsps * pe
        }
    
    def _estimate_lut_usage(self, kernel_params: Dict[str, Any]) -> int:
        """Estimate LUT usage for resource constraint checking."""
        resources = self._estimate_resources(kernel_params, "generic")
        return resources['luts']
    
    def validate_parameter_mapping(self, design_params: Dict[str, Any], 
                                  kernel_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that parameter mapping is consistent.
        
        Args:
            design_params: Original design space parameters
            kernel_params: Mapped kernel parameters
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check PE parameter consistency
        if 'kernels' in design_params and 'pe' in design_params['kernels']:
            expected_pe = design_params['kernels']['pe']
            actual_pe = kernel_params.get('pe', 1)
            if expected_pe != actual_pe:
                errors.append(f"PE mismatch: expected {expected_pe}, got {actual_pe}")
        
        # Check SIMD parameter consistency
        if 'kernels' in design_params and 'simd' in design_params['kernels']:
            expected_simd = design_params['kernels']['simd']
            actual_simd = kernel_params.get('simd', 1)
            if expected_simd != actual_simd:
                errors.append(f"SIMD mismatch: expected {expected_simd}, got {actual_simd}")
        
        # Check precision consistency
        if 'kernels' in design_params and 'precision' in design_params['kernels']:
            expected_precision = design_params['kernels']['precision']
            actual_precision = kernel_params.get('precision', 'int8')
            if expected_precision != actual_precision:
                errors.append(f"Precision mismatch: expected {expected_precision}, got {actual_precision}")
        
        return len(errors) == 0, errors
    
    def get_parameter_ranges(self, kernel_name: str) -> Dict[str, Any]:
        """
        Get valid parameter ranges for a kernel.
        
        Args:
            kernel_name: Kernel name
            
        Returns:
            Dictionary of parameter ranges
        """
        # Default ranges (would be kernel-specific in practice)
        return {
            'pe': {
                'type': 'integer',
                'min': 1,
                'max': 32,
                'values': [1, 2, 4, 8, 16, 32]
            },
            'simd': {
                'type': 'integer',
                'min': 1,
                'max': 16,
                'values': [1, 2, 4, 8, 16]
            },
            'precision': {
                'type': 'categorical',
                'values': ['int8', 'int16', 'int32']
            }
        }