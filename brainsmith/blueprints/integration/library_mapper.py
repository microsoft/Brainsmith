"""
Blueprint to library parameter mapping.

Maps blueprint specifications to library-specific configurations,
integrating with the Week 2 library system.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

from ..core.blueprint import Blueprint

logger = logging.getLogger(__name__)


class LibraryMapper:
    """
    Maps blueprint specifications to library configurations.
    
    Bridges the gap between high-level blueprint specifications
    and the specific parameter formats expected by the Week 2 libraries.
    """
    
    def __init__(self):
        """Initialize library mapper."""
        self.logger = logging.getLogger("brainsmith.blueprints.integration.mapper")
    
    def map_blueprint_to_libraries(self, blueprint: Blueprint) -> Dict[str, Dict[str, Any]]:
        """
        Map blueprint to library configurations.
        
        Args:
            blueprint: Blueprint to map
            
        Returns:
            Dictionary of library configurations
        """
        library_configs = {}
        
        # Map each library configuration
        for lib_name, lib_config in blueprint.libraries.items():
            if lib_name == 'kernels':
                library_configs[lib_name] = self._map_kernels_config(lib_config, blueprint)
            elif lib_name == 'transforms':
                library_configs[lib_name] = self._map_transforms_config(lib_config, blueprint)
            elif lib_name == 'hw_optim':
                library_configs[lib_name] = self._map_hw_optim_config(lib_config, blueprint)
            elif lib_name == 'analysis':
                library_configs[lib_name] = self._map_analysis_config(lib_config, blueprint)
        
        return library_configs
    
    def _map_kernels_config(self, config: Dict[str, Any], blueprint: Blueprint) -> Dict[str, Any]:
        """Map blueprint config to kernels library format."""
        mapped_config = {}
        
        # Map PE configuration
        if 'pe_range' in config:
            mapped_config['pe_values'] = config['pe_range']
        elif 'pe_values' in config:
            mapped_config['pe_values'] = config['pe_values']
        else:
            mapped_config['pe_values'] = [1, 2, 4, 8]  # Default
        
        # Map SIMD configuration
        if 'simd_range' in config:
            mapped_config['simd_values'] = config['simd_range']
        elif 'simd_values' in config:
            mapped_config['simd_values'] = config['simd_values']
        else:
            mapped_config['simd_values'] = [1, 2, 4]  # Default
        
        # Map precision configuration
        if 'precision' in config:
            if isinstance(config['precision'], list):
                mapped_config['precision_options'] = config['precision']
            else:
                mapped_config['precision_options'] = [config['precision']]
        else:
            mapped_config['precision_options'] = ['int8']  # Default
        
        # Add optimization hints
        if 'optimization_hint' in config:
            mapped_config['optimization_hint'] = config['optimization_hint']
        
        # Apply resource constraints from blueprint
        resource_constraints = blueprint.get_resource_constraints()
        if resource_constraints:
            mapped_config['resource_constraints'] = resource_constraints
        
        return mapped_config
    
    def _map_transforms_config(self, config: Dict[str, Any], blueprint: Blueprint) -> Dict[str, Any]:
        """Map blueprint config to transforms library format."""
        mapped_config = {}
        
        # Map pipeline configuration
        if 'pipeline_depth' in config:
            if isinstance(config['pipeline_depth'], list):
                mapped_config['pipeline_depths'] = config['pipeline_depth']
            else:
                mapped_config['pipeline_depths'] = [config['pipeline_depth']]
        
        # Map folding factors
        if 'folding_factors' in config:
            mapped_config['folding_factors'] = config['folding_factors']
        
        # Map memory optimization
        if 'memory_optimization' in config:
            mapped_config['memory_optimization_level'] = config['memory_optimization']
        
        # Map transform sequence if specified
        if 'transform_sequence' in config:
            mapped_config['transform_sequence'] = config['transform_sequence']
        
        return mapped_config
    
    def _map_hw_optim_config(self, config: Dict[str, Any], blueprint: Blueprint) -> Dict[str, Any]:
        """Map blueprint config to hardware optimization library format."""
        mapped_config = {}
        
        # Map target frequency
        if 'target_frequency' in config:
            mapped_config['target_frequency_mhz'] = config['target_frequency']
        
        # Map resource budget
        if 'resource_budget' in config:
            mapped_config['resource_budget'] = config['resource_budget']
        
        # Map optimization strategy
        if 'optimization_strategy' in config:
            mapped_config['strategy'] = config['optimization_strategy']
        
        # Add constraints from blueprint
        constraints = blueprint.constraints
        if 'performance_requirements' in constraints:
            mapped_config['performance_targets'] = constraints['performance_requirements']
        
        return mapped_config
    
    def _map_analysis_config(self, config: Dict[str, Any], blueprint: Blueprint) -> Dict[str, Any]:
        """Map blueprint config to analysis library format."""
        mapped_config = {}
        
        # Map performance metrics
        if 'performance_metrics' in config:
            mapped_config['metrics_to_analyze'] = config['performance_metrics']
        
        # Map analysis options
        if 'roofline_analysis' in config:
            mapped_config['enable_roofline'] = config['roofline_analysis']
        
        if 'power_estimation' in config:
            mapped_config['enable_power_analysis'] = config['power_estimation']
        
        if 'accuracy_analysis' in config:
            mapped_config['enable_accuracy_analysis'] = config['accuracy_analysis']
        
        # Add objectives from blueprint for analysis
        objectives = blueprint.get_optimization_objectives()
        if objectives:
            mapped_config['optimization_objectives'] = [obj['name'] if isinstance(obj, dict) else obj for obj in objectives]
        
        return mapped_config
    
    def extract_design_space_parameters(self, blueprint: Blueprint) -> Dict[str, Any]:
        """
        Extract design space parameters from blueprint.
        
        Args:
            blueprint: Blueprint to extract from
            
        Returns:
            Design space parameter definitions
        """
        design_space_params = {}
        
        # Extract from kernels configuration
        kernels_config = blueprint.get_kernels_config()
        if kernels_config:
            if 'pe_range' in kernels_config:
                design_space_params['kernels_pe'] = {
                    'type': 'categorical',
                    'values': kernels_config['pe_range'],
                    'description': 'Processing element parallelism'
                }
            
            if 'simd_range' in kernels_config:
                design_space_params['kernels_simd'] = {
                    'type': 'categorical', 
                    'values': kernels_config['simd_range'],
                    'description': 'SIMD parallelism'
                }
            
            if 'precision' in kernels_config:
                precision_values = kernels_config['precision'] if isinstance(kernels_config['precision'], list) else [kernels_config['precision']]
                design_space_params['kernels_precision'] = {
                    'type': 'categorical',
                    'values': precision_values,
                    'description': 'Data precision'
                }
        
        # Extract from transforms configuration
        transforms_config = blueprint.get_transforms_config()
        if transforms_config:
            if 'pipeline_depth' in transforms_config:
                depth_values = transforms_config['pipeline_depth'] if isinstance(transforms_config['pipeline_depth'], list) else [transforms_config['pipeline_depth']]
                design_space_params['transforms_pipeline_depth'] = {
                    'type': 'categorical',
                    'values': depth_values,
                    'description': 'Pipeline depth'
                }
        
        # Extract from hw_optim configuration
        hw_optim_config = blueprint.get_hw_optim_config()
        if hw_optim_config:
            if 'target_frequency' in hw_optim_config:
                design_space_params['hw_optim_frequency'] = {
                    'type': 'continuous',
                    'min': hw_optim_config['target_frequency'] * 0.8,
                    'max': hw_optim_config['target_frequency'] * 1.2,
                    'description': 'Target frequency (MHz)'
                }
        
        return design_space_params
    
    def create_library_execution_plan(self, blueprint: Blueprint) -> Dict[str, Any]:
        """
        Create execution plan for libraries based on blueprint.
        
        Args:
            blueprint: Blueprint to create plan from
            
        Returns:
            Execution plan dictionary
        """
        execution_plan = {
            'libraries': list(blueprint.libraries.keys()),
            'execution_order': self._determine_execution_order(blueprint),
            'library_configs': self.map_blueprint_to_libraries(blueprint),
            'constraints': blueprint.constraints,
            'objectives': blueprint.get_optimization_objectives(),
            'design_space_config': blueprint.design_space
        }
        
        return execution_plan
    
    def _determine_execution_order(self, blueprint: Blueprint) -> List[str]:
        """Determine optimal execution order for libraries."""
        # Standard execution order based on typical DSE flow
        all_libraries = ['kernels', 'transforms', 'hw_optim', 'analysis']
        available_libraries = [lib for lib in all_libraries if lib in blueprint.libraries]
        
        return available_libraries
    
    def validate_library_compatibility(self, blueprint: Blueprint) -> Tuple[bool, List[str]]:
        """
        Validate blueprint compatibility with available libraries.
        
        Args:
            blueprint: Blueprint to validate
            
        Returns:
            Tuple of (is_compatible, list_of_warnings)
        """
        warnings = []
        
        # Check if all specified libraries are available
        # (In a real implementation, this would check against actual library availability)
        available_libraries = {'kernels', 'transforms', 'hw_optim', 'analysis'}
        required_libraries = set(blueprint.libraries.keys())
        
        missing_libraries = required_libraries - available_libraries
        if missing_libraries:
            warnings.extend([f"Library not available: {lib}" for lib in missing_libraries])
        
        # Check for conflicting configurations
        kernels_config = blueprint.get_kernels_config()
        hw_optim_config = blueprint.get_hw_optim_config()
        
        if kernels_config and hw_optim_config:
            # Check if resource budget is compatible with kernel requirements
            if 'resource_budget' in hw_optim_config and 'pe_range' in kernels_config:
                max_pe = max(kernels_config['pe_range']) if isinstance(kernels_config['pe_range'], list) else kernels_config['pe_range']
                budget = hw_optim_config['resource_budget']
                
                # Simple heuristic check
                if 'luts' in budget and budget['luts'] < max_pe * 1000:
                    warnings.append("Resource budget may be insufficient for maximum PE configuration")
        
        return len(warnings) == 0, warnings