"""
Blueprint validation system.

Comprehensive validation for blueprint specifications including
library compatibility, constraint validation, and dependency checking.
"""

from typing import Dict, List, Any, Tuple, Optional
import logging

from .blueprint import Blueprint

logger = logging.getLogger(__name__)


class BlueprintValidator:
    """
    Comprehensive blueprint validation system.
    """
    
    def __init__(self):
        """Initialize validator."""
        self.logger = logging.getLogger("brainsmith.blueprints.validator")
        
        # Valid configuration keys for each library
        self.valid_library_configs = {
            'kernels': {
                'pe_range', 'simd_range', 'precision', 'optimization_hint',
                'pe_values', 'simd_values', 'precision_options'
            },
            'transforms': {
                'pipeline_depth', 'folding_factors', 'memory_optimization',
                'transform_sequence', 'optimization_level'
            },
            'hw_optim': {
                'target_frequency', 'resource_budget', 'optimization_strategy',
                'clock_constraints', 'power_constraints'
            },
            'analysis': {
                'performance_metrics', 'roofline_analysis', 'power_estimation',
                'accuracy_analysis', 'report_format'
            }
        }
        
        # Valid constraint categories
        self.valid_constraints = {
            'resource_limits', 'performance_requirements', 
            'power_constraints', 'timing_constraints'
        }
        
        # Valid optimization objectives - expanded list
        self.valid_objectives = {
            'throughput', 'latency', 'resource_efficiency', 'power_efficiency',
            'accuracy', 'area', 'performance', 'energy', 'area_efficiency',
            'power', 'efficiency', 'utilization'
        }
    
    def validate(self, blueprint: Blueprint) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive blueprint validation.
        
        Args:
            blueprint: Blueprint to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic structure validation
        errors.extend(self._validate_basic_structure(blueprint))
        
        # Library configuration validation
        errors.extend(self._validate_library_configs(blueprint))
        
        # Constraint validation
        errors.extend(self._validate_constraints(blueprint))
        
        # Objective validation
        errors.extend(self._validate_objectives(blueprint))
        
        # Design space validation
        errors.extend(self._validate_design_space(blueprint))
        
        # Cross-validation (dependencies, conflicts)
        errors.extend(self._validate_cross_dependencies(blueprint))
        
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info(f"Blueprint '{blueprint.name}' validation: PASSED")
        else:
            self.logger.warning(f"Blueprint '{blueprint.name}' validation: FAILED ({len(errors)} errors)")
            for error in errors:
                self.logger.warning(f"  - {error}")
        
        return is_valid, errors
    
    def _validate_basic_structure(self, blueprint: Blueprint) -> List[str]:
        """Validate basic blueprint structure."""
        errors = []
        
        # Required fields
        if not blueprint.name:
            errors.append("Blueprint name is required")
        elif not isinstance(blueprint.name, str):
            errors.append("Blueprint name must be a string")
        
        if not blueprint.version:
            errors.append("Blueprint version is required")
        elif not isinstance(blueprint.version, str):
            errors.append("Blueprint version must be a string")
        
        # Optional but should be string if present
        if blueprint.description and not isinstance(blueprint.description, str):
            errors.append("Blueprint description must be a string")
        
        # Metadata should be dict
        if blueprint.metadata and not isinstance(blueprint.metadata, dict):
            errors.append("Blueprint metadata must be a dictionary")
        
        return errors
    
    def _validate_library_configs(self, blueprint: Blueprint) -> List[str]:
        """Validate library configurations."""
        errors = []
        
        if not blueprint.libraries:
            errors.append("At least one library configuration is required")
            return errors
        
        # Validate each library configuration
        for lib_name, lib_config in blueprint.libraries.items():
            if lib_name not in self.valid_library_configs:
                errors.append(f"Unknown library: {lib_name}")
                continue
            
            if not isinstance(lib_config, dict):
                errors.append(f"Library '{lib_name}' configuration must be a dictionary")
                continue
            
            # Validate library-specific configuration keys
            valid_keys = self.valid_library_configs[lib_name]
            for config_key in lib_config.keys():
                if config_key not in valid_keys:
                    errors.append(f"Unknown configuration key '{config_key}' for library '{lib_name}'")
            
            # Library-specific validation
            errors.extend(self._validate_specific_library(lib_name, lib_config))
        
        return errors
    
    def _validate_specific_library(self, lib_name: str, config: Dict[str, Any]) -> List[str]:
        """Validate specific library configuration."""
        errors = []
        
        if lib_name == 'kernels':
            errors.extend(self._validate_kernels_config(config))
        elif lib_name == 'transforms':
            errors.extend(self._validate_transforms_config(config))
        elif lib_name == 'hw_optim':
            errors.extend(self._validate_hw_optim_config(config))
        elif lib_name == 'analysis':
            errors.extend(self._validate_analysis_config(config))
        
        return errors
    
    def _validate_kernels_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate kernels library configuration."""
        errors = []
        
        # Validate PE range/values
        if 'pe_range' in config:
            pe_range = config['pe_range']
            if not isinstance(pe_range, list) or len(pe_range) < 1:
                errors.append("kernels.pe_range must be a non-empty list")
            elif not all(isinstance(x, int) and x > 0 for x in pe_range):
                errors.append("kernels.pe_range must contain positive integers")
        
        # Validate SIMD range/values
        if 'simd_range' in config:
            simd_range = config['simd_range']
            if not isinstance(simd_range, list) or len(simd_range) < 1:
                errors.append("kernels.simd_range must be a non-empty list")
            elif not all(isinstance(x, int) and x > 0 for x in simd_range):
                errors.append("kernels.simd_range must contain positive integers")
        
        # Validate precision options
        if 'precision' in config:
            precision = config['precision']
            valid_precisions = {'int8', 'int16', 'int32', 'float16', 'float32'}
            if isinstance(precision, str):
                if precision not in valid_precisions:
                    errors.append(f"Invalid precision: {precision}")
            elif isinstance(precision, list):
                for p in precision:
                    if p not in valid_precisions:
                        errors.append(f"Invalid precision: {p}")
            else:
                errors.append("kernels.precision must be string or list of strings")
        
        return errors
    
    def _validate_transforms_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate transforms library configuration."""
        errors = []
        
        # Validate pipeline depth
        if 'pipeline_depth' in config:
            depth = config['pipeline_depth']
            if isinstance(depth, int):
                if depth <= 0:
                    errors.append("transforms.pipeline_depth must be positive")
            elif isinstance(depth, list):
                if not all(isinstance(x, int) and x > 0 for x in depth):
                    errors.append("transforms.pipeline_depth list must contain positive integers")
            else:
                errors.append("transforms.pipeline_depth must be integer or list of integers")
        
        # Validate memory optimization
        if 'memory_optimization' in config:
            mem_opt = config['memory_optimization']
            valid_options = {'conservative', 'balanced', 'aggressive'}
            if mem_opt not in valid_options:
                errors.append(f"Invalid memory_optimization: {mem_opt}")
        
        return errors
    
    def _validate_hw_optim_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate hardware optimization library configuration."""
        errors = []
        
        # Validate target frequency
        if 'target_frequency' in config:
            freq = config['target_frequency']
            if not isinstance(freq, (int, float)) or freq <= 0:
                errors.append("hw_optim.target_frequency must be a positive number")
        
        # Validate resource budget
        if 'resource_budget' in config:
            budget = config['resource_budget']
            if not isinstance(budget, dict):
                errors.append("hw_optim.resource_budget must be a dictionary")
            else:
                valid_resources = {'luts', 'ffs', 'brams', 'dsps', 'uram'}
                for resource, value in budget.items():
                    if resource not in valid_resources:
                        errors.append(f"Unknown resource type: {resource}")
                    elif not isinstance(value, (int, float)) or value <= 0:
                        errors.append(f"Resource budget for {resource} must be positive")
        
        return errors
    
    def _validate_analysis_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate analysis library configuration."""
        errors = []
        
        # Validate performance metrics
        if 'performance_metrics' in config:
            metrics = config['performance_metrics']
            if not isinstance(metrics, list):
                errors.append("analysis.performance_metrics must be a list")
            else:
                valid_metrics = {'throughput', 'latency', 'efficiency', 'power', 'accuracy'}
                for metric in metrics:
                    if metric not in valid_metrics:
                        errors.append(f"Unknown performance metric: {metric}")
        
        return errors
    
    def _validate_constraints(self, blueprint: Blueprint) -> List[str]:
        """Validate constraints."""
        errors = []
        
        for constraint_name, constraint_value in blueprint.constraints.items():
            if constraint_name not in self.valid_constraints:
                errors.append(f"Unknown constraint category: {constraint_name}")
                continue
            
            # Constraint-specific validation
            if constraint_name == 'resource_limits':
                errors.extend(self._validate_resource_limits(constraint_value))
            elif constraint_name == 'performance_requirements':
                errors.extend(self._validate_performance_requirements(constraint_value))
        
        return errors
    
    def _validate_resource_limits(self, limits: Any) -> List[str]:
        """Validate resource limits constraint."""
        errors = []
        
        if not isinstance(limits, dict):
            errors.append("resource_limits must be a dictionary")
            return errors
        
        for resource, limit in limits.items():
            if not isinstance(limit, (int, float)) or limit <= 0:
                errors.append(f"Resource limit for {resource} must be a positive number")
        
        return errors
    
    def _validate_performance_requirements(self, requirements: Any) -> List[str]:
        """Validate performance requirements constraint."""
        errors = []
        
        if not isinstance(requirements, dict):
            errors.append("performance_requirements must be a dictionary")
            return errors
        
        for req_name, req_value in requirements.items():
            if not isinstance(req_value, (int, float)):
                errors.append(f"Performance requirement {req_name} must be a number")
        
        return errors
    
    def _validate_objectives(self, blueprint: Blueprint) -> List[str]:
        """Validate optimization objectives."""
        errors = []
        
        if not blueprint.objectives:
            errors.append("At least one optimization objective is required")
            return errors
        
        for i, objective in enumerate(blueprint.objectives):
            if isinstance(objective, str):
                # Simple objective name
                if objective not in self.valid_objectives:
                    errors.append(f"Unknown objective: {objective}")
            elif isinstance(objective, dict):
                # Detailed objective specification
                if 'name' not in objective:
                    errors.append(f"Objective {i} must have 'name' field")
                elif objective['name'] not in self.valid_objectives:
                    errors.append(f"Unknown objective: {objective['name']}")
                
                if 'type' in objective:
                    obj_type = objective['type']
                    if obj_type not in {'maximize', 'minimize'}:
                        errors.append(f"Objective type must be 'maximize' or 'minimize', got: {obj_type}")
            else:
                errors.append(f"Objective {i} must be string or dictionary")
        
        return errors
    
    def _validate_design_space(self, blueprint: Blueprint) -> List[str]:
        """Validate design space configuration."""
        errors = []
        
        if not blueprint.design_space:
            return errors  # Design space config is optional
        
        design_space = blueprint.design_space
        
        # Validate exploration strategy
        if 'exploration_strategy' in design_space:
            strategy = design_space['exploration_strategy']
            valid_strategies = {'pareto_optimal', 'random', 'grid', 'genetic', 'bayesian'}
            if strategy not in valid_strategies:
                errors.append(f"Unknown exploration strategy: {strategy}")
        
        # Validate max evaluations
        if 'max_evaluations' in design_space:
            max_evals = design_space['max_evaluations']
            if not isinstance(max_evals, int) or max_evals <= 0:
                errors.append("max_evaluations must be a positive integer")
        
        return errors
    
    def _validate_cross_dependencies(self, blueprint: Blueprint) -> List[str]:
        """Validate cross-dependencies and conflicts."""
        errors = []
        
        # Check for conflicting configurations
        if 'kernels' in blueprint.libraries and 'hw_optim' in blueprint.libraries:
            kernels_config = blueprint.libraries['kernels']
            hw_optim_config = blueprint.libraries['hw_optim']
            
            # Check if resource budget conflicts with kernel requirements
            if 'resource_budget' in hw_optim_config and ('pe_range' in kernels_config or 'simd_range' in kernels_config):
                # This is a simplified check - in practice, you'd estimate resource usage
                pass
        
        # Check objective consistency
        objectives = [obj if isinstance(obj, str) else obj.get('name', '') for obj in blueprint.objectives]
        if 'throughput' in objectives and 'latency' in objectives:
            # Check if both are being maximized (potential conflict)
            pass
        
        return errors
    
    def validate_library_compatibility(self, blueprint: Blueprint) -> Tuple[bool, List[str]]:
        """
        Validate compatibility with available libraries.
        
        Args:
            blueprint: Blueprint to validate
            
        Returns:
            Tuple of (is_compatible, list_of_warnings)
        """
        warnings = []
        
        # This would check against actual library availability
        # For now, we assume all libraries are available
        
        required_libraries = set(blueprint.libraries.keys())
        available_libraries = {'kernels', 'transforms', 'hw_optim', 'analysis'}
        
        missing_libraries = required_libraries - available_libraries
        if missing_libraries:
            warnings.extend([f"Library not available: {lib}" for lib in missing_libraries])
        
        return len(warnings) == 0, warnings