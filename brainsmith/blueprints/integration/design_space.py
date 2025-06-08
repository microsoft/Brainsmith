"""
Design space generation from blueprints.

Generates design spaces automatically from blueprint specifications,
integrating with the Week 1 design space system.
"""

from typing import Dict, List, Any, Optional, Tuple
import itertools
import logging

from ..core.blueprint import Blueprint
from .library_mapper import LibraryMapper

logger = logging.getLogger(__name__)


class DesignSpaceGenerator:
    """
    Generates design spaces from blueprint specifications.
    
    Creates comprehensive design spaces by combining parameters from
    all configured libraries in a blueprint.
    """
    
    def __init__(self):
        """Initialize design space generator."""
        self.logger = logging.getLogger("brainsmith.blueprints.integration.design_space")
        self.library_mapper = LibraryMapper()
    
    def generate_from_blueprint(self, blueprint: Blueprint) -> Dict[str, Any]:
        """
        Generate design space from blueprint.
        
        Args:
            blueprint: Blueprint specification
            
        Returns:
            Design space dictionary
        """
        self.logger.info(f"Generating design space from blueprint: {blueprint.name}")
        
        # Extract parameters from blueprint
        parameters = self.library_mapper.extract_design_space_parameters(blueprint)
        
        # Generate design points
        design_points = self._generate_design_points(parameters, blueprint)
        
        # Apply constraints
        filtered_points = self._apply_constraints(design_points, blueprint)
        
        # Create design space specification
        design_space = {
            'name': f"{blueprint.name}_design_space",
            'parameters': parameters,
            'design_points': filtered_points,
            'total_points': len(filtered_points),
            'constraints': blueprint.constraints,
            'objectives': blueprint.get_optimization_objectives(),
            'exploration_config': blueprint.design_space,
            'source_blueprint': blueprint.name
        }
        
        self.logger.info(f"Generated design space with {len(filtered_points)} points")
        return design_space
    
    def _generate_design_points(self, parameters: Dict[str, Any], blueprint: Blueprint) -> List[Dict[str, Any]]:
        """Generate all possible design points from parameters."""
        if not parameters:
            return []
        
        # Get parameter values
        param_names = []
        param_values = []
        
        for param_name, param_def in parameters.items():
            param_names.append(param_name)
            
            if param_def['type'] == 'categorical':
                param_values.append(param_def['values'])
            elif param_def['type'] == 'continuous':
                # For continuous parameters, create discrete samples
                min_val = param_def['min']
                max_val = param_def['max']
                num_samples = param_def.get('samples', 5)
                
                if num_samples == 1:
                    values = [(min_val + max_val) / 2]
                else:
                    step = (max_val - min_val) / (num_samples - 1)
                    values = [min_val + i * step for i in range(num_samples)]
                
                param_values.append(values)
            elif param_def['type'] == 'integer':
                min_val = param_def.get('min', 1)
                max_val = param_def.get('max', 10)
                step = param_def.get('step', 1)
                values = list(range(min_val, max_val + 1, step))
                param_values.append(values)
        
        # Generate cartesian product of all parameter values
        design_points = []
        for combination in itertools.product(*param_values):
            point = dict(zip(param_names, combination))
            design_points.append(point)
        
        # Limit number of points if specified
        max_evaluations = blueprint.design_space.get('max_evaluations')
        if max_evaluations and len(design_points) > max_evaluations:
            # Use sampling strategy specified in blueprint
            strategy = blueprint.design_space.get('exploration_strategy', 'random')
            design_points = self._sample_design_points(design_points, max_evaluations, strategy)
        
        return design_points
    
    def _sample_design_points(self, design_points: List[Dict[str, Any]], 
                            max_points: int, strategy: str) -> List[Dict[str, Any]]:
        """Sample design points according to strategy."""
        if len(design_points) <= max_points:
            return design_points
        
        if strategy == 'random':
            import random
            return random.sample(design_points, max_points)
        elif strategy == 'grid':
            # Take evenly spaced points
            step = len(design_points) // max_points
            return design_points[::step][:max_points]
        elif strategy == 'pareto_optimal':
            # For now, use random sampling (Pareto filtering would require evaluation)
            import random
            return random.sample(design_points, max_points)
        else:
            # Default to taking first max_points
            return design_points[:max_points]
    
    def _apply_constraints(self, design_points: List[Dict[str, Any]], 
                          blueprint: Blueprint) -> List[Dict[str, Any]]:
        """Apply blueprint constraints to filter design points."""
        if not blueprint.constraints:
            return design_points
        
        filtered_points = []
        
        for point in design_points:
            if self._point_satisfies_constraints(point, blueprint.constraints):
                filtered_points.append(point)
        
        return filtered_points
    
    def _point_satisfies_constraints(self, point: Dict[str, Any], 
                                   constraints: Dict[str, Any]) -> bool:
        """Check if a design point satisfies constraints."""
        # Resource constraints
        if 'resource_limits' in constraints:
            resource_limits = constraints['resource_limits']
            estimated_resources = self._estimate_point_resources(point)
            
            for resource, limit in resource_limits.items():
                if resource in estimated_resources:
                    if estimated_resources[resource] > limit:
                        return False
        
        # Performance constraints
        if 'performance_requirements' in constraints:
            perf_requirements = constraints['performance_requirements']
            estimated_performance = self._estimate_point_performance(point)
            
            for metric, requirement in perf_requirements.items():
                if metric in estimated_performance:
                    if metric.startswith('min_') and estimated_performance[metric] < requirement:
                        return False
                    elif metric.startswith('max_') and estimated_performance[metric] > requirement:
                        return False
        
        return True
    
    def _estimate_point_resources(self, point: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resource usage for a design point."""
        # Simplified resource estimation based on parameters
        resources = {'luts': 0, 'brams': 0, 'dsps': 0}
        
        # Estimate based on kernels parameters
        pe = point.get('kernels_pe', 1)
        simd = point.get('kernels_simd', 1)
        
        # Simple linear scaling model
        base_luts = 1000
        base_brams = 5
        base_dsps = 2
        
        resources['luts'] = base_luts * pe * simd
        resources['brams'] = base_brams * max(1, pe // 2)
        resources['dsps'] = base_dsps * pe
        
        return resources
    
    def _estimate_point_performance(self, point: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance for a design point."""
        # Simplified performance estimation
        performance = {}
        
        # Estimate throughput based on parallelism
        pe = point.get('kernels_pe', 1)
        simd = point.get('kernels_simd', 1)
        frequency = point.get('hw_optim_frequency', 250)
        
        # Simple throughput model
        base_throughput = 100
        performance['throughput'] = base_throughput * pe * simd * (frequency / 250)
        
        # Estimate latency (inverse relationship with throughput)
        performance['latency'] = 1000 / performance['throughput']
        
        return performance
    
    def generate_exploration_plan(self, blueprint: Blueprint) -> Dict[str, Any]:
        """
        Generate exploration plan from blueprint.
        
        Args:
            blueprint: Blueprint specification
            
        Returns:
            Exploration plan dictionary
        """
        design_space = self.generate_from_blueprint(blueprint)
        
        exploration_plan = {
            'name': f"{blueprint.name}_exploration",
            'design_space': design_space,
            'exploration_strategy': blueprint.design_space.get('exploration_strategy', 'grid'),
            'max_evaluations': blueprint.design_space.get('max_evaluations', 100),
            'objectives': blueprint.get_optimization_objectives(),
            'library_execution_plan': self.library_mapper.create_library_execution_plan(blueprint),
            'constraints': blueprint.constraints,
            'evaluation_metrics': self._extract_evaluation_metrics(blueprint)
        }
        
        return exploration_plan
    
    def _extract_evaluation_metrics(self, blueprint: Blueprint) -> List[str]:
        """Extract metrics to evaluate from blueprint."""
        metrics = set()
        
        # Add metrics from objectives
        for objective in blueprint.get_optimization_objectives():
            if isinstance(objective, dict):
                metrics.add(objective['name'])
            else:
                metrics.add(objective)
        
        # Add metrics from analysis configuration
        analysis_config = blueprint.get_analysis_config()
        if 'performance_metrics' in analysis_config:
            metrics.update(analysis_config['performance_metrics'])
        
        # Add default metrics
        metrics.update(['throughput', 'latency', 'resource_efficiency'])
        
        return list(metrics)
    
    def optimize_design_space(self, blueprint: Blueprint) -> Dict[str, Any]:
        """
        Generate optimized design space focusing on promising regions.
        
        Args:
            blueprint: Blueprint specification
            
        Returns:
            Optimized design space
        """
        # Generate initial design space
        base_design_space = self.generate_from_blueprint(blueprint)
        
        # Apply optimization strategies based on objectives
        objectives = blueprint.get_optimization_objectives()
        
        optimized_points = []
        
        for point in base_design_space['design_points']:
            # Score point based on objectives
            score = self._score_design_point(point, objectives)
            point['_optimization_score'] = score
            optimized_points.append(point)
        
        # Sort by score and take top points
        optimized_points.sort(key=lambda p: p['_optimization_score'], reverse=True)
        
        # Limit to reasonable number of points
        max_points = blueprint.design_space.get('max_evaluations', 50)
        optimized_points = optimized_points[:max_points]
        
        # Remove score from final points
        for point in optimized_points:
            del point['_optimization_score']
        
        optimized_design_space = base_design_space.copy()
        optimized_design_space['design_points'] = optimized_points
        optimized_design_space['total_points'] = len(optimized_points)
        optimized_design_space['optimization_applied'] = True
        
        return optimized_design_space
    
    def _score_design_point(self, point: Dict[str, Any], 
                           objectives: List[Any]) -> float:
        """Score a design point based on objectives."""
        score = 0.0
        
        for objective in objectives:
            obj_name = objective if isinstance(objective, str) else objective.get('name', '')
            obj_type = 'maximize' if isinstance(objective, str) else objective.get('type', 'maximize')
            
            if obj_name == 'throughput':
                # Higher PE and SIMD generally mean higher throughput
                pe = point.get('kernels_pe', 1)
                simd = point.get('kernels_simd', 1)
                throughput_score = pe * simd
                score += throughput_score if obj_type == 'maximize' else -throughput_score
                
            elif obj_name == 'resource_efficiency':
                # Balance between performance and resource usage
                pe = point.get('kernels_pe', 1)
                simd = point.get('kernels_simd', 1)
                efficiency_score = (pe * simd) / (pe + simd)  # Simple efficiency metric
                score += efficiency_score if obj_type == 'maximize' else -efficiency_score
                
            elif obj_name == 'latency':
                # Lower latency preferred (inverse of throughput approximation)
                pe = point.get('kernels_pe', 1)
                simd = point.get('kernels_simd', 1)
                latency_score = 1.0 / (pe * simd)
                score += latency_score if obj_type == 'minimize' else -latency_score
        
        return score