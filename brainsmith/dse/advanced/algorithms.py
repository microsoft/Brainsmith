"""
FPGA-Specific Optimization Algorithms
Genetic algorithms, simulated annealing, PSO, and hybrid frameworks tailored for FPGA design optimization.
"""

import os
import sys
import time
import logging
import random
import math
import copy
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .multi_objective import ParetoSolution, MultiObjectiveOptimizer

logger = logging.getLogger(__name__)


@dataclass
class FPGADesignCandidate:
    """Represents an FPGA design candidate with specific parameters."""
    parameters: Dict[str, Any]
    architecture: str = "generic"
    transformation_sequence: List[str] = field(default_factory=list)
    resource_budget: Dict[str, int] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_pareto_solution(self, objective_values: List[float], constraints: List[float] = None) -> ParetoSolution:
        """Convert to ParetoSolution for multi-objective optimization."""
        return ParetoSolution(
            design_parameters=self.parameters.copy(),
            objective_values=objective_values,
            constraint_violations=constraints or [],
            metadata={
                'architecture': self.architecture,
                'transformation_sequence': self.transformation_sequence,
                'resource_budget': self.resource_budget,
                'performance_targets': self.performance_targets
            }
        )


class FPGAGeneticOperators:
    """FPGA-specific genetic operators for design optimization."""
    
    def __init__(self):
        self.crossover_strategies = {
            'uniform': self._uniform_crossover,
            'single_point': self._single_point_crossover,
            'transformation_aware': self._transformation_aware_crossover,
            'resource_balanced': self._resource_balanced_crossover
        }
        
        self.mutation_strategies = {
            'parameter_mutation': self._parameter_mutation,
            'transformation_mutation': self._transformation_mutation,
            'resource_aware_mutation': self._resource_aware_mutation,
            'architecture_preserving_mutation': self._architecture_preserving_mutation
        }
    
    def crossover(self, parent1: FPGADesignCandidate, parent2: FPGADesignCandidate, 
                 strategy: str = 'transformation_aware') -> Tuple[FPGADesignCandidate, FPGADesignCandidate]:
        """Perform crossover between two FPGA design candidates."""
        crossover_func = self.crossover_strategies.get(strategy, self._uniform_crossover)
        return crossover_func(parent1, parent2)
    
    def mutate(self, candidate: FPGADesignCandidate, design_space: Dict[str, Any],
              strategy: str = 'resource_aware_mutation', mutation_rate: float = 0.1) -> FPGADesignCandidate:
        """Mutate an FPGA design candidate."""
        mutation_func = self.mutation_strategies.get(strategy, self._parameter_mutation)
        return mutation_func(candidate, design_space, mutation_rate)
    
    def _uniform_crossover(self, parent1: FPGADesignCandidate, parent2: FPGADesignCandidate) -> Tuple[FPGADesignCandidate, FPGADesignCandidate]:
        """Uniform crossover for design parameters."""
        child1_params = {}
        child2_params = {}
        
        all_params = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        
        for param in all_params:
            if random.random() < 0.5:
                child1_params[param] = parent1.parameters.get(param, parent2.parameters.get(param))
                child2_params[param] = parent2.parameters.get(param, parent1.parameters.get(param))
            else:
                child1_params[param] = parent2.parameters.get(param, parent1.parameters.get(param))
                child2_params[param] = parent1.parameters.get(param, parent2.parameters.get(param))
        
        child1 = FPGADesignCandidate(
            parameters=child1_params,
            architecture=parent1.architecture,
            resource_budget=parent1.resource_budget.copy()
        )
        
        child2 = FPGADesignCandidate(
            parameters=child2_params,
            architecture=parent2.architecture,
            resource_budget=parent2.resource_budget.copy()
        )
        
        return child1, child2
    
    def _single_point_crossover(self, parent1: FPGADesignCandidate, parent2: FPGADesignCandidate) -> Tuple[FPGADesignCandidate, FPGADesignCandidate]:
        """Single-point crossover for parameters."""
        param_list = list(parent1.parameters.keys())
        if not param_list:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        crossover_point = random.randint(0, len(param_list))
        
        child1_params = {}
        child2_params = {}
        
        for i, param in enumerate(param_list):
            if i < crossover_point:
                child1_params[param] = parent1.parameters[param]
                child2_params[param] = parent2.parameters[param]
            else:
                child1_params[param] = parent2.parameters[param]
                child2_params[param] = parent1.parameters[param]
        
        child1 = FPGADesignCandidate(
            parameters=child1_params,
            architecture=parent1.architecture,
            transformation_sequence=parent1.transformation_sequence.copy()
        )
        
        child2 = FPGADesignCandidate(
            parameters=child2_params,
            architecture=parent2.architecture,
            transformation_sequence=parent2.transformation_sequence.copy()
        )
        
        return child1, child2
    
    def _transformation_aware_crossover(self, parent1: FPGADesignCandidate, parent2: FPGADesignCandidate) -> Tuple[FPGADesignCandidate, FPGADesignCandidate]:
        """Crossover that preserves valid FINN transformation sequences."""
        
        # Crossover parameters with transformation compatibility
        child1_params = {}
        child2_params = {}
        
        # Group parameters by transformation they affect
        transformation_groups = self._group_parameters_by_transformation(parent1.parameters, parent1.transformation_sequence)
        
        for group_name, param_names in transformation_groups.items():
            if random.random() < 0.5:
                # Take entire group from parent1
                for param in param_names:
                    if param in parent1.parameters:
                        child1_params[param] = parent1.parameters[param]
                    if param in parent2.parameters:
                        child2_params[param] = parent2.parameters[param]
            else:
                # Take entire group from parent2
                for param in param_names:
                    if param in parent2.parameters:
                        child1_params[param] = parent2.parameters[param]
                    if param in parent1.parameters:
                        child2_params[param] = parent1.parameters[param]
        
        # Handle remaining parameters
        for param in parent1.parameters:
            if param not in child1_params:
                child1_params[param] = parent1.parameters[param]
        for param in parent2.parameters:
            if param not in child2_params:
                child2_params[param] = parent2.parameters[param]
        
        # Choose transformation sequence
        child1_transforms = parent1.transformation_sequence if random.random() < 0.5 else parent2.transformation_sequence
        child2_transforms = parent2.transformation_sequence if random.random() < 0.5 else parent1.transformation_sequence
        
        child1 = FPGADesignCandidate(
            parameters=child1_params,
            architecture=parent1.architecture,
            transformation_sequence=child1_transforms.copy()
        )
        
        child2 = FPGADesignCandidate(
            parameters=child2_params,
            architecture=parent2.architecture,
            transformation_sequence=child2_transforms.copy()
        )
        
        return child1, child2
    
    def _resource_balanced_crossover(self, parent1: FPGADesignCandidate, parent2: FPGADesignCandidate) -> Tuple[FPGADesignCandidate, FPGADesignCandidate]:
        """Crossover that maintains resource balance."""
        
        # Analyze resource impact of parameters
        resource_impact = self._analyze_parameter_resource_impact(parent1.parameters, parent1.resource_budget)
        
        child1_params = {}
        child2_params = {}
        
        # Sort parameters by resource impact
        sorted_params = sorted(parent1.parameters.keys(), 
                             key=lambda p: resource_impact.get(p, 0), reverse=True)
        
        child1_resources = {'lut': 0, 'dsp': 0, 'bram': 0}
        child2_resources = {'lut': 0, 'dsp': 0, 'bram': 0}
        
        # Assign parameters to children based on resource balance
        for param in sorted_params:
            param_impact = resource_impact.get(param, {})
            
            # Calculate current resource usage
            child1_usage = sum(child1_resources.values())
            child2_usage = sum(child2_resources.values())
            
            if child1_usage <= child2_usage:
                # Assign to child1
                child1_params[param] = parent1.parameters[param]
                for resource, impact in param_impact.items():
                    child1_resources[resource] = child1_resources.get(resource, 0) + impact
                
                if param in parent2.parameters:
                    child2_params[param] = parent2.parameters[param]
            else:
                # Assign to child2
                child2_params[param] = parent2.parameters[param] if param in parent2.parameters else parent1.parameters[param]
                for resource, impact in param_impact.items():
                    child2_resources[resource] = child2_resources.get(resource, 0) + impact
                
                child1_params[param] = parent1.parameters[param]
        
        child1 = FPGADesignCandidate(
            parameters=child1_params,
            architecture=parent1.architecture,
            resource_budget=parent1.resource_budget.copy()
        )
        
        child2 = FPGADesignCandidate(
            parameters=child2_params,
            architecture=parent2.architecture,
            resource_budget=parent2.resource_budget.copy()
        )
        
        return child1, child2
    
    def _parameter_mutation(self, candidate: FPGADesignCandidate, design_space: Dict[str, Any], 
                          mutation_rate: float) -> FPGADesignCandidate:
        """Basic parameter mutation."""
        mutated_params = candidate.parameters.copy()
        
        for param_name, param_value in mutated_params.items():
            if random.random() < mutation_rate:
                if param_name in design_space:
                    param_range = design_space[param_name]
                    
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        min_val, max_val = param_range
                        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                            # Numerical parameter
                            if isinstance(min_val, int) and isinstance(max_val, int):
                                mutated_params[param_name] = random.randint(min_val, max_val)
                            else:
                                mutated_params[param_name] = random.uniform(min_val, max_val)
                    elif isinstance(param_range, (list, tuple)):
                        # Discrete choices
                        mutated_params[param_name] = random.choice(param_range)
        
        return FPGADesignCandidate(
            parameters=mutated_params,
            architecture=candidate.architecture,
            transformation_sequence=candidate.transformation_sequence.copy(),
            resource_budget=candidate.resource_budget.copy()
        )
    
    def _transformation_mutation(self, candidate: FPGADesignCandidate, design_space: Dict[str, Any],
                                mutation_rate: float) -> FPGADesignCandidate:
        """Mutation that can modify transformation sequence."""
        mutated_candidate = self._parameter_mutation(candidate, design_space, mutation_rate)
        
        # Mutate transformation sequence
        if random.random() < mutation_rate and 'available_transformations' in design_space:
            available_transforms = design_space['available_transformations']
            
            if random.random() < 0.3:  # Add transformation
                new_transform = random.choice(available_transforms)
                insertion_point = random.randint(0, len(mutated_candidate.transformation_sequence))
                mutated_candidate.transformation_sequence.insert(insertion_point, new_transform)
            
            elif random.random() < 0.3 and mutated_candidate.transformation_sequence:  # Remove transformation
                removal_point = random.randint(0, len(mutated_candidate.transformation_sequence) - 1)
                mutated_candidate.transformation_sequence.pop(removal_point)
            
            elif mutated_candidate.transformation_sequence:  # Replace transformation
                replacement_point = random.randint(0, len(mutated_candidate.transformation_sequence) - 1)
                new_transform = random.choice(available_transforms)
                mutated_candidate.transformation_sequence[replacement_point] = new_transform
        
        return mutated_candidate
    
    def _resource_aware_mutation(self, candidate: FPGADesignCandidate, design_space: Dict[str, Any],
                               mutation_rate: float) -> FPGADesignCandidate:
        """Mutation that considers resource constraints."""
        mutated_params = candidate.parameters.copy()
        
        # Estimate current resource usage
        current_usage = self._estimate_resource_usage(candidate.parameters)
        resource_budget = candidate.resource_budget
        
        for param_name, param_value in mutated_params.items():
            if random.random() < mutation_rate and param_name in design_space:
                param_range = design_space[param_name]
                
                # Estimate resource impact of parameter change
                param_impact = self._estimate_parameter_resource_impact(param_name, param_value)
                
                # Check if mutation would violate resource constraints
                projected_usage = {k: current_usage.get(k, 0) + param_impact.get(k, 0) 
                                 for k in set(current_usage.keys()) | set(param_impact.keys())}
                
                violates_constraints = any(
                    projected_usage.get(resource, 0) > resource_budget.get(resource, float('inf'))
                    for resource in projected_usage
                )
                
                if not violates_constraints:
                    # Safe to mutate
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        min_val, max_val = param_range
                        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                            if isinstance(min_val, int) and isinstance(max_val, int):
                                mutated_params[param_name] = random.randint(min_val, max_val)
                            else:
                                mutated_params[param_name] = random.uniform(min_val, max_val)
                    elif isinstance(param_range, (list, tuple)):
                        mutated_params[param_name] = random.choice(param_range)
        
        return FPGADesignCandidate(
            parameters=mutated_params,
            architecture=candidate.architecture,
            transformation_sequence=candidate.transformation_sequence.copy(),
            resource_budget=candidate.resource_budget.copy()
        )
    
    def _architecture_preserving_mutation(self, candidate: FPGADesignCandidate, design_space: Dict[str, Any],
                                        mutation_rate: float) -> FPGADesignCandidate:
        """Mutation that preserves architectural validity."""
        mutated_candidate = self._resource_aware_mutation(candidate, design_space, mutation_rate)
        
        # Validate and fix architectural constraints
        mutated_candidate = self._validate_and_fix_architecture(mutated_candidate, design_space)
        
        return mutated_candidate
    
    def _group_parameters_by_transformation(self, parameters: Dict[str, Any], 
                                           transformations: List[str]) -> Dict[str, List[str]]:
        """Group parameters by the transformation they primarily affect."""
        groups = {'general': []}
        
        # Define parameter-transformation relationships
        transform_params = {
            'ConvertONNXToFINN': ['input_shape', 'data_type'],
            'CreateDataflowPartition': ['partition_size', 'memory_mode'],
            'GiveUniqueNodeNames': [],
            'GiveReadableTensorNames': [],
            'InferShapes': [],
            'FoldConstants': [],
            'InsertTopK': ['k_value'],
            'InsertIODMA': ['dma_config'],
            'AnnotateCycles': ['cycle_config'],
            'CodeGen_cppsim': ['sim_config'],
            'Compile_cppsim': ['compile_config'],
            'Set_exec_mode': ['exec_mode'],
            'Execute_cppsim': ['input_data'],
            'CreateStitchedIP': ['ip_config']
        }
        
        # Group parameters
        for transform in transformations:
            if transform in transform_params:
                groups[transform] = transform_params[transform]
        
        # Assign ungrouped parameters to general
        all_grouped = set()
        for param_list in groups.values():
            all_grouped.update(param_list)
        
        for param in parameters:
            if param not in all_grouped:
                groups['general'].append(param)
        
        return groups
    
    def _analyze_parameter_resource_impact(self, parameters: Dict[str, Any], 
                                         resource_budget: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """Analyze resource impact of each parameter."""
        impact = {}
        
        # Simplified resource impact analysis
        # In practice, this would use detailed models or historical data
        for param_name, param_value in parameters.items():
            param_impact = {}
            
            if 'parallelism' in param_name.lower() or 'pe' in param_name.lower():
                # Parallelism parameters typically increase LUT and DSP usage
                param_impact['lut'] = int(param_value) * 10 if isinstance(param_value, (int, float)) else 10
                param_impact['dsp'] = int(param_value) * 2 if isinstance(param_value, (int, float)) else 2
            
            elif 'memory' in param_name.lower() or 'buffer' in param_name.lower():
                # Memory parameters typically increase BRAM usage
                param_impact['bram'] = int(param_value) if isinstance(param_value, (int, float)) else 5
            
            elif 'precision' in param_name.lower() or 'width' in param_name.lower():
                # Precision parameters affect resource usage
                if isinstance(param_value, (int, float)):
                    width_factor = max(1, int(param_value) // 8)
                    param_impact['lut'] = width_factor * 5
                    param_impact['dsp'] = width_factor
            
            impact[param_name] = param_impact
        
        return impact
    
    def _estimate_resource_usage(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Estimate total resource usage for parameters."""
        usage = {'lut': 0, 'dsp': 0, 'bram': 0}
        
        impact_analysis = self._analyze_parameter_resource_impact(parameters, {})
        
        for param_impact in impact_analysis.values():
            for resource, impact in param_impact.items():
                usage[resource] = usage.get(resource, 0) + impact
        
        return usage
    
    def _estimate_parameter_resource_impact(self, param_name: str, param_value: Any) -> Dict[str, int]:
        """Estimate resource impact of a single parameter."""
        impact = {'lut': 0, 'dsp': 0, 'bram': 0}
        
        if 'parallelism' in param_name.lower():
            if isinstance(param_value, (int, float)):
                impact['lut'] = int(param_value) * 10
                impact['dsp'] = int(param_value) * 2
        
        elif 'memory' in param_name.lower():
            if isinstance(param_value, (int, float)):
                impact['bram'] = int(param_value)
        
        return impact
    
    def _validate_and_fix_architecture(self, candidate: FPGADesignCandidate, 
                                     design_space: Dict[str, Any]) -> FPGADesignCandidate:
        """Validate and fix architectural constraints."""
        # Check transformation sequence validity
        valid_sequences = design_space.get('valid_transformation_sequences', [])
        
        if valid_sequences and candidate.transformation_sequence not in valid_sequences:
            # Fix by choosing a similar valid sequence
            candidate.transformation_sequence = self._find_similar_valid_sequence(
                candidate.transformation_sequence, valid_sequences
            )
        
        # Check parameter compatibility
        for i, transform in enumerate(candidate.transformation_sequence):
            compatible_params = design_space.get(f'{transform}_compatible_params', [])
            if compatible_params:
                # Ensure parameters are compatible with transformation
                for param_name in list(candidate.parameters.keys()):
                    if param_name not in compatible_params:
                        # Remove incompatible parameter or set to default
                        if f'{transform}_default_params' in design_space:
                            defaults = design_space[f'{transform}_default_params']
                            if param_name in defaults:
                                candidate.parameters[param_name] = defaults[param_name]
        
        return candidate
    
    def _find_similar_valid_sequence(self, sequence: List[str], valid_sequences: List[List[str]]) -> List[str]:
        """Find most similar valid transformation sequence."""
        if not valid_sequences:
            return sequence
        
        # Simple similarity: count common transformations
        best_sequence = valid_sequences[0]
        best_similarity = 0
        
        for valid_seq in valid_sequences:
            common = len(set(sequence) & set(valid_seq))
            if common > best_similarity:
                best_similarity = common
                best_sequence = valid_seq
        
        return best_sequence


class FPGAGeneticAlgorithm:
    """FPGA-specific genetic algorithm with custom operators and selection."""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 selection_pressure: float = 2.0):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        
        self.genetic_operators = FPGAGeneticOperators()
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.generation_count = 0
    
    def evolve(self, 
              fitness_function: Callable[[FPGADesignCandidate], float],
              design_space: Dict[str, Any],
              constraint_functions: List[Callable] = None) -> FPGADesignCandidate:
        """Evolve population to find optimal FPGA design."""
        
        logger.info(f"Starting FPGA genetic algorithm: {self.population_size} population, {self.max_generations} generations")
        
        # Initialize population
        self._initialize_population(design_space)
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation_count = generation
            
            # Evaluate fitness
            self._evaluate_population(fitness_function, constraint_functions)
            
            # Track best individual
            current_best = max(self.population, key=lambda x: x.metadata.get('fitness', -float('inf')))
            if self.best_individual is None or current_best.metadata.get('fitness', -float('inf')) > self.best_individual.metadata.get('fitness', -float('inf')):
                self.best_individual = copy.deepcopy(current_best)
            
            # Create new generation
            new_population = self._create_new_generation(design_space)
            self.population = new_population
            
            # Log progress
            if generation % 10 == 0:
                avg_fitness = np.mean([ind.metadata.get('fitness', 0) for ind in self.population])
                best_fitness = self.best_individual.metadata.get('fitness', 0)
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}, Avg fitness = {avg_fitness:.4f}")
            
            # Store generation statistics
            generation_stats = {
                'generation': generation,
                'best_fitness': self.best_individual.metadata.get('fitness', 0),
                'avg_fitness': np.mean([ind.metadata.get('fitness', 0) for ind in self.population]),
                'population_diversity': self._calculate_population_diversity()
            }
            self.fitness_history.append(generation_stats)
        
        logger.info(f"FPGA GA completed: Best fitness = {self.best_individual.metadata.get('fitness', 0):.4f}")
        return self.best_individual
    
    def _initialize_population(self, design_space: Dict[str, Any]):
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            # Generate random parameters
            parameters = {}
            for param_name, param_range in design_space.items():
                if param_name.startswith('_'):  # Skip internal design space parameters
                    continue
                
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    min_val, max_val = param_range
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            parameters[param_name] = random.randint(min_val, max_val)
                        else:
                            parameters[param_name] = random.uniform(min_val, max_val)
                elif isinstance(param_range, (list, tuple)):
                    parameters[param_name] = random.choice(param_range)
            
            # Generate random transformation sequence
            available_transforms = design_space.get('available_transformations', [])
            if available_transforms:
                seq_length = random.randint(1, min(10, len(available_transforms)))
                transformation_sequence = random.sample(available_transforms, seq_length)
            else:
                transformation_sequence = []
            
            # Create candidate
            candidate = FPGADesignCandidate(
                parameters=parameters,
                architecture=design_space.get('target_architecture', 'generic'),
                transformation_sequence=transformation_sequence,
                resource_budget=design_space.get('resource_budget', {})
            )
            
            self.population.append(candidate)
    
    def _evaluate_population(self, fitness_function: Callable, constraint_functions: List[Callable] = None):
        """Evaluate fitness for entire population."""
        
        for individual in self.population:
            if 'fitness' not in individual.metadata:  # Only evaluate if not already evaluated
                try:
                    # Calculate base fitness
                    fitness = fitness_function(individual)
                    
                    # Apply constraint penalties
                    if constraint_functions:
                        penalty = 0.0
                        for constraint_func in constraint_functions:
                            violation = constraint_func(individual)
                            if violation > 0:
                                penalty += violation
                        
                        # Apply penalty to fitness
                        fitness = fitness - penalty
                    
                    individual.metadata['fitness'] = fitness
                    individual.metadata['constraint_penalty'] = penalty if constraint_functions else 0.0
                    
                except Exception as e:
                    logger.error(f"Fitness evaluation failed: {e}")
                    individual.metadata['fitness'] = -float('inf')
                    individual.metadata['constraint_penalty'] = float('inf')
    
    def _create_new_generation(self, design_space: Dict[str, Any]) -> List[FPGADesignCandidate]:
        """Create new generation through selection, crossover, and mutation."""
        
        new_population = []
        
        # Elitism: keep best individuals
        elite_size = max(1, int(0.1 * self.population_size))
        elite = sorted(self.population, key=lambda x: x.metadata.get('fitness', -float('inf')), reverse=True)[:elite_size]
        new_population.extend([copy.deepcopy(ind) for ind in elite])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Parent selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self.genetic_operators.crossover(parent1, parent2, 'transformation_aware')
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self.genetic_operators.mutate(child1, design_space, 'resource_aware_mutation', self.mutation_rate)
            
            if random.random() < self.mutation_rate:
                child2 = self.genetic_operators.mutate(child2, design_space, 'resource_aware_mutation', self.mutation_rate)
            
            # Clear fitness (will be re-evaluated)
            child1.metadata.pop('fitness', None)
            child2.metadata.pop('fitness', None)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> FPGADesignCandidate:
        """Tournament selection for parent selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.metadata.get('fitness', -float('inf')))
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity based on parameter variance."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate variance for numerical parameters
        param_variances = []
        
        # Get all parameter names
        all_params = set()
        for individual in self.population:
            all_params.update(individual.parameters.keys())
        
        for param_name in all_params:
            param_values = []
            for individual in self.population:
                if param_name in individual.parameters:
                    value = individual.parameters[param_name]
                    if isinstance(value, (int, float)):
                        param_values.append(value)
            
            if len(param_values) > 1:
                variance = np.var(param_values)
                param_variances.append(variance)
        
        return np.mean(param_variances) if param_variances else 0.0


class AdaptiveSimulatedAnnealing:
    """Adaptive simulated annealing for FPGA design optimization."""
    
    def __init__(self, 
                 initial_temperature: float = 100.0,
                 final_temperature: float = 0.01,
                 max_iterations: int = 10000,
                 cooling_schedule: str = 'exponential'):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        
        self.current_solution = None
        self.best_solution = None
        self.temperature = initial_temperature
        self.iteration = 0
        self.acceptance_history = []
    
    def optimize(self, 
                objective_function: Callable[[Dict[str, Any]], float],
                design_space: Dict[str, Any],
                initial_solution: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run simulated annealing optimization."""
        
        logger.info(f"Starting adaptive simulated annealing: {self.max_iterations} iterations")
        
        # Initialize solution
        if initial_solution:
            self.current_solution = initial_solution.copy()
        else:
            self.current_solution = self._generate_random_solution(design_space)
        
        current_objective = objective_function(self.current_solution)
        
        self.best_solution = self.current_solution.copy()
        best_objective = current_objective
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            # Generate neighbor solution
            neighbor = self._generate_neighbor(self.current_solution, design_space)
            neighbor_objective = objective_function(neighbor)
            
            # Calculate acceptance probability
            if neighbor_objective > current_objective:
                # Better solution - always accept
                accept = True
                acceptance_prob = 1.0
            else:
                # Worse solution - accept with probability based on temperature
                delta = neighbor_objective - current_objective
                acceptance_prob = math.exp(delta / self.temperature)
                accept = random.random() < acceptance_prob
            
            # Update solution
            if accept:
                self.current_solution = neighbor
                current_objective = neighbor_objective
                
                # Update best solution
                if neighbor_objective > best_objective:
                    self.best_solution = neighbor.copy()
                    best_objective = neighbor_objective
            
            # Update temperature
            self.temperature = self._update_temperature(iteration)
            
            # Track acceptance history
            self.acceptance_history.append({
                'iteration': iteration,
                'temperature': self.temperature,
                'current_objective': current_objective,
                'best_objective': best_objective,
                'accepted': accept,
                'acceptance_probability': acceptance_prob
            })
            
            # Log progress
            if iteration % 1000 == 0:
                logger.info(f"Iteration {iteration}: Best = {best_objective:.4f}, Current = {current_objective:.4f}, T = {self.temperature:.4f}")
        
        logger.info(f"Simulated annealing completed: Best objective = {best_objective:.4f}")
        return self.best_solution
    
    def _generate_random_solution(self, design_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random solution within design space."""
        solution = {}
        
        for param_name, param_range in design_space.items():
            if param_name.startswith('_'):
                continue
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        solution[param_name] = random.randint(min_val, max_val)
                    else:
                        solution[param_name] = random.uniform(min_val, max_val)
            elif isinstance(param_range, (list, tuple)):
                solution[param_name] = random.choice(param_range)
        
        return solution
    
    def _generate_neighbor(self, solution: Dict[str, Any], design_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate neighbor solution by perturbing current solution."""
        neighbor = solution.copy()
        
        # Select random parameter to modify
        param_names = [name for name in solution.keys() if not name.startswith('_')]
        if not param_names:
            return neighbor
        
        param_to_modify = random.choice(param_names)
        
        if param_to_modify in design_space:
            param_range = design_space[param_to_modify]
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    current_val = solution[param_to_modify]
                    
                    # Adaptive step size based on temperature
                    step_size = (max_val - min_val) * (self.temperature / self.initial_temperature) * 0.1
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        step = max(1, int(step_size))
                        new_val = current_val + random.randint(-step, step)
                        neighbor[param_to_modify] = max(min_val, min(max_val, new_val))
                    else:
                        new_val = current_val + random.uniform(-step_size, step_size)
                        neighbor[param_to_modify] = max(min_val, min(max_val, new_val))
            
            elif isinstance(param_range, (list, tuple)):
                neighbor[param_to_modify] = random.choice(param_range)
        
        return neighbor
    
    def _update_temperature(self, iteration: int) -> float:
        """Update temperature according to cooling schedule."""
        progress = iteration / self.max_iterations
        
        if self.cooling_schedule == 'exponential':
            alpha = (self.final_temperature / self.initial_temperature) ** (1.0 / self.max_iterations)
            return self.initial_temperature * (alpha ** iteration)
        
        elif self.cooling_schedule == 'linear':
            return self.initial_temperature * (1.0 - progress) + self.final_temperature * progress
        
        elif self.cooling_schedule == 'adaptive':
            # Adaptive cooling based on acceptance rate
            recent_acceptances = self.acceptance_history[-100:] if len(self.acceptance_history) >= 100 else self.acceptance_history
            acceptance_rate = sum(1 for record in recent_acceptances if record['accepted']) / max(1, len(recent_acceptances))
            
            # Slow cooling if acceptance rate is good, fast cooling if too low
            if acceptance_rate > 0.5:
                cooling_factor = 0.95  # Slow cooling
            elif acceptance_rate > 0.1:
                cooling_factor = 0.90  # Normal cooling
            else:
                cooling_factor = 0.85  # Fast cooling
            
            return max(self.final_temperature, self.temperature * cooling_factor)
        
        else:
            # Default to exponential
            alpha = (self.final_temperature / self.initial_temperature) ** (1.0 / self.max_iterations)
            return self.initial_temperature * (alpha ** iteration)


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for continuous FPGA design parameters."""
    
    def __init__(self, 
                 swarm_size: int = 50,
                 max_iterations: int = 1000,
                 inertia: float = 0.7,
                 cognitive_factor: float = 2.0,
                 social_factor: float = 2.0):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive_factor = cognitive_factor
        self.social_factor = social_factor
        
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = -float('inf')
        self.fitness_history = []
    
    def optimize(self, 
                objective_function: Callable[[Dict[str, Any]], float],
                design_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run particle swarm optimization."""
        
        logger.info(f"Starting particle swarm optimization: {self.swarm_size} particles, {self.max_iterations} iterations")
        
        # Initialize swarm
        self._initialize_swarm(design_space)
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            # Evaluate particles
            for particle in self.particles:
                fitness = objective_function(particle['position'])
                particle['fitness'] = fitness
                
                # Update personal best
                if fitness > particle['best_fitness']:
                    particle['best_position'] = particle['position'].copy()
                    particle['best_fitness'] = fitness
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_position = particle['position'].copy()
                    self.global_best_fitness = fitness
            
            # Update particle velocities and positions
            self._update_particles(design_space)
            
            # Log progress
            if iteration % 100 == 0:
                avg_fitness = np.mean([p['fitness'] for p in self.particles])
                logger.info(f"Iteration {iteration}: Best = {self.global_best_fitness:.4f}, Avg = {avg_fitness:.4f}")
            
            # Store iteration statistics
            self.fitness_history.append({
                'iteration': iteration,
                'best_fitness': self.global_best_fitness,
                'avg_fitness': np.mean([p['fitness'] for p in self.particles])
            })
        
        logger.info(f"PSO completed: Best fitness = {self.global_best_fitness:.4f}")
        return self.global_best_position
    
    def _initialize_swarm(self, design_space: Dict[str, Any]):
        """Initialize particle swarm."""
        self.particles = []
        
        # Extract continuous parameters
        continuous_params = {}
        for param_name, param_range in design_space.items():
            if param_name.startswith('_'):
                continue
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    continuous_params[param_name] = (min_val, max_val)
        
        for _ in range(self.swarm_size):
            # Initialize position
            position = {}
            velocity = {}
            
            for param_name, (min_val, max_val) in continuous_params.items():
                position[param_name] = random.uniform(min_val, max_val)
                velocity[param_name] = random.uniform(-(max_val - min_val) * 0.1, (max_val - min_val) * 0.1)
            
            particle = {
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': -float('inf'),
                'fitness': -float('inf')
            }
            
            self.particles.append(particle)
    
    def _update_particles(self, design_space: Dict[str, Any]):
        """Update particle velocities and positions."""
        
        for particle in self.particles:
            for param_name in particle['position']:
                if param_name in design_space:
                    min_val, max_val = design_space[param_name]
                    
                    # Update velocity
                    r1, r2 = random.random(), random.random()
                    
                    cognitive_component = self.cognitive_factor * r1 * (
                        particle['best_position'][param_name] - particle['position'][param_name]
                    )
                    
                    social_component = self.social_factor * r2 * (
                        self.global_best_position[param_name] - particle['position'][param_name]
                    ) if self.global_best_position else 0.0
                    
                    particle['velocity'][param_name] = (
                        self.inertia * particle['velocity'][param_name] +
                        cognitive_component + social_component
                    )
                    
                    # Limit velocity
                    max_velocity = (max_val - min_val) * 0.2
                    particle['velocity'][param_name] = max(-max_velocity, min(max_velocity, particle['velocity'][param_name]))
                    
                    # Update position
                    particle['position'][param_name] += particle['velocity'][param_name]
                    
                    # Keep within bounds
                    particle['position'][param_name] = max(min_val, min(max_val, particle['position'][param_name]))


class HybridDSEFramework:
    """Hybrid optimization framework combining multiple algorithms."""
    
    def __init__(self):
        self.algorithms = {
            'genetic': FPGAGeneticAlgorithm(),
            'simulated_annealing': AdaptiveSimulatedAnnealing(),
            'particle_swarm': ParticleSwarmOptimizer()
        }
        
        self.algorithm_performance = {}
        self.optimization_history = []
    
    def optimize(self, 
                objective_function: Callable,
                design_space: Dict[str, Any],
                strategy: str = 'adaptive',
                max_time: float = 3600.0) -> Dict[str, Any]:
        """Run hybrid optimization with multiple algorithms."""
        
        logger.info(f"Starting hybrid DSE optimization with strategy: {strategy}")
        
        start_time = time.time()
        best_solution = None
        best_fitness = -float('inf')
        
        if strategy == 'sequential':
            # Run algorithms sequentially
            best_solution = self._sequential_optimization(objective_function, design_space, max_time)
        
        elif strategy == 'parallel':
            # Run algorithms in parallel
            best_solution = self._parallel_optimization(objective_function, design_space, max_time)
        
        elif strategy == 'adaptive':
            # Adaptively switch between algorithms
            best_solution = self._adaptive_optimization(objective_function, design_space, max_time)
        
        else:
            # Default to genetic algorithm
            ga = FPGAGeneticAlgorithm(max_generations=100)
            best_solution = ga.evolve(
                lambda candidate: objective_function(candidate.parameters),
                design_space
            )
            best_solution = best_solution.parameters
        
        total_time = time.time() - start_time
        logger.info(f"Hybrid DSE completed in {total_time:.2f}s")
        
        return best_solution
    
    def _sequential_optimization(self, objective_function: Callable, design_space: Dict[str, Any], max_time: float) -> Dict[str, Any]:
        """Run algorithms sequentially, using output of one as input to next."""
        
        time_per_algorithm = max_time / len(self.algorithms)
        current_solution = None
        
        for alg_name, algorithm in self.algorithms.items():
            logger.info(f"Running {alg_name} for {time_per_algorithm:.1f}s")
            
            if alg_name == 'genetic':
                algorithm.max_generations = min(50, algorithm.max_generations)
                result = algorithm.evolve(
                    lambda candidate: objective_function(candidate.parameters),
                    design_space
                )
                current_solution = result.parameters
            
            elif alg_name == 'simulated_annealing':
                algorithm.max_iterations = min(5000, algorithm.max_iterations)
                current_solution = algorithm.optimize(objective_function, design_space, current_solution)
            
            elif alg_name == 'particle_swarm':
                algorithm.max_iterations = min(500, algorithm.max_iterations)
                current_solution = algorithm.optimize(objective_function, design_space)
        
        return current_solution
    
    def _parallel_optimization(self, objective_function: Callable, design_space: Dict[str, Any], max_time: float) -> Dict[str, Any]:
        """Run algorithms in parallel and return best result."""
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.algorithms)) as executor:
            # Submit all algorithms
            futures = {}
            
            for alg_name, algorithm in self.algorithms.items():
                if alg_name == 'genetic':
                    algorithm.max_generations = 50
                    future = executor.submit(
                        lambda: algorithm.evolve(
                            lambda candidate: objective_function(candidate.parameters),
                            design_space
                        ).parameters
                    )
                
                elif alg_name == 'simulated_annealing':
                    algorithm.max_iterations = 5000
                    future = executor.submit(algorithm.optimize, objective_function, design_space)
                
                elif alg_name == 'particle_swarm':
                    algorithm.max_iterations = 500
                    future = executor.submit(algorithm.optimize, objective_function, design_space)
                
                futures[future] = alg_name
            
            # Collect results
            for future in as_completed(futures, timeout=max_time):
                alg_name = futures[future]
                try:
                    result = future.result()
                    fitness = objective_function(result)
                    results[alg_name] = {'solution': result, 'fitness': fitness}
                    logger.info(f"{alg_name} completed with fitness: {fitness:.4f}")
                except Exception as e:
                    logger.error(f"{alg_name} failed: {e}")
        
        # Return best result
        if results:
            best_alg = max(results.keys(), key=lambda k: results[k]['fitness'])
            return results[best_alg]['solution']
        else:
            # Fallback to random solution
            return self._generate_random_solution(design_space)
    
    def _adaptive_optimization(self, objective_function: Callable, design_space: Dict[str, Any], max_time: float) -> Dict[str, Any]:
        """Adaptively switch between algorithms based on performance."""
        
        start_time = time.time()
        phase_time = max_time / 3  # Three phases
        
        best_solution = None
        best_fitness = -float('inf')
        
        # Phase 1: Exploration with GA
        logger.info("Phase 1: Genetic Algorithm exploration")
        ga = FPGAGeneticAlgorithm(max_generations=30)
        ga_result = ga.evolve(
            lambda candidate: objective_function(candidate.parameters),
            design_space
        )
        
        if ga_result:
            current_solution = ga_result.parameters
            current_fitness = objective_function(current_solution)
            
            if current_fitness > best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness
        
        # Phase 2: Local search with SA
        if time.time() - start_time < max_time * 0.67:
            logger.info("Phase 2: Simulated Annealing local search")
            sa = AdaptiveSimulatedAnnealing(max_iterations=3000)
            sa_result = sa.optimize(objective_function, design_space, current_solution)
            
            sa_fitness = objective_function(sa_result)
            if sa_fitness > best_fitness:
                best_solution = sa_result
                best_fitness = sa_fitness
                current_solution = sa_result
        
        # Phase 3: Fine-tuning with PSO
        if time.time() - start_time < max_time * 0.9:
            logger.info("Phase 3: Particle Swarm Optimization fine-tuning")
            pso = ParticleSwarmOptimizer(max_iterations=300)
            pso_result = pso.optimize(objective_function, design_space)
            
            pso_fitness = objective_function(pso_result)
            if pso_fitness > best_fitness:
                best_solution = pso_result
                best_fitness = pso_fitness
        
        return best_solution
    
    def _generate_random_solution(self, design_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random solution as fallback."""
        solution = {}
        
        for param_name, param_range in design_space.items():
            if param_name.startswith('_'):
                continue
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        solution[param_name] = random.randint(min_val, max_val)
                    else:
                        solution[param_name] = random.uniform(min_val, max_val)
            elif isinstance(param_range, (list, tuple)):
                solution[param_name] = random.choice(param_range)
        
        return solution