"""
Multi-Objective Optimization Framework
NSGA-II, SPEA2, MOEA/D algorithms with Pareto frontier management and hypervolume calculation.
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
import threading

logger = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    """Represents a solution in the Pareto frontier."""
    design_parameters: Dict[str, Any]
    objective_values: List[float]
    constraint_violations: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: float = 0.0
    solution_id: str = ""
    
    def __post_init__(self):
        if not self.solution_id:
            self.solution_id = f"sol_{int(time.time() * 1000000) % 1000000}"
    
    @property
    def is_feasible(self) -> bool:
        """Check if solution satisfies all constraints."""
        return len(self.constraint_violations) == 0 or all(v <= 0 for v in self.constraint_violations)
    
    @property
    def constraint_violation_sum(self) -> float:
        """Sum of constraint violations."""
        return sum(max(0, v) for v in self.constraint_violations)
    
    def dominates(self, other: 'ParetoSolution', minimize_objectives: List[bool] = None) -> bool:
        """Check if this solution dominates another."""
        if minimize_objectives is None:
            minimize_objectives = [True] * len(self.objective_values)
        
        assert len(self.objective_values) == len(other.objective_values), "Objective count mismatch"
        assert len(minimize_objectives) == len(self.objective_values), "Minimize flags count mismatch"
        
        better_in_any = False
        for i, (obj1, obj2, minimize) in enumerate(zip(self.objective_values, other.objective_values, minimize_objectives)):
            if minimize:
                if obj1 > obj2:  # Worse in this objective
                    return False
                if obj1 < obj2:  # Better in this objective
                    better_in_any = True
            else:  # Maximize
                if obj1 < obj2:  # Worse in this objective
                    return False
                if obj1 > obj2:  # Better in this objective
                    better_in_any = True
        
        return better_in_any


class ParetoArchive:
    """Manages a collection of Pareto-optimal solutions."""
    
    def __init__(self, max_size: Optional[int] = None):
        self.solutions = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.minimize_objectives = None
    
    def add_solution(self, solution: ParetoSolution, minimize_objectives: List[bool] = None) -> bool:
        """Add solution to archive, maintaining Pareto optimality."""
        with self.lock:
            if minimize_objectives is not None:
                self.minimize_objectives = minimize_objectives
            
            # Check if solution is dominated by any existing solution
            for existing in self.solutions:
                if existing.dominates(solution, self.minimize_objectives):
                    return False  # Solution is dominated, don't add
            
            # Remove solutions dominated by new solution
            self.solutions = [s for s in self.solutions if not solution.dominates(s, self.minimize_objectives)]
            
            # Add new solution
            self.solutions.append(solution)
            
            # Maintain size limit if specified
            if self.max_size and len(self.solutions) > self.max_size:
                self._reduce_archive_size()
            
            return True
    
    def _reduce_archive_size(self):
        """Reduce archive size using crowding distance."""
        if len(self.solutions) <= self.max_size:
            return
        
        # Calculate crowding distances
        crowding_distances = self._calculate_crowding_distances()
        
        # Sort by crowding distance (descending) and keep top solutions
        indexed_solutions = list(zip(self.solutions, crowding_distances))
        indexed_solutions.sort(key=lambda x: x[1], reverse=True)
        
        self.solutions = [sol for sol, _ in indexed_solutions[:self.max_size]]
    
    def _calculate_crowding_distances(self) -> List[float]:
        """Calculate crowding distance for each solution."""
        if len(self.solutions) <= 2:
            return [float('inf')] * len(self.solutions)
        
        num_objectives = len(self.solutions[0].objective_values)
        distances = [0.0] * len(self.solutions)
        
        for obj_idx in range(num_objectives):
            # Sort solutions by this objective
            indexed_solutions = list(enumerate(self.solutions))
            indexed_solutions.sort(key=lambda x: x[1].objective_values[obj_idx])
            
            # Set boundary solutions to infinite distance
            distances[indexed_solutions[0][0]] = float('inf')
            distances[indexed_solutions[-1][0]] = float('inf')
            
            # Calculate objective range
            obj_values = [sol.objective_values[obj_idx] for _, sol in indexed_solutions]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range == 0:
                continue
            
            # Calculate distances for intermediate solutions
            for i in range(1, len(indexed_solutions) - 1):
                idx = indexed_solutions[i][0]
                if distances[idx] != float('inf'):
                    prev_val = indexed_solutions[i-1][1].objective_values[obj_idx]
                    next_val = indexed_solutions[i+1][1].objective_values[obj_idx]
                    distances[idx] += (next_val - prev_val) / obj_range
        
        return distances
    
    def get_pareto_frontier(self) -> List[ParetoSolution]:
        """Get current Pareto frontier."""
        with self.lock:
            return copy.deepcopy(self.solutions)
    
    def size(self) -> int:
        """Get number of solutions in archive."""
        return len(self.solutions)
    
    def clear(self):
        """Clear all solutions."""
        with self.lock:
            self.solutions.clear()


class HypervolumeCalculator:
    """Calculate hypervolume indicator for multi-objective optimization."""
    
    def __init__(self, reference_point: List[float]):
        self.reference_point = reference_point
    
    def calculate(self, solutions: List[ParetoSolution], minimize_objectives: List[bool] = None) -> float:
        """Calculate hypervolume for a set of solutions."""
        if not solutions:
            return 0.0
        
        if minimize_objectives is None:
            minimize_objectives = [True] * len(solutions[0].objective_values)
        
        # Convert to numpy array for easier manipulation
        objectives_matrix = np.array([sol.objective_values for sol in solutions])
        
        # For minimization objectives, we need to transform the problem
        # Hypervolume is typically calculated for maximization
        transformed_objectives = objectives_matrix.copy()
        transformed_reference = np.array(self.reference_point.copy())
        
        for i, minimize in enumerate(minimize_objectives):
            if minimize:
                # Transform minimization to maximization
                max_val = max(transformed_reference[i], np.max(transformed_objectives[:, i]))
                transformed_objectives[:, i] = max_val - transformed_objectives[:, i]
                transformed_reference[i] = max_val - transformed_reference[i]
        
        return self._calculate_hypervolume_recursive(transformed_objectives, transformed_reference)
    
    def _calculate_hypervolume_recursive(self, points: np.ndarray, reference: np.ndarray) -> float:
        """Recursive hypervolume calculation using WFG algorithm."""
        if len(points) == 0:
            return 0.0
        
        if points.shape[1] == 1:
            # 1D case
            return max(0, np.max(points[:, 0]) - reference[0])
        
        if len(points) == 1:
            # Single point case
            volume = 1.0
            for i in range(len(reference)):
                volume *= max(0, points[0, i] - reference[i])
            return volume
        
        # Multi-dimensional case - use sweep line algorithm
        # Sort points by last objective
        last_obj_idx = points.shape[1] - 1
        sorted_indices = np.argsort(points[:, last_obj_idx])
        sorted_points = points[sorted_indices]
        
        total_volume = 0.0
        prev_value = reference[last_obj_idx]
        
        for i, point in enumerate(sorted_points):
            if point[last_obj_idx] > prev_value:
                # Calculate hypervolume contribution
                height = point[last_obj_idx] - prev_value
                
                # Get dominated points (points that are dominated by current point in all other objectives)
                dominated_points = []
                for j in range(i + 1):
                    other_point = sorted_points[j]
                    if all(other_point[k] >= point[k] for k in range(last_obj_idx)):
                        dominated_points.append(other_point[:last_obj_idx])
                
                if dominated_points:
                    dominated_points = np.array(dominated_points)
                    # Remove duplicates
                    dominated_points = np.unique(dominated_points, axis=0)
                    
                    # Recursive call for lower dimensional problem
                    lower_dim_volume = self._calculate_hypervolume_recursive(
                        dominated_points, reference[:last_obj_idx]
                    )
                    total_volume += height * lower_dim_volume
                
                prev_value = point[last_obj_idx]
        
        return total_volume


class MultiObjectiveOptimizer(ABC):
    """Abstract base class for multi-objective optimizers."""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_generations: int = 100,
                 minimize_objectives: List[bool] = None):
        self.population_size = population_size
        self.max_generations = max_generations
        self.minimize_objectives = minimize_objectives
        self.pareto_archive = ParetoArchive()
        self.evaluation_count = 0
        self.generation_count = 0
        self.optimization_history = []
    
    @abstractmethod
    def optimize(self, 
                objective_functions: List[Callable],
                design_space: Dict[str, Tuple[Any, Any]],
                constraints: List[Callable] = None) -> List[ParetoSolution]:
        """Run optimization and return Pareto frontier."""
        pass
    
    def _evaluate_solution(self, 
                          design_parameters: Dict[str, Any],
                          objective_functions: List[Callable],
                          constraints: List[Callable] = None) -> ParetoSolution:
        """Evaluate a single solution."""
        start_time = time.time()
        
        # Evaluate objectives
        objective_values = []
        for obj_func in objective_functions:
            try:
                value = obj_func(design_parameters)
                objective_values.append(float(value))
            except Exception as e:
                logger.error(f"Objective evaluation failed: {e}")
                objective_values.append(float('inf'))
        
        # Evaluate constraints
        constraint_violations = []
        if constraints:
            for constraint in constraints:
                try:
                    violation = constraint(design_parameters)
                    constraint_violations.append(float(violation))
                except Exception as e:
                    logger.error(f"Constraint evaluation failed: {e}")
                    constraint_violations.append(float('inf'))
        
        evaluation_time = time.time() - start_time
        self.evaluation_count += 1
        
        return ParetoSolution(
            design_parameters=design_parameters.copy(),
            objective_values=objective_values,
            constraint_violations=constraint_violations,
            evaluation_time=evaluation_time
        )
    
    def _generate_random_solution(self, design_space: Dict[str, Tuple[Any, Any]]) -> Dict[str, Any]:
        """Generate random solution within design space."""
        solution = {}
        
        for param_name, (min_val, max_val) in design_space.items():
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                if isinstance(min_val, int) and isinstance(max_val, int):
                    solution[param_name] = random.randint(min_val, max_val)
                else:
                    solution[param_name] = random.uniform(min_val, max_val)
            elif isinstance(min_val, (list, tuple)):
                # Discrete choice
                solution[param_name] = random.choice(min_val)
            else:
                # String or other type - assume it's a choice
                if hasattr(min_val, '__iter__') and not isinstance(min_val, str):
                    solution[param_name] = random.choice(list(min_val))
                else:
                    solution[param_name] = min_val
        
        return solution


class NSGA2(MultiObjectiveOptimizer):
    """Non-dominated Sorting Genetic Algorithm II (NSGA-II)."""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_generations: int = 100,
                 crossover_probability: float = 0.8,
                 mutation_probability: float = 0.1,
                 minimize_objectives: List[bool] = None):
        super().__init__(population_size, max_generations, minimize_objectives)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
    
    def optimize(self, 
                objective_functions: List[Callable],
                design_space: Dict[str, Tuple[Any, Any]],
                constraints: List[Callable] = None) -> List[ParetoSolution]:
        """Run NSGA-II optimization."""
        
        logger.info(f"Starting NSGA-II optimization: {self.population_size} population, {self.max_generations} generations")
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            design_params = self._generate_random_solution(design_space)
            solution = self._evaluate_solution(design_params, objective_functions, constraints)
            population.append(solution)
        
        # Set minimize objectives if not specified
        if self.minimize_objectives is None:
            self.minimize_objectives = [True] * len(objective_functions)
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation_count = generation
            
            # Create offspring through crossover and mutation
            offspring = self._create_offspring(population, design_space)
            
            # Evaluate offspring
            for child in offspring:
                if hasattr(child, 'objective_values'):
                    continue  # Already evaluated
                
                solution = self._evaluate_solution(child.design_parameters, objective_functions, constraints)
                child.objective_values = solution.objective_values
                child.constraint_violations = solution.constraint_violations
                child.evaluation_time = solution.evaluation_time
            
            # Combine population and offspring
            combined_population = population + offspring
            
            # Non-dominated sorting and selection
            population = self._environmental_selection(combined_population)
            
            # Update Pareto archive
            for solution in population:
                if solution.is_feasible:
                    self.pareto_archive.add_solution(solution, self.minimize_objectives)
            
            # Log progress
            if generation % 10 == 0:
                best_objectives = self._get_best_objectives(population)
                logger.info(f"Generation {generation}: Best objectives = {best_objectives}")
            
            # Store generation history
            self.optimization_history.append({
                'generation': generation,
                'evaluation_count': self.evaluation_count,
                'pareto_size': self.pareto_archive.size(),
                'population_size': len(population)
            })
        
        logger.info(f"NSGA-II completed: {self.evaluation_count} evaluations, {self.pareto_archive.size()} Pareto solutions")
        return self.pareto_archive.get_pareto_frontier()
    
    def _create_offspring(self, population: List[ParetoSolution], design_space: Dict[str, Tuple[Any, Any]]) -> List[ParetoSolution]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        for _ in range(self.population_size):
            # Select parents via tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_probability:
                child_params = self._crossover(parent1.design_parameters, parent2.design_parameters, design_space)
            else:
                child_params = parent1.design_parameters.copy()
            
            # Mutation
            if random.random() < self.mutation_probability:
                child_params = self._mutate(child_params, design_space)
            
            offspring.append(ParetoSolution(design_parameters=child_params, objective_values=[]))
        
        return offspring
    
    def _tournament_selection(self, population: List[ParetoSolution], tournament_size: int = 2) -> ParetoSolution:
        """Tournament selection for parent selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Prefer feasible solutions
        feasible = [sol for sol in tournament if sol.is_feasible]
        if feasible:
            tournament = feasible
        
        # Select based on dominance and crowding distance
        best = tournament[0]
        for candidate in tournament[1:]:
            if candidate.dominates(best, self.minimize_objectives):
                best = candidate
            elif not best.dominates(candidate, self.minimize_objectives):
                # Neither dominates - prefer less crowded solution
                # This is a simplified version - in practice would calculate actual crowding distance
                if random.random() < 0.5:
                    best = candidate
        
        return best
    
    def _crossover(self, parent1_params: Dict[str, Any], parent2_params: Dict[str, Any], 
                  design_space: Dict[str, Tuple[Any, Any]]) -> Dict[str, Any]:
        """Crossover operation between two parents."""
        child_params = {}
        
        for param_name in parent1_params:
            if random.random() < 0.5:
                child_params[param_name] = parent1_params[param_name]
            else:
                child_params[param_name] = parent2_params[param_name]
            
            # For numerical parameters, apply simulated binary crossover
            if param_name in design_space:
                min_val, max_val = design_space[param_name]
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    p1_val = parent1_params[param_name]
                    p2_val = parent2_params[param_name]
                    
                    if random.random() < 0.5:  # Apply SBX
                        eta = 2.0  # Distribution index
                        u = random.random()
                        
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        
                        child_val = 0.5 * ((1 + beta) * p1_val + (1 - beta) * p2_val)
                        child_val = max(min_val, min(max_val, child_val))
                        
                        if isinstance(min_val, int):
                            child_val = int(round(child_val))
                        
                        child_params[param_name] = child_val
        
        return child_params
    
    def _mutate(self, params: Dict[str, Any], design_space: Dict[str, Tuple[Any, Any]]) -> Dict[str, Any]:
        """Mutation operation."""
        mutated_params = params.copy()
        
        for param_name in params:
            if param_name in design_space and random.random() < 0.1:  # Per-parameter mutation probability
                min_val, max_val = design_space[param_name]
                
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    # Polynomial mutation for numerical parameters
                    eta = 20.0  # Distribution index
                    current_val = params[param_name]
                    
                    delta1 = (current_val - min_val) / (max_val - min_val)
                    delta2 = (max_val - current_val) / (max_val - min_val)
                    
                    u = random.random()
                    
                    if u <= 0.5:
                        xy = 1.0 - delta1
                        val = (2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1)) ** (1.0 / (eta + 1)) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1)) ** (1.0 / (eta + 1))
                    
                    mutated_val = current_val + val * (max_val - min_val)
                    mutated_val = max(min_val, min(max_val, mutated_val))
                    
                    if isinstance(min_val, int):
                        mutated_val = int(round(mutated_val))
                    
                    mutated_params[param_name] = mutated_val
                
                elif isinstance(min_val, (list, tuple)):
                    # Random choice for discrete parameters
                    mutated_params[param_name] = random.choice(min_val)
        
        return mutated_params
    
    def _environmental_selection(self, population: List[ParetoSolution]) -> List[ParetoSolution]:
        """Environmental selection using non-dominated sorting and crowding distance."""
        
        # Non-dominated sorting
        fronts = self._non_dominated_sorting(population)
        
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                # Add part of the front based on crowding distance
                remaining_slots = self.population_size - len(selected)
                if remaining_slots > 0:
                    crowding_distances = self._calculate_crowding_distances_for_front(front)
                    
                    # Sort by crowding distance (descending)
                    front_with_distances = list(zip(front, crowding_distances))
                    front_with_distances.sort(key=lambda x: x[1], reverse=True)
                    
                    selected.extend([sol for sol, _ in front_with_distances[:remaining_slots]])
                break
        
        return selected
    
    def _non_dominated_sorting(self, population: List[ParetoSolution]) -> List[List[ParetoSolution]]:
        """Perform non-dominated sorting on population."""
        fronts = []
        domination_counts = {}  # Number of solutions that dominate this solution
        dominated_solutions = {}  # Solutions dominated by this solution
        
        # Initialize
        for sol in population:
            domination_counts[sol.solution_id] = 0
            dominated_solutions[sol.solution_id] = []
        
        # Calculate domination relationships
        for i, sol1 in enumerate(population):
            for j, sol2 in enumerate(population):
                if i != j:
                    if sol1.dominates(sol2, self.minimize_objectives):
                        dominated_solutions[sol1.solution_id].append(sol2)
                    elif sol2.dominates(sol1, self.minimize_objectives):
                        domination_counts[sol1.solution_id] += 1
        
        # Create first front
        current_front = []
        for sol in population:
            if domination_counts[sol.solution_id] == 0:
                current_front.append(sol)
        
        fronts.append(current_front)
        
        # Create subsequent fronts
        while current_front:
            next_front = []
            for sol in current_front:
                for dominated_sol in dominated_solutions[sol.solution_id]:
                    domination_counts[dominated_sol.solution_id] -= 1
                    if domination_counts[dominated_sol.solution_id] == 0:
                        next_front.append(dominated_sol)
            
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts
    
    def _calculate_crowding_distances_for_front(self, front: List[ParetoSolution]) -> List[float]:
        """Calculate crowding distances for solutions in a front."""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        num_objectives = len(front[0].objective_values)
        distances = [0.0] * len(front)
        
        for obj_idx in range(num_objectives):
            # Sort front by this objective
            indexed_front = list(enumerate(front))
            indexed_front.sort(key=lambda x: x[1].objective_values[obj_idx])
            
            # Set boundary solutions to infinite distance
            distances[indexed_front[0][0]] = float('inf')
            distances[indexed_front[-1][0]] = float('inf')
            
            # Calculate objective range
            obj_values = [sol.objective_values[obj_idx] for _, sol in indexed_front]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range == 0:
                continue
            
            # Calculate distances for intermediate solutions
            for i in range(1, len(indexed_front) - 1):
                idx = indexed_front[i][0]
                if distances[idx] != float('inf'):
                    prev_val = indexed_front[i-1][1].objective_values[obj_idx]
                    next_val = indexed_front[i+1][1].objective_values[obj_idx]
                    distances[idx] += (next_val - prev_val) / obj_range
        
        return distances
    
    def _get_best_objectives(self, population: List[ParetoSolution]) -> List[float]:
        """Get best objective values from population."""
        if not population:
            return []
        
        num_objectives = len(population[0].objective_values)
        best_objectives = []
        
        for obj_idx in range(num_objectives):
            obj_values = [sol.objective_values[obj_idx] for sol in population if sol.is_feasible]
            if obj_values:
                if self.minimize_objectives[obj_idx]:
                    best_objectives.append(min(obj_values))
                else:
                    best_objectives.append(max(obj_values))
            else:
                best_objectives.append(float('inf') if self.minimize_objectives[obj_idx] else float('-inf'))
        
        return best_objectives


class SPEA2(MultiObjectiveOptimizer):
    """Strength Pareto Evolutionary Algorithm 2 (SPEA2)."""
    
    def __init__(self, 
                 population_size: int = 100,
                 archive_size: int = 100,
                 max_generations: int = 100,
                 minimize_objectives: List[bool] = None):
        super().__init__(population_size, max_generations, minimize_objectives)
        self.archive_size = archive_size
        self.archive = []
    
    def optimize(self, 
                objective_functions: List[Callable],
                design_space: Dict[str, Tuple[Any, Any]],
                constraints: List[Callable] = None) -> List[ParetoSolution]:
        """Run SPEA2 optimization."""
        
        logger.info(f"Starting SPEA2 optimization: {self.population_size} population, {self.archive_size} archive, {self.max_generations} generations")
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            design_params = self._generate_random_solution(design_space)
            solution = self._evaluate_solution(design_params, objective_functions, constraints)
            population.append(solution)
        
        # Set minimize objectives if not specified
        if self.minimize_objectives is None:
            self.minimize_objectives = [True] * len(objective_functions)
        
        # Initialize archive
        self.archive = []
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation_count = generation
            
            # Update archive
            self._update_archive(population)
            
            # Calculate fitness
            self._calculate_fitness(population + self.archive)
            
            # Environmental selection
            mating_pool = self._environmental_selection(population + self.archive)
            
            # Create new population
            population = self._create_offspring_spea2(mating_pool, design_space, objective_functions, constraints)
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Archive size = {len(self.archive)}")
            
            # Store generation history
            self.optimization_history.append({
                'generation': generation,
                'evaluation_count': self.evaluation_count,
                'archive_size': len(self.archive),
                'population_size': len(population)
            })
        
        # Final archive update
        self._update_archive(population)
        
        logger.info(f"SPEA2 completed: {self.evaluation_count} evaluations, {len(self.archive)} archive solutions")
        return self.archive.copy()
    
    def _update_archive(self, population: List[ParetoSolution]):
        """Update archive with non-dominated solutions."""
        combined = population + self.archive
        
        # Find non-dominated solutions
        non_dominated = []
        for sol in combined:
            is_dominated = False
            for other in combined:
                if other != sol and other.dominates(sol, self.minimize_objectives):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(sol)
        
        # Update archive
        if len(non_dominated) <= self.archive_size:
            self.archive = non_dominated
        else:
            # Truncate using clustering
            self.archive = self._truncate_archive(non_dominated)
    
    def _truncate_archive(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Truncate archive using clustering method."""
        # Simple truncation - in practice would use more sophisticated clustering
        # For now, use crowding distance similar to NSGA-II
        if len(solutions) <= self.archive_size:
            return solutions
        
        crowding_distances = self._calculate_crowding_distances_for_front(solutions)
        
        # Sort by crowding distance (descending) and keep top solutions
        indexed_solutions = list(zip(solutions, crowding_distances))
        indexed_solutions.sort(key=lambda x: x[1], reverse=True)
        
        return [sol for sol, _ in indexed_solutions[:self.archive_size]]
    
    def _calculate_fitness(self, solutions: List[ParetoSolution]):
        """Calculate SPEA2 fitness for all solutions."""
        # Calculate strength (number of solutions dominated)
        strengths = {}
        for sol in solutions:
            strength = 0
            for other in solutions:
                if sol != other and sol.dominates(other, self.minimize_objectives):
                    strength += 1
            strengths[sol.solution_id] = strength
        
        # Calculate raw fitness (sum of strengths of dominating solutions)
        for sol in solutions:
            raw_fitness = 0
            for other in solutions:
                if other != sol and other.dominates(sol, self.minimize_objectives):
                    raw_fitness += strengths[other.solution_id]
            sol.metadata['raw_fitness'] = raw_fitness
        
        # Calculate density (k-th nearest neighbor distance)
        # Simplified version - in practice would calculate actual k-NN distance
        for sol in solutions:
            distances = []
            for other in solutions:
                if other != sol:
                    dist = self._euclidean_distance(sol.objective_values, other.objective_values)
                    distances.append(dist)
            
            if distances:
                distances.sort()
                k = min(int(math.sqrt(len(solutions))), len(distances))
                density = 1.0 / (distances[k-1] + 2.0)  # Add 2 to ensure denominator > 1
            else:
                density = 0.0
            
            sol.metadata['density'] = density
            sol.metadata['fitness'] = sol.metadata['raw_fitness'] + density
    
    def _euclidean_distance(self, obj1: List[float], obj2: List[float]) -> float:
        """Calculate Euclidean distance between two objective vectors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(obj1, obj2)))
    
    def _environmental_selection(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Environmental selection for SPEA2."""
        # Archive becomes the mating pool
        return self.archive.copy()
    
    def _create_offspring_spea2(self, mating_pool: List[ParetoSolution], 
                               design_space: Dict[str, Tuple[Any, Any]],
                               objective_functions: List[Callable],
                               constraints: List[Callable] = None) -> List[ParetoSolution]:
        """Create offspring for SPEA2."""
        offspring = []
        
        for _ in range(self.population_size):
            # Binary tournament selection based on fitness
            parent1 = self._binary_tournament_fitness(mating_pool)
            parent2 = self._binary_tournament_fitness(mating_pool)
            
            # Crossover
            child_params = self._crossover(parent1.design_parameters, parent2.design_parameters, design_space)
            
            # Mutation
            if random.random() < 0.1:
                child_params = self._mutate(child_params, design_space)
            
            # Evaluate offspring
            child = self._evaluate_solution(child_params, objective_functions, constraints)
            offspring.append(child)
        
        return offspring
    
    def _binary_tournament_fitness(self, population: List[ParetoSolution]) -> ParetoSolution:
        """Binary tournament selection based on fitness."""
        candidate1 = random.choice(population)
        candidate2 = random.choice(population)
        
        fitness1 = candidate1.metadata.get('fitness', float('inf'))
        fitness2 = candidate2.metadata.get('fitness', float('inf'))
        
        # Lower fitness is better in SPEA2
        return candidate1 if fitness1 < fitness2 else candidate2


class MOEAD(MultiObjectiveOptimizer):
    """Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)."""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_generations: int = 100,
                 neighborhood_size: int = 20,
                 minimize_objectives: List[bool] = None):
        super().__init__(population_size, max_generations, minimize_objectives)
        self.neighborhood_size = neighborhood_size
        self.weight_vectors = []
        self.neighbors = []
        self.ideal_point = None
    
    def optimize(self, 
                objective_functions: List[Callable],
                design_space: Dict[str, Tuple[Any, Any]],
                constraints: List[Callable] = None) -> List[ParetoSolution]:
        """Run MOEA/D optimization."""
        
        logger.info(f"Starting MOEA/D optimization: {self.population_size} population, {self.max_generations} generations")
        
        # Generate weight vectors
        num_objectives = len(objective_functions)
        self.weight_vectors = self._generate_weight_vectors(num_objectives, self.population_size)
        
        # Calculate neighborhoods
        self.neighbors = self._calculate_neighborhoods()
        
        # Initialize population
        population = []
        for i in range(self.population_size):
            design_params = self._generate_random_solution(design_space)
            solution = self._evaluate_solution(design_params, objective_functions, constraints)
            population.append(solution)
        
        # Set minimize objectives if not specified
        if self.minimize_objectives is None:
            self.minimize_objectives = [True] * len(objective_functions)
        
        # Initialize ideal point
        self._update_ideal_point(population)
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation_count = generation
            
            for i in range(self.population_size):
                # Generate offspring
                parents = self._select_parents(i, population)
                child_params = self._crossover(parents[0].design_parameters, parents[1].design_parameters, design_space)
                
                if random.random() < 0.1:
                    child_params = self._mutate(child_params, design_space)
                
                child = self._evaluate_solution(child_params, objective_functions, constraints)
                
                # Update ideal point
                self._update_ideal_point([child])
                
                # Update neighbors
                self._update_neighbors(i, child, population)
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Ideal point = {self.ideal_point}")
            
            # Store generation history
            self.optimization_history.append({
                'generation': generation,
                'evaluation_count': self.evaluation_count,
                'ideal_point': self.ideal_point.copy() if self.ideal_point else None
            })
        
        # Update Pareto archive with final population
        for solution in population:
            if solution.is_feasible:
                self.pareto_archive.add_solution(solution, self.minimize_objectives)
        
        logger.info(f"MOEA/D completed: {self.evaluation_count} evaluations, {self.pareto_archive.size()} Pareto solutions")
        return self.pareto_archive.get_pareto_frontier()
    
    def _generate_weight_vectors(self, num_objectives: int, num_vectors: int) -> List[List[float]]:
        """Generate weight vectors for decomposition."""
        if num_objectives == 2:
            # For 2 objectives, use uniform distribution
            weights = []
            for i in range(num_vectors):
                w1 = i / (num_vectors - 1)
                w2 = 1.0 - w1
                weights.append([w1, w2])
            return weights
        
        else:
            # For more objectives, use random generation
            # In practice, would use more sophisticated methods like Das-Dennis
            weights = []
            for _ in range(num_vectors):
                weight = [random.random() for _ in range(num_objectives)]
                weight_sum = sum(weight)
                weight = [w / weight_sum for w in weight]
                weights.append(weight)
            return weights
    
    def _calculate_neighborhoods(self) -> List[List[int]]:
        """Calculate neighborhoods based on weight vector distances."""
        neighbors = []
        
        for i, weight_i in enumerate(self.weight_vectors):
            distances = []
            for j, weight_j in enumerate(self.weight_vectors):
                if i != j:
                    dist = self._euclidean_distance(weight_i, weight_j)
                    distances.append((j, dist))
            
            # Sort by distance and select nearest neighbors
            distances.sort(key=lambda x: x[1])
            neighbor_indices = [idx for idx, _ in distances[:self.neighborhood_size]]
            neighbors.append(neighbor_indices)
        
        return neighbors
    
    def _update_ideal_point(self, solutions: List[ParetoSolution]):
        """Update ideal point with new solutions."""
        if not solutions:
            return
        
        if self.ideal_point is None:
            self.ideal_point = solutions[0].objective_values.copy()
        
        for solution in solutions:
            for i, obj_val in enumerate(solution.objective_values):
                if self.minimize_objectives[i]:
                    self.ideal_point[i] = min(self.ideal_point[i], obj_val)
                else:
                    self.ideal_point[i] = max(self.ideal_point[i], obj_val)
    
    def _select_parents(self, subproblem_index: int, population: List[ParetoSolution]) -> List[ParetoSolution]:
        """Select parents for reproduction."""
        # Select from neighborhood
        neighbor_indices = self.neighbors[subproblem_index]
        
        if random.random() < 0.8:  # 80% chance to select from neighborhood
            parent_indices = random.sample(neighbor_indices, min(2, len(neighbor_indices)))
        else:  # 20% chance to select from entire population
            parent_indices = random.sample(range(len(population)), 2)
        
        if len(parent_indices) == 1:
            parent_indices.append(random.choice(range(len(population))))
        
        return [population[idx] for idx in parent_indices]
    
    def _update_neighbors(self, subproblem_index: int, child: ParetoSolution, population: List[ParetoSolution]):
        """Update neighbors with new child solution."""
        neighbor_indices = self.neighbors[subproblem_index]
        
        for neighbor_idx in neighbor_indices:
            # Calculate Tchebycheff values
            current_tcheby = self._tchebycheff_value(population[neighbor_idx], self.weight_vectors[neighbor_idx])
            child_tcheby = self._tchebycheff_value(child, self.weight_vectors[neighbor_idx])
            
            # Replace if child is better
            if child_tcheby < current_tcheby:
                population[neighbor_idx] = child
    
    def _tchebycheff_value(self, solution: ParetoSolution, weight_vector: List[float]) -> float:
        """Calculate Tchebycheff aggregation value."""
        if self.ideal_point is None:
            return float('inf')
        
        max_val = 0.0
        for i, (obj_val, weight, ideal_val) in enumerate(zip(solution.objective_values, weight_vector, self.ideal_point)):
            if self.minimize_objectives[i]:
                normalized = abs(obj_val - ideal_val)
            else:
                normalized = abs(ideal_val - obj_val)
            
            if weight > 0:
                max_val = max(max_val, normalized / weight)
            else:
                max_val = max(max_val, normalized * 1000)  # Large penalty for zero weight
        
        return max_val