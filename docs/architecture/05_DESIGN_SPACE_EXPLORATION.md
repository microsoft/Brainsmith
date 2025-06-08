# 🎯 Design Space Exploration Architecture
## Advanced Multi-Objective Optimization Framework

---

## 🎯 DSE System Overview

The Design Space Exploration (DSE) system is the core optimization engine of the Brainsmith platform. It provides intelligent, automated exploration of the FPGA accelerator design space using advanced optimization algorithms and multi-objective analysis.

### Key Capabilities

- **6+ Optimization Strategies**: From simple random sampling to advanced genetic algorithms
- **Multi-Objective Optimization**: Simultaneous optimization of performance, power, and resource usage
- **Intelligent Strategy Selection**: Automatic algorithm recommendation based on problem characteristics
- **Pareto Frontier Analysis**: True multi-objective trade-off identification
- **External Framework Integration**: Support for scikit-optimize, optuna, deap, hyperopt
- **Constraint Handling**: Resource and timing constraint enforcement

---

## 🏗️ DSE Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│            DESIGN SPACE EXPLORATION SYSTEM              │
├─────────────────────────────────────────────────────────┤
│                  Strategy Engine                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Strategy Selection                     │ │
│  │  • Problem characterization                         │ │
│  │  • Algorithm recommendation                         │ │
│  │  • Performance prediction                           │ │
│  │  • Resource requirement estimation                  │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Optimization Algorithms                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│  │     Random      │ │   Quasi-Random  │ │   Adaptive  │ │
│  │    Sampling     │ │    Sampling     │ │   Methods   │ │
│  │                 │ │                 │ │             │ │
│  │ • Pure random   │ │ • Latin Hyper   │ │ • Gaussian  │ │
│  │ • Uniform dist  │ │ • Sobol seq     │ │   Process   │ │
│  │ • Seed control  │ │ • Halton seq    │ │ • Tree-based│ │
│  └─────────────────┘ └─────────────────┘ └─────────────┘ │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│  │   Evolutionary  │ │   External      │ │  Analysis   │ │
│  │   Algorithms    │ │   Frameworks    │ │   Tools     │ │
│  │                 │ │                 │ │             │ │
│  │ • Genetic Algo  │ │ • scikit-opt    │ │ • Pareto    │ │
│  │ • NSGA-II       │ │ • Optuna        │ │ • Dominance │ │
│  │ • Differential  │ │ • DEAP          │ │ • Trade-off │ │
│  └─────────────────┘ └─────────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Multi-Objective Analysis                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Pareto Analysis Engine                 │ │
│  │  • Non-dominated sorting                            │ │
│  │  • Crowding distance computation                    │ │
│  │  • Hypervolume calculation                          │ │
│  │  • Trade-off visualization                          │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### DSE Engine Interface

```python
class DSEEngine(ABC):
    """Base interface for design space exploration engines."""
    
    def __init__(self, strategy_name: str, design_space: DesignSpace, 
                 config: DSEConfig):
        self.strategy_name = strategy_name
        self.design_space = design_space
        self.config = config
        self.history = []
    
    @abstractmethod
    def suggest(self, n_points: int = 1) -> List[DesignPoint]:
        """Suggest next design points to evaluate."""
        pass
    
    @abstractmethod
    def update(self, point: DesignPoint, results: Dict[str, Any]):
        """Update engine with evaluation results."""
        pass
    
    @abstractmethod
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        pass
    
    def get_best_points(self, n_points: int = 5) -> List[DesignPoint]:
        """Get best points found so far."""
        pass
    
    def get_pareto_frontier(self, objectives: List[str]) -> List[DesignPoint]:
        """Compute Pareto frontier for specified objectives."""
        pass
```

---

## 🎲 Optimization Strategies

### Strategy Selection Framework

```python
class StrategySelector:
    """Intelligent strategy selection based on problem characteristics."""
    
    def recommend_strategy(self, design_space: DesignSpace, 
                          config: DSEConfig) -> str:
        """Recommend optimal strategy for given problem."""
        
        problem_characteristics = self._analyze_problem(design_space, config)
        
        # Decision tree for strategy selection
        if problem_characteristics['parameter_count'] <= 3:
            if config.max_evaluations >= 100:
                return 'grid'  # Exhaustive search for small spaces
            else:
                return 'random'
        
        elif problem_characteristics['parameter_count'] <= 8:
            if len(config.objectives) == 1:
                return 'adaptive'  # Bayesian optimization for single objective
            else:
                return 'genetic'  # Multi-objective genetic algorithm
        
        else:
            if config.max_evaluations >= 500:
                return 'genetic'  # Genetic for large spaces with budget
            else:
                return 'latin_hypercube'  # Good coverage with limited budget
    
    def _analyze_problem(self, design_space: DesignSpace, 
                        config: DSEConfig) -> Dict[str, Any]:
        """Analyze problem characteristics for strategy selection."""
        return {
            'parameter_count': len(design_space.parameters),
            'continuous_params': self._count_continuous_params(design_space),
            'categorical_params': self._count_categorical_params(design_space),
            'objective_count': len(config.objectives),
            'evaluation_budget': config.max_evaluations,
            'constraint_count': len(getattr(design_space, 'constraints', []))
        }
```

### Random Sampling Strategy

```python
class RandomSamplingStrategy(DSEEngine):
    """Pure random sampling from design space."""
    
    def __init__(self, design_space: DesignSpace, config: DSEConfig):
        super().__init__("random", design_space, config)
        self.rng = np.random.Generator(np.random.PCG64(config.random_seed))
        self.evaluated_points = []
    
    def suggest(self, n_points: int = 1) -> List[DesignPoint]:
        """Generate random design points."""
        points = []
        
        for _ in range(n_points):
            point_params = {}
            
            for param_name, param_def in self.design_space.parameters.items():
                if param_def.type == ParameterType.INTEGER:
                    value = self.rng.integers(
                        param_def.range_min, param_def.range_max + 1
                    )
                elif param_def.type == ParameterType.FLOAT:
                    value = self.rng.uniform(
                        param_def.range_min, param_def.range_max
                    )
                elif param_def.type == ParameterType.CATEGORICAL:
                    value = self.rng.choice(param_def.values)
                elif param_def.type == ParameterType.BOOLEAN:
                    value = self.rng.choice([True, False])
                
                point_params[param_name] = value
            
            points.append(DesignPoint(point_params))
        
        return points
    
    def update(self, point: DesignPoint, results: Dict[str, Any]):
        """Update with evaluation results."""
        point.results.update(results)
        self.evaluated_points.append(point)
        self.history.append({
            'iteration': len(self.history),
            'point': point,
            'results': results
        })
    
    def is_converged(self) -> bool:
        """Random sampling doesn't converge, relies on budget."""
        return len(self.evaluated_points) >= self.config.max_evaluations
```

### Latin Hypercube Sampling

```python
class LatinHypercubeStrategy(DSEEngine):
    """Latin Hypercube Sampling for good space coverage."""
    
    def __init__(self, design_space: DesignSpace, config: DSEConfig):
        super().__init__("latin_hypercube", design_space, config)
        self.sample_points = self._generate_lhs_samples()
        self.current_index = 0
    
    def _generate_lhs_samples(self) -> List[DesignPoint]:
        """Pre-generate Latin Hypercube samples."""
        try:
            from scipy.stats import qmc
            
            # Count continuous parameters
            continuous_params = [
                (name, param) for name, param in self.design_space.parameters.items()
                if param.type in [ParameterType.FLOAT, ParameterType.INTEGER]
            ]
            
            if not continuous_params:
                # Fall back to random for all-categorical spaces
                return []
            
            # Generate Latin Hypercube samples
            sampler = qmc.LatinHypercube(
                d=len(continuous_params), 
                seed=self.config.random_seed
            )
            unit_samples = sampler.random(n=self.config.max_evaluations)
            
            # Transform to parameter ranges
            points = []
            for sample in unit_samples:
                point_params = {}
                
                # Map continuous parameters
                for i, (param_name, param_def) in enumerate(continuous_params):
                    if param_def.type == ParameterType.FLOAT:
                        value = param_def.range_min + sample[i] * (
                            param_def.range_max - param_def.range_min
                        )
                    else:  # INTEGER
                        value = int(param_def.range_min + sample[i] * (
                            param_def.range_max - param_def.range_min + 1
                        ))
                    point_params[param_name] = value
                
                # Handle categorical parameters with random selection
                for param_name, param_def in self.design_space.parameters.items():
                    if param_def.type == ParameterType.CATEGORICAL:
                        point_params[param_name] = np.random.choice(param_def.values)
                    elif param_def.type == ParameterType.BOOLEAN:
                        point_params[param_name] = np.random.choice([True, False])
                
                points.append(DesignPoint(point_params))
            
            return points
            
        except ImportError:
            # Fall back to random sampling if scipy not available
            return []
    
    def suggest(self, n_points: int = 1) -> List[DesignPoint]:
        """Return next LHS points."""
        if not self.sample_points:
            # Fall back to random sampling
            random_strategy = RandomSamplingStrategy(self.design_space, self.config)
            return random_strategy.suggest(n_points)
        
        points = []
        for _ in range(min(n_points, len(self.sample_points) - self.current_index)):
            points.append(self.sample_points[self.current_index])
            self.current_index += 1
        
        return points
```

### Adaptive Bayesian Optimization

```python
class AdaptiveStrategy(DSEEngine):
    """Adaptive sampling using surrogate models."""
    
    def __init__(self, design_space: DesignSpace, config: DSEConfig):
        super().__init__("adaptive", design_space, config)
        self.surrogate_model = None
        self.acquisition_function = "expected_improvement"
        self.evaluated_points = []
        self.warm_up_samples = min(10, config.max_evaluations // 4)
        
    def suggest(self, n_points: int = 1) -> List[DesignPoint]:
        """Suggest points using adaptive strategy."""
        
        if len(self.evaluated_points) < self.warm_up_samples:
            # Warm-up phase: use Latin Hypercube or random sampling
            lhs_strategy = LatinHypercubeStrategy(self.design_space, self.config)
            return lhs_strategy.suggest(n_points)
        else:
            # Adaptive phase: use surrogate model
            return self._suggest_adaptive(n_points)
    
    def _suggest_adaptive(self, n_points: int) -> List[DesignPoint]:
        """Use surrogate model for adaptive suggestions."""
        
        # Update surrogate model with recent data
        self._update_surrogate_model()
        
        # Generate candidate points
        candidate_points = self._generate_candidates(n_points * 100)
        
        # Score candidates using acquisition function
        scores = self._evaluate_acquisition_function(candidate_points)
        
        # Select best candidates
        best_indices = np.argsort(scores)[-n_points:]
        selected_points = [candidate_points[i] for i in best_indices]
        
        return selected_points
    
    def _update_surrogate_model(self):
        """Update surrogate model with evaluation data."""
        if len(self.evaluated_points) < 3:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for point in self.evaluated_points:
            # Convert point parameters to feature vector
            features = self._point_to_features(point)
            X.append(features)
            
            # Extract primary objective
            primary_objective = self.config.objectives[0]
            objective_value = point.objectives.get(primary_objective, 0)
            y.append(objective_value)
        
        # Train simple surrogate (placeholder for more sophisticated models)
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            
            kernel = ConstantKernel(1.0) * RBF(1.0)
            self.surrogate_model = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, random_state=self.config.random_seed
            )
            self.surrogate_model.fit(X, y)
            
        except ImportError:
            # Simple linear model fallback
            self.surrogate_model = None
```

### Genetic Algorithm Implementation

```python
class GeneticAlgorithmStrategy(DSEEngine):
    """Multi-objective genetic algorithm (NSGA-II)."""
    
    def __init__(self, design_space: DesignSpace, config: DSEConfig):
        super().__init__("genetic", design_space, config)
        self.population_size = min(50, config.max_evaluations // 5)
        self.current_generation = 0
        self.population = []
        self.pareto_archive = []
        
        # GA parameters
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        
    def suggest(self, n_points: int = 1) -> List[DesignPoint]:
        """Suggest points using genetic algorithm."""
        
        if self.current_generation == 0:
            # Initial population
            return self._generate_initial_population()
        else:
            # Evolve population
            return self._evolve_population()
    
    def _generate_initial_population(self) -> List[DesignPoint]:
        """Generate initial random population."""
        random_strategy = RandomSamplingStrategy(self.design_space, self.config)
        initial_pop = random_strategy.suggest(self.population_size)
        return initial_pop
    
    def _evolve_population(self) -> List[DesignPoint]:
        """Evolve current population using genetic operators."""
        
        # Selection
        parents = self._tournament_selection(self.population, self.population_size)
        
        # Crossover and mutation
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def update(self, point: DesignPoint, results: Dict[str, Any]):
        """Update population with evaluation results."""
        point.results.update(results)
        
        # Add objectives from results
        for obj_name in self.config.objectives:
            if obj_name in results:
                point.set_objective(obj_name, results[obj_name])
        
        self.population.append(point)
        
        # Check if generation is complete
        if len(self.population) >= self.population_size:
            self._complete_generation()
    
    def _complete_generation(self):
        """Complete current generation and prepare for next."""
        
        # Non-dominated sorting
        fronts = self._non_dominated_sort(self.population)
        
        # Update Pareto archive
        if fronts:
            self.pareto_archive.extend(fronts[0])
            # Remove dominated points from archive
            self.pareto_archive = self._get_non_dominated_points(self.pareto_archive)
        
        # Prepare for next generation
        self.current_generation += 1
        self.population = []
```

---

## 🎯 Multi-Objective Optimization

### Pareto Analysis Framework

```python
class ParetoAnalyzer:
    """Comprehensive Pareto frontier analysis."""
    
    def compute_pareto_frontier(self, points: List[DesignPoint], 
                               objectives: List[str],
                               directions: List[str] = None) -> List[DesignPoint]:
        """Compute Pareto-optimal points."""
        
        if directions is None:
            directions = ['maximize'] * len(objectives)
        
        # Filter points with all objectives
        valid_points = [
            p for p in points 
            if all(obj in p.objectives for obj in objectives)
        ]
        
        if not valid_points:
            return []
        
        # Compute dominance relationships
        pareto_points = []
        for point in valid_points:
            is_dominated = False
            
            for other_point in valid_points:
                if other_point != point and self._dominates(
                    other_point, point, objectives, directions
                ):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_points.append(point)
        
        return pareto_points
    
    def _dominates(self, point1: DesignPoint, point2: DesignPoint,
                  objectives: List[str], directions: List[str]) -> bool:
        """Check if point1 dominates point2."""
        
        at_least_one_better = False
        
        for obj, direction in zip(objectives, directions):
            val1 = point1.objectives.get(obj, 0)
            val2 = point2.objectives.get(obj, 0)
            
            if direction == 'maximize':
                if val1 < val2:
                    return False  # point1 is worse in this objective
                elif val1 > val2:
                    at_least_one_better = True
            else:  # minimize
                if val1 > val2:
                    return False  # point1 is worse in this objective
                elif val1 < val2:
                    at_least_one_better = True
        
        return at_least_one_better
    
    def compute_hypervolume(self, pareto_points: List[DesignPoint],
                           objectives: List[str],
                           reference_point: Dict[str, float] = None) -> float:
        """Compute hypervolume indicator for Pareto set quality."""
        
        if not pareto_points or len(objectives) > 3:
            return 0.0  # Hypervolume computation limited to ≤3 objectives
        
        # Set default reference point if not provided
        if reference_point is None:
            reference_point = {}
            for obj in objectives:
                all_values = [p.objectives.get(obj, 0) for p in pareto_points]
                reference_point[obj] = min(all_values) - 1.0
        
        # Extract objective vectors
        objective_vectors = []
        for point in pareto_points:
            vector = [point.objectives.get(obj, 0) for obj in objectives]
            objective_vectors.append(vector)
        
        # Compute hypervolume (simplified implementation)
        return self._hypervolume_calculation(objective_vectors, reference_point, objectives)
    
    def analyze_trade_offs(self, pareto_points: List[DesignPoint],
                          objectives: List[str]) -> Dict[str, Any]:
        """Analyze trade-offs in Pareto frontier."""
        
        if len(pareto_points) < 2:
            return {"trade_offs": [], "diversity": 0.0}
        
        trade_offs = []
        
        # Analyze pairwise trade-offs
        for i, point1 in enumerate(pareto_points):
            for j, point2 in enumerate(pareto_points[i+1:], i+1):
                trade_off = self._compute_trade_off(point1, point2, objectives)
                trade_offs.append(trade_off)
        
        # Compute diversity metrics
        diversity = self._compute_diversity(pareto_points, objectives)
        
        return {
            "trade_offs": trade_offs,
            "diversity": diversity,
            "frontier_size": len(pareto_points),
            "objective_ranges": self._compute_objective_ranges(pareto_points, objectives)
        }
```

---

## 📊 Performance Analysis and Visualization

### Convergence Analysis

```python
class ConvergenceAnalyzer:
    """Analysis of optimization convergence characteristics."""
    
    def analyze_convergence(self, dse_history: List[Dict], 
                           objectives: List[str]) -> Dict[str, Any]:
        """Analyze convergence of optimization process."""
        
        convergence_data = {
            'iterations': [],
            'best_values': {obj: [] for obj in objectives},
            'hypervolume': [],
            'diversity': [],
            'improvement_rate': []
        }
        
        for iteration, entry in enumerate(dse_history):
            convergence_data['iterations'].append(iteration)
            
            # Track best values for each objective
            for obj in objectives:
                if 'results' in entry and obj in entry['results']:
                    current_best = max(convergence_data['best_values'][obj]) if convergence_data['best_values'][obj] else 0
                    new_value = entry['results'][obj]
                    convergence_data['best_values'][obj].append(max(current_best, new_value))
                else:
                    last_best = convergence_data['best_values'][obj][-1] if convergence_data['best_values'][obj] else 0
                    convergence_data['best_values'][obj].append(last_best)
        
        # Detect convergence
        convergence_status = self._detect_convergence(convergence_data, objectives)
        
        return {
            'convergence_data': convergence_data,
            'is_converged': convergence_status['converged'],
            'convergence_iteration': convergence_status['iteration'],
            'final_improvements': convergence_status['final_improvements']
        }
    
    def _detect_convergence(self, convergence_data: Dict, 
                           objectives: List[str], 
                           patience: int = 10, 
                           min_improvement: float = 0.01) -> Dict[str, Any]:
        """Detect if optimization has converged."""
        
        if len(convergence_data['iterations']) < patience:
            return {'converged': False, 'iteration': None, 'final_improvements': {}}
        
        # Check improvement in recent iterations
        recent_improvements = {}
        for obj in objectives:
            values = convergence_data['best_values'][obj]
            if len(values) >= patience:
                recent_start = values[-patience]
                recent_end = values[-1]
                improvement = (recent_end - recent_start) / (recent_start + 1e-8)
                recent_improvements[obj] = improvement
        
        # Convergence if all objectives show minimal improvement
        converged = all(
            improvement < min_improvement 
            for improvement in recent_improvements.values()
        )
        
        convergence_iteration = len(convergence_data['iterations']) - patience if converged else None
        
        return {
            'converged': converged,
            'iteration': convergence_iteration,
            'final_improvements': recent_improvements
        }
```

---

*Next: [Blueprint System](06_BLUEPRINT_SYSTEM.md)*