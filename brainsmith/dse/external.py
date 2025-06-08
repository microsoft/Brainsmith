"""
ExternalDSEAdapter for integrating with external optimization frameworks.

This module provides adapters for popular optimization libraries including
Bayesian optimization, genetic algorithms, and other research frameworks.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import numpy as np

from .interface import DSEEngine, DSEConfiguration, OptimizationObjective
from ..core.design_space import DesignSpace, DesignPoint, ParameterType
from ..core.result import BrainsmithResult


class ExternalDSEAdapter(DSEEngine):
    """
    Adapter for external optimization frameworks.
    
    Supports:
    - Bayesian Optimization (scikit-optimize, Optuna)
    - Genetic Algorithms (DEAP)
    - Hyperparameter optimization (Hyperopt)
    - Custom framework adapters
    """
    
    def __init__(self, framework: str, config: DSEConfiguration):
        super().__init__(f"External_{framework}", config)
        self.framework = framework
        self.design_space: Optional[DesignSpace] = None
        self.external_optimizer = None
        self.framework_space = None
        self.param_mapping: Dict[str, int] = {}
        self.reverse_mapping: Dict[int, str] = {}
        
        # Framework availability
        self.framework_available = self._check_framework_availability()
        
        if not self.framework_available:
            logging.warning(f"Framework '{framework}' not available, will fallback to SimpleDSE")
    
    def _check_framework_availability(self) -> bool:
        """Check if the specified framework is available."""
        try:
            if self.framework in ["bayesian", "skopt", "scikit-optimize"]:
                import skopt
                return True
            elif self.framework == "optuna":
                import optuna
                return True
            elif self.framework in ["genetic", "deap"]:
                import deap
                return True
            elif self.framework == "hyperopt":
                import hyperopt
                return True
            else:
                return False
        except ImportError:
            return False
    
    def initialize(self, design_space: DesignSpace):
        """Initialize adapter with design space."""
        self.design_space = design_space
        self._setup_parameter_mapping()
        
        if self.framework_available:
            self._setup_external_optimizer()
        else:
            # Fallback to simple random sampling
            from .simple import SimpleDSEEngine
            self.fallback_engine = SimpleDSEEngine("random", self.config)
            self.fallback_engine.initialize(design_space)
    
    def _setup_parameter_mapping(self):
        """Create mapping between Brainsmith parameters and framework indices."""
        self.param_mapping = {}
        self.reverse_mapping = {}
        
        for i, param_name in enumerate(self.design_space.parameters.keys()):
            self.param_mapping[param_name] = i
            self.reverse_mapping[i] = param_name
    
    def _setup_external_optimizer(self):
        """Setup the external optimization framework."""
        if self.framework in ["bayesian", "skopt", "scikit-optimize"]:
            self._setup_skopt()
        elif self.framework == "optuna":
            self._setup_optuna()
        elif self.framework in ["genetic", "deap"]:
            self._setup_deap()
        elif self.framework == "hyperopt":
            self._setup_hyperopt()
    
    def _setup_skopt(self):
        """Setup scikit-optimize Bayesian optimization."""
        try:
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            from skopt import gp_minimize, forest_minimize
            
            # Convert design space to skopt space
            dimensions = []
            for param_name, param_def in self.design_space.parameters.items():
                if param_def.type == ParameterType.CONTINUOUS:
                    dim = Real(param_def.range[0], param_def.range[1], name=param_name)
                elif param_def.type == ParameterType.INTEGER:
                    dim = Integer(param_def.range[0], param_def.range[1], name=param_name)
                elif param_def.type == ParameterType.CATEGORICAL:
                    dim = Categorical(param_def.values, name=param_name)
                elif param_def.type == ParameterType.BOOLEAN:
                    dim = Categorical([True, False], name=param_name)
                else:
                    continue
                dimensions.append(dim)
            
            self.framework_space = dimensions
            
            # Setup optimizer function
            acquisition_function = self.config.strategy_config.get("acquisition_function", "EI")
            n_initial_points = self.config.strategy_config.get("n_initial_points", 10)
            
            self.external_optimizer = {
                "space": dimensions,
                "acquisition_function": acquisition_function,
                "n_initial_points": n_initial_points,
                "func": gp_minimize if "gp" in self.config.strategy_config.get("base_estimator", "gp") else forest_minimize
            }
            
        except ImportError:
            logging.error("scikit-optimize not available")
            self.framework_available = False
    
    def _setup_optuna(self):
        """Setup Optuna optimization."""
        try:
            import optuna
            
            # Create study
            direction = "maximize" if len(self.config.objectives) == 1 and self.config.objectives[0].direction == OptimizationObjective.MAXIMIZE else "minimize"
            
            self.external_optimizer = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=self.config.seed),
                pruner=optuna.pruners.MedianPruner()
            )
            
        except ImportError:
            logging.error("Optuna not available")
            self.framework_available = False
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm."""
        try:
            import deap
            from deap import creator, base, tools
            
            # Create fitness and individual classes
            if len(self.config.objectives) == 1:
                if self.config.objectives[0].direction == OptimizationObjective.MAXIMIZE:
                    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                else:
                    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax if self.config.objectives[0].direction == OptimizationObjective.MAXIMIZE else creator.FitnessMin)
            else:
                # Multi-objective
                weights = tuple(1.0 if obj.direction == OptimizationObjective.MAXIMIZE else -1.0 for obj in self.config.objectives)
                creator.create("FitnessMulti", base.Fitness, weights=weights)
                creator.create("Individual", list, fitness=creator.FitnessMulti)
            
            # Setup toolbox
            toolbox = base.Toolbox()
            
            # Register parameter generators
            for i, (param_name, param_def) in enumerate(self.design_space.parameters.items()):
                if param_def.type == ParameterType.CONTINUOUS:
                    toolbox.register(f"attr_{i}", np.random.uniform, param_def.range[0], param_def.range[1])
                elif param_def.type == ParameterType.INTEGER:
                    toolbox.register(f"attr_{i}", np.random.randint, param_def.range[0], param_def.range[1] + 1)
                elif param_def.type == ParameterType.CATEGORICAL:
                    toolbox.register(f"attr_{i}", np.random.choice, param_def.values)
                elif param_def.type == ParameterType.BOOLEAN:
                    toolbox.register(f"attr_{i}", np.random.choice, [True, False])
            
            # Register individual and population
            n_params = len(self.design_space.parameters)
            attrs = [getattr(toolbox, f"attr_{i}") for i in range(n_params)]
            toolbox.register("individual", tools.initCycle, creator.Individual, attrs, n=1)
            
            pop_size = self.config.strategy_config.get("population_size", 50)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Setup genetic operators
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            self.external_optimizer = {
                "toolbox": toolbox,
                "population_size": pop_size,
                "mutation_prob": self.config.strategy_config.get("mutation_prob", 0.2),
                "crossover_prob": self.config.strategy_config.get("crossover_prob", 0.8)
            }
            
        except ImportError:
            logging.error("DEAP not available")
            self.framework_available = False
    
    def _setup_hyperopt(self):
        """Setup Hyperopt optimization."""
        try:
            import hyperopt
            from hyperopt import hp
            
            # Convert design space to hyperopt space
            space = {}
            for param_name, param_def in self.design_space.parameters.items():
                if param_def.type == ParameterType.CONTINUOUS:
                    space[param_name] = hp.uniform(param_name, param_def.range[0], param_def.range[1])
                elif param_def.type == ParameterType.INTEGER:
                    space[param_name] = hp.randint(param_name, param_def.range[0], param_def.range[1] + 1)
                elif param_def.type == ParameterType.CATEGORICAL:
                    space[param_name] = hp.choice(param_name, param_def.values)
                elif param_def.type == ParameterType.BOOLEAN:
                    space[param_name] = hp.choice(param_name, [True, False])
            
            self.framework_space = space
            
            # Setup algorithm
            algorithm = self.config.strategy_config.get("algorithm", "tpe")
            if algorithm == "tpe":
                algo = hyperopt.tpe.suggest
            elif algorithm == "random":
                algo = hyperopt.rand.suggest
            elif algorithm == "adaptive_tpe":
                algo = hyperopt.atpe.suggest
            else:
                algo = hyperopt.tpe.suggest
            
            self.external_optimizer = {
                "space": space,
                "algo": algo,
                "trials": hyperopt.Trials()
            }
            
        except ImportError:
            logging.error("Hyperopt not available")
            self.framework_available = False
    
    def suggest_next_points(self, n_points: int = 1) -> List[DesignPoint]:
        """Suggest next design points using external framework."""
        if not self.framework_available:
            # Fallback to simple engine
            return self.fallback_engine.suggest_next_points(n_points)
        
        if self.framework in ["bayesian", "skopt", "scikit-optimize"]:
            return self._suggest_skopt_points(n_points)
        elif self.framework == "optuna":
            return self._suggest_optuna_points(n_points)
        elif self.framework in ["genetic", "deap"]:
            return self._suggest_deap_points(n_points)
        elif self.framework == "hyperopt":
            return self._suggest_hyperopt_points(n_points)
        else:
            return []
    
    def _suggest_skopt_points(self, n_points: int) -> List[DesignPoint]:
        """Suggest points using scikit-optimize."""
        if len(self.evaluation_history) < self.external_optimizer["n_initial_points"]:
            # Initial random points
            from skopt.sampler import Lhs
            sampler = Lhs(criterion="maximin", iterations=1000)
            points = sampler.generate(self.external_optimizer["space"], n_points)
        else:
            # Use Bayesian optimization
            # Convert history to format expected by skopt
            X = []
            y = []
            for point, result in self.evaluation_history:
                x_vals = []
                for param_name in self.design_space.parameters.keys():
                    x_vals.append(point.parameters[param_name])
                X.append(x_vals)
                
                # Get objective value
                objective_value = self._get_objective_value(result)
                y.append(objective_value)
            
            # Use skopt's ask interface
            try:
                from skopt import Optimizer
                opt = Optimizer(
                    dimensions=self.external_optimizer["space"],
                    base_estimator="GP",
                    acq_func=self.external_optimizer["acquisition_function"]
                )
                
                # Tell optimizer about previous evaluations
                opt.tell(X, y)
                
                # Ask for new points
                points = [opt.ask() for _ in range(n_points)]
            except:
                # Fallback to random sampling
                points = [self.design_space.sample_random_point().get_parameter_values() for _ in range(n_points)]
        
        # Convert to DesignPoint objects
        design_points = []
        for point in points:
            dp = DesignPoint()
            for i, param_name in enumerate(self.design_space.parameters.keys()):
                dp.set_parameter(param_name, point[i])
            design_points.append(dp)
        
        return design_points
    
    def _suggest_optuna_points(self, n_points: int) -> List[DesignPoint]:
        """Suggest points using Optuna."""
        design_points = []
        
        for _ in range(n_points):
            trial = self.external_optimizer.ask()
            
            dp = DesignPoint()
            for param_name, param_def in self.design_space.parameters.items():
                if param_def.type == ParameterType.CONTINUOUS:
                    value = trial.suggest_float(param_name, param_def.range[0], param_def.range[1])
                elif param_def.type == ParameterType.INTEGER:
                    value = trial.suggest_int(param_name, param_def.range[0], param_def.range[1])
                elif param_def.type == ParameterType.CATEGORICAL:
                    value = trial.suggest_categorical(param_name, param_def.values)
                elif param_def.type == ParameterType.BOOLEAN:
                    value = trial.suggest_categorical(param_name, [True, False])
                else:
                    continue
                
                dp.set_parameter(param_name, value)
            
            dp.trial = trial  # Store trial for later use
            design_points.append(dp)
        
        return design_points
    
    def _suggest_deap_points(self, n_points: int) -> List[DesignPoint]:
        """Suggest points using DEAP genetic algorithm."""
        # For DEAP, we generate a population and return best individuals
        toolbox = self.external_optimizer["toolbox"]
        
        if len(self.evaluation_history) == 0:
            # Initial population
            population = toolbox.population(n=n_points)
        else:
            # Evolve population based on previous results
            pop_size = self.external_optimizer["population_size"]
            population = toolbox.population(n=pop_size)
            
            # Run one generation of evolution
            import deap.algorithms as algorithms
            offspring = algorithms.varAnd(population, toolbox, 
                                        self.external_optimizer["crossover_prob"],
                                        self.external_optimizer["mutation_prob"])
            
            # Select best individuals
            population = toolbox.select(offspring, n_points)
        
        # Convert to DesignPoint objects
        design_points = []
        for individual in population:
            dp = DesignPoint()
            for i, param_name in enumerate(self.design_space.parameters.keys()):
                dp.set_parameter(param_name, individual[i])
            design_points.append(dp)
        
        return design_points
    
    def _suggest_hyperopt_points(self, n_points: int) -> List[DesignPoint]:
        """Suggest points using Hyperopt."""
        design_points = []
        
        for _ in range(n_points):
            # Use hyperopt's suggest mechanism
            import hyperopt
            trials = self.external_optimizer["trials"]
            
            # Get suggestion from algorithm
            new_ids = trials.new_trial_ids(1)
            trials.refresh()
            
            suggestions = self.external_optimizer["algo"](
                new_ids, trials.domain, trials, self.config.seed
            )
            
            if suggestions:
                suggestion = suggestions[0]
                vals = suggestion['misc']['vals']
                
                dp = DesignPoint()
                for param_name in self.design_space.parameters.keys():
                    if param_name in vals and vals[param_name]:
                        value = vals[param_name][0]
                        dp.set_parameter(param_name, value)
                
                design_points.append(dp)
        
        return design_points
    
    def update_with_result(self, design_point: DesignPoint, result: BrainsmithResult):
        """Update external optimizer with evaluation result."""
        super().update_with_result(design_point, result)
        
        if not self.framework_available:
            if hasattr(self, 'fallback_engine'):
                self.fallback_engine.update_with_result(design_point, result)
            return
        
        objective_value = self._get_objective_value(result)
        
        if self.framework == "optuna" and hasattr(design_point, 'trial'):
            # Tell Optuna about the result
            trial = design_point.trial
            self.external_optimizer.tell(trial, objective_value)
        
        # Other frameworks are updated implicitly through evaluation_history
    
    def _get_objective_value(self, result: BrainsmithResult) -> float:
        """Extract objective value from result for external optimizer."""
        if len(self.config.objectives) == 1:
            objective = self.config.objectives[0]
            value = objective.evaluate(result.metrics)
            # Convert minimization to maximization if needed
            if objective.direction == OptimizationObjective.MINIMIZE:
                value = -value
            return value
        else:
            # Multi-objective: return weighted sum
            total_value = 0.0
            for objective in self.config.objectives:
                value = objective.evaluate(result.metrics)
                if objective.direction == OptimizationObjective.MINIMIZE:
                    value = -value
                total_value += objective.weight * value
            return total_value
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get information about the external framework."""
        return {
            "framework": self.framework,
            "available": self.framework_available,
            "fallback_active": not self.framework_available,
            "evaluations_completed": len(self.evaluation_history),
            "strategy_config": self.config.strategy_config
        }


# Utility functions for framework integration
def check_framework_availability() -> Dict[str, bool]:
    """Check availability of all supported frameworks."""
    frameworks = {
        "scikit-optimize": False,
        "optuna": False,
        "deap": False,
        "hyperopt": False
    }
    
    try:
        import skopt
        frameworks["scikit-optimize"] = True
    except ImportError:
        pass
    
    try:
        import optuna
        frameworks["optuna"] = True
    except ImportError:
        pass
    
    try:
        import deap
        frameworks["deap"] = True
    except ImportError:
        pass
    
    try:
        import hyperopt
        frameworks["hyperopt"] = True
    except ImportError:
        pass
    
    return frameworks


def get_recommended_framework_config(framework: str) -> Dict[str, Any]:
    """Get recommended configuration for a framework."""
    configs = {
        "scikit-optimize": {
            "acquisition_function": "EI",
            "n_initial_points": 10,
            "base_estimator": "gp"
        },
        "optuna": {
            "sampler": "TPE",
            "pruner": "MedianPruner"
        },
        "deap": {
            "population_size": 50,
            "mutation_prob": 0.2,
            "crossover_prob": 0.8,
            "tournament_size": 3
        },
        "hyperopt": {
            "algorithm": "tpe",
            "max_evals": 100
        }
    }
    
    return configs.get(framework, {})