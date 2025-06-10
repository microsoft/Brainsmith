# 🤖 **Automation Module Simplification - Implementation Plan**
## BrainSmith API Simplification - Replace Enterprise Bloat with Simple Helpers

**Date**: June 10, 2025  
**Implementation**: Automation Module Simplification  
**Goal**: Replace 1,400+ lines of enterprise bloat with ~200 lines of focused helpers  

---

## 🎯 **Implementation Overview**

Replace the entire `brainsmith/automation` module with simple automation utilities that help users run `forge()` multiple times with different parameters, instead of the current enterprise workflow orchestration system.

### **Current State:**
- 9 files, 1,400+ lines of enterprise workflow bloat
- 36+ exported classes/functions for workflow orchestration
- Academic research features (ML learning, quality frameworks)
- Enterprise-grade automation engine

### **Target State:**
- 4 files, ~200 lines of focused automation helpers
- 4-6 simple functions for practical automation
- Parameter sweep and batch processing utilities
- Leverage existing `forge()` function

---

## 📋 **Implementation Steps**

### **Phase 1: Remove Enterprise Bloat** ⏱️ 15 minutes

#### **Step 1.1: Delete Bloated Files**
```bash
# Remove enterprise workflow files
rm brainsmith/automation/engine.py          # 662 lines of workflow engine
rm brainsmith/automation/models.py          # 453 lines of enterprise models  
rm brainsmith/automation/integration.py     # 25 lines of integration layer
rm brainsmith/automation/learning.py        # 25 lines of ML learning
rm brainsmith/automation/quality.py         # 28 lines of quality control
rm brainsmith/automation/recommendations.py # 21 lines of AI recommendations
rm brainsmith/automation/workflows.py       # 35 lines of workflow orchestration
```

#### **Step 1.2: Check Dependencies**
- Verify no other modules import from deleted automation files
- Update any remaining imports to use new simplified API

### **Phase 2: Implement Simple Automation Helpers** ⏱️ 45 minutes

#### **Step 2.1: Create Parameter Sweep Utilities (`parameter_sweep.py`)**
```python
def parameter_sweep(
    model_path: str,
    blueprint_path: str,
    parameter_ranges: Dict[str, List[Any]],
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """Simple parameter sweep automation."""

def grid_search(
    model_path: str,
    blueprint_path: str,
    parameter_grid: Dict[str, List[Any]]
) -> Dict[str, Any]:
    """Grid search with best result selection."""

def random_search(
    model_path: str,
    blueprint_path: str,
    parameter_distributions: Dict[str, Any],
    n_iterations: int = 20
) -> Dict[str, Any]:
    """Random parameter search."""
```

#### **Step 2.2: Create Batch Processing Utilities (`batch_processing.py`)**
```python
def batch_process(
    model_blueprint_pairs: List[Tuple[str, str]],
    common_config: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Batch process multiple model/blueprint pairs."""

def multi_objective_runs(
    model_path: str,
    blueprint_path: str,
    objective_sets: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Run forge() with different objective configurations."""

def configuration_sweep(
    model_path: str,
    blueprint_configs: List[str]
) -> List[Dict[str, Any]]:
    """Sweep across different blueprint configurations."""
```

#### **Step 2.3: Update Utilities (`utils.py`)**
```python
def generate_parameter_combinations(parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple forge() runs."""

def find_best_result(results: List[Dict[str, Any]], metric: str = 'throughput') -> Dict[str, Any]:
    """Find best result based on metric."""

def save_automation_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save automation results to file."""
```

#### **Step 2.4: Create Simple Exports (`__init__.py`)**
```python
from .parameter_sweep import parameter_sweep, grid_search, random_search
from .batch_processing import batch_process, multi_objective_runs, configuration_sweep
from .utils import aggregate_results, find_best_result, save_automation_results

__all__ = [
    'parameter_sweep', 'grid_search', 'random_search',
    'batch_process', 'multi_objective_runs', 'configuration_sweep',
    'aggregate_results', 'find_best_result', 'save_automation_results'
]
```

### **Phase 3: Testing and Validation** ⏱️ 20 minutes

#### **Step 3.1: Create Test File (`tests/test_automation_helpers.py`)**
- Test parameter sweep functionality
- Test batch processing
- Test results aggregation
- Integration tests with forge() function

#### **Step 3.2: Create Demo Script (`automation_helpers_demo.py`)**
- Show before/after comparison
- Demonstrate simple automation patterns
- Example workflows users actually need

### **Phase 4: Documentation** ⏱️ 15 minutes

#### **Step 4.1: Create README (`brainsmith/automation/README.md`)**
- Explain simplified automation philosophy
- Usage examples for each helper function
- Migration guide from complex automation

#### **Step 4.2: Update Main Documentation**
- Update `brainsmith/__init__.py` exports
- Remove enterprise automation from documentation

---

## 📁 **Detailed File Implementations**

### **File: `brainsmith/automation/parameter_sweep.py`**
```python
"""
Parameter Sweep Automation

Simple utilities for exploring design parameter spaces by running forge()
with different parameter combinations.
"""

import itertools
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


def parameter_sweep(
    model_path: str,
    blueprint_path: str,
    parameter_ranges: Dict[str, List[Any]],
    max_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Run parameter sweep by calling forge() with different parameter combinations.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameter_ranges: Dict mapping parameter names to lists of values
        max_workers: Number of parallel workers
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of forge() results with parameter information
        
    Example:
        results = parameter_sweep(
            "model.onnx", 
            "blueprint.yaml",
            {
                'pe_count': [4, 8, 16, 32],
                'simd_width': [2, 4, 8, 16],
                'frequency': [100, 150, 200]
            }
        )
    """
    from ..core.api import forge
    
    # Generate all parameter combinations
    combinations = generate_parameter_combinations(parameter_ranges)
    total_combinations = len(combinations)
    
    logger.info(f"Starting parameter sweep with {total_combinations} combinations")
    
    results = []
    
    def run_single_combination(params: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Run forge() with single parameter combination."""
        try:
            # Create objectives/constraints from parameters
            objectives = _extract_objectives_from_params(params)
            constraints = _extract_constraints_from_params(params)
            
            # Run forge with parameters
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                objectives=objectives,
                constraints=constraints
            )
            
            # Add parameter information to result
            result['sweep_parameters'] = params
            result['sweep_index'] = index
            result['success'] = True
            
            if progress_callback:
                progress_callback(index + 1, total_combinations, params)
            
            return result
            
        except Exception as e:
            logger.error(f"Parameter combination {index} failed: {e}")
            return {
                'sweep_parameters': params,
                'sweep_index': index,
                'success': False,
                'error': str(e)
            }
    
    # Run parameter sweep with parallel execution
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {
                executor.submit(run_single_combination, params, i): (params, i)
                for i, params in enumerate(combinations)
            }
            
            for future in as_completed(future_to_params):
                result = future.result()
                results.append(result)
    else:
        # Sequential execution
        for i, params in enumerate(combinations):
            result = run_single_combination(params, i)
            results.append(result)
    
    # Sort results by sweep index
    results.sort(key=lambda x: x.get('sweep_index', 0))
    
    successful_runs = sum(1 for r in results if r.get('success', False))
    logger.info(f"Parameter sweep completed: {successful_runs}/{total_combinations} successful")
    
    return results


def grid_search(
    model_path: str,
    blueprint_path: str,
    parameter_grid: Dict[str, List[Any]],
    metric: str = 'throughput',
    maximize: bool = True
) -> Dict[str, Any]:
    """
    Grid search to find best parameter combination.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameter_grid: Grid of parameters to search
        metric: Metric to optimize
        maximize: Whether to maximize (True) or minimize (False) metric
        
    Returns:
        Best result with parameters
    """
    # Run parameter sweep
    results = parameter_sweep(model_path, blueprint_path, parameter_grid)
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        raise ValueError("No successful parameter combinations found")
    
    # Find best result based on metric
    def get_metric_value(result):
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        return performance.get(metric, 0.0)
    
    if maximize:
        best_result = max(successful_results, key=get_metric_value)
    else:
        best_result = min(successful_results, key=get_metric_value)
    
    # Add grid search metadata
    best_result['grid_search'] = {
        'total_combinations': len(results),
        'successful_combinations': len(successful_results),
        'optimization_metric': metric,
        'maximize': maximize,
        'best_metric_value': get_metric_value(best_result)
    }
    
    return best_result


def random_search(
    model_path: str,
    blueprint_path: str,
    parameter_distributions: Dict[str, Any],
    n_iterations: int = 20,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Random search over parameter space.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        parameter_distributions: Parameter distributions (ranges or choices)
        n_iterations: Number of random samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Best result from random search
    """
    import random
    
    if random_seed is not None:
        random.seed(random_seed)
    
    # Generate random parameter combinations
    random_combinations = []
    for _ in range(n_iterations):
        combination = {}
        for param_name, distribution in parameter_distributions.items():
            if isinstance(distribution, list):
                # Choose randomly from list
                combination[param_name] = random.choice(distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                # Random value in range
                min_val, max_val = distribution
                if isinstance(min_val, int) and isinstance(max_val, int):
                    combination[param_name] = random.randint(min_val, max_val)
                else:
                    combination[param_name] = random.uniform(min_val, max_val)
            else:
                raise ValueError(f"Invalid distribution for parameter {param_name}")
        
        random_combinations.append(combination)
    
    # Convert to parameter ranges format for sweep
    parameter_ranges = {}
    for param_name in parameter_distributions.keys():
        parameter_ranges[param_name] = [combo[param_name] for combo in random_combinations]
    
    # Run parameter sweep with generated combinations
    results = []
    for i, params in enumerate(random_combinations):
        try:
            from ..core.api import forge
            
            objectives = _extract_objectives_from_params(params)
            constraints = _extract_constraints_from_params(params)
            
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                objectives=objectives,
                constraints=constraints
            )
            
            result['random_parameters'] = params
            result['iteration'] = i
            result['success'] = True
            results.append(result)
            
        except Exception as e:
            results.append({
                'random_parameters': params,
                'iteration': i,
                'success': False,
                'error': str(e)
            })
    
    # Find best result
    successful_results = [r for r in results if r.get('success', False)]
    if not successful_results:
        raise ValueError("No successful random search iterations")
    
    # Use first metric found for optimization
    def get_first_metric_value(result):
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        if performance:
            return list(performance.values())[0]
        return 0.0
    
    best_result = max(successful_results, key=get_first_metric_value)
    
    # Add random search metadata
    best_result['random_search'] = {
        'total_iterations': n_iterations,
        'successful_iterations': len(successful_results),
        'random_seed': random_seed
    }
    
    return best_result


def generate_parameter_combinations(parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    keys = list(parameter_ranges.keys())
    values = list(parameter_ranges.values())
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def _extract_objectives_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract objectives from parameters (if any)."""
    objectives = {}
    
    # Map common parameter names to objectives
    param_to_objective = {
        'target_throughput': 'throughput',
        'target_latency': 'latency', 
        'target_power': 'power'
    }
    
    for param_name, obj_name in param_to_objective.items():
        if param_name in params:
            objectives[obj_name] = {'direction': 'maximize' if obj_name == 'throughput' else 'minimize'}
    
    return objectives


def _extract_constraints_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract constraints from parameters (if any)."""
    constraints = {}
    
    # Map common parameter names to constraints
    param_to_constraint = {
        'max_luts': 'max_luts',
        'max_dsps': 'max_dsps',
        'max_power': 'max_power',
        'target_frequency': 'target_frequency'
    }
    
    for param_name, constraint_name in param_to_constraint.items():
        if param_name in params:
            constraints[constraint_name] = params[param_name]
    
    return constraints
```

### **File: `brainsmith/automation/batch_processing.py`**
```python
"""
Batch Processing Automation

Simple utilities for processing multiple models or configurations in batch.
"""

from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def batch_process(
    model_blueprint_pairs: List[Tuple[str, str]],
    common_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple model/blueprint pairs in batch.
    
    Args:
        model_blueprint_pairs: List of (model_path, blueprint_path) tuples
        common_config: Common configuration for all runs
        max_workers: Number of parallel workers
        progress_callback: Optional progress callback
        
    Returns:
        List of forge() results
        
    Example:
        results = batch_process([
            ("model1.onnx", "blueprint1.yaml"),
            ("model2.onnx", "blueprint2.yaml"),
            ("model3.onnx", "blueprint3.yaml")
        ])
    """
    from ..core.api import forge
    
    total_pairs = len(model_blueprint_pairs)
    logger.info(f"Starting batch processing of {total_pairs} model/blueprint pairs")
    
    common_config = common_config or {}
    results = []
    
    def process_single_pair(pair: Tuple[str, str], index: int) -> Dict[str, Any]:
        """Process single model/blueprint pair."""
        model_path, blueprint_path = pair
        
        try:
            # Run forge with common configuration
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                **common_config
            )
            
            # Add batch processing metadata
            result['batch_info'] = {
                'model_path': model_path,
                'blueprint_path': blueprint_path,
                'batch_index': index,
                'success': True
            }
            
            if progress_callback:
                progress_callback(index + 1, total_pairs, (model_path, blueprint_path))
            
            return result
            
        except Exception as e:
            logger.error(f"Batch processing failed for {model_path}: {e}")
            return {
                'batch_info': {
                    'model_path': model_path,
                    'blueprint_path': blueprint_path,
                    'batch_index': index,
                    'success': False,
                    'error': str(e)
                }
            }
    
    # Process with parallel execution
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(process_single_pair, pair, i): (pair, i)
                for i, pair in enumerate(model_blueprint_pairs)
            }
            
            for future in as_completed(future_to_pair):
                result = future.result()
                results.append(result)
    else:
        # Sequential processing
        for i, pair in enumerate(model_blueprint_pairs):
            result = process_single_pair(pair, i)
            results.append(result)
    
    # Sort results by batch index
    results.sort(key=lambda x: x.get('batch_info', {}).get('batch_index', 0))
    
    successful_runs = sum(1 for r in results if r.get('batch_info', {}).get('success', False))
    logger.info(f"Batch processing completed: {successful_runs}/{total_pairs} successful")
    
    return results


def multi_objective_runs(
    model_path: str,
    blueprint_path: str,
    objective_sets: List[Dict[str, Any]],
    base_constraints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Run forge() with different objective configurations.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        objective_sets: List of objective dictionaries
        base_constraints: Common constraints for all runs
        
    Returns:
        List of forge() results for each objective set
        
    Example:
        results = multi_objective_runs(
            "model.onnx", 
            "blueprint.yaml",
            [
                {'throughput': {'direction': 'maximize'}},
                {'power': {'direction': 'minimize'}},
                {'latency': {'direction': 'minimize'}}
            ]
        )
    """
    from ..core.api import forge
    
    results = []
    base_constraints = base_constraints or {}
    
    for i, objectives in enumerate(objective_sets):
        try:
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                objectives=objectives,
                constraints=base_constraints
            )
            
            # Add multi-objective metadata
            result['multi_objective_info'] = {
                'objective_set': objectives,
                'run_index': i,
                'success': True
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Multi-objective run {i} failed: {e}")
            results.append({
                'multi_objective_info': {
                    'objective_set': objectives,
                    'run_index': i,
                    'success': False,
                    'error': str(e)
                }
            })
    
    successful_runs = sum(1 for r in results if r.get('multi_objective_info', {}).get('success', False))
    logger.info(f"Multi-objective runs completed: {successful_runs}/{len(objective_sets)} successful")
    
    return results


def configuration_sweep(
    model_path: str,
    blueprint_configs: List[str],
    common_objectives: Optional[Dict[str, Any]] = None,
    common_constraints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Sweep across different blueprint configurations.
    
    Args:
        model_path: Path to ONNX model
        blueprint_configs: List of blueprint configuration paths
        common_objectives: Common objectives for all runs
        common_constraints: Common constraints for all runs
        
    Returns:
        List of forge() results for each configuration
        
    Example:
        results = configuration_sweep(
            "model.onnx",
            ["config1.yaml", "config2.yaml", "config3.yaml"]
        )
    """
    from ..core.api import forge
    
    results = []
    common_objectives = common_objectives or {}
    common_constraints = common_constraints or {}
    
    for i, blueprint_path in enumerate(blueprint_configs):
        try:
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                objectives=common_objectives,
                constraints=common_constraints
            )
            
            # Add configuration sweep metadata
            result['config_sweep_info'] = {
                'blueprint_path': blueprint_path,
                'config_index': i,
                'success': True
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Configuration sweep for {blueprint_path} failed: {e}")
            results.append({
                'config_sweep_info': {
                    'blueprint_path': blueprint_path,
                    'config_index': i,
                    'success': False,
                    'error': str(e)
                }
            })
    
    successful_runs = sum(1 for r in results if r.get('config_sweep_info', {}).get('success', False))
    logger.info(f"Configuration sweep completed: {successful_runs}/{len(blueprint_configs)} successful")
    
    return results
```

### **File: `brainsmith/automation/utils.py`**
```python
"""
Automation utilities for result aggregation and analysis.
"""

import json
import itertools
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_parameter_combinations(parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters for parameter sweep.
    
    Args:
        parameter_ranges: Dict mapping parameter names to lists of values
        
    Returns:
        List of parameter dictionaries
        
    Example:
        combinations = generate_parameter_combinations({
            'pe_count': [4, 8, 16],
            'simd_width': [2, 4]
        })
        # Returns: [
        #   {'pe_count': 4, 'simd_width': 2},
        #   {'pe_count': 4, 'simd_width': 4},
        #   {'pe_count': 8, 'simd_width': 2},
        #   ...
        # ]
    """
    if not parameter_ranges:
        return [{}]
    
    keys = list(parameter_ranges.keys())
    values = list(parameter_ranges.values())
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple forge() runs.
    
    Args:
        results: List of forge() results
        
    Returns:
        Aggregated analysis with statistics and best results
    """
    if not results:
        return {'error': 'No results to aggregate'}
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', True) and 'error' not in r]
    
    if not successful_results:
        return {
            'total_runs': len(results),
            'successful_runs': 0,
            'success_rate': 0.0,
            'error': 'No successful results found'
        }
    
    # Extract metrics from successful results
    all_metrics = []
    for result in successful_results:
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        if performance:
            all_metrics.append(performance)
    
    # Calculate aggregate statistics
    aggregated_stats = {}
    if all_metrics:
        # Get all metric names
        all_metric_names = set()
        for metrics in all_metrics:
            all_metric_names.update(metrics.keys())
        
        # Calculate statistics for each metric
        for metric_name in all_metric_names:
            values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
            if values:
                aggregated_stats[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
                
                if len(values) > 1:
                    variance = sum((x - aggregated_stats[metric_name]['mean']) ** 2 for x in values) / (len(values) - 1)
                    aggregated_stats[metric_name]['std'] = variance ** 0.5
                else:
                    aggregated_stats[metric_name]['std'] = 0.0
    
    return {
        'total_runs': len(results),
        'successful_runs': len(successful_results),
        'success_rate': len(successful_results) / len(results),
        'aggregated_metrics': aggregated_stats,
        'best_results': find_top_results(successful_results, n=3)
    }


def find_best_result(results: List[Dict[str, Any]], 
                    metric: str = 'throughput',
                    maximize: bool = True) -> Optional[Dict[str, Any]]:
    """
    Find best result based on specified metric.
    
    Args:
        results: List of forge() results
        metric: Metric name to optimize
        maximize: Whether to maximize (True) or minimize (False) the metric
        
    Returns:
        Best result or None if no valid results found
    """
    if not results:
        return None
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', True) and 'error' not in r]
    
    if not successful_results:
        return None
    
    def get_metric_value(result: Dict[str, Any]) -> float:
        """Extract metric value from result."""
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        return performance.get(metric, 0.0)
    
    # Find best result
    if maximize:
        best_result = max(successful_results, key=get_metric_value)
    else:
        best_result = min(successful_results, key=get_metric_value)
    
    # Add optimization metadata
    best_result['optimization_info'] = {
        'optimized_metric': metric,
        'maximize': maximize,
        'metric_value': get_metric_value(best_result),
        'total_candidates': len(results),
        'successful_candidates': len(successful_results)
    }
    
    return best_result


def find_top_results(results: List[Dict[str, Any]], 
                    n: int = 5,
                    metric: str = 'throughput') -> List[Dict[str, Any]]:
    """
    Find top N results based on metric.
    
    Args:
        results: List of forge() results
        n: Number of top results to return
        metric: Metric to rank by
        
    Returns:
        List of top N results
    """
    if not results:
        return []
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', True) and 'error' not in r]
    
    def get_metric_value(result: Dict[str, Any]) -> float:
        metrics = result.get('metrics', {})
        performance = metrics.get('performance', {})
        return performance.get(metric, 0.0)
    
    # Sort by metric (descending)
    sorted_results = sorted(successful_results, key=get_metric_value, reverse=True)
    
    # Return top N
    top_results = sorted_results[:n]
    
    # Add ranking metadata
    for i, result in enumerate(top_results):
        result['ranking_info'] = {
            'rank': i + 1,
            'ranked_by': metric,
            'metric_value': get_metric_value(result)
        }
    
    return top_results


def save_automation_results(results: List[Dict[str, Any]], 
                           output_path: str,
                           include_analysis: bool = True) -> None:
    """
    Save automation results to file.
    
    Args:
        results: List of forge() results
        output_path: Path to save results
        include_analysis: Whether to include aggregated analysis
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'automation_results': results,
        'summary': {
            'total_runs': len(results),
            'successful_runs': sum(1 for r in results if r.get('success', True) and 'error' not in r),
            'timestamp': str(Path().resolve()),  # Current timestamp would be better
        }
    }
    
    # Add aggregated analysis if requested
    if include_analysis:
        save_data['aggregated_analysis'] = aggregate_results(results)
    
    # Save as JSON
    try:
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Automation results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save automation results: {e}")
        raise


def load_automation_results(file_path: str) -> Dict[str, Any]:
    """
    Load automation results from file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Loaded automation results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded automation results from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load automation results: {e}")
        raise


def compare_automation_runs(results1: List[Dict[str, Any]], 
                           results2: List[Dict[str, Any]],
                           metric: str = 'throughput') -> Dict[str, Any]:
    """
    Compare results from two automation runs.
    
    Args:
        results1: First set of results
        results2: Second set of results
        metric: Metric to compare
        
    Returns:
        Comparison analysis
    """
    def get_metric_values(results):
        values = []
        for result in results:
            if result.get('success', True) and 'error' not in result:
                metrics = result.get('metrics', {})
                performance = metrics.get('performance', {})
                if metric in performance:
                    values.append(performance[metric])
        return values
    
    values1 = get_metric_values(results1)
    values2 = get_metric_values(results2)
    
    if not values1 or not values2:
        return {'error': 'Insufficient data for comparison'}
    
    mean1 = sum(values1) / len(values1)
    mean2 = sum(values2) / len(values2)
    
    improvement = ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0
    
    return {
        'metric': metric,
        'run1': {
            'mean': mean1,
            'best': max(values1),
            'count': len(values1)
        },
        'run2': {
            'mean': mean2,
            'best': max(values2),
            'count': len(values2)
        },
        'improvement_percent': improvement,
        'better_run': 'run2' if improvement > 0 else 'run1'
    }
```

### **File: `brainsmith/automation/__init__.py`**
```python
"""
BrainSmith Simple Automation Helpers

Provides simple utilities for common automation patterns in FPGA design space exploration.
Instead of complex workflow orchestration, these helpers make it easy to run forge() 
multiple times with different parameters or configurations.

Key Philosophy:
- Simple helpers that leverage existing forge() function
- No enterprise workflow orchestration
- Focus on practical automation patterns users actually need
- Minimal complexity, maximum utility

Example Usage:
    from brainsmith.automation import parameter_sweep, batch_process
    
    # Parameter sweep
    results = parameter_sweep(
        "model.onnx", 
        "blueprint.yaml",
        {'pe_count': [4, 8, 16], 'simd_width': [2, 4, 8]}
    )
    
    # Batch processing
    results = batch_process([
        ("model1.onnx", "blueprint1.yaml"),
        ("model2.onnx", "blueprint2.yaml")
    ])
"""

from .parameter_sweep import (
    parameter_sweep,
    grid_search,
    random_search
)

from .batch_processing import (
    batch_process,
    multi_objective_runs,
    configuration_sweep
)

from .utils import (
    aggregate_results,
    find_best_result,
    find_top_results,
    save_automation_results,
    load_automation_results,
    compare_automation_runs
)

__version__ = "0.1.0"
__author__ = "BrainSmith Development Team"

# Export simple automation helpers
__all__ = [
    # Parameter exploration
    'parameter_sweep',
    'grid_search', 
    'random_search',
    
    # Batch processing
    'batch_process',
    'multi_objective_runs',
    'configuration_sweep',
    
    # Result analysis
    'aggregate_results',
    'find_best_result',
    'find_top_results',
    'save_automation_results',
    'load_automation_results',
    'compare_automation_runs'
]

# Initialize logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BrainSmith Simple Automation v{__version__} initialized")
logger.info("Available helpers: parameter_sweep, batch_process, result aggregation")
```

---

## 🧪 **Testing Strategy**

### **Create Test File: `tests/test_automation_helpers.py`**
```python
def test_parameter_sweep():
    """Test parameter sweep functionality."""
    
def test_grid_search():
    """Test grid search optimization."""
    
def test_batch_processing():
    """Test batch processing multiple models."""
    
def test_result_aggregation():
    """Test result aggregation utilities."""
    
def test_integration_with_forge():
    """Test integration with forge() function."""
```

---

## 📊 **Implementation Impact**

### **Before (Enterprise Bloat):**
- **Files**: 9 files, 1,400+ lines
- **Exports**: 36+ enterprise classes/functions
- **Complexity**: Workflow orchestration, ML learning, quality frameworks
- **Usage**: Complex enterprise automation engine

### **After (Simple Helpers):**
- **Files**: 4 files, ~210 lines total
- **Exports**: 11 focused helper functions
- **Complexity**: Simple utilities that call forge()
- **Usage**: Straightforward automation patterns

### **Benefits:**
- **85% code reduction**
- **Simple, focused API**
- **Leverages existing forge() function**
- **Practical automation users actually need**

---

## ⏱️ **Implementation Timeline**

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | 15 min | Remove enterprise bloat files |
| Phase 2 | 45 min | Implement simple automation helpers |
| Phase 3 | 20 min | Create tests and validation |
| Phase 4 | 15 min | Documentation and examples |
| **Total** | **95 min** | **Complete transformation** |

---

## 🎯 **Success Criteria**

1. ✅ **Code reduction**: 1,400+ lines → ~210 lines (85% reduction)
2. ✅ **API simplification**: 36+ exports → 11 focused functions
3. ✅ **Practical focus**: Parameter sweeps, batch processing, result aggregation
4. ✅ **Integration**: Leverages existing forge() function
5. ✅ **User-friendly**: Simple patterns users actually need
6. ✅ **Maintainable**: Minimal complexity, clear purpose

**Ready to implement the automation module simplification!** 🚀