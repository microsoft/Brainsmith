"""
Core functionality tests for Week 3 Advanced DSE Framework
Tests core components without external dependencies.
"""

import os
import sys
import unittest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test core imports
def test_imports():
    """Test that all core components can be imported."""
    try:
        # Core multi-objective optimization
        from brainsmith.dse.advanced.multi_objective import (
            ParetoSolution, ParetoArchive, MultiObjectiveOptimizer,
            NSGA2, HypervolumeCalculator
        )
        
        # FPGA algorithms
        from brainsmith.dse.advanced.algorithms import (
            FPGADesignCandidate, FPGAGeneticOperators,
            FPGAGeneticAlgorithm, AdaptiveSimulatedAnnealing
        )
        
        # Objectives and constraints
        from brainsmith.dse.advanced.objectives import (
            ObjectiveDefinition, ConstraintDefinition,
            ObjectiveRegistry
        )
        
        # Learning components
        from brainsmith.dse.advanced.learning import (
            SearchPattern, SearchMemory, AdaptiveStrategySelector
        )
        
        # Analysis tools
        from brainsmith.dse.advanced.analysis import (
            DesignSpaceAnalyzer, ParetoFrontierAnalyzer
        )
        
        # Integration framework
        from brainsmith.dse.advanced.integration import (
            DSEResults, DesignProblem, OptimizationConfiguration
        )
        
        print("✅ All core imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_core_data_structures():
    """Test core data structures."""
    try:
        from brainsmith.dse.advanced.multi_objective import ParetoSolution, ParetoArchive
        
        # Test ParetoSolution
        solution = ParetoSolution(
            design_parameters={'pe_parallelism': 4, 'memory_width': 64},
            objective_values=[100.0, 50.0],
            constraint_violations=[],
            metadata={'generation': 0}
        )
        
        assert solution.design_parameters['pe_parallelism'] == 4
        assert len(solution.objective_values) == 2
        assert solution.is_feasible
        
        # Test ParetoArchive
        archive = ParetoArchive(max_size=10)
        archive.add_solution(solution)
        
        assert len(archive.get_solutions()) == 1
        
        print("✅ Core data structures working")
        return True
        
    except Exception as e:
        print(f"❌ Core data structures failed: {e}")
        return False


def test_objective_registry():
    """Test objective registry functionality."""
    try:
        from brainsmith.dse.advanced.objectives import ObjectiveRegistry, ObjectiveDefinition
        
        registry = ObjectiveRegistry()
        
        # Test predefined objectives
        maximize_throughput = registry.get_objective('maximize_throughput')
        assert maximize_throughput is not None
        assert maximize_throughput.optimization_direction == 'maximize'
        
        # Test predefined constraints
        lut_budget = registry.get_constraint('lut_budget')
        assert lut_budget is not None
        assert lut_budget.constraint_type == 'resource'
        
        print("✅ Objective registry working")
        return True
        
    except Exception as e:
        print(f"❌ Objective registry failed: {e}")
        return False


def test_fpga_design_candidate():
    """Test FPGA design candidate."""
    try:
        from brainsmith.dse.advanced.algorithms import FPGADesignCandidate
        
        candidate = FPGADesignCandidate(
            parameters={'pe_parallelism': 4, 'memory_width': 64},
            architecture='finn_dataflow',
            transformation_sequence=['ConvertONNXToFINN'],
            resource_budget={'lut': 50000, 'dsp': 500}
        )
        
        assert candidate.parameters['pe_parallelism'] == 4
        assert candidate.architecture == 'finn_dataflow'
        
        # Test conversion to Pareto solution
        pareto_sol = candidate.to_pareto_solution([100.0, 50.0])
        assert len(pareto_sol.objective_values) == 2
        
        print("✅ FPGA design candidate working")
        return True
        
    except Exception as e:
        print(f"❌ FPGA design candidate failed: {e}")
        return False


def test_search_memory():
    """Test search memory functionality."""
    try:
        from brainsmith.dse.advanced.learning import SearchMemory
        
        memory = SearchMemory(memory_size=100)
        
        # Store a pattern
        design_params = {'param1': 2.0, 'param2': 15.0}
        objectives = [0.8, 0.6]
        
        pattern_id = memory.store_pattern(design_params, objectives, {'test': True})
        
        assert pattern_id in memory.patterns
        assert memory.patterns[pattern_id].success_rate == 1.0
        
        # Retrieve similar patterns
        query_params = {'param1': 2.1, 'param2': 14.5}
        similar = memory.retrieve_similar_patterns(query_params, top_k=1)
        
        assert len(similar) <= 1
        
        print("✅ Search memory working")
        return True
        
    except Exception as e:
        print(f"❌ Search memory failed: {e}")
        return False


def test_design_space_analyzer():
    """Test design space analyzer (basic functionality)."""
    try:
        from brainsmith.dse.advanced.analysis import DesignSpaceAnalyzer
        
        analyzer = DesignSpaceAnalyzer()
        
        design_space = {
            'param1': (0.0, 10.0),
            'param2': [1, 2, 4, 8],
            'param3': (50.0, 200.0)
        }
        
        # Test sample generation
        samples = analyzer._generate_samples(design_space, 10)
        
        assert len(samples) == 10
        for sample in samples:
            assert 'param1' in sample
            assert 'param2' in sample
            assert 'param3' in sample
            assert 0.0 <= sample['param1'] <= 10.0
            assert sample['param2'] in [1, 2, 4, 8]
            assert 50.0 <= sample['param3'] <= 200.0
        
        print("✅ Design space analyzer working")
        return True
        
    except Exception as e:
        print(f"❌ Design space analyzer failed: {e}")
        return False


def test_configuration_objects():
    """Test configuration and problem objects."""
    try:
        from brainsmith.dse.advanced.integration import (
            DesignProblem, OptimizationConfiguration
        )
        
        # Test OptimizationConfiguration
        config = OptimizationConfiguration(
            algorithm='nsga2',
            population_size=50,
            max_generations=100,
            learning_enabled=True
        )
        
        assert config.algorithm == 'nsga2'
        assert config.population_size == 50
        assert config.learning_enabled == True
        
        # Test DesignProblem
        problem = DesignProblem(
            model_path='/test/model.onnx',
            design_space={'param1': (1, 10)},
            objectives=['maximize_throughput'],
            constraints=['lut_budget'],
            device_target='xczu7ev',
            optimization_config={}
        )
        
        assert problem.model_path == '/test/model.onnx'
        assert len(problem.objectives) == 1
        assert problem.device_target == 'xczu7ev'
        
        print("✅ Configuration objects working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration objects failed: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions from main module."""
    try:
        from brainsmith.dse.advanced import (
            create_dse_configuration, create_design_problem,
            QUICK_DSE_CONFIG, PERFORMANCE_OBJECTIVES
        )
        
        # Test configuration creation
        config = create_dse_configuration(
            algorithm='genetic_algorithm',
            population_size=20,
            learning_enabled=False
        )
        
        assert config.algorithm == 'genetic_algorithm'
        assert config.population_size == 20
        assert config.learning_enabled == False
        
        # Test design problem creation
        problem = create_design_problem(
            model_path='/test/model.onnx',
            objectives=['maximize_throughput_ops', 'minimize_power_mw'],
            device_target='xczu7ev'
        )
        
        assert problem.model_path == '/test/model.onnx'
        assert len(problem.objectives) == 2
        
        # Test predefined configurations
        assert QUICK_DSE_CONFIG.algorithm == 'adaptive'
        assert isinstance(PERFORMANCE_OBJECTIVES, list)
        assert len(PERFORMANCE_OBJECTIVES) > 0
        
        print("✅ Convenience functions working")
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
        return False


def run_core_tests():
    """Run all core functionality tests."""
    print("Testing Week 3 Advanced DSE Core Functionality")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Core Data Structures", test_core_data_structures),
        ("Objective Registry", test_objective_registry),
        ("FPGA Design Candidate", test_fpga_design_candidate),
        ("Search Memory", test_search_memory),
        ("Design Space Analyzer", test_design_space_analyzer),
        ("Configuration Objects", test_configuration_objects),
        ("Convenience Functions", test_convenience_functions)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Core Functionality Test Results")
    print(f"{'='*50}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 All core functionality tests passed!")
        print(f"Advanced DSE Framework is ready for use!")
    else:
        print(f"\n⚠️  Some tests failed - check implementation")
    
    return failed == 0


if __name__ == '__main__':
    success = run_core_tests()
    sys.exit(0 if success else 1)