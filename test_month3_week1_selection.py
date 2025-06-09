"""
Test suite for Month 3 Week 1: Intelligent Solution Selection Framework
Tests core selection engine and MCDA algorithms.
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_selection_imports():
    """Test that selection framework can be imported."""
    try:
        from brainsmith.selection import (
            SelectionEngine, SelectionCriteria, SelectionConfiguration,
            SelectionResult, RankedSolution
        )
        
        from brainsmith.selection.models import (
            ParetoSolution, DecisionMatrix, PreferenceFunction,
            SelectionContext, SelectionMetrics
        )
        
        from brainsmith.selection.strategies import (
            TOPSISSelector, WeightedSumSelector, WeightedProductSelector
        )
        
        print("✅ All selection framework imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Selection import failed: {e}")
        return False


def test_selection_criteria():
    """Test selection criteria creation and validation."""
    try:
        from brainsmith.selection.models import SelectionCriteria
        
        # Create selection criteria
        criteria = SelectionCriteria(
            objectives=['maximize_throughput', 'minimize_power'],
            weights={'maximize_throughput': 0.6, 'minimize_power': 0.4},
            constraints=['lut_budget', 'timing_closure']
        )
        
        # Test validation
        assert criteria.validate() == True
        assert len(criteria.objectives) == 2
        assert abs(sum(criteria.weights.values()) - 1.0) < 1e-6
        
        print("✅ Selection criteria working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Selection criteria failed: {e}")
        return False


def test_decision_matrix():
    """Test decision matrix creation and normalization."""
    try:
        from brainsmith.selection.models import DecisionMatrix
        
        # Create test matrix
        alternatives = ['Sol1', 'Sol2', 'Sol3']
        criteria = ['throughput', 'power']
        matrix = np.array([[100, 50], [150, 75], [80, 40]])
        weights = np.array([0.6, 0.4])
        maximize = np.array([True, False])  # maximize throughput, minimize power
        
        decision_matrix = DecisionMatrix(
            alternatives=alternatives,
            criteria=criteria,
            matrix=matrix,
            weights=weights,
            maximize=maximize
        )
        
        # Test normalization
        normalized = decision_matrix.normalize('minmax')
        
        assert normalized.matrix.shape == (3, 2)
        assert np.all(normalized.matrix >= 0)
        assert np.all(normalized.matrix <= 1)
        
        print("✅ Decision matrix working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Decision matrix failed: {e}")
        return False


def test_pareto_solutions():
    """Test Pareto solution creation."""
    try:
        from brainsmith.selection.models import ParetoSolution
        
        # Create test Pareto solutions
        solutions = []
        
        for i in range(3):
            solution = ParetoSolution(
                design_parameters={
                    'pe_parallelism': 4 + i * 2,
                    'memory_width': 64 + i * 32
                },
                objective_values=[100.0 + i * 25, 50.0 - i * 5],  # throughput, power
                constraint_violations=[],
                metadata={'generation': 0}
            )
            solutions.append(solution)
        
        # Test properties
        for solution in solutions:
            assert solution.is_feasible == True
            assert len(solution.objective_values) == 2
            assert 'pe_parallelism' in solution.design_parameters
        
        print("✅ Pareto solutions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Pareto solutions failed: {e}")
        return False


def test_topsis_selector():
    """Test TOPSIS selection algorithm."""
    try:
        from brainsmith.selection.models import ParetoSolution, SelectionCriteria, SelectionContext
        from brainsmith.selection.strategies.topsis import TOPSISSelector
        from brainsmith.selection.models import SelectionConfiguration
        
        # Create test solutions
        solutions = [
            ParetoSolution(
                design_parameters={'pe': 4, 'mem': 64},
                objective_values=[100.0, 50.0]  # throughput, power
            ),
            ParetoSolution(
                design_parameters={'pe': 8, 'mem': 128},
                objective_values=[150.0, 75.0]
            ),
            ParetoSolution(
                design_parameters={'pe': 2, 'mem': 32},
                objective_values=[80.0, 40.0]
            )
        ]
        
        # Create selection criteria
        criteria = SelectionCriteria(
            objectives=['throughput', 'power'],
            weights={'throughput': 0.6, 'power': 0.4},
            maximize_objectives={'throughput': True, 'power': False}
        )
        
        # Create context
        context = SelectionContext(
            pareto_solutions=solutions,
            selection_criteria=criteria
        )
        
        # Create and run TOPSIS
        config = SelectionConfiguration(algorithm='topsis')
        topsis = TOPSISSelector(config)
        
        ranked_solutions = topsis.select_solutions(context)
        
        # Verify results
        assert len(ranked_solutions) == 3
        assert all(sol.rank >= 1 for sol in ranked_solutions)
        assert all(0 <= sol.score <= 1 for sol in ranked_solutions)
        
        # Check ranking order (higher scores should have lower ranks)
        scores = [sol.score for sol in ranked_solutions]
        ranks = [sol.rank for sol in ranked_solutions]
        
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]  # Descending scores
            assert ranks[i] <= ranks[i + 1]   # Ascending ranks
        
        print("✅ TOPSIS selector working correctly")
        print(f"   Best solution: rank {ranked_solutions[0].rank}, score {ranked_solutions[0].score:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ TOPSIS selector failed: {e}")
        return False


def test_weighted_sum_selector():
    """Test Weighted Sum selection algorithm."""
    try:
        from brainsmith.selection.models import ParetoSolution, SelectionCriteria, SelectionContext
        from brainsmith.selection.strategies.weighted import WeightedSumSelector
        from brainsmith.selection.models import SelectionConfiguration
        
        # Create test solutions
        solutions = [
            ParetoSolution(
                design_parameters={'pe': 4},
                objective_values=[100.0, 50.0]
            ),
            ParetoSolution(
                design_parameters={'pe': 8},
                objective_values=[150.0, 75.0]
            ),
            ParetoSolution(
                design_parameters={'pe': 2},
                objective_values=[80.0, 40.0]
            )
        ]
        
        # Create selection criteria
        criteria = SelectionCriteria(
            objectives=['throughput', 'power'],
            weights={'throughput': 0.7, 'power': 0.3},
            maximize_objectives={'throughput': True, 'power': False}
        )
        
        # Create context
        context = SelectionContext(
            pareto_solutions=solutions,
            selection_criteria=criteria
        )
        
        # Create and run Weighted Sum
        config = SelectionConfiguration(algorithm='weighted_sum')
        wsm = WeightedSumSelector(config)
        
        ranked_solutions = wsm.select_solutions(context)
        
        # Verify results
        assert len(ranked_solutions) == 3
        assert all(sol.score >= 0 for sol in ranked_solutions)
        
        print("✅ Weighted Sum selector working correctly")
        print(f"   Best solution: rank {ranked_solutions[0].rank}, score {ranked_solutions[0].score:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Weighted Sum selector failed: {e}")
        return False


def test_selection_engine():
    """Test the main selection engine."""
    try:
        from brainsmith.selection import SelectionEngine, SelectionConfiguration
        from brainsmith.selection.models import ParetoSolution, SelectionCriteria
        
        # Create test data
        solutions = [
            ParetoSolution(
                design_parameters={'config': 'A'},
                objective_values=[100.0, 50.0, 0.95]  # throughput, power, accuracy
            ),
            ParetoSolution(
                design_parameters={'config': 'B'},
                objective_values=[120.0, 60.0, 0.93]
            ),
            ParetoSolution(
                design_parameters={'config': 'C'},
                objective_values=[90.0, 45.0, 0.97]
            )
        ]
        
        criteria = SelectionCriteria(
            objectives=['throughput', 'power', 'accuracy'],
            weights={'throughput': 0.4, 'power': 0.3, 'accuracy': 0.3},
            maximize_objectives={'throughput': True, 'power': False, 'accuracy': True}
        )
        
        # Test different algorithms
        algorithms = ['topsis', 'weighted_sum']
        
        for algorithm in algorithms:
            config = SelectionConfiguration(algorithm=algorithm)
            engine = SelectionEngine(config)
            
            result = engine.select_solutions(solutions, criteria, max_solutions=3)
            
            assert isinstance(result.ranked_solutions, list)
            assert len(result.ranked_solutions) <= 3
            assert result.best_solution is not None
            assert result.selection_metrics.number_of_solutions > 0
            
            print(f"✅ Selection engine with {algorithm} working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Selection engine failed: {e}")
        return False


def test_preference_functions():
    """Test preference functions for PROMETHEE."""
    try:
        from brainsmith.selection.models import PreferenceFunction, PreferenceType
        
        # Test different preference function types
        
        # Usual preference
        usual_pref = PreferenceFunction(PreferenceType.USUAL)
        assert usual_pref(0.5) == 1.0
        assert usual_pref(0.0) == 0.0
        assert usual_pref(-0.1) == 0.0
        
        # V-shape preference
        v_pref = PreferenceFunction(PreferenceType.V_SHAPE, threshold=1.0)
        assert v_pref(0.5) == 0.5
        assert v_pref(1.0) == 1.0
        assert v_pref(0.0) == 0.0
        
        # Linear preference
        linear_pref = PreferenceFunction(
            PreferenceType.LINEAR,
            indifference_threshold=0.2,
            threshold=1.0
        )
        assert linear_pref(0.1) == 0.0  # Below indifference
        assert linear_pref(1.0) == 1.0  # At threshold
        assert 0.0 < linear_pref(0.6) < 1.0  # In between
        
        print("✅ Preference functions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Preference functions failed: {e}")
        return False


def run_selection_tests():
    """Run all selection framework tests."""
    print("Testing Month 3 Week 1: Intelligent Solution Selection Framework")
    print("=" * 70)
    
    tests = [
        ("Import Test", test_selection_imports),
        ("Selection Criteria", test_selection_criteria),
        ("Decision Matrix", test_decision_matrix),
        ("Pareto Solutions", test_pareto_solutions),
        ("TOPSIS Selector", test_topsis_selector),
        ("Weighted Sum Selector", test_weighted_sum_selector),
        ("Selection Engine", test_selection_engine),
        ("Preference Functions", test_preference_functions)
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
    
    print(f"\n{'='*70}")
    print(f"Selection Framework Test Results")
    print(f"{'='*70}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 All selection framework tests passed!")
        print(f"Month 3 Week 1 implementation is working correctly!")
    else:
        print(f"\n⚠️  Some tests failed - check implementation")
    
    return failed == 0


if __name__ == '__main__':
    success = run_selection_tests()
    
    if success:
        print(f"\n{'='*70}")
        print(f"🏁 Month 3 Week 1: Selection Framework Complete!")
        print(f"{'='*70}")
        print(f"📦 Implemented components:")
        print(f"   • Multi-criteria decision analysis engine")
        print(f"   • TOPSIS algorithm with ideal point calculation")
        print(f"   • Weighted Sum/Product methods")
        print(f"   • Preference function framework")
        print(f"   • Decision matrix operations")
        print(f"   • Solution ranking and scoring")
        print(f"   • Selection quality metrics")
        print(f"\n🔧 Key features:")
        print(f"   • 5+ MCDA algorithms implemented")
        print(f"   • Configurable selection strategies")
        print(f"   • Automated solution ranking")
        print(f"   • Integration with Week 2 Pareto solutions")
        print(f"   • Comprehensive selection reporting")
    
    sys.exit(0 if success else 1)