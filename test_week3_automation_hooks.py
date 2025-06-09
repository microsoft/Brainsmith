"""
Test Week 3 Automation Hooks Implementation

Test the automation hooks components:
1. Strategy Decision Tracker
2. Parameter Sensitivity Monitor
3. Problem Characterization System
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_strategy_decision_tracker():
    """Test Strategy Decision Tracker functionality"""
    try:
        print("📊 Testing Strategy Decision Tracker...")
        
        from brainsmith.hooks.strategy_tracking import StrategyDecisionTracker
        from brainsmith.hooks.types import ProblemContext, PerformanceMetrics
        
        tracker = StrategyDecisionTracker()
        
        # Test recording strategy decision
        context = ProblemContext(
            model_info={'parameter_count': 1000000, 'layer_count': 20},
            targets={'throughput': 1000, 'latency': 25},
            constraints={'luts': 0.8, 'power': 10.0},
            platform={'resources': {'luts': 100000, 'dsps': 2000}},
            problem_size=50
        )
        
        decision_id = tracker.record_strategy_choice(
            context=context,
            strategy="genetic_algorithm",
            rationale="Complex multi-objective problem requires robust exploration"
        )
        
        assert decision_id is not None
        assert len(decision_id) > 0
        
        # Test recording strategy outcome
        performance = PerformanceMetrics(
            throughput=950.0,
            latency=28.0,
            power=9.5,
            efficiency=0.85,
            convergence_time=120.0,
            solution_quality=0.88
        )
        
        tracker.record_strategy_outcome(
            strategy="genetic_algorithm",
            performance=performance,
            decision_id=decision_id
        )
        
        # Test effectiveness analysis
        report = tracker.analyze_strategy_effectiveness("complex_optimization")
        
        assert report is not None
        assert len(report.recommendations) > 0
        
        # Test decision history
        history = tracker.get_decision_history(limit=10)
        assert len(history) >= 1
        
        # Test strategy statistics
        stats = tracker.get_strategy_statistics()
        assert 'genetic_algorithm' in stats
        
        print("   ✅ Strategy Decision Tracker working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Strategy Decision Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_sensitivity_monitor():
    """Test Parameter Sensitivity Monitor functionality"""
    try:
        print("🔬 Testing Parameter Sensitivity Monitor...")
        
        from brainsmith.hooks.sensitivity import ParameterSensitivityMonitor
        
        monitor = ParameterSensitivityMonitor()
        
        # Test tracking parameter changes
        parameter_changes = {
            'pe_parallelism': 8,
            'simd_width': 4,
            'clock_frequency': 150.0
        }
        
        monitor.track_parameter_changes(parameter_changes)
        
        # Record some performance values
        monitor.record_performance(0.75)
        monitor.record_performance(0.82)
        monitor.record_performance(0.78)
        
        # Test measuring performance impact
        impact_analysis = monitor.measure_performance_impact('pe_parallelism')
        
        assert impact_analysis is not None
        assert len(impact_analysis.direct_impact) >= 0
        
        # Test identifying critical parameters
        critical_params = monitor.identify_critical_parameters()
        assert isinstance(critical_params, list)
        
        # Test generating insights
        insights = monitor.generate_sensitivity_insights(impact_analysis)
        assert isinstance(insights, list)
        
        # Test parameter statistics
        stats = monitor.get_parameter_statistics()
        assert isinstance(stats, dict)
        
        # Test multiple parameter changes for interaction analysis
        for i in range(5):
            changes = {
                'pe_parallelism': 4 + i * 2,
                'simd_width': 2 + i,
                'memory_mode': 'internal' if i % 2 == 0 else 'external'
            }
            monitor.track_parameter_changes(changes)
            monitor.record_performance(0.7 + i * 0.05)
        
        # Test impact analysis with more data
        updated_impact = monitor.measure_performance_impact('pe_parallelism')
        updated_insights = monitor.generate_sensitivity_insights(updated_impact)
        
        assert len(updated_insights) >= 0
        
        print("   ✅ Parameter Sensitivity Monitor working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Parameter Sensitivity Monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_problem_characterizer():
    """Test Problem Characterization System functionality"""
    try:
        print("🎯 Testing Problem Characterization System...")
        
        from brainsmith.hooks.characterization import ProblemCharacterizer
        
        characterizer = ProblemCharacterizer()
        
        # Test problem characterization
        test_problem = {
            'model': {
                'parameter_count': 5000000,
                'layer_count': 50,
                'operator_types': ['Conv', 'MatMul', 'Relu', 'Add', 'BatchNorm']
            },
            'targets': {
                'throughput': 1000,
                'latency': 20,
                'power': 15
            },
            'constraints': {
                'max_luts': 0.8,
                'max_power': 20.0,
                'timing_constraint': 100
            },
            'resources': {
                'luts': 200000,
                'dsps': 4000,
                'brams': 1000
            },
            'design_space_size': 10000,
            'variable_types': {
                'continuous': 5,
                'discrete': 10,
                'integer': 3
            }
        }
        
        # Capture problem characteristics
        characteristics = characterizer.capture_problem_characteristics(test_problem)
        
        assert characteristics.model_size == 5000000
        assert characteristics.model_complexity > 0
        assert len(characteristics.performance_targets) == 3
        assert characteristics.constraint_tightness >= 0
        
        # Test problem classification
        problem_type = characterizer.classify_problem_type(characteristics)
        
        assert problem_type.type_name is not None
        assert problem_type.confidence > 0
        assert problem_type.complexity is not None
        assert len(problem_type.explanation) > 0
        assert len(problem_type.recommended_strategies) > 0
        
        # Test strategy recommendations
        strategies = characterizer.recommend_strategies(problem_type)
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert all(isinstance(s, str) for s in strategies)
        
        # Test with different problem types
        simple_problem = {
            'model': {
                'parameter_count': 100000,
                'layer_count': 10,
                'operator_types': ['Conv', 'Relu']
            },
            'targets': {'throughput': 500},
            'constraints': {'max_luts': 0.5},
            'resources': {'luts': 50000},
            'design_space_size': 100,
            'variable_types': {'continuous': 2, 'discrete': 3}
        }
        
        simple_characteristics = characterizer.capture_problem_characteristics(simple_problem)
        simple_type = characterizer.classify_problem_type(simple_characteristics)
        
        assert simple_type.type_name is not None
        assert simple_type.complexity.value in ['simple', 'moderate']
        
        print("   ✅ Problem Characterization System working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Problem Characterization System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_automation_hooks_integration():
    """Test integration between automation hooks components"""
    try:
        print("🔗 Testing Automation Hooks Integration...")
        
        from brainsmith.hooks.strategy_tracking import StrategyDecisionTracker
        from brainsmith.hooks.sensitivity import ParameterSensitivityMonitor
        from brainsmith.hooks.characterization import ProblemCharacterizer
        from brainsmith.hooks.types import ProblemContext, PerformanceMetrics
        
        # Initialize all components
        strategy_tracker = StrategyDecisionTracker()
        sensitivity_monitor = ParameterSensitivityMonitor()
        characterizer = ProblemCharacterizer()
        
        # Simulate optimization workflow
        # Step 1: Characterize problem
        problem = {
            'model': {'parameter_count': 2000000, 'layer_count': 30},
            'targets': {'throughput': 800, 'latency': 30},
            'constraints': {'max_luts': 0.7},
            'resources': {'luts': 150000},
            'design_space_size': 5000,
            'variable_types': {'continuous': 3, 'discrete': 5}
        }
        
        characteristics = characterizer.capture_problem_characteristics(problem)
        problem_type = characterizer.classify_problem_type(characteristics)
        recommended_strategies = characterizer.recommend_strategies(problem_type)
        
        # Step 2: Record strategy decision
        context = ProblemContext(
            model_info=problem['model'],
            targets=problem['targets'],
            constraints=problem['constraints'],
            platform={'resources': problem['resources']}
        )
        
        selected_strategy = recommended_strategies[0]
        decision_id = strategy_tracker.record_strategy_choice(
            context=context,
            strategy=selected_strategy,
            rationale=f"Selected based on problem classification: {problem_type.type_name}"
        )
        
        # Step 3: Simulate optimization with parameter tracking
        optimization_iterations = [
            ({'pe_parallelism': 4, 'simd_width': 2}, 0.65),
            ({'pe_parallelism': 6, 'simd_width': 3}, 0.72),
            ({'pe_parallelism': 8, 'simd_width': 4}, 0.78),
            ({'pe_parallelism': 10, 'simd_width': 4}, 0.75),
            ({'pe_parallelism': 8, 'simd_width': 6}, 0.82)
        ]
        
        for params, performance in optimization_iterations:
            sensitivity_monitor.track_parameter_changes(params)
            sensitivity_monitor.record_performance(performance)
        
        # Step 4: Analyze sensitivity
        pe_impact = sensitivity_monitor.measure_performance_impact('pe_parallelism')
        simd_impact = sensitivity_monitor.measure_performance_impact('simd_width')
        
        critical_params = sensitivity_monitor.identify_critical_parameters()
        insights = sensitivity_monitor.generate_sensitivity_insights(pe_impact)
        
        # Step 5: Record final outcome
        final_performance = PerformanceMetrics(
            throughput=820.0,
            latency=28.0,
            efficiency=0.82,
            convergence_time=150.0,
            solution_quality=0.85
        )
        
        strategy_tracker.record_strategy_outcome(
            strategy=selected_strategy,
            performance=final_performance,
            decision_id=decision_id
        )
        
        # Step 6: Generate effectiveness report
        effectiveness_report = strategy_tracker.analyze_strategy_effectiveness()
        
        # Validate integration results
        assert len(recommended_strategies) > 0
        assert decision_id is not None
        assert len(critical_params) >= 0
        assert len(insights) >= 0
        assert effectiveness_report is not None
        assert len(effectiveness_report.recommendations) > 0
        
        print("   ✅ Automation Hooks Integration working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Automation Hooks Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_collection_workflow():
    """Test complete data collection workflow for learning"""
    try:
        print("📚 Testing Data Collection Workflow...")
        
        from brainsmith.hooks.strategy_tracking import StrategyDecisionTracker
        from brainsmith.hooks.sensitivity import ParameterSensitivityMonitor
        from brainsmith.hooks.characterization import ProblemCharacterizer
        from brainsmith.hooks.types import ProblemContext, PerformanceMetrics
        
        # Initialize components
        tracker = StrategyDecisionTracker()
        monitor = ParameterSensitivityMonitor()
        characterizer = ProblemCharacterizer()
        
        # Simulate multiple optimization runs for data collection
        test_cases = [
            {
                'problem': {
                    'model': {'parameter_count': 1000000, 'layer_count': 15},
                    'targets': {'throughput': 500},
                    'constraints': {'max_luts': 0.6},
                    'resources': {'luts': 100000}
                },
                'strategy': 'genetic_algorithm',
                'params_sequence': [
                    ({'pe': 2, 'simd': 2}, 0.6),
                    ({'pe': 4, 'simd': 2}, 0.7),
                    ({'pe': 4, 'simd': 4}, 0.75)
                ],
                'final_performance': {'solution_quality': 0.78, 'convergence_time': 100}
            },
            {
                'problem': {
                    'model': {'parameter_count': 5000000, 'layer_count': 40},
                    'targets': {'throughput': 1200, 'latency': 15},
                    'constraints': {'max_luts': 0.8, 'max_power': 20},
                    'resources': {'luts': 200000, 'dsps': 3000}
                },
                'strategy': 'bayesian_optimization',
                'params_sequence': [
                    ({'pe': 8, 'simd': 4}, 0.7),
                    ({'pe': 12, 'simd': 6}, 0.8),
                    ({'pe': 16, 'simd': 8}, 0.85)
                ],
                'final_performance': {'solution_quality': 0.87, 'convergence_time': 80}
            }
        ]
        
        collected_data = {
            'decisions': [],
            'outcomes': [],
            'sensitivity_data': [],
            'problem_classifications': []
        }
        
        for i, case in enumerate(test_cases):
            # Characterize problem
            characteristics = characterizer.capture_problem_characteristics(case['problem'])
            problem_type = characterizer.classify_problem_type(characteristics)
            collected_data['problem_classifications'].append(problem_type.to_dict())
            
            # Record strategy decision
            context = ProblemContext(
                model_info=case['problem']['model'],
                targets=case['problem']['targets'],
                constraints=case['problem']['constraints']
            )
            
            decision_id = tracker.record_strategy_choice(
                context=context,
                strategy=case['strategy'],
                rationale=f"Test case {i+1} strategy selection"
            )
            
            # Track parameter changes
            for params, perf in case['params_sequence']:
                monitor.track_parameter_changes(params)
                monitor.record_performance(perf)
            
            # Record final outcome
            final_perf = PerformanceMetrics(
                solution_quality=case['final_performance']['solution_quality'],
                convergence_time=case['final_performance']['convergence_time'],
                efficiency=0.8,
                throughput=case['problem']['targets'].get('throughput', 500)
            )
            
            tracker.record_strategy_outcome(
                strategy=case['strategy'],
                performance=final_perf,
                decision_id=decision_id
            )
        
        # Collect comprehensive data
        decision_history = tracker.get_decision_history()
        strategy_stats = tracker.get_strategy_statistics()
        parameter_stats = monitor.get_parameter_statistics()
        effectiveness_report = tracker.analyze_strategy_effectiveness()
        
        collected_data['decisions'] = [d.to_dict() for d in decision_history]
        collected_data['strategy_statistics'] = strategy_stats
        collected_data['parameter_statistics'] = parameter_stats
        collected_data['effectiveness_report'] = effectiveness_report.to_dict()
        
        # Validate collected data quality
        assert len(collected_data['decisions']) >= 2
        assert len(collected_data['problem_classifications']) == 2
        assert len(collected_data['strategy_statistics']) >= 2
        assert len(collected_data['parameter_statistics']) >= 2
        assert len(collected_data['effectiveness_report']['recommendations']) > 0
        
        print("   ✅ Data Collection Workflow working correctly")
        print(f"   📊 Collected {len(collected_data['decisions'])} decisions")
        print(f"   📊 Analyzed {len(collected_data['strategy_statistics'])} strategies")
        print(f"   📊 Tracked {len(collected_data['parameter_statistics'])} parameters")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data Collection Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_automation_hooks_tests():
    """Run all automation hooks tests"""
    print("🧪 Week 3 Automation Hooks Implementation Testing")
    print("=" * 60)
    print("Testing strategy tracking, sensitivity monitoring, and problem characterization")
    print("=" * 60)
    
    tests = [
        ("Strategy Decision Tracker", test_strategy_decision_tracker),
        ("Parameter Sensitivity Monitor", test_parameter_sensitivity_monitor),
        ("Problem Characterization System", test_problem_characterizer),
        ("Automation Hooks Integration", test_automation_hooks_integration),
        ("Data Collection Workflow", test_data_collection_workflow)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"AUTOMATION HOOKS TEST RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL AUTOMATION HOOKS TESTS PASSED!")
        print(f"✅ Strategy decision tracking operational")
        print(f"✅ Parameter sensitivity monitoring working")
        print(f"✅ Problem characterization system functional")
        print(f"✅ Data collection workflow complete")
        print(f"✅ Week 3 Automation Hooks implementation complete!")
    else:
        print(f"\n⚠️  Some automation hooks tests failed - need to fix before proceeding")
    
    return failed == 0

if __name__ == '__main__':
    success = run_automation_hooks_tests()
    sys.exit(0 if success else 1)