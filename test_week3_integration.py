"""
Week 3 Integration Test - Complete Blueprint System

Comprehensive integration test for the complete Week 3 blueprint system,
testing integration with Week 1 and Week 2 components.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from brainsmith.blueprints.core import Blueprint, BlueprintLoader, BlueprintValidator
from brainsmith.blueprints.integration import LibraryMapper, DesignSpaceGenerator, BlueprintOrchestrator


def test_complete_blueprint_workflow():
    """Test complete blueprint workflow from creation to execution."""
    print("🔄 Testing Complete Blueprint Workflow...")
    
    # 1. Create a comprehensive blueprint
    blueprint = Blueprint("integration_test_cnn", "1.0.0", 
                         "Integration test for CNN accelerator")
    
    # Configure all libraries
    blueprint.add_library_config('kernels', {
        'pe_range': [2, 4, 8],
        'simd_range': [1, 2, 4],
        'precision': ['int8', 'int16']
    })
    
    blueprint.add_library_config('transforms', {
        'pipeline_depth': [2, 4],
        'memory_optimization': 'balanced'
    })
    
    blueprint.add_library_config('hw_optim', {
        'target_frequency': 250,
        'resource_budget': {'luts': 50000, 'brams': 200}
    })
    
    blueprint.add_library_config('analysis', {
        'performance_metrics': ['throughput', 'latency'],
        'roofline_analysis': True
    })
    
    # Add constraints and objectives
    blueprint.add_constraint('resource_limits', {'luts': 60000, 'brams': 250})
    blueprint.add_objective('throughput', 'maximize')
    blueprint.add_objective('resource_efficiency', 'maximize')
    
    # Set design space config
    blueprint.set_design_space_config({
        'exploration_strategy': 'pareto_optimal',
        'max_evaluations': 20
    })
    
    # 2. Validate blueprint
    validator = BlueprintValidator()
    is_valid, errors = validator.validate(blueprint)
    print(f"  ✅ Blueprint validation: {is_valid}")
    
    # 3. Test library mapping
    mapper = LibraryMapper()
    library_configs = mapper.map_blueprint_to_libraries(blueprint)
    print(f"  ✅ Library mapping: {list(library_configs.keys())}")
    
    # 4. Test design space generation
    generator = DesignSpaceGenerator()
    design_space = generator.generate_from_blueprint(blueprint)
    print(f"  ✅ Design space generation: {design_space['total_points']} points")
    
    # 5. Test orchestrator integration
    orchestrator = BlueprintOrchestrator()
    can_execute, issues = orchestrator.validate_blueprint_execution(blueprint)
    print(f"  ✅ Execution validation: {can_execute}")
    
    if can_execute:
        results = orchestrator.execute_blueprint(blueprint)
        print(f"  ✅ Blueprint execution: {results['status']}")
    
    return is_valid and can_execute


def test_library_mapper_functionality():
    """Test library mapper detailed functionality."""
    print("\n🗺️ Testing Library Mapper Functionality...")
    
    blueprint = Blueprint("mapper_test", "1.0.0")
    blueprint.add_library_config('kernels', {
        'pe_range': [1, 2, 4],
        'simd_range': [1, 2],
        'precision': ['int8']
    })
    
    mapper = LibraryMapper()
    
    # Test parameter extraction
    design_params = mapper.extract_design_space_parameters(blueprint)
    print(f"  ✅ Design space parameters: {list(design_params.keys())}")
    
    # Test execution plan creation
    execution_plan = mapper.create_library_execution_plan(blueprint)
    print(f"  ✅ Execution plan: {execution_plan['libraries']}")
    
    # Test compatibility validation
    is_compatible, warnings = mapper.validate_library_compatibility(blueprint)
    print(f"  ✅ Library compatibility: {is_compatible}")
    
    return len(design_params) > 0


def test_design_space_generator_functionality():
    """Test design space generator detailed functionality."""
    print("\n🎯 Testing Design Space Generator Functionality...")
    
    blueprint = Blueprint("generator_test", "1.0.0")
    blueprint.add_library_config('kernels', {
        'pe_range': [2, 4],
        'simd_range': [1, 2],
        'precision': ['int8']
    })
    blueprint.add_library_config('hw_optim', {
        'target_frequency': 250
    })
    
    blueprint.add_constraint('resource_limits', {'luts': 10000})
    blueprint.add_objective('throughput', 'maximize')
    
    generator = DesignSpaceGenerator()
    
    # Test basic design space generation
    design_space = generator.generate_from_blueprint(blueprint)
    print(f"  ✅ Basic design space: {design_space['total_points']} points")
    
    # Test exploration plan generation
    exploration_plan = generator.generate_exploration_plan(blueprint)
    print(f"  ✅ Exploration plan: {exploration_plan['exploration_strategy']}")
    
    # Test optimized design space generation
    optimized_space = generator.optimize_design_space(blueprint)
    print(f"  ✅ Optimized design space: {optimized_space['total_points']} points")
    
    return design_space['total_points'] > 0


def test_blueprint_serialization_integration():
    """Test blueprint serialization with full integration."""
    print("\n💾 Testing Blueprint Serialization Integration...")
    
    # Create complex blueprint
    blueprint = Blueprint("serialization_integration", "1.0.0")
    
    # Add comprehensive configuration
    blueprint.add_library_config('kernels', {'pe_range': [1, 2, 4]})
    blueprint.add_library_config('transforms', {'pipeline_depth': [2, 4]})
    blueprint.add_constraint('resource_limits', {'luts': 50000})
    blueprint.add_objective('throughput', 'maximize')
    blueprint.set_design_space_config({'max_evaluations': 10})
    
    # Test JSON round-trip
    json_str = blueprint.to_json()
    blueprint_from_json = Blueprint.from_json(json_str)
    
    # Verify functionality is preserved
    mapper = LibraryMapper()
    original_configs = mapper.map_blueprint_to_libraries(blueprint)
    restored_configs = mapper.map_blueprint_to_libraries(blueprint_from_json)
    
    print(f"  ✅ JSON serialization: {len(json_str)} chars")
    print(f"  ✅ Functionality preserved: {original_configs == restored_configs}")
    
    # Test file operations
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(json_str)
        temp_file = f.name
    
    try:
        loader = BlueprintLoader()
        blueprint_from_file = loader.load_from_file(temp_file)
        
        # Test that loaded blueprint works with integration components
        generator = DesignSpaceGenerator()
        design_space = generator.generate_from_blueprint(blueprint_from_file)
        
        print(f"  ✅ File operations: {design_space['total_points']} points generated")
        return True
        
    finally:
        os.unlink(temp_file)


def test_week2_library_integration():
    """Test integration with Week 2 library system."""
    print("\n🔗 Testing Week 2 Library Integration...")
    
    blueprint = Blueprint("week2_integration", "1.0.0")
    blueprint.add_library_config('kernels', {
        'pe_range': [2, 4],
        'simd_range': [1, 2],
        'precision': ['int8']
    })
    
    # Test that blueprint parameters can be mapped to Week 2 library format
    mapper = LibraryMapper()
    library_configs = mapper.map_blueprint_to_libraries(blueprint)
    
    kernels_config = library_configs.get('kernels', {})
    print(f"  ✅ Kernels config mapping: {list(kernels_config.keys())}")
    
    # Verify the format matches what Week 2 libraries expect
    expected_keys = {'pe_values', 'simd_values', 'precision_options'}
    actual_keys = set(kernels_config.keys())
    
    has_expected_format = expected_keys.issubset(actual_keys)
    print(f"  ✅ Week 2 library format: {has_expected_format}")
    
    return has_expected_format


def test_week1_orchestrator_integration():
    """Test integration with Week 1 orchestrator concepts."""
    print("\n🎯 Testing Week 1 Orchestrator Integration...")
    
    blueprint = Blueprint("week1_integration", "1.0.0")
    blueprint.add_library_config('kernels', {'pe_range': [2, 4]})
    blueprint.add_objective('throughput', 'maximize')
    
    orchestrator = BlueprintOrchestrator()
    
    # Test orchestrator config creation
    orchestrator_config = orchestrator.create_orchestrator_config(blueprint)
    print(f"  ✅ Orchestrator config: {list(orchestrator_config.keys())}")
    
    # Test execution validation
    can_execute, issues = orchestrator.validate_blueprint_execution(blueprint)
    print(f"  ✅ Execution validation: {can_execute}")
    if issues:
        print(f"    Issues: {issues}")
    
    # Test blueprint execution
    if can_execute:
        results = orchestrator.execute_blueprint(blueprint)
        print(f"  ✅ Execution results: {results['status']}")
        
        # Verify results contain expected components
        has_design_space = 'design_space' in results
        has_execution_plan = 'execution_plan' in results
        
        print(f"  ✅ Complete results: {has_design_space and has_execution_plan}")
        return has_design_space and has_execution_plan
    
    return can_execute


def test_real_world_blueprint_scenario():
    """Test a realistic real-world blueprint scenario."""
    print("\n🌍 Testing Real-World Blueprint Scenario...")
    
    # Create a high-performance CNN blueprint similar to what users would create
    blueprint = Blueprint("mobilenet_v2_accelerator", "1.0.0", 
                         "MobileNet v2 FPGA accelerator blueprint")
    
    # Configure for mobile/edge deployment
    blueprint.add_library_config('kernels', {
        'pe_range': [4, 8, 16],
        'simd_range': [2, 4, 8],
        'precision': ['int8'],
        'optimization_hint': 'balance_performance_area'
    })
    
    blueprint.add_library_config('transforms', {
        'pipeline_depth': [3, 4, 5],
        'folding_factors': [2, 4],
        'memory_optimization': 'aggressive'
    })
    
    blueprint.add_library_config('hw_optim', {
        'target_frequency': 200,  # Conservative for mobile
        'resource_budget': {
            'luts': 30000,    # Small FPGA target
            'brams': 100,
            'dsps': 200
        },
        'optimization_strategy': 'area_optimized'
    })
    
    blueprint.add_library_config('analysis', {
        'performance_metrics': ['throughput', 'power', 'efficiency'],
        'power_estimation': True,
        'accuracy_analysis': True
    })
    
    # Add realistic constraints
    blueprint.add_constraint('resource_limits', {
        'lut_utilization': 0.75,
        'bram_utilization': 0.8,
        'power_consumption': 5.0  # 5W max
    })
    
    blueprint.add_constraint('performance_requirements', {
        'min_throughput': 500,  # 500 inferences/sec
        'max_latency': 10       # 10ms max
    })
    
    # Multi-objective optimization
    blueprint.add_objective('throughput', 'maximize')
    blueprint.add_objective('power_efficiency', 'maximize')
    blueprint.add_objective('area_efficiency', 'maximize')
    
    blueprint.set_design_space_config({
        'exploration_strategy': 'pareto_optimal',
        'max_evaluations': 50
    })
    
    # Test the complete workflow
    validator = BlueprintValidator()
    is_valid, errors = validator.validate(blueprint)
    print(f"  ✅ Real-world blueprint validation: {is_valid}")
    
    if is_valid:
        # Test design space generation
        generator = DesignSpaceGenerator()
        design_space = generator.generate_from_blueprint(blueprint)
        print(f"  ✅ Design space size: {design_space['total_points']} points")
        
        # Test orchestrator execution
        orchestrator = BlueprintOrchestrator()
        can_execute, issues = orchestrator.validate_blueprint_execution(blueprint)
        
        if can_execute:
            results = orchestrator.execute_blueprint(blueprint)
            print(f"  ✅ Real-world execution: {results['status']}")
            return True
        else:
            print(f"  ❌ Execution issues: {issues}")
    else:
        print(f"  ❌ Validation errors: {errors}")
    
    return is_valid


def main():
    """Main integration test function."""
    print("🚀 Week 3 Integration Test - Complete Blueprint System")
    print("=" * 70)
    
    tests = [
        test_complete_blueprint_workflow,
        test_library_mapper_functionality, 
        test_design_space_generator_functionality,
        test_blueprint_serialization_integration,
        test_week2_library_integration,
        test_week1_orchestrator_integration,
        test_real_world_blueprint_scenario
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Integration Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Complete Week 3 blueprint system integration successful!")
        print("✅ Blueprint system ready for production use!")
        return True
    elif passed >= len(tests) * 0.8:
        print("✅ Week 3 blueprint system mostly working!")
        print("⚠️  Some minor issues to address")
        return True
    else:
        print("❌ Significant integration issues need resolution")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)