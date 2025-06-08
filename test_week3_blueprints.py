"""
Week 3 Implementation Test - Blueprint System

Test the blueprint system implementation to verify it successfully
provides blueprint-driven design space exploration.
"""

import sys
import os
import json
sys.path.insert(0, os.path.abspath('.'))

from brainsmith.blueprints.core import Blueprint, BlueprintLoader, BlueprintValidator


def test_blueprint_creation():
    """Test blueprint creation and basic functionality."""
    print("🧪 Testing Blueprint Creation...")
    
    # Create a blueprint programmatically
    blueprint = Blueprint("test_cnn", "1.0.0", "Test CNN blueprint")
    
    # Add library configurations
    blueprint.add_library_config('kernels', {
        'pe_range': [1, 2, 4, 8],
        'simd_range': [1, 2, 4],
        'precision': ['int8', 'int16']
    })
    
    blueprint.add_library_config('transforms', {
        'pipeline_depth': [2, 4, 6],
        'memory_optimization': 'balanced'
    })
    
    # Add constraints
    blueprint.add_constraint('resource_limits', {
        'luts': 50000,
        'brams': 200
    })
    
    # Add objectives
    blueprint.add_objective('throughput', 'maximize')
    blueprint.add_objective('resource_efficiency', 'maximize')
    
    print(f"  ✅ Created blueprint: {blueprint.name}")
    print(f"  ✅ Libraries configured: {list(blueprint.libraries.keys())}")
    print(f"  ✅ Constraints: {len(blueprint.constraints)}")
    print(f"  ✅ Objectives: {len(blueprint.objectives)}")
    
    return True


def test_blueprint_serialization():
    """Test blueprint serialization and deserialization."""
    print("\n📄 Testing Blueprint Serialization...")
    
    # Create blueprint
    blueprint = Blueprint("serialization_test", "1.0.0")
    blueprint.add_library_config('kernels', {'pe_range': [1, 2, 4]})
    blueprint.add_objective('throughput')
    
    # Test JSON serialization
    json_str = blueprint.to_json()
    print(f"  ✅ JSON serialization: {len(json_str)} characters")
    
    # Test JSON deserialization
    blueprint_from_json = Blueprint.from_json(json_str)
    print(f"  ✅ JSON deserialization: {blueprint_from_json.name}")
    
    # Verify content matches
    assert blueprint_from_json.name == blueprint.name
    assert blueprint_from_json.libraries == blueprint.libraries
    
    # Test YAML serialization
    yaml_str = blueprint.to_yaml()
    print(f"  ✅ YAML serialization: {len(yaml_str)} characters")
    
    # Test YAML deserialization
    blueprint_from_yaml = Blueprint.from_yaml(yaml_str)
    print(f"  ✅ YAML deserialization: {blueprint_from_yaml.name}")
    
    return True


def test_blueprint_validation():
    """Test blueprint validation system."""
    print("\n✅ Testing Blueprint Validation...")
    
    validator = BlueprintValidator()
    
    # Test valid blueprint
    valid_blueprint = Blueprint("valid_test", "1.0.0")
    valid_blueprint.add_library_config('kernels', {
        'pe_range': [1, 2, 4],
        'precision': ['int8']
    })
    valid_blueprint.add_objective('throughput')
    
    is_valid, errors = validator.validate(valid_blueprint)
    print(f"  ✅ Valid blueprint validation: {is_valid}")
    if errors:
        print(f"    Unexpected errors: {errors}")
    
    # Test invalid blueprint
    invalid_blueprint = Blueprint("", "")  # Missing name and version
    invalid_blueprint.add_library_config('invalid_library', {})  # Invalid library
    
    is_valid, errors = validator.validate(invalid_blueprint)
    print(f"  ✅ Invalid blueprint validation: {not is_valid}")
    print(f"  ✅ Found {len(errors)} validation errors")
    
    return True


def test_blueprint_loading():
    """Test blueprint loading from various sources."""
    print("\n📁 Testing Blueprint Loading...")
    
    loader = BlueprintLoader()
    
    # Test loading from dictionary
    blueprint_data = {
        'name': 'loader_test',
        'version': '1.0.0',
        'description': 'Test blueprint for loader',
        'libraries': {
            'kernels': {'pe_range': [1, 2, 4]}
        },
        'objectives': ['throughput']
    }
    
    blueprint = loader.load_from_dict(blueprint_data)
    print(f"  ✅ Loaded from dict: {blueprint.name}")
    
    # Test loading from JSON string
    json_str = json.dumps(blueprint_data)
    blueprint_from_json = loader.load_from_json(json_str)
    print(f"  ✅ Loaded from JSON string: {blueprint_from_json.name}")
    
    # Test saving and loading from file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(json_str)
        temp_file = f.name
    
    try:
        blueprint_from_file = loader.load_from_file(temp_file)
        print(f"  ✅ Loaded from file: {blueprint_from_file.name}")
    finally:
        os.unlink(temp_file)
    
    return True


def test_blueprint_library_integration():
    """Test blueprint integration with Week 2 library system."""
    print("\n🔗 Testing Blueprint-Library Integration...")
    
    # Create a comprehensive blueprint
    blueprint = Blueprint("library_integration_test", "1.0.0")
    
    # Configure all four libraries
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
    
    # Test library configuration extraction
    kernels_config = blueprint.get_kernels_config()
    transforms_config = blueprint.get_transforms_config()
    hw_optim_config = blueprint.get_hw_optim_config()
    analysis_config = blueprint.get_analysis_config()
    
    print(f"  ✅ Kernels config: {list(kernels_config.keys())}")
    print(f"  ✅ Transforms config: {list(transforms_config.keys())}")
    print(f"  ✅ HW Optim config: {list(hw_optim_config.keys())}")
    print(f"  ✅ Analysis config: {list(analysis_config.keys())}")
    
    # Test constraint extraction
    resource_constraints = blueprint.get_resource_constraints()
    print(f"  ✅ Resource constraints: {list(resource_constraints.keys())}")
    
    return True


def test_example_blueprint_creation():
    """Test creation of a realistic example blueprint."""
    print("\n🎯 Testing Example Blueprint Creation...")
    
    # Create a high-performance CNN blueprint
    blueprint = Blueprint("high_performance_cnn", "1.0.0", 
                         "High-performance CNN accelerator blueprint")
    
    # Configure for high performance
    blueprint.add_library_config('kernels', {
        'pe_range': [8, 16, 32],
        'simd_range': [4, 8, 16],
        'precision': ['int8'],
        'optimization_hint': 'maximize_throughput'
    })
    
    blueprint.add_library_config('transforms', {
        'pipeline_depth': [4, 6, 8],
        'folding_factors': [2, 4, 8],
        'memory_optimization': 'aggressive'
    })
    
    blueprint.add_library_config('hw_optim', {
        'target_frequency': 300,
        'resource_budget': {
            'luts': 100000,
            'brams': 500,
            'dsps': 1000
        },
        'optimization_strategy': 'performance'
    })
    
    blueprint.add_library_config('analysis', {
        'performance_metrics': ['throughput', 'latency', 'efficiency'],
        'roofline_analysis': True,
        'power_estimation': True
    })
    
    # Add comprehensive constraints
    blueprint.add_constraint('resource_limits', {
        'lut_utilization': 0.8,
        'bram_utilization': 0.7,
        'dsp_utilization': 0.9
    })
    
    blueprint.add_constraint('performance_requirements', {
        'min_throughput': 2000,
        'max_latency': 5
    })
    
    # Add optimization objectives
    blueprint.add_objective('throughput', 'maximize')
    blueprint.add_objective('resource_efficiency', 'maximize')
    
    # Set design space configuration
    blueprint.set_design_space_config({
        'exploration_strategy': 'pareto_optimal',
        'max_evaluations': 200,
        'objectives': ['performance', 'resource_efficiency']
    })
    
    # Validate the blueprint
    validator = BlueprintValidator()
    is_valid, errors = validator.validate(blueprint)
    
    print(f"  ✅ Example blueprint created: {blueprint.name}")
    print(f"  ✅ Blueprint validation: {is_valid}")
    if errors:
        print(f"    Validation errors: {errors}")
    
    # Get blueprint summary
    summary = blueprint.get_summary()
    print(f"  ✅ Blueprint summary: {summary['libraries_configured']} libraries")
    
    return is_valid


def main():
    """Main test function."""
    print("🚀 Week 3 Implementation Test - Blueprint System")
    print("=" * 60)
    
    tests = [
        test_blueprint_creation,
        test_blueprint_serialization,
        test_blueprint_validation,
        test_blueprint_loading,
        test_blueprint_library_integration,
        test_example_blueprint_creation
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All blueprint system tests passed!")
        print("✅ Blueprint system implementation successful!")
        return True
    else:
        print("⚠️  Some tests failed - needs investigation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)