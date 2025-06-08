"""
Week 4 Test - Transforms Library

Test the transforms library implementation to verify it successfully
organizes existing steps/ functionality and integrates with blueprints.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from brainsmith.libraries.transforms import TransformsLibrary, TransformRegistry, TransformPipeline
from brainsmith.blueprints.core import Blueprint


def test_transforms_library_initialization():
    """Test transforms library initialization."""
    print("🧪 Testing Transforms Library Initialization...")
    
    # Create transforms library
    transforms_lib = TransformsLibrary()
    
    # Test basic properties
    assert transforms_lib.name == "transforms"
    assert transforms_lib.version == "1.0.0"
    assert not transforms_lib.initialized
    
    # Initialize library
    config = {
        'search_paths': ['./steps/', './brainsmith/libraries/transforms/steps/']
    }
    success = transforms_lib.initialize(config)
    
    print(f"  ✅ Library initialized: {success}")
    print(f"  ✅ Capabilities: {transforms_lib.get_capabilities()}")
    
    return success


def test_transform_registry():
    """Test transform registry functionality."""
    print("\n📚 Testing Transform Registry...")
    
    registry = TransformRegistry()
    
    # Test transform registration
    test_transform = {
        'name': 'Test Transform',
        'description': 'A test transform',
        'category': 'testing',
        'parameters': ['param1', 'param2']
    }
    
    registry.register_transform('test_transform', test_transform)
    
    # Test retrieval
    retrieved = registry.get_transform('test_transform')
    print(f"  ✅ Transform registered and retrieved: {retrieved['name']}")
    
    # Test listing
    available = registry.get_available_transforms()
    print(f"  ✅ Available transforms: {available}")
    
    return 'test_transform' in available


def test_transform_pipeline():
    """Test transform pipeline functionality."""
    print("\n🔄 Testing Transform Pipeline...")
    
    # Create pipeline
    pipeline = TransformPipeline("test_pipeline")
    
    # Configure pipeline from blueprint-style config
    config = {
        'pipeline_depth': 4,
        'folding_factors': [2, 4],
        'memory_optimization': 'balanced',
        'streaming_enabled': True
    }
    
    pipeline.configure(config)
    
    # Test pipeline properties
    sequence = pipeline.get_transform_sequence()
    estimated_depth = pipeline.get_estimated_depth()
    resource_estimate = pipeline.get_resource_estimate()
    
    print(f"  ✅ Pipeline configured: {len(sequence)} steps")
    print(f"  ✅ Transform sequence: {sequence}")
    print(f"  ✅ Estimated depth: {estimated_depth}")
    print(f"  ✅ Resource estimates: {list(resource_estimate.keys())}")
    
    # Test pipeline validation
    is_valid, errors = pipeline.validate()
    print(f"  ✅ Pipeline validation: {is_valid}")
    if errors:
        print(f"    Errors: {errors}")
    
    return len(sequence) > 0


def test_blueprint_integration():
    """Test transforms library integration with blueprints."""
    print("\n🔗 Testing Blueprint Integration...")
    
    # Create blueprint with transforms configuration
    blueprint = Blueprint("transforms_test", "1.0.0")
    blueprint.add_library_config('transforms', {
        'pipeline_depth': [2, 4, 6],
        'folding_factors': [2, 4, 8],
        'memory_optimization': 'aggressive'
    })
    
    # Test transforms library with blueprint
    transforms_lib = TransformsLibrary()
    transforms_lib.initialize()
    
    # Test design space generation
    design_space = transforms_lib.get_design_space_parameters()
    print(f"  ✅ Design space parameters: {list(design_space['transforms'].keys())}")
    
    # Test blueprint configuration extraction
    transforms_config = blueprint.get_transforms_config()
    print(f"  ✅ Blueprint transforms config: {list(transforms_config.keys())}")
    
    # Test parameter validation
    is_valid, errors = transforms_lib.validate_parameters({'transforms': transforms_config})
    print(f"  ✅ Parameter validation: {is_valid}")
    
    return len(design_space['transforms']) > 0


def test_transform_operations():
    """Test transform library operations."""
    print("\n⚙️ Testing Transform Operations...")
    
    transforms_lib = TransformsLibrary()
    transforms_lib.initialize()
    
    # Test getting design space
    result = transforms_lib.execute("get_design_space", {})
    print(f"  ✅ Design space: {result['total_transforms']} transforms")
    
    # Test applying transforms
    transform_config = {
        'transforms': {
            'pipeline_depth': 4,
            'folding_factors': [2, 4],
            'memory_optimization': 'balanced'
        }
    }
    
    result = transforms_lib.execute("apply_transforms", {
        'model': 'test_model',
        'transforms': transform_config['transforms']
    })
    
    print(f"  ✅ Transforms applied: {len(result['applied_transforms'])}")
    print(f"  ✅ Estimated improvement: {result['estimated_improvement']['throughput_gain']}")
    
    # Test pipeline creation
    result = transforms_lib.execute("create_pipeline", {
        'pipeline_config': transform_config['transforms']
    })
    
    print(f"  ✅ Pipeline created: {result['pipeline_id']}")
    print(f"  ✅ Transform sequence: {result['transform_sequence']}")
    
    # Test resource estimation
    result = transforms_lib.execute("estimate_resources", transform_config)
    print(f"  ✅ Resource estimation: {result['resource_overhead']['luts']} LUTs")
    
    return True


def test_end_to_end_transforms_workflow():
    """Test complete transforms workflow."""
    print("\n🌟 Testing End-to-End Transforms Workflow...")
    
    # 1. Create blueprint with comprehensive transforms config
    blueprint = Blueprint("end_to_end_transforms", "1.0.0")
    blueprint.add_library_config('transforms', {
        'pipeline_depth': 5,
        'folding_factors': [2, 4, 8],
        'memory_optimization': 'aggressive',
        'streaming_enabled': True
    })
    
    # 2. Initialize transforms library
    transforms_lib = TransformsLibrary()
    success = transforms_lib.initialize()
    if not success:
        return False
    
    # 3. Extract configuration from blueprint
    transforms_config = blueprint.get_transforms_config()
    
    # 4. Create and configure pipeline
    pipeline = TransformPipeline()
    pipeline.configure(transforms_config)
    
    # 5. Validate pipeline
    is_valid, errors = pipeline.validate()
    print(f"  ✅ Pipeline validation: {is_valid}")
    
    # 6. Execute transforms
    result = transforms_lib.execute("apply_transforms", {
        'model': 'test_model',
        'transforms': transforms_config
    })
    
    # 7. Get resource estimates
    resource_result = transforms_lib.execute("estimate_resources", {
        'transforms': transforms_config
    })
    
    print(f"  ✅ Workflow completed successfully")
    print(f"  ✅ Applied transforms: {len(result['applied_transforms'])}")
    print(f"  ✅ Resource overhead: {resource_result['resource_overhead']}")
    print(f"  ✅ Performance impact: {resource_result['performance_impact']}")
    
    return is_valid and len(result['applied_transforms']) > 0


def main():
    """Main test function."""
    print("🚀 Week 4 Test - Transforms Library")
    print("=" * 60)
    
    tests = [
        test_transforms_library_initialization,
        test_transform_registry,
        test_transform_pipeline,
        test_blueprint_integration,
        test_transform_operations,
        test_end_to_end_transforms_workflow
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
        print("🎉 Transforms library implementation successful!")
        print("✅ Week 4 Day 1: Transforms Library completed!")
        return True
    else:
        print("⚠️  Some tests failed - needs investigation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)