#!/usr/bin/env python3
"""
Simple test script to verify the step library and YAML blueprint system works.
"""

import sys
import os

# Add brainsmith to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_step_registry():
    """Test the step registry functionality."""
    print("Testing Step Registry...")
    
    from brainsmith.steps import list_steps, get_step, validate_steps
    
    # List available steps
    print("\n=== Available Steps ===")
    common_steps = list_steps("common")
    transformer_steps = list_steps("transformer")
    
    print(f"Common steps: {common_steps}")
    print(f"Transformer steps: {transformer_steps}")
    
    # Test getting a specific step
    print("\n=== Testing Step Retrieval ===")
    try:
        cleanup_step = get_step("common.cleanup")
        print(f"✓ Successfully retrieved 'common.cleanup': {cleanup_step}")
    except Exception as e:
        print(f"✗ Failed to retrieve 'common.cleanup': {e}")
    
    # Test FINN step fallback
    print("\n=== Testing FINN Step Fallback ===")
    try:
        finn_step = get_step("step_create_dataflow_partition")
        print(f"✓ Successfully retrieved FINN step: {finn_step}")
    except Exception as e:
        print(f"✗ Failed to retrieve FINN step: {e}")
    
    # Test step validation
    print("\n=== Testing Step Validation ===")
    test_steps = ["common.cleanup", "transformer.remove_head", "step_create_dataflow_partition"]
    errors = validate_steps(test_steps)
    if errors:
        print(f"✗ Validation errors: {errors}")
    else:
        print(f"✓ Step sequence validated successfully")

def test_blueprint_manager():
    """Test the blueprint manager functionality."""
    print("\n\nTesting Blueprint Manager...")
    
    from brainsmith.blueprints.manager import list_blueprints, load_blueprint
    
    # List available blueprints
    print("\n=== Available Blueprints ===")
    blueprints = list_blueprints()
    print(f"Available blueprints: {blueprints}")
    
    # Test loading BERT blueprint
    print("\n=== Testing Blueprint Loading ===")
    try:
        bert_blueprint = load_blueprint("bert")
        print(f"✓ Successfully loaded BERT blueprint:")
        print(f"  Name: {bert_blueprint.name}")
        print(f"  Description: {bert_blueprint.description}")
        print(f"  Architecture: {bert_blueprint.architecture}")
        print(f"  Steps: {len(bert_blueprint.build_steps)}")
        print(f"  First 5 steps: {bert_blueprint.build_steps[:5]}")
    except Exception as e:
        print(f"✗ Failed to load BERT blueprint: {e}")

def test_backward_compatibility():
    """Test backward compatibility with the original BERT blueprint."""
    print("\n\nTesting Backward Compatibility...")
    
    try:
        from brainsmith.blueprints.bert import BUILD_STEPS
        print(f"✓ Successfully imported BUILD_STEPS from legacy BERT blueprint")
        print(f"  BUILD_STEPS type: {type(BUILD_STEPS)}")
        print(f"  Number of steps: {len(BUILD_STEPS)}")
        print(f"  First step: {BUILD_STEPS[0]}")
    except Exception as e:
        print(f"✗ Failed to import legacy BUILD_STEPS: {e}")

if __name__ == "__main__":
    print("Brainsmith Step Library Test")
    print("=" * 50)
    
    try:
        test_step_registry()
        test_blueprint_manager()
        test_backward_compatibility()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)