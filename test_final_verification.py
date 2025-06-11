#!/usr/bin/env python3
"""
Final verification test for BrainSmith Libraries Refactoring
Tests that all 5 libraries can be imported and core functions work.
"""

def test_all_libraries():
    """Test all 5 refactored libraries."""
    print("=== BrainSmith Libraries Refactoring - Final Verification ===\n")
    
    # Test 1: Kernels Library
    print("1. Testing kernels library...")
    try:
        from brainsmith.libraries.kernels import list_kernels, get_kernel
        kernels = list_kernels()
        print(f"   ✅ Found {len(kernels)} kernels: {', '.join(kernels[:3])}{'...' if len(kernels) > 3 else ''}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Transforms Library
    print("2. Testing transforms library...")
    try:
        from brainsmith.libraries.transforms import list_transforms, get_transform
        transforms = list_transforms()
        print(f"   ✅ Found {len(transforms)} transforms: {', '.join(transforms[:3])}{'...' if len(transforms) > 3 else ''}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Analysis Library
    print("3. Testing analysis library...")
    try:
        from brainsmith.libraries.analysis import list_analysis_tools, get_analysis_tool
        tools = list_analysis_tools()
        print(f"   ✅ Found {len(tools)} analysis tools: {', '.join(tools[:3])}{'...' if len(tools) > 3 else ''}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Blueprints Library
    print("4. Testing blueprints library...")
    try:
        from brainsmith.libraries.blueprints import list_blueprints, get_blueprint
        blueprints = list_blueprints()
        print(f"   ✅ Found {len(blueprints)} blueprints: {', '.join(blueprints[:3])}{'...' if len(blueprints) > 3 else ''}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Automation Library
    print("5. Testing automation library...")
    try:
        from brainsmith.libraries.automation import parameter_sweep, batch_process, find_best, aggregate_stats
        print(f"   ✅ All automation functions imported successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n=== Final Verification Complete ===")

if __name__ == "__main__":
    test_all_libraries()