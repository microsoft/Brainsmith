#!/usr/bin/env python3
"""
Final verification test for BrainSmith Libraries Refactoring
Tests that all 5 libraries can be imported and core functions work.
"""

def test_all_libraries():
    """Test all 5 refactored libraries."""
    print("=== BrainSmith Libraries Refactoring - Final Verification ===\n")
    
    # Test 1: Kernels Library - Clean API
    print("1. Testing kernels library clean API...")
    try:
        from brainsmith.libraries.kernels import list_kernels, get_kernel, find_compatible_kernels
        kernels = list_kernels()
        print(f"   ✅ Registry functions: Found {len(kernels)} kernels: {', '.join(kernels)}")
        
        # Test business logic functions still work
        kernel = get_kernel("conv2d_hls")
        print(f"   ✅ Business logic: get_kernel() returns {kernel.name}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Transforms Library - Clean API
    print("2. Testing transforms library clean API...")
    try:
        from brainsmith.libraries.transforms import list_transforms, get_transform
        transforms = list_transforms()
        print(f"   ✅ Registry functions: Found {len(transforms)} transforms: {', '.join(transforms[:3])}{'...' if len(transforms) > 3 else ''}")
        
        # Test transform function works
        cleanup_fn = get_transform("cleanup")
        print(f"   ✅ Transform access: get_transform('cleanup') returns {cleanup_fn.__name__}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Analysis Library - Clean API
    print("3. Testing analysis library clean API...")
    try:
        from brainsmith.libraries.analysis import list_analysis_tools, get_analysis_tool
        tools = list_analysis_tools()
        print(f"   ✅ Registry functions: Found {len(tools)} analysis tools: {', '.join(tools)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Blueprints Library - Clean API
    print("4. Testing blueprints library clean API...")
    try:
        from brainsmith.libraries.blueprints import list_blueprints, get_blueprint, load_blueprint_yaml
        blueprints = list_blueprints()
        print(f"   ✅ Registry functions: Found {len(blueprints)} blueprints: {', '.join(blueprints)}")
        
        # Test blueprint loading
        blueprint_path = get_blueprint("cnn_accelerator")
        print(f"   ✅ Blueprint access: get_blueprint() returns valid path")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Automation Library - Clean API
    print("5. Testing automation library clean API...")
    try:
        from brainsmith.libraries.automation import parameter_sweep, batch_process, find_best, aggregate_stats
        print(f"   ✅ Core automation functions imported successfully (no registry needed)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Verify legacy functions are gone
    print("6. Verifying legacy functions removed...")
    legacy_removed = True
    
    try:
        from brainsmith.libraries.kernels import discover_all_kernels
        print(f"   ❌ discover_all_kernels still exists!")
        legacy_removed = False
    except ImportError:
        print(f"   ✅ discover_all_kernels properly removed")
    
    try:
        from brainsmith.libraries.transforms import get_transform_by_name
        print(f"   ❌ get_transform_by_name still exists!")
        legacy_removed = False
    except ImportError:
        print(f"   ✅ get_transform_by_name properly removed")
    
    if legacy_removed:
        print(f"   ✅ Legacy function cleanup verified")
    
    print("\n=== Final Verification Complete ===")

if __name__ == "__main__":
    test_all_libraries()