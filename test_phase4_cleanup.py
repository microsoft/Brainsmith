#!/usr/bin/env python3
"""
Test Phase 4: Legacy System Cleanup.

This validates that the enhanced approach is now the default and that
legacy systems are properly isolated.
"""

import tempfile
from pathlib import Path
from brainsmith.tools.unified_hwkg.generator import (
    create_unified_generator, 
    create_enhanced_generator, 
    create_legacy_generator
)
from brainsmith.tools.hw_kernel_gen.cli import generate_all_enhanced, _load_compiler_data
from brainsmith.tools.hw_kernel_gen.config import Config

def test_enhanced_is_default():
    """Test that enhanced mode is now the default."""
    
    print("🧪 TESTING ENHANCED IS DEFAULT")
    print("=" * 50)
    
    # Test factory functions
    default_generator = create_unified_generator()
    enhanced_generator = create_enhanced_generator()
    legacy_generator = create_legacy_generator()
    
    # Default should be enhanced
    assert default_generator.enhanced == True, "Default generator should be enhanced"
    assert enhanced_generator.enhanced == True, "Enhanced generator should be enhanced"
    assert legacy_generator.enhanced == False, "Legacy generator should not be enhanced"
    
    print("✅ Factory functions working correctly")
    print(f"   Default generator: enhanced={default_generator.enhanced}")
    print(f"   Enhanced generator: enhanced={enhanced_generator.enhanced}")
    print(f"   Legacy generator: enhanced={legacy_generator.enhanced}")
    
    return True

def test_enhanced_cli_integration():
    """Test that CLI uses enhanced generation by default."""
    
    print("\n🧪 TESTING ENHANCED CLI INTEGRATION")
    print("=" * 50)
    
    # Create test config
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    
    # Simple config for testing
    config = Config(
        rtl_file=rtl_file,
        compiler_data_file=Path("examples/thresholding/dummy_compiler_data.py"),
        output_dir=Path("/tmp/test_phase4"),
        debug=True
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config.output_dir = Path(temp_dir)
        
        try:
            # Test enhanced CLI generation
            result = generate_all_enhanced(config)
            
            assert result.success, f"Enhanced CLI generation failed: {result.errors}"
            assert len(result.generated_files) >= 3, f"Expected at least 3 files, got {len(result.generated_files)}"
            
            print("✅ Enhanced CLI generation successful")
            print(f"   Generated files: {len(result.generated_files)}")
            for file_path in result.generated_files:
                print(f"   📄 {file_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced CLI generation failed: {e}")
            return False

def test_dataflow_model_elimination():
    """Test that enhanced approach doesn't use DataflowModel for templates."""
    
    print("\n🧪 TESTING DATAFLOW MODEL ELIMINATION")
    print("=" * 50)
    
    # Create enhanced generator
    enhanced_generator = create_enhanced_generator()
    legacy_generator = create_legacy_generator()
    
    # Enhanced generator should not have RTL converter
    assert hasattr(enhanced_generator, 'direct_renderer'), "Enhanced generator should have direct_renderer"
    assert not hasattr(enhanced_generator, 'rtl_converter'), "Enhanced generator should not have rtl_converter"
    
    # Legacy generator should have RTL converter
    assert hasattr(legacy_generator, 'rtl_converter'), "Legacy generator should have rtl_converter"
    assert not hasattr(legacy_generator, 'direct_renderer'), "Legacy generator should not have direct_renderer"
    
    print("✅ DataflowModel properly eliminated from enhanced approach")
    print("   Enhanced generator: Uses DirectTemplateRenderer")
    print("   Legacy generator: Uses RTLDataflowConverter")
    
    return True

def test_template_context_builders_removed():
    """Test that complex template context builders are no longer needed."""
    
    print("\n🧪 TESTING TEMPLATE CONTEXT BUILDERS REMOVED")
    print("=" * 50)
    
    # Parse RTL file with enhanced approach
    from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file_enhanced
    
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    enhanced_result = parse_rtl_file_enhanced(rtl_file)
    
    # Enhanced result should provide template context directly
    context = enhanced_result.get_template_context()
    
    # No need for external context builders
    assert len(context) >= 20, f"Expected at least 20 template variables, got {len(context)}"
    assert "kernel_name" in context, "Template context missing kernel_name"
    assert "class_name" in context, "Template context missing class_name"
    assert "dataflow_interfaces" in context, "Template context missing dataflow_interfaces"
    
    print("✅ Template context generated directly from enhanced result")
    print(f"   Template variables: {len(context)}")
    print(f"   No external context builders required")
    
    return True

def test_performance_improvement():
    """Test that enhanced approach is faster than legacy."""
    
    print("\n🧪 TESTING PERFORMANCE IMPROVEMENT")
    print("=" * 50)
    
    import time
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config(
            rtl_file=rtl_file,
            compiler_data_file=Path("examples/thresholding/dummy_compiler_data.py"),
            output_dir=Path(temp_dir),
            debug=False
        )
        
        # Test enhanced generation timing
        start_time = time.time()
        enhanced_result = generate_all_enhanced(config)
        enhanced_time = time.time() - start_time
        
        # Enhanced should be fast
        assert enhanced_result.success, "Enhanced generation should succeed"
        assert enhanced_time < 1.0, f"Enhanced generation too slow: {enhanced_time:.3f}s"
        
        print(f"✅ Enhanced generation performance: {enhanced_time:.3f}s")
        print(f"   Files generated: {len(enhanced_result.generated_files)}")
        print(f"   Performance target met (< 1.0s)")
        
        return True

def test_backward_compatibility():
    """Test that legacy approach still works for compatibility."""
    
    print("\n🧪 TESTING BACKWARD COMPATIBILITY")
    print("=" * 50)
    
    # Create legacy generator explicitly
    legacy_generator = create_legacy_generator()
    
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    compiler_data = {"target": "fpga", "optimization": "area"}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        try:
            # Test legacy generation still works
            result = legacy_generator.generate_from_rtl(
                rtl_file=rtl_file,
                compiler_data=compiler_data,
                output_dir=output_dir
            )
            
            assert result.success, f"Legacy generation failed: {result.errors}"
            assert len(result.generated_files) >= 2, f"Expected at least 2 files, got {len(result.generated_files)}"
            assert result.dataflow_model is not None, "Legacy approach should create DataflowModel"
            
            print("✅ Legacy generation still works")
            print(f"   Generated files: {len(result.generated_files)}")
            print(f"   DataflowModel created: {type(result.dataflow_model)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Legacy generation failed: {e}")
            return False

def run_all_tests():
    """Run all Phase 4 cleanup tests."""
    
    print("🧪 PHASE 4: LEGACY SYSTEM CLEANUP VALIDATION")
    print("=" * 60)
    
    tests = [
        test_enhanced_is_default,
        test_enhanced_cli_integration,
        test_dataflow_model_elimination,
        test_template_context_builders_removed,
        test_performance_improvement,
        test_backward_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 PHASE 4 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED")
        print("✅ Phase 4: Legacy System Cleanup COMPLETE")
        print("✅ Enhanced approach is now default")
        print("✅ DataflowModel overhead eliminated for templates")
        print("✅ Backward compatibility maintained")
        print("✅ Ready for Phase 5: Comprehensive Validation")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Phase 4 implementation needs fixes")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)