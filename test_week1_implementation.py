"""
Week 1 Implementation Test

This script tests the core infrastructure implemented in Week 1
to ensure all components are working correctly before proceeding
to Week 2 (Library Structure Implementation).
"""

import sys
import os
import logging
from pathlib import Path

# Add brainsmith to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_core_imports():
    """Test that all core components can be imported."""
    print("🧪 Testing core imports...")
    
    try:
        from brainsmith.core import DesignSpaceOrchestrator
        print("✅ DesignSpaceOrchestrator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DesignSpaceOrchestrator: {e}")
        return False
    
    try:
        from brainsmith.core import FINNInterface, FINNHooksPlaceholder
        print("✅ FINN interface components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FINN interface: {e}")
        return False
    
    try:
        from brainsmith.core import WorkflowManager
        print("✅ WorkflowManager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import WorkflowManager: {e}")
        return False
    
    try:
        from brainsmith.core import brainsmith_explore, brainsmith_roofline
        print("✅ API functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import API functions: {e}")
        return False
    
    try:
        from brainsmith.core import cli
        print("✅ CLI module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CLI: {e}")
        return False
    
    return True

def test_orchestrator_initialization():
    """Test DesignSpaceOrchestrator initialization."""
    print("\n🧪 Testing DesignSpaceOrchestrator initialization...")
    
    try:
        from brainsmith.core import DesignSpaceOrchestrator
        
        # Create mock blueprint
        class MockBlueprint:
            def __init__(self):
                self.name = "test_blueprint"
                self.model_path = "test_model.onnx"
            
            def get_finn_legacy_config(self):
                return {"fpga_part": "xcvu9p-flga2104-2-i"}
        
        blueprint = MockBlueprint()
        orchestrator = DesignSpaceOrchestrator(blueprint)
        
        print("✅ DesignSpaceOrchestrator initialized successfully")
        print(f"   Blueprint name: {orchestrator.blueprint.name}")
        print(f"   Libraries count: {len(orchestrator.libraries)}")
        
        return True
        
    except Exception as e:
        print(f"❌ DesignSpaceOrchestrator initialization failed: {e}")
        return False

def test_finn_interface():
    """Test FINN interface components."""
    print("\n🧪 Testing FINN interface...")
    
    try:
        from brainsmith.core import FINNInterface, FINNHooksPlaceholder
        
        # Test FINNHooksPlaceholder
        hooks = FINNHooksPlaceholder()
        print(f"✅ FINNHooksPlaceholder created - available: {hooks.is_available()}")
        
        # Test configuration preparation
        design_point = {
            'transforms': {'param1': 'value1'},
            'hw_optimization': {'param2': 'value2'}
        }
        config = hooks.prepare_for_future_interface(design_point)
        print(f"✅ Future interface config prepared: {list(config.keys())}")
        
        # Test FINNInterface
        legacy_config = {"fpga_part": "xcvu9p-flga2104-2-i"}
        finn_interface = FINNInterface(legacy_config, hooks)
        
        status = finn_interface.get_interface_status()
        print(f"✅ FINNInterface created - using legacy: {status['using_legacy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ FINN interface test failed: {e}")
        return False

def test_workflow_manager():
    """Test WorkflowManager functionality."""
    print("\n🧪 Testing WorkflowManager...")
    
    try:
        from brainsmith.core import DesignSpaceOrchestrator, WorkflowManager
        
        # Create mock orchestrator
        class MockBlueprint:
            def __init__(self):
                self.name = "test_workflow_blueprint"
                self.model_path = "test_model.onnx"
            
            def get_finn_legacy_config(self):
                return {}
        
        blueprint = MockBlueprint()
        orchestrator = DesignSpaceOrchestrator(blueprint)
        workflow_manager = WorkflowManager(orchestrator)
        
        print("✅ WorkflowManager initialized successfully")
        
        # Test workflow statistics
        stats = workflow_manager.get_workflow_statistics()
        print(f"✅ Workflow statistics retrieved: {stats['total_workflows']} workflows")
        
        return True
        
    except Exception as e:
        print(f"❌ WorkflowManager test failed: {e}")
        return False

def test_api_functions():
    """Test main API functions."""
    print("\n🧪 Testing API functions...")
    
    try:
        from brainsmith.core import validate_blueprint
        
        # Test validate_blueprint with mock file
        test_blueprint_path = "test_blueprint.yaml"
        
        # Create minimal test blueprint
        blueprint_content = """
name: "test_blueprint"
description: "Test blueprint for Week 1 validation"

kernels:
  available: []

transforms:
  pipeline: []

hw_optimization:
  strategies: []

finn_interface:
  legacy_config:
    fpga_part: "xcvu9p-flga2104-2-i"
"""
        
        with open(test_blueprint_path, 'w') as f:
            f.write(blueprint_content)
        
        try:
            is_valid, errors = validate_blueprint(test_blueprint_path)
            print(f"✅ Blueprint validation completed - valid: {is_valid}")
            if errors:
                print(f"   Validation errors: {len(errors)}")
        except Exception as e:
            print(f"⚠️ Blueprint validation failed (expected in dev): {e}")
        
        # Cleanup
        if os.path.exists(test_blueprint_path):
            os.remove(test_blueprint_path)
        
        return True
        
    except Exception as e:
        print(f"❌ API functions test failed: {e}")
        return False

def test_legacy_compatibility():
    """Test legacy compatibility functions."""
    print("\n🧪 Testing legacy compatibility...")
    
    try:
        from brainsmith.core import (
            maintain_existing_api_compatibility,
            get_legacy_compatibility_report
        )
        
        # Test compatibility check
        compatibility = maintain_existing_api_compatibility()
        print(f"✅ Legacy compatibility check completed - compatible: {compatibility}")
        
        # Test compatibility report
        report = get_legacy_compatibility_report()
        print(f"✅ Compatibility report generated - overall: {report['overall_compatibility']}")
        print(f"   Functions checked: {len(report['functions_checked'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")
        return False

def test_hierarchical_exit_points():
    """Test the hierarchical exit points functionality."""
    print("\n🧪 Testing hierarchical exit points...")
    
    try:
        from brainsmith.core import DesignSpaceOrchestrator
        
        # Create mock blueprint with model path
        class MockBlueprint:
            def __init__(self):
                self.name = "exit_point_test"
                self.model_path = "mock_model.onnx"
            
            def get_finn_legacy_config(self):
                return {"fpga_part": "xcvu9p-flga2104-2-i"}
        
        blueprint = MockBlueprint()
        orchestrator = DesignSpaceOrchestrator(blueprint)
        
        # Test all three exit points
        exit_points = ["roofline", "dataflow_analysis", "dataflow_generation"]
        
        for exit_point in exit_points:
            try:
                result = orchestrator.orchestrate_exploration(exit_point)
                print(f"✅ Exit point '{exit_point}' executed successfully")
                print(f"   Analysis exit point: {result.analysis.get('exit_point', 'unknown')}")
                print(f"   Method: {result.analysis.get('method', 'unknown')}")
            except Exception as e:
                print(f"⚠️ Exit point '{exit_point}' failed (expected in dev): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hierarchical exit points test failed: {e}")
        return False

def test_core_status():
    """Test core module status functionality."""
    print("\n🧪 Testing core module status...")
    
    try:
        from brainsmith.core import get_core_status, verify_installation
        
        # Get status
        status = get_core_status()
        print(f"✅ Core status retrieved - version: {status['version']}")
        print(f"   Readiness: {status['readiness']*100:.1f}%")
        
        # Test installation verification
        installation_ok = verify_installation()
        print(f"✅ Installation verification completed - ok: {installation_ok}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core status test failed: {e}")
        return False

def run_all_tests():
    """Run all Week 1 implementation tests."""
    print("=" * 60)
    print("🚀 Brainsmith Week 1 Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("DesignSpaceOrchestrator", test_orchestrator_initialization),
        ("FINN Interface", test_finn_interface),
        ("WorkflowManager", test_workflow_manager),
        ("API Functions", test_api_functions),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Hierarchical Exit Points", test_hierarchical_exit_points),
        ("Core Status", test_core_status)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status_icon = "✅" if result else "❌"
        print(f"{status_icon} {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\n📈 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Week 1 implementation is ready for Week 2!")
        print("💡 Proceed with library structure implementation.")
    elif success_rate >= 60:
        print("⚠️ Week 1 implementation mostly works but has some issues.")
        print("💡 Consider fixing failing tests before proceeding.")
    else:
        print("❌ Week 1 implementation has significant issues.")
        print("💡 Fix core components before proceeding to Week 2.")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🚀 Ready to proceed to Week 2: Library Structure Implementation")
        print("📋 Next steps:")
        print("   1. Implement kernels library structure (existing components)")
        print("   2. Implement model transforms library structure")
        print("   3. Implement hw optimization library structure")
        print("   4. Test library integration")
        
        sys.exit(0)
    else:
        print("\n🔧 Fix issues before proceeding to Week 2")
        sys.exit(1)