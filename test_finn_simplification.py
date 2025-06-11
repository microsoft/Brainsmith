#!/usr/bin/env python3
"""
Test script for FINN module simplification validation.

This script tests the simplified FINN interface without requiring
a full FINN environment, focusing on interface compatibility and
structure validation.
"""

import sys
import os
import logging
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all simplified FINN components can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test main module imports
        from brainsmith.finn import (
            FINNInterface, build_accelerator, validate_finn_config, 
            prepare_4hooks_config, FINNConfig, FINNResult, FINNHooksConfig
        )
        logger.info("✅ All main imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_finn_config():
    """Test FINNConfig functionality."""
    logger.info("Testing FINNConfig...")
    
    try:
        from brainsmith.finn import FINNConfig
        
        # Test default configuration
        config = FINNConfig()
        assert config.target_device == "U250"
        assert config.target_fps == 1000
        assert config.clock_period == 3.33
        
        # Test custom configuration
        custom_config = FINNConfig(
            target_device="U280",
            target_fps=2000,
            clock_period=2.5
        )
        assert custom_config.target_device == "U280"
        assert custom_config.target_fps == 2000
        
        # Test conversion methods
        core_dict = config.to_core_dict()
        assert isinstance(core_dict, dict)
        assert 'target_device' in core_dict
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'target_device' in config_dict
        
        logger.info("✅ FINNConfig tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ FINNConfig test failed: {e}")
        return False


def test_finn_result():
    """Test FINNResult functionality."""
    logger.info("Testing FINNResult...")
    
    try:
        from brainsmith.finn import FINNResult
        
        # Test basic result creation
        result = FINNResult(
            success=True,
            model_path="/path/to/model.onnx",
            output_dir="./output"
        )
        assert result.success == True
        assert result.model_path == "/path/to/model.onnx"
        
        # Test from_core_result conversion
        core_result = {
            'success': True,
            'model_path': '/test/model.onnx',
            'output_dir': './test_output',
            'performance_metrics': {'throughput_fps': 1000},
            'resource_usage': {'lut_count': 50000}
        }
        
        finn_result = FINNResult.from_core_result(core_result)
        assert finn_result.success == True
        assert finn_result.throughput_fps == 1000
        assert finn_result.lut_count == 50000
        
        # Test to_dict conversion
        result_dict = finn_result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['success'] == True
        
        logger.info("✅ FINNResult tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ FINNResult test failed: {e}")
        return False


def test_4hooks_config():
    """Test 4-hooks preparation functionality."""
    logger.info("Testing 4-hooks configuration...")
    
    try:
        from brainsmith.finn import FINNHooksConfig, prepare_4hooks_config
        
        # Test hooks config creation
        hooks_config = FINNHooksConfig()
        assert hooks_config.preprocessing_enabled == True
        assert hooks_config.transformation_enabled == True
        assert hooks_config.optimization_enabled == True
        assert hooks_config.generation_enabled == True
        
        # Test 4-hooks preparation
        design_point = {
            'preprocessing': {'param1': 'value1'},
            'transforms': {'optimization_level': 'aggressive'},
            'hw_optimization': {'strategy': 'throughput'},
            'generation': {'bitstream': True}
        }
        
        hooks_dict = hooks_config.prepare_config(design_point)
        assert 'preprocessing' in hooks_dict
        assert 'transformation' in hooks_dict
        assert 'optimization' in hooks_dict
        assert 'generation' in hooks_dict
        
        # Test convenience function
        prepared = prepare_4hooks_config(design_point)
        assert isinstance(prepared, dict)
        
        # Test that 4-hooks is not ready yet (until FINN implements it)
        assert hooks_config.is_4hooks_ready() == False
        
        # Test enabled hooks list
        enabled_hooks = hooks_config.get_enabled_hooks()
        assert len(enabled_hooks) == 4
        
        logger.info("✅ 4-hooks configuration tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ 4-hooks configuration test failed: {e}")
        return False


def test_interface_structure():
    """Test the simplified interface structure."""
    logger.info("Testing interface structure...")
    
    try:
        from brainsmith.finn import FINNInterface
        
        # Test interface creation
        interface = FINNInterface()
        assert hasattr(interface, 'config')
        assert hasattr(interface, 'core_interface')
        assert hasattr(interface, 'hooks_config')
        
        # Test methods exist
        assert hasattr(interface, 'build_accelerator')
        assert hasattr(interface, 'prepare_4hooks_config')
        assert hasattr(interface, 'validate_config')
        assert hasattr(interface, 'get_supported_devices')
        
        # Test supported devices
        devices = interface.get_supported_devices()
        assert isinstance(devices, list)
        assert len(devices) > 0
        
        logger.info("✅ Interface structure tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Interface structure test failed: {e}")
        return False


def test_module_info():
    """Test module information and exports."""
    logger.info("Testing module information...")
    
    try:
        import brainsmith.finn as finn_module
        
        # Test version information
        assert hasattr(finn_module, '__version__')
        assert finn_module.__version__ == "2.0.0"
        
        # Test module info
        module_info = finn_module.get_module_info()
        assert isinstance(module_info, dict)
        assert 'name' in module_info
        assert 'reduction_from_v1' in module_info
        
        # Test __all__ exports
        expected_exports = [
            'FINNInterface', 'build_accelerator', 'validate_finn_config',
            'prepare_4hooks_config', 'FINNConfig', 'FINNResult', 'FINNHooksConfig'
        ]
        
        for export in expected_exports:
            assert hasattr(finn_module, export), f"Missing export: {export}"
        
        # Test that we have clean exports (not too many)
        actual_exports = finn_module.__all__
        assert len(actual_exports) <= 8, f"Too many exports: {len(actual_exports)}"
        
        logger.info("✅ Module information tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Module information test failed: {e}")
        return False


def test_integration_with_core():
    """Test integration with core API."""
    logger.info("Testing integration with core...")
    
    try:
        # Test that core can import from simplified FINN
        from brainsmith.finn import build_accelerator
        
        # Test function signature
        import inspect
        sig = inspect.signature(build_accelerator)
        params = list(sig.parameters.keys())
        assert 'model_path' in params
        assert 'blueprint_config' in params
        assert 'output_dir' in params
        
        logger.info("✅ Core integration tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Core integration test failed: {e}")
        return False


def test_file_structure():
    """Test that we have the expected simplified file structure."""
    logger.info("Testing file structure...")
    
    try:
        import os
        finn_dir = "brainsmith/finn"
        
        # Check that we only have the expected files
        expected_files = {'__init__.py', 'interface.py', 'types.py'}
        actual_files = set(f for f in os.listdir(finn_dir) if f.endswith('.py'))
        
        assert actual_files == expected_files, f"Unexpected files: {actual_files - expected_files}"
        
        # Check that enterprise files are gone
        enterprise_files = {
            'engine.py', 'orchestration.py', 'monitoring.py', 'workflow.py',
            'environment.py', 'model_ops_manager.py', 'model_transforms_manager.py',
            'hw_kernels_manager.py', 'hw_optimization_manager.py'
        }
        
        for enterprise_file in enterprise_files:
            file_path = os.path.join(finn_dir, enterprise_file)
            assert not os.path.exists(file_path), f"Enterprise file still exists: {enterprise_file}"
        
        logger.info("✅ File structure tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ File structure test failed: {e}")
        return False


def main():
    """Run all tests and report results."""
    logger.info("🚀 Starting FINN simplification validation tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("FINNConfig", test_finn_config),
        ("FINNResult", test_finn_result),
        ("4-Hooks Config", test_4hooks_config),
        ("Interface Structure", test_interface_structure),
        ("Module Info", test_module_info),
        ("Core Integration", test_integration_with_core),
        ("File Structure", test_file_structure)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} tests...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! FINN simplification is working correctly.")
        return 0
    else:
        logger.error(f"💥 {failed} tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())