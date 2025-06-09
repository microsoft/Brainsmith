#!/usr/bin/env python3
"""
BrainSmith Test Suite - Smoke Test Runner

Quick validation of the comprehensive test suite functionality.
"""

import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(cmd, timeout=60):
    """Run a command with timeout and return success status."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=str(project_root)
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def print_test_result(test_name, success, duration=None):
    """Print formatted test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    duration_str = f" ({duration:.1f}s)" if duration else ""
    print(f"{status} {test_name}{duration_str}")

def main():
    """Run smoke tests for the BrainSmith test suite."""
    print("🚀 BrainSmith Comprehensive Test Suite - Smoke Tests")
    print(f"📁 Project Root: {project_root}")
    
    # Test configuration
    smoke_tests = [
        {
            'name': 'Import BrainSmith Module',
            'cmd': 'python -c "import brainsmith; print(f\'BrainSmith {brainsmith.__version__} imported successfully\')"',
            'timeout': 10
        },
        {
            'name': 'API Module Import Test',
            'cmd': 'python -m pytest tests/functional/api/test_highlevel_api.py::TestHighLevelAPI::test_api_module_imports -v',
            'timeout': 30
        },
        {
            'name': 'Core API Functions Test',
            'cmd': 'python -m pytest tests/functional/api/test_highlevel_api.py::TestHighLevelAPI::test_core_api_functions_available -v',
            'timeout': 30
        },
        {
            'name': 'Strategy Listing Test',
            'cmd': 'python -m pytest tests/functional/api/test_highlevel_api.py::TestHighLevelAPI::test_list_available_strategies -v',
            'timeout': 30
        },
        {
            'name': 'Build Model Test',
            'cmd': 'python -m pytest tests/functional/api/test_highlevel_api.py::TestHighLevelAPI::test_build_model_basic_usage -v',
            'timeout': 30
        },
        {
            'name': 'Performance API Test',
            'cmd': 'python -m pytest tests/performance/test_performance_benchmarks.py::TestThroughputBenchmarks::test_api_response_time_strategies -v',
            'timeout': 60
        }
    ]
    
    # Run smoke tests
    print_section("SMOKE TESTS")
    
    passed_tests = 0
    total_tests = len(smoke_tests)
    
    for test in smoke_tests:
        print(f"\n🔍 Running: {test['name']}")
        
        start_time = time.time()
        success, stdout, stderr = run_command(test['cmd'], test['timeout'])
        duration = time.time() - start_time
        
        print_test_result(test['name'], success, duration)
        
        if success:
            passed_tests += 1
        else:
            print(f"   ❌ Error: {stderr[:200]}...")
            if stdout:
                print(f"   📝 Output: {stdout[:200]}...")
    
    # Summary
    print_section("SMOKE TEST SUMMARY")
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"📊 Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All smoke tests PASSED! Test suite is functional.")
        return_code = 0
    else:
        print("⚠️  Some smoke tests FAILED. Test suite needs attention.")
        return_code = 1
    
    # Additional validation
    print_section("ENVIRONMENT VALIDATION")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 7):
        print("✅ Python version OK")
    else:
        print("⚠️  Python version may be too old")
    
    # Check test directory structure
    test_dirs = [
        'tests/functional',
        'tests/functional/api',
        'tests/performance',
        'tests/configs'
    ]
    
    print("\n📁 Test Directory Structure:")
    for test_dir in test_dirs:
        dir_path = project_root / test_dir
        if dir_path.exists():
            print(f"✅ {test_dir}/")
        else:
            print(f"❌ {test_dir}/ - MISSING")
    
    # Check key test files
    test_files = [
        'tests/pytest.ini',
        'tests/conftest.py',
        'tests/functional/api/test_highlevel_api.py',
        'tests/performance/test_performance_benchmarks.py'
    ]
    
    print("\n📄 Key Test Files:")
    for test_file in test_files:
        file_path = project_root / test_file
        if file_path.exists():
            print(f"✅ {test_file}")
        else:
            print(f"❌ {test_file} - MISSING")
    
    # Final recommendations
    print_section("RECOMMENDATIONS")
    
    if return_code == 0:
        print("🚀 Test suite is ready for full execution!")
        print("💡 Next steps:")
        print("   • Run full test suite: python -m pytest tests/")
        print("   • Run performance tests: python -m pytest tests/performance/ -m performance")
        print("   • Run integration tests: python -m pytest tests/ -m integration")
    else:
        print("🔧 Test suite needs fixes:")
        print("   • Check failed tests above")
        print("   • Verify BrainSmith installation")
        print("   • Install missing dependencies")
        print("   • Run: pip install -r tests/requirements-test.txt")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())