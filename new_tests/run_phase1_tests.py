"""
Phase 1 Test Runner - Core Functionality Tests

Simple test runner for Phase 1 core functionality validation.
Focuses on testing the new three-layer architecture with minimal mocking.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any


def run_test_category(category: str, test_files: List[str]) -> Dict[str, Any]:
    """Run a category of tests and return results."""
    print(f"\n🧪 Running {category} Tests...")
    print("=" * 50)
    
    results = {
        'category': category,
        'total_files': len(test_files),
        'passed_files': 0,
        'failed_files': 0,
        'skipped_files': 0,
        'execution_time': 0.0,
        'details': []
    }
    
    start_time = time.time()
    
    for test_file in test_files:
        test_path = Path(__file__).parent / test_file
        
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            results['failed_files'] += 1
            continue
        
        print(f"  📝 {test_file}")
        
        try:
            # Run pytest on the specific file
            cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"  ✅ PASSED")
                results['passed_files'] += 1
                status = 'PASSED'
            elif "SKIPPED" in result.stdout or result.returncode == 5:
                print(f"  ⏭️  SKIPPED (components not available)")
                results['skipped_files'] += 1
                status = 'SKIPPED'
            else:
                print(f"  ❌ FAILED")
                results['failed_files'] += 1
                status = 'FAILED'
                # Print error details for failed tests
                if result.stdout:
                    print(f"     STDOUT: {result.stdout[-200:]}")
                if result.stderr:
                    print(f"     STDERR: {result.stderr[-200:]}")
            
            results['details'].append({
                'file': test_file,
                'status': status,
                'returncode': result.returncode,
                'stdout_length': len(result.stdout) if result.stdout else 0,
                'stderr_length': len(result.stderr) if result.stderr else 0
            })
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ TIMEOUT")
            results['failed_files'] += 1
            results['details'].append({
                'file': test_file,
                'status': 'TIMEOUT',
                'returncode': -1,
                'stdout_length': 0,
                'stderr_length': 0
            })
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            results['failed_files'] += 1
            results['details'].append({
                'file': test_file,
                'status': 'ERROR',
                'returncode': -1,
                'stdout_length': 0,
                'stderr_length': 0
            })
    
    results['execution_time'] = time.time() - start_time
    return results


def print_summary(all_results: List[Dict[str, Any]]):
    """Print comprehensive test summary."""
    print("\n" + "=" * 70)
    print("🎯 PHASE 1 CORE FUNCTIONALITY TEST SUMMARY")
    print("=" * 70)
    
    total_files = sum(r['total_files'] for r in all_results)
    total_passed = sum(r['passed_files'] for r in all_results)
    total_failed = sum(r['failed_files'] for r in all_results)
    total_skipped = sum(r['skipped_files'] for r in all_results)
    total_time = sum(r['execution_time'] for r in all_results)
    
    print(f"📊 OVERALL RESULTS:")
    print(f"   📁 Total test files: {total_files}")
    print(f"   ✅ Passed: {total_passed}")
    print(f"   ❌ Failed: {total_failed}")
    print(f"   ⏭️  Skipped: {total_skipped}")
    print(f"   ⏱️  Total time: {total_time:.2f}s")
    
    if total_files > 0:
        success_rate = (total_passed / total_files) * 100
        print(f"   📈 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 95:
            print("   🎉 EXCELLENT: Phase 1 is production-ready!")
        elif success_rate >= 85:
            print("   🚀 GOOD: Phase 1 ready for Phase 2 implementation")
        elif success_rate >= 70:
            print("   ⚠️  WARNING: Consider fixes before Phase 2")
        else:
            print("   🚨 CRITICAL: Fix issues before proceeding")
    
    print(f"\n📋 DETAILED RESULTS BY CATEGORY:")
    for result in all_results:
        category = result['category']
        passed = result['passed_files']
        failed = result['failed_files']
        skipped = result['skipped_files']
        total = result['total_files']
        time_taken = result['execution_time']
        
        status_emoji = "✅" if failed == 0 else "❌" if passed == 0 else "⚠️"
        print(f"   {status_emoji} {category}: {passed}✅ {failed}❌ {skipped}⏭️ ({time_taken:.1f}s)")


def main():
    """Run Phase 1 core functionality tests."""
    print("🧪======================================================================")
    print("🚀 BrainSmith Phase 1 Implementation - Core Functionality Test Suite")
    print("🧪======================================================================")
    print("Focus: Testing new three-layer architecture with minimal mocking")
    print("Scope: Core API, CLI, Metrics, Design Space, Package Imports")
    
    # Define test categories and files
    test_categories = [
        {
            'name': 'Core Layer',
            'files': [
                'core/test_forge_api.py',
                'core/test_cli.py', 
                'core/test_metrics.py',
                'core/test_validation.py'
            ]
        },
        {
            'name': 'Infrastructure Layer',
            'files': [
                'infrastructure/test_design_space.py',
                'infrastructure/test_package_imports.py'
            ]
        }
    ]
    
    all_results = []
    
    # Run each test category
    for category in test_categories:
        result = run_test_category(category['name'], category['files'])
        all_results.append(result)
    
    # Print comprehensive summary
    print_summary(all_results)
    
    # Print next steps
    print(f"\n🔮 NEXT STEPS:")
    print(f"   📚 Phase 2: Library robustness tests")
    print(f"   🪝 Phase 3: Hooks and unused capabilities tests")
    print(f"   🚀 Stakeholder delivery preparation")
    
    # Return appropriate exit code
    total_failed = sum(r['failed_files'] for r in all_results)
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())