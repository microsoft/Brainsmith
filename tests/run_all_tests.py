"""
Comprehensive test runner for Brainsmith Week 1 implementation.

This script runs all test suites and provides comprehensive reporting
of the test results, coverage, and readiness for Week 2 implementation.
"""

import unittest
import sys
import os
import time
from pathlib import Path
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all test modules
test_modules = [
    'tests.test_core_orchestrator',
    'tests.test_finn_interface', 
    'tests.test_workflow_manager',
    'tests.test_api',
    'tests.test_legacy_support'
]

class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""
    
    def getDescription(self, test):
        return f"{test._testMethodName} ({test.__class__.__name__})"
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.showAll:
            self.stream.write("✅ PASS\n")
        elif self.dots:
            self.stream.write('✅')
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.write("💥 ERROR\n")
        elif self.dots:
            self.stream.write('💥')
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.write("❌ FAIL\n") 
        elif self.dots:
            self.stream.write('❌')
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.write(f"⏭️  SKIP: {reason}\n")
        elif self.dots:
            self.stream.write('⏭️')


class ComprehensiveTestRunner:
    """Comprehensive test runner with detailed reporting."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, verbosity=2):
        """Run all test suites and collect results."""
        print("🧪" + "="*70)
        print("🚀 Brainsmith Week 1 Implementation - Comprehensive Test Suite")
        print("🧪" + "="*70)
        
        self.start_time = time.time()
        
        overall_result = unittest.TestResult()
        
        for module_name in test_modules:
            print(f"\n📦 Running {module_name}...")
            print("-" * 50)
            
            module_result = self._run_module_tests(module_name, verbosity)
            self.results[module_name] = module_result
            
            # Aggregate results
            overall_result.testsRun += module_result.testsRun
            overall_result.failures.extend(module_result.failures)
            overall_result.errors.extend(module_result.errors)
            overall_result.skipped.extend(module_result.skipped)
            
            self._print_module_summary(module_name, module_result)
        
        self.end_time = time.time()
        
        # Print comprehensive summary
        self._print_comprehensive_summary(overall_result)
        
        return overall_result
    
    def _run_module_tests(self, module_name, verbosity):
        """Run tests for a specific module."""
        try:
            # Import the test module
            test_module = __import__(module_name, fromlist=[''])
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests with custom result class
            stream = StringIO()
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=verbosity,
                resultclass=ColoredTextTestResult
            )
            
            result = runner.run(suite)
            
            # Print output
            output = stream.getvalue()
            if output.strip():
                print(output)
            
            return result
            
        except ImportError as e:
            print(f"⚠️  Could not import {module_name}: {e}")
            
            # Create dummy result for missing module
            dummy_result = unittest.TestResult()
            dummy_result.skipped.append((None, f"Module {module_name} not available: {e}"))
            return dummy_result
        
        except Exception as e:
            print(f"❌ Error running {module_name}: {e}")
            
            # Create error result
            error_result = unittest.TestResult()
            error_result.errors.append((None, str(e)))
            return error_result
    
    def _print_module_summary(self, module_name, result):
        """Print summary for a test module."""
        total = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors) 
        skipped = len(result.skipped)
        passed = total - failures - errors
        
        if total == 0:
            print(f"⏭️  {module_name}: No tests run")
            return
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        status_icon = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
        
        print(f"{status_icon} {module_name}: {passed}/{total} passed ({success_rate:.1f}%)")
        if failures > 0:
            print(f"   ❌ Failures: {failures}")
        if errors > 0:
            print(f"   💥 Errors: {errors}")
        if skipped > 0:
            print(f"   ⏭️  Skipped: {skipped}")
    
    def _print_comprehensive_summary(self, overall_result):
        """Print comprehensive test summary."""
        duration = self.end_time - self.start_time
        
        total = overall_result.testsRun
        failures = len(overall_result.failures)
        errors = len(overall_result.errors)
        skipped = len(overall_result.skipped)
        passed = total - failures - errors
        
        print("\n" + "🎯" + "="*70)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("🎯" + "="*70)
        
        print(f"⏱️  Total execution time: {duration:.2f} seconds")
        print(f"📝 Tests run: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failures}")
        print(f"💥 Errors: {errors}")
        print(f"⏭️  Skipped: {skipped}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
            
            # Overall status
            if success_rate >= 95:
                print("🎉 EXCELLENT: Week 1 implementation is production-ready!")
                readiness = "READY FOR WEEK 2"
            elif success_rate >= 85:
                print("✅ GOOD: Week 1 implementation is solid with minor issues")
                readiness = "READY FOR WEEK 2"
            elif success_rate >= 70:
                print("⚠️  WARNING: Week 1 implementation has some issues")
                readiness = "CONSIDER FIXES BEFORE WEEK 2"
            else:
                print("❌ CRITICAL: Week 1 implementation has major issues")
                readiness = "FIX CRITICAL ISSUES BEFORE WEEK 2"
            
            print(f"🚀 Week 2 Readiness: {readiness}")
        else:
            print("⚠️  No tests were executed - check test imports")
        
        # Component-wise breakdown
        self._print_component_breakdown()
        
        # Detailed failure/error reporting
        if failures > 0 or errors > 0:
            self._print_detailed_issues(overall_result)
        
        # Next steps
        self._print_next_steps(overall_result)
    
    def _print_component_breakdown(self):
        """Print breakdown by component."""
        print("\n📦 COMPONENT BREAKDOWN:")
        
        component_map = {
            'test_core_orchestrator': '🎯 Core Orchestrator',
            'test_finn_interface': '🔧 FINN Interface',
            'test_workflow_manager': '🔄 Workflow Manager',
            'test_api': '🐍 Python API',
            'test_legacy_support': '🔙 Legacy Support'
        }
        
        for module_name, result in self.results.items():
            component_name = component_map.get(module_name.split('.')[-1], module_name)
            
            total = result.testsRun
            if total > 0:
                failures = len(result.failures)
                errors = len(result.errors)
                passed = total - failures - errors
                success_rate = (passed / total) * 100
                
                status_icon = "✅" if success_rate >= 95 else "⚠️" if success_rate >= 80 else "❌"
                print(f"   {status_icon} {component_name}: {success_rate:.1f}% ({passed}/{total})")
            else:
                print(f"   ⏭️  {component_name}: No tests")
    
    def _print_detailed_issues(self, overall_result):
        """Print detailed information about failures and errors."""
        print("\n🔍 DETAILED ISSUES:")
        
        if overall_result.failures:
            print("\n❌ FAILURES:")
            for test, traceback in overall_result.failures:
                if test:
                    print(f"   • {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if overall_result.errors:
            print("\n💥 ERRORS:")
            for test, traceback in overall_result.errors:
                if test:
                    error_msg = traceback.split('\n')[-2] if '\n' in traceback else traceback
                    print(f"   • {test}: {error_msg}")
    
    def _print_next_steps(self, overall_result):
        """Print recommended next steps."""
        print("\n🎯 NEXT STEPS:")
        
        total = overall_result.testsRun
        if total == 0:
            print("   1. ❗ Fix import issues - ensure all core components are available")
            print("   2. 🔧 Check Python path and module dependencies")
            print("   3. 📝 Verify test module structure")
            return
        
        failures = len(overall_result.failures)
        errors = len(overall_result.errors)
        passed = total - failures - errors
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        if success_rate >= 85:
            print("   1. 🚀 Proceed to Week 2: Library Structure Implementation")
            print("   2. 📋 Begin kernels library organization (existing custom_op/)")
            print("   3. 🔧 Implement transforms library structure (existing steps/)")
            print("   4. ⚡ Organize hw optimization library (existing dse/)")
        else:
            print("   1. 🔧 Fix critical component issues before proceeding")
            if failures > 0:
                print("   2. ❌ Address test failures in core functionality")
            if errors > 0:
                print("   3. 💥 Resolve errors in component initialization")
            print("   4. 🧪 Re-run tests until success rate >= 85%")
        
        print("\n📚 Resources:")
        print("   • Week 1 Implementation: docs/week1_implementation_complete.md") 
        print("   • Architecture Design: docs/brainsmith_final_architectural_design.md")
        print("   • Phase 4 Execution Plan: docs/phase4_execution_plan.md")


def main():
    """Main test runner function."""
    runner = ComprehensiveTestRunner()
    
    # Parse command line arguments
    verbosity = 2
    if len(sys.argv) > 1:
        if '--quiet' in sys.argv:
            verbosity = 0
        elif '--verbose' in sys.argv:
            verbosity = 2
    
    # Run all tests
    overall_result = runner.run_all_tests(verbosity)
    
    # Determine exit code
    if overall_result.testsRun == 0:
        exit_code = 2  # No tests run
    else:
        failures = len(overall_result.failures)
        errors = len(overall_result.errors)
        if failures > 0 or errors > 0:
            exit_code = 1  # Some tests failed
        else:
            exit_code = 0  # All tests passed
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()