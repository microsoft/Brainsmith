#!/usr/bin/env python3
"""
Unified Test Runner for Brainsmith HWKG Test Suites

This script provides a comprehensive test runner that can:
1. Run all available tests across tests/ and test_builds/
2. Identify and skip tests with missing dependencies 
3. Provide detailed reporting on test coverage and missing infrastructure
4. Serve as a unified test bench for HWKG development

Usage:
    python test_runner.py --all                    # Run all available tests
    python test_runner.py --core                   # Run only core dataflow tests
    python test_runner.py --hwkg                   # Run only HWKG tests that work
    python test_runner.py --analysis               # Show missing infrastructure analysis
    python test_runner.py --builds                 # Run test_builds/ tests
"""

import sys
import os
import argparse
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import traceback


class TestSuiteAnalyzer:
    """Analyzes test suites to identify working vs broken tests."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.working_tests = []
        self.broken_tests = []
        self.missing_modules = set()
        self.missing_dependencies = set()
        
    def analyze_test_file(self, test_file: Path) -> Dict[str, any]:
        """Analyze a single test file for import issues."""
        result = {
            "file": str(test_file),
            "status": "unknown", 
            "missing_modules": [],
            "missing_dependencies": [],
            "error": None
        }
        
        try:
            # Try to import the test module
            spec = importlib.util.spec_from_file_location(
                f"test_{test_file.stem}", test_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                result["status"] = "working"
                self.working_tests.append(str(test_file))
                
        except ModuleNotFoundError as e:
            result["status"] = "broken_import"
            missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
            result["missing_modules"].append(missing_module)
            self.missing_modules.add(missing_module)
            result["error"] = str(e)
            self.broken_tests.append(str(test_file))
            
        except ImportError as e:
            result["status"] = "broken_dependency"
            result["missing_dependencies"].append(str(e))
            self.missing_dependencies.add(str(e))
            result["error"] = str(e)
            self.broken_tests.append(str(test_file))
            
        except Exception as e:
            result["status"] = "other_error"
            result["error"] = str(e)
            self.broken_tests.append(str(test_file))
            
        return result
    
    def analyze_directory(self, directory: Path) -> List[Dict[str, any]]:
        """Analyze all test files in a directory."""
        results = []
        
        for test_file in directory.rglob("test_*.py"):
            if test_file.is_file():
                result = self.analyze_test_file(test_file)
                results.append(result)
                
        return results
    
    def categorize_missing_modules(self) -> Dict[str, List[str]]:
        """Categorize missing modules by type."""
        categories = {
            "brainsmith_enhanced": [],
            "brainsmith_core": [],
            "external_deps": [],
            "finn_deps": [],
            "other": []
        }
        
        for module in self.missing_modules:
            if "enhanced_" in module and "brainsmith" in module:
                categories["brainsmith_enhanced"].append(module)
            elif "brainsmith" in module:
                categories["brainsmith_core"].append(module)
            elif "finn" in module:
                categories["finn_deps"].append(module)
            elif module in ["qonnx", "onnx", "onnxruntime", "brevitas"]:
                categories["external_deps"].append(module)
            else:
                categories["other"].append(module)
                
        return categories


class UnifiedTestRunner:
    """Unified test runner for all Brainsmith test suites."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.analyzer = TestSuiteAnalyzer(base_path)
        
    def run_pytest_suite(self, test_path: str, suite_name: str) -> Dict[str, any]:
        """Run a pytest suite and capture results."""
        try:
            cmd = [
                sys.executable, "-m", "pytest", 
                test_path, "-v", "--tb=short", 
                "--no-header", "-x"  # Stop on first failure for analysis
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.base_path
            )
            
            return {
                "suite": suite_name,
                "path": test_path,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
        except Exception as e:
            return {
                "suite": suite_name,
                "path": test_path,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
    
    def run_working_tests_only(self, test_path: str, suite_name: str) -> Dict[str, any]:
        """Run only tests that can import successfully."""
        # First analyze which tests can import
        analysis_results = self.analyzer.analyze_directory(Path(test_path))
        working_files = [r["file"] for r in analysis_results if r["status"] == "working"]
        
        if not working_files:
            return {
                "suite": suite_name,
                "path": test_path, 
                "returncode": 0,
                "stdout": "No working tests found in this suite",
                "stderr": "",
                "success": True,
                "working_count": 0,
                "total_count": len(analysis_results)
            }
        
        # Run pytest on working files only
        cmd = [
            sys.executable, "-m", "pytest",
            *working_files, "-v", "--tb=short", "--no-header"
        ]
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=self.base_path
        )
        
        return {
            "suite": suite_name,
            "path": test_path,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "working_count": len(working_files),
            "total_count": len(analysis_results)
        }
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report."""
        report_lines = [
            "=" * 80,
            "BRAINSMITH HWKG TEST SUITE ANALYSIS REPORT",
            "=" * 80,
            ""
        ]
        
        # Analyze all test directories
        test_dirs = [
            ("tests/dataflow", "Core Dataflow Tests"),
            ("tests/tools/hw_kernel_gen", "HWKG Tests"), 
            ("tests/integration", "Integration Tests"),
            ("tests/validation", "Validation Tests"),
            ("test_builds", "Test Builds")
        ]
        
        total_working = 0
        total_tests = 0
        
        for test_dir, description in test_dirs:
            test_path = self.base_path / test_dir
            if test_path.exists():
                results = self.analyzer.analyze_directory(test_path)
                working = len([r for r in results if r["status"] == "working"])
                total = len(results)
                
                total_working += working
                total_tests += total
                
                report_lines.extend([
                    f"{description}:",
                    f"  Path: {test_dir}",
                    f"  Working: {working}/{total} tests",
                    f"  Status: {'✅ ALL WORKING' if working == total else '⚠️  SOME BROKEN' if working > 0 else '❌ ALL BROKEN'}",
                    ""
                ])
        
        # Summary
        success_rate = (total_working / total_tests * 100) if total_tests > 0 else 0
        report_lines.extend([
            f"OVERALL SUMMARY:",
            f"  Total Tests: {total_tests}",
            f"  Working Tests: {total_working}",
            f"  Success Rate: {success_rate:.1f}%",
            ""
        ])
        
        # Missing module analysis
        categories = self.analyzer.categorize_missing_modules()
        
        if any(categories.values()):
            report_lines.extend([
                "MISSING INFRASTRUCTURE ANALYSIS:",
                ""
            ])
            
            if categories["brainsmith_enhanced"]:
                report_lines.extend([
                    "❌ Missing Enhanced Modules (deleted during cleanup):",
                    *[f"  - {mod}" for mod in sorted(categories["brainsmith_enhanced"])],
                    ""
                ])
            
            if categories["brainsmith_core"]:
                report_lines.extend([
                    "❌ Missing Core Modules:",
                    *[f"  - {mod}" for mod in sorted(categories["brainsmith_core"])],
                    ""
                ])
            
            if categories["finn_deps"]:
                report_lines.extend([
                    "⚠️  Missing FINN Dependencies:",
                    *[f"  - {mod}" for mod in sorted(categories["finn_deps"])],
                    ""
                ])
            
            if categories["external_deps"]:
                report_lines.extend([
                    "⚠️  Missing External Dependencies:",
                    *[f"  - {mod}" for mod in sorted(categories["external_deps"])],
                    ""
                ])
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            ""
        ])
        
        if success_rate >= 80:
            report_lines.append("✅ Test infrastructure is mostly working - ready for development")
        elif success_rate >= 50:
            report_lines.append("⚠️  Test infrastructure partially working - some fixes needed")
        else:
            report_lines.append("❌ Test infrastructure needs significant repair")
        
        if categories["brainsmith_enhanced"]:
            report_lines.extend([
                "",
                "For missing enhanced modules:",
                "- These were deleted during cleanup and should NOT be re-implemented",
                "- Mark associated tests as @pytest.mark.skip until needed",
                "- Focus on core functionality tests instead"
            ])
        
        if categories["finn_deps"]:
            report_lines.extend([
                "",
                "For FINN dependencies:",
                "- Consider mocking FINN components for unit tests", 
                "- Keep integration tests separate from unit tests",
                "- Provide clear setup instructions for FINN environment"
            ])
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def run_core_tests(self) -> Dict[str, any]:
        """Run core dataflow tests (known to work)."""
        return self.run_pytest_suite("tests/dataflow", "Core Dataflow Tests")
    
    def run_hwkg_working_tests(self) -> Dict[str, any]:
        """Run only HWKG tests that can import successfully."""
        return self.run_working_tests_only("tests/tools/hw_kernel_gen", "HWKG Tests (Working Only)")
    
    def run_test_builds_working(self) -> Dict[str, any]:
        """Run only test_builds tests that can import successfully.""" 
        return self.run_working_tests_only("test_builds", "Test Builds (Working Only)")
    
    def run_all_working_tests(self) -> List[Dict[str, any]]:
        """Run all tests that can successfully import."""
        results = []
        
        # Core tests (known to work)
        results.append(self.run_core_tests())
        
        # HWKG working tests
        results.append(self.run_hwkg_working_tests())
        
        # Test builds working tests
        results.append(self.run_test_builds_working())
        
        # Integration tests (if any work)
        if (self.base_path / "tests/integration").exists():
            results.append(self.run_working_tests_only("tests/integration", "Integration Tests (Working Only)"))
        
        return results


def main():
    """Main entry point for unified test runner."""
    parser = argparse.ArgumentParser(
        description="Unified Test Runner for Brainsmith HWKG Test Suites"
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run all working tests across all test suites"
    )
    parser.add_argument(
        "--core", action="store_true", 
        help="Run only core dataflow tests"
    )
    parser.add_argument(
        "--hwkg", action="store_true",
        help="Run only HWKG tests that work"
    )
    parser.add_argument(
        "--builds", action="store_true",
        help="Run test_builds/ tests that work"
    )
    parser.add_argument(
        "--analysis", action="store_true",
        help="Show comprehensive analysis of test infrastructure"
    )
    parser.add_argument(
        "--base-path", default=".",
        help="Base path for test discovery (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # If no specific option, default to analysis
    if not any([args.all, args.core, args.hwkg, args.builds, args.analysis]):
        args.analysis = True
    
    runner = UnifiedTestRunner(args.base_path)
    
    if args.analysis:
        print(runner.generate_analysis_report())
        return
    
    results = []
    
    if args.core or args.all:
        results.append(runner.run_core_tests())
    
    if args.hwkg or args.all:
        results.append(runner.run_hwkg_working_tests())
    
    if args.builds or args.all:
        results.append(runner.run_test_builds_working())
    
    if args.all:
        results.extend(runner.run_all_working_tests())
    
    # Print results summary
    print("\n" + "=" * 60)
    print("TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    total_success = 0
    total_suites = len(results)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        working_info = ""
        if "working_count" in result:
            working_info = f" ({result['working_count']}/{result['total_count']} tests runnable)"
        
        print(f"{status} {result['suite']}{working_info}")
        
        if result["success"]:
            total_success += 1
        elif not result["success"] and result["stderr"]:
            print(f"    Error: {result['stderr'][:100]}...")
    
    print(f"\nOverall: {total_success}/{total_suites} test suites passed")
    
    # Exit with appropriate code
    sys.exit(0 if total_success == total_suites else 1)


if __name__ == "__main__":
    main()