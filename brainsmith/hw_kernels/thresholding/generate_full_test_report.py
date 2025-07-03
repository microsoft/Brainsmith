############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Comprehensive test report generator for AutoHWCustomOp validation.

This script runs all available tests and generates a consolidated report
showing the complete validation status of the AutoHWCustomOp system.
"""

import sys
import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

class TestResult:
    """Container for individual test results."""
    
    def __init__(self, name: str, passed: bool, duration: float, details: str = ""):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.details = details
        self.timestamp = datetime.now()

class ComprehensiveTestRunner:
    """Runs all AutoHWCustomOp tests and generates comprehensive report."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def run_single_test(self, test_name: str, test_file: str) -> TestResult:
        """Run a single test and capture results."""
        print(f"🧪 Running {test_name}...")
        
        start_time = time.time()
        try:
            # Run test using smithy exec
            project_root = self.test_dir.parent.parent.parent
            # Split command properly for subprocess
            cmd = ["./smithy", "exec", "python", test_file]
            result = subprocess.run(
                cmd,
                cwd=project_root,  # Project root
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"   ✅ {test_name} passed ({duration:.1f}s)")
                return TestResult(test_name, True, duration, result.stdout)
            else:
                print(f"   ❌ {test_name} failed ({duration:.1f}s)")
                return TestResult(test_name, False, duration, result.stderr)
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"   ⏰ {test_name} timed out ({duration:.1f}s)")
            return TestResult(test_name, False, duration, "Test timed out after 5 minutes")
        except Exception as e:
            duration = time.time() - start_time
            print(f"   💥 {test_name} crashed ({duration:.1f}s): {e}")
            return TestResult(test_name, False, duration, f"Test crashed: {e}")
    
    def run_all_tests(self) -> None:
        """Run all available tests."""
        test_suite = [
            ("Parity Test", "brainsmith/hw_kernels/thresholding/test_thresholding_comparison_v2.py"),
            ("Behavioral Execution", "brainsmith/hw_kernels/thresholding/test_behavioral_execution.py"),
            ("RTL Generation", "brainsmith/hw_kernels/thresholding/test_rtl_generation.py"),
            ("CPPSIM Testing", "brainsmith/hw_kernels/thresholding/test_cppsim.py"),
            ("FINN Pipeline", "brainsmith/hw_kernels/thresholding/test_finn_pipeline.py"),
        ]
        
        print("🚀 Starting Comprehensive AutoHWCustomOp Test Suite...")
        print("=" * 80)
        
        for test_name, test_file in test_suite:
            result = self.run_single_test(test_name, test_file)
            self.results.append(result)
            print()  # Add spacing between tests
    
    def generate_summary_report(self) -> str:
        """Generate summary report."""
        total_time = time.time() - self.start_time
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE AUTOHWCUSTOMOP TEST REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"⏱️ Total Runtime: {total_time:.1f} seconds")
        report.append(f"📊 Overall Result: {len(passed_tests)}/{len(self.results)} tests passed")
        report.append("")
        
        # Test Summary
        report.append("## Test Summary")
        report.append("")
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report.append(f"- {result.name:<25} {status:<10} ({result.duration:.1f}s)")
        report.append("")
        
        # Detailed Results
        if passed_tests:
            report.append("## ✅ Passed Tests")
            report.append("")
            for result in passed_tests:
                report.append(f"### {result.name}")
                report.append(f"- Duration: {result.duration:.1f} seconds")
                report.append(f"- Status: Successful execution")
                if "All tests completed successfully!" in result.details:
                    report.append(f"- Details: All validations passed")
                report.append("")
        
        if failed_tests:
            report.append("## ❌ Failed Tests")
            report.append("")
            for result in failed_tests:
                report.append(f"### {result.name}")
                report.append(f"- Duration: {result.duration:.1f} seconds")
                report.append(f"- Status: Failed execution")
                
                # Extract key error information
                error_lines = result.details.split('\n')
                key_errors = [line for line in error_lines if 'error' in line.lower() or 'failed' in line.lower()]
                if key_errors:
                    report.append(f"- Error: {key_errors[-1].strip()}")
                report.append("")
        
        # System Information
        report.append("## System Information")
        report.append("")
        report.append("- **Environment**: Brainsmith experimental/hwkg branch")
        report.append("- **Container**: Docker with FINN dependencies")
        report.append("- **Python Path**: Includes project root and FINN")
        report.append("- **Test Framework**: Custom test suite with FINN integration")
        report.append("")
        
        # Architecture Validation
        report.append("## Architecture Validation Status")
        report.append("")
        
        parity_passed = any(r.name == "Parity Test" and r.passed for r in self.results)
        behavioral_passed = any(r.name == "Behavioral Execution" and r.passed for r in self.results)
        cppsim_passed = any(r.name == "CPPSIM Testing" and r.passed for r in self.results)
        pipeline_passed = any(r.name == "FINN Pipeline" and r.passed for r in self.results)
        
        validation_status = [
            ("Shape Extraction", "✅ VALIDATED" if parity_passed else "❌ FAILED", 
             "AutoHWCustomOp correctly extracts shapes from ONNX"),
            ("Functional Parity", "✅ VALIDATED" if parity_passed else "❌ FAILED", 
             "Auto-generated implementation matches manual behavior"),
            ("FINN Integration", "✅ VALIDATED" if cppsim_passed else "❌ FAILED", 
             "Integration with FINN transformation pipeline"),
            ("Execution Capability", "✅ VALIDATED" if pipeline_passed else "❌ FAILED", 
             "End-to-end execution in FINN environment"),
        ]
        
        for component, status, description in validation_status:
            report.append(f"- **{component}**: {status}")
            report.append(f"  - {description}")
        
        report.append("")
        
        # Conclusions
        report.append("## Conclusions")
        report.append("")
        
        if len(passed_tests) == len(self.results):
            report.append("🎉 **COMPLETE SUCCESS**: All AutoHWCustomOp tests passed!")
            report.append("")
            report.append("The AutoHWCustomOp system is **production ready** with:")
            report.append("- ✅ Automatic shape extraction from ONNX graphs")
            report.append("- ✅ Functional equivalence with manual implementations")
            report.append("- ✅ Full FINN pipeline integration")
            report.append("- ✅ CPPSIM execution capability")
            report.append("- ✅ Robust validation framework")
        elif len(passed_tests) >= len(self.results) * 0.8:
            report.append("🟡 **MOSTLY SUCCESSFUL**: Most AutoHWCustomOp tests passed!")
            report.append("")
            report.append("The system shows **strong validation** with minor issues:")
            report.append(f"- ✅ {len(passed_tests)} tests passed")
            report.append(f"- ❌ {len(failed_tests)} tests failed")
            report.append("- 🔧 Requires attention to failed components")
        else:
            report.append("❌ **NEEDS ATTENTION**: Several AutoHWCustomOp tests failed!")
            report.append("")
            report.append("The system requires **additional development**:")
            report.append(f"- ✅ {len(passed_tests)} tests passed")
            report.append(f"- ❌ {len(failed_tests)} tests failed")
            report.append("- 🚨 Critical issues need resolution")
        
        report.append("")
        report.append("## Next Steps")
        report.append("")
        
        if len(passed_tests) == len(self.results):
            report.append("1. **Deploy to Production**: System ready for broader use")
            report.append("2. **Add More Kernels**: Apply AutoHWCustomOp to other operations")
            report.append("3. **Performance Optimization**: Fine-tune for specific use cases")
        else:
            report.append("1. **Address Failed Tests**: Investigate and fix failing components")
            report.append("2. **Improve Error Handling**: Add robustness for edge cases")
            report.append("3. **Re-run Validation**: Verify fixes with complete test suite")
        
        report.append("")
        report.append("=" * 80)
        report.append("End of Report")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, report: str) -> str:
        """Save report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.test_dir / f"AutoHWCustomOp_Test_Report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        return str(report_file)
    
    def run_comprehensive_validation(self) -> bool:
        """Run complete validation suite and generate report."""
        self.run_all_tests()
        
        # Generate and save report
        report = self.generate_summary_report()
        report_file = self.save_report(report)
        
        # Print summary
        print(report)
        print(f"\n📄 Full report saved to: {report_file}")
        
        # Return overall success
        return all(result.passed for result in self.results)


def main():
    """Main function."""
    runner = ComprehensiveTestRunner()
    success = runner.run_comprehensive_validation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())