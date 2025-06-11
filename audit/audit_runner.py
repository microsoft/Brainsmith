#!/usr/bin/env python3
"""
Brainsmith Repository Audit Runner

Main execution script for comprehensive repository audit focusing on:
- Functional completeness (ensuring all components work)
- Integration testing (ensuring layers work together seamlessly) 
- Extension mechanisms (validating contrib/ directories and registries)
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path to import brainsmith modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('audit/reports/audit_execution.log')
    ]
)

logger = logging.getLogger(__name__)

class AuditRunner:
    """Main audit execution coordinator."""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {
            'phase1': {},
            'phase2': {},
            'phase3': {},
            'summary': {},
            'execution_time': 0
        }
        self.test_results = []
        self.issues_found = []
        
        # Ensure reports directory exists
        os.makedirs('audit/reports', exist_ok=True)
        
        logger.info("Brainsmith Repository Audit Started")
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Execute complete audit across all phases."""
        try:
            # Phase 1: Functional Completeness Audit
            logger.info("Starting Phase 1: Functional Completeness Audit")
            self.results['phase1'] = self._run_phase1()
            
            # Phase 2: Integration Testing
            logger.info("Starting Phase 2: Integration Testing")
            self.results['phase2'] = self._run_phase2()
            
            # Phase 3: Extension Mechanisms Audit
            logger.info("Starting Phase 3: Extension Mechanisms Audit")
            self.results['phase3'] = self._run_phase3()
            
            # Generate summary
            self._generate_summary()
            
            # Save results
            self._save_results()
            
            # Generate reports
            self._generate_reports()
            
        except Exception as e:
            logger.error(f"Audit execution failed: {e}")
            self.results['error'] = str(e)
        
        finally:
            self.results['execution_time'] = time.time() - self.start_time
            logger.info(f"Audit completed in {self.results['execution_time']:.2f} seconds")
        
        return self.results
    
    def _run_phase1(self) -> Dict[str, Any]:
        """Phase 1: Functional Completeness Audit."""
        phase1_results = {
            'core_layer': {},
            'infrastructure_layer': {},
            'libraries_layer': {}
        }
        
        try:
            # Import test modules
            from audit.tests.test_core_layer import CoreLayerTester
            from audit.tests.test_infrastructure import InfrastructureTester
            from audit.tests.test_libraries import LibrariesTester
            
            # Core layer testing
            logger.info("Testing Core Layer...")
            core_tester = CoreLayerTester()
            phase1_results['core_layer'] = core_tester.run_all_tests()
            
            # Infrastructure layer testing
            logger.info("Testing Infrastructure Layer...")
            infra_tester = InfrastructureTester()
            phase1_results['infrastructure_layer'] = infra_tester.run_all_tests()
            
            # Libraries layer testing
            logger.info("Testing Libraries Layer...")
            libs_tester = LibrariesTester()
            phase1_results['libraries_layer'] = libs_tester.run_all_tests()
            
        except Exception as e:
            logger.error(f"Phase 1 execution failed: {e}")
            phase1_results['error'] = str(e)
        
        return phase1_results
    
    def _run_phase2(self) -> Dict[str, Any]:
        """Phase 2: Integration Testing."""
        phase2_results = {
            'cross_layer_integration': {},
            'blueprint_management_integration': {},
            'registry_integration': {},
            'import_dependency_health': {}
        }
        
        try:
            from audit.tests.test_integration import IntegrationTester
            
            logger.info("Running Integration Tests...")
            integration_tester = IntegrationTester()
            phase2_results = integration_tester.run_all_tests()
            
        except Exception as e:
            logger.error(f"Phase 2 execution failed: {e}")
            phase2_results['error'] = str(e)
        
        return phase2_results
    
    def _run_phase3(self) -> Dict[str, Any]:
        """Phase 3: Extension Mechanisms Audit."""
        phase3_results = {
            'registry_auto_discovery': {},
            'contrib_directory_structure': {},
            'plugin_system_validation': {},
            'extension_point_testing': {}
        }
        
        try:
            from audit.tests.test_extensions import ExtensionsTester
            
            logger.info("Running Extension Mechanism Tests...")
            extensions_tester = ExtensionsTester()
            phase3_results = extensions_tester.run_all_tests()
            
        except Exception as e:
            logger.error(f"Phase 3 execution failed: {e}")
            phase3_results['error'] = str(e)
        
        return phase3_results
    
    def _generate_summary(self):
        """Generate overall audit summary."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        issues = []
        
        # Count results from all phases
        for phase_name, phase_results in self.results.items():
            if phase_name in ['phase1', 'phase2', 'phase3']:
                for category, category_results in phase_results.items():
                    if isinstance(category_results, dict):
                        for test, result in category_results.items():
                            if isinstance(result, dict) and 'passed' in result:
                                total_tests += 1
                                if result['passed']:
                                    passed_tests += 1
                                else:
                                    failed_tests += 1
                                    if 'issues' in result:
                                        issues.extend(result['issues'])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate_percent': success_rate,
            'critical_issues': len([i for i in issues if i.get('severity', '') == 'critical']),
            'total_issues': len(issues),
            'overall_status': 'PASS' if success_rate >= 90 else 'FAIL'
        }
        
        self.issues_found = issues
    
    def _save_results(self):
        """Save detailed results to JSON file."""
        results_file = 'audit/reports/test_results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Detailed results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _generate_reports(self):
        """Generate human-readable audit reports."""
        try:
            from utils.report_generator import ReportGenerator
            
            generator = ReportGenerator(self.results, self.issues_found)
            generator.generate_audit_report()
            generator.generate_recommendations()
            
            logger.info("Reports generated successfully")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")

def main():
    """Main entry point for audit execution."""
    print("🔍 Brainsmith Repository Audit")
    print("=" * 50)
    print("Focus Areas:")
    print("- Functional Completeness")
    print("- Integration Testing") 
    print("- Extension Mechanisms")
    print("=" * 50)
    
    # Create and run audit
    runner = AuditRunner()
    results = runner.run_full_audit()
    
    # Print summary
    summary = results.get('summary', {})
    print(f"\n📊 Audit Summary")
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"Passed: {summary.get('passed_tests', 0)}")
    print(f"Failed: {summary.get('failed_tests', 0)}")
    print(f"Success Rate: {summary.get('success_rate_percent', 0):.1f}%")
    print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    print(f"Execution Time: {results.get('execution_time', 0):.2f} seconds")
    
    if summary.get('total_issues', 0) > 0:
        print(f"\n⚠️ Issues Found: {summary.get('total_issues', 0)}")
        print(f"Critical Issues: {summary.get('critical_issues', 0)}")
    
    print(f"\n📋 Detailed reports available in audit/reports/")
    
    return 0 if summary.get('overall_status') == 'PASS' else 1

if __name__ == '__main__':
    sys.exit(main())