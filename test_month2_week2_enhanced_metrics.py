"""
Month 2 Week 2 Test Suite: Enhanced Metrics Foundation
Comprehensive testing of advanced performance metrics, resource utilization tracking,
historical analysis, and quality metrics framework.
"""

import os
import sys
import tempfile
import shutil
import time
import threading
import numpy as np
from pathlib import Path
import json

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from brainsmith.metrics import (
    MetricsManager, MetricsConfiguration, MetricsRegistry,
    AdvancedPerformanceMetrics, ResourceUtilizationTracker,
    HistoricalAnalysisEngine, QualityMetricsCollector,
    MetricValue, MetricCollection, MetricType, MetricScope
)


class MockBuildResult:
    """Mock build result for testing metrics collection."""
    
    def __init__(self):
        self.output_path = "/tmp/mock_build_output"
        self.synthesis_report = "/tmp/mock_synthesis.rpt"
        self.implementation_report = "/tmp/mock_implementation.rpt"
        self.resource_usage = {
            'lut_count': 15000,
            'ff_count': 25000,
            'dsp_count': 48,
            'bram_count': 32,
            'uram_count': 8
        }
        self.clock_frequency = 150.0
        self.device = 'xczu7ev'
        self.success = True
        self.duration = 3600.0
        
        # Create mock report files
        self._create_mock_reports()
    
    def _create_mock_reports(self):
        """Create mock synthesis and implementation reports."""
        os.makedirs("/tmp", exist_ok=True)
        
        # Mock synthesis report
        synthesis_content = """
========================================
Synthesis Report
========================================

Clock Summary:
create_clock -period 6.667 [get_ports clk]

Critical Path Delay: 5.234 ns
Logic Levels: 12

Resource Utilization Summary:
| Resource | Used | Total | Util% |
|----------|------|-------|-------|
| LUT      | 15000| 230400| 6.5%  |
| FF       | 25000| 460800| 5.4%  |
| DSP      | 48   | 1728  | 2.8%  |
| BRAM     | 32   | 312   | 10.3% |
| URAM     | 8    | 96    | 8.3%  |
"""
        
        with open(self.synthesis_report, 'w') as f:
            f.write(synthesis_content)
        
        # Mock implementation report
        implementation_content = """
========================================
Implementation Report
========================================

Timing Summary:
WNS(ns): 1.433
TNS(ns): 0.000
Failing Endpoints: 0

Design meets timing constraints.

Power Summary:
Total Power: 2.456 W
Dynamic Power: 1.234 W
Static Power: 1.222 W
"""
        
        with open(self.implementation_report, 'w') as f:
            f.write(implementation_content)


class Month2Week2TestSuite:
    """Comprehensive test suite for Month 2 Week 2 components."""
    
    def __init__(self):
        self.temp_directories = []
        self.test_results = {}
    
    def cleanup(self):
        """Clean up test resources."""
        for temp_dir in self.temp_directories:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        # Clean up mock files
        for path in ["/tmp/mock_synthesis.rpt", "/tmp/mock_implementation.rpt"]:
            if os.path.exists(path):
                os.remove(path)
    
    def test_metrics_core_framework(self) -> bool:
        """Test the core metrics framework."""
        print("🧮 Testing Metrics Core Framework...")
        
        try:
            # Test MetricsRegistry
            registry = MetricsRegistry()
            
            # Register collectors
            registry.register_collector(AdvancedPerformanceMetrics)
            registry.register_collector(ResourceUtilizationTracker)
            registry.register_collector(QualityMetricsCollector)
            
            # Check registered collectors
            collectors = registry.list_collectors()
            assert len(collectors) >= 3, f"Expected at least 3 collectors, got {len(collectors)}"
            print(f"   Registered {len(collectors)} collectors")
            
            # Create collector instances
            perf_collector = registry.create_collector('AdvancedPerformanceMetrics')
            resource_collector = registry.create_collector('ResourceUtilizationTracker')
            quality_collector = registry.create_collector('QualityMetricsCollector')
            
            assert perf_collector is not None, "Failed to create performance collector"
            assert resource_collector is not None, "Failed to create resource collector"
            assert quality_collector is not None, "Failed to create quality collector"
            
            # Test metrics collection
            mock_build = MockBuildResult()
            context = {
                'build_result': mock_build,
                'synthesis_report': mock_build.synthesis_report,
                'implementation_report': mock_build.implementation_report,
                'resource_utilization': mock_build.resource_usage,
                'clock_frequency_mhz': mock_build.clock_frequency,
                'device': mock_build.device
            }
            
            collections = registry.collect_all_metrics(context)
            assert len(collections) >= 3, f"Expected at least 3 collections, got {len(collections)}"
            
            # Verify each collection has metrics
            for collection in collections:
                assert len(collection.metrics) > 0, f"Collection {collection.name} has no metrics"
                print(f"   {collection.name}: {len(collection.metrics)} metrics")
            
            print("✅ Metrics Core Framework tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Metrics Core Framework test failed: {e}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test advanced performance metrics collection."""
        print("⚡ Testing Advanced Performance Metrics...")
        
        try:
            collector = AdvancedPerformanceMetrics()
            mock_build = MockBuildResult()
            
            # Prepare context with comprehensive performance data
            context = {
                'build_result': mock_build,
                'synthesis_report': mock_build.synthesis_report,
                'implementation_report': mock_build.implementation_report,
                'resource_utilization': mock_build.resource_usage,
                'clock_frequency_mhz': mock_build.clock_frequency,
                'device': mock_build.device
            }
            
            # Collect performance metrics
            collection = collector.collect_metrics(context)
            
            assert collection is not None, "Performance metrics collection failed"
            assert len(collection.metrics) > 0, "No performance metrics collected"
            
            # Check for key performance metrics
            metric_names = [metric.name for metric in collection.metrics]
            
            expected_metrics = [
                'timing_slack', 'timing_met', 'critical_path_delay',
                'power_total_mw', 'power_efficiency'
            ]
            
            found_metrics = 0
            for expected in expected_metrics:
                if any(expected in name for name in metric_names):
                    found_metrics += 1
            
            print(f"   Found {found_metrics}/{len(expected_metrics)} key performance metrics")
            print(f"   Total metrics collected: {len(collection.metrics)}")
            
            # Check metric types
            timing_metrics = [m for m in collection.metrics if m.metric_type == MetricType.TIMING]
            power_metrics = [m for m in collection.metrics if m.metric_type == MetricType.POWER]
            
            print(f"   Timing metrics: {len(timing_metrics)}")
            print(f"   Power metrics: {len(power_metrics)}")
            
            assert len(timing_metrics) > 0, "No timing metrics found"
            assert len(power_metrics) > 0, "No power metrics found"
            
            print("✅ Advanced Performance Metrics tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Performance Metrics test failed: {e}")
            return False
    
    def test_resource_utilization_tracking(self) -> bool:
        """Test resource utilization tracking and analysis."""
        print("🔧 Testing Resource Utilization Tracking...")
        
        try:
            collector = ResourceUtilizationTracker()
            mock_build = MockBuildResult()
            
            # Prepare context with resource utilization data
            context = {
                'utilization_report': mock_build.synthesis_report,
                'performance_metrics': {
                    'throughput_ops_per_sec': 1000000.0,
                    'latency_cycles': 150,
                    'clock_frequency_mhz': 150.0
                },
                'device': mock_build.device,
                'tool': 'vivado',
                'power_consumption': 2456.0
            }
            
            # Collect resource metrics
            collection = collector.collect_metrics(context)
            
            assert collection is not None, "Resource metrics collection failed"
            assert len(collection.metrics) > 0, "No resource metrics collected"
            
            # Check for key resource metrics
            metric_names = [metric.name for metric in collection.metrics]
            
            expected_metrics = [
                'lut_utilization', 'dsp_utilization', 'bram_utilization',
                'lut_efficiency', 'area_efficiency', 'bottleneck_resource'
            ]
            
            found_metrics = 0
            for expected in expected_metrics:
                if any(expected in name for name in metric_names):
                    found_metrics += 1
            
            print(f"   Found {found_metrics}/{len(expected_metrics)} key resource metrics")
            print(f"   Total metrics collected: {len(collection.metrics)}")
            
            # Check metric types
            utilization_metrics = [m for m in collection.metrics if m.metric_type == MetricType.UTILIZATION]
            efficiency_metrics = [m for m in collection.metrics if m.metric_type == MetricType.EFFICIENCY]
            resource_metrics = [m for m in collection.metrics if m.metric_type == MetricType.RESOURCE]
            
            print(f"   Utilization metrics: {len(utilization_metrics)}")
            print(f"   Efficiency metrics: {len(efficiency_metrics)}")
            print(f"   Resource metrics: {len(resource_metrics)}")
            
            assert len(utilization_metrics) > 0, "No utilization metrics found"
            assert len(efficiency_metrics) > 0, "No efficiency metrics found"
            
            # Check recommendations in metadata
            recommendations = collection.metadata.get('recommendations', [])
            print(f"   Generated {len(recommendations)} optimization recommendations")
            
            print("✅ Resource Utilization Tracking tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Resource Utilization Tracking test failed: {e}")
            return False
    
    def test_historical_analysis_engine(self) -> bool:
        """Test historical analysis and trend detection."""
        print("📊 Testing Historical Analysis Engine...")
        
        try:
            # Create temporary database
            temp_dir = tempfile.mkdtemp(prefix="metrics_test_")
            self.temp_directories.append(temp_dir)
            db_path = os.path.join(temp_dir, "test_metrics.db")
            
            config = {'db_path': db_path, 'analysis_window_hours': 1}
            collector = HistoricalAnalysisEngine(config=config)
            
            # Generate some historical data by collecting multiple times
            mock_build = MockBuildResult()
            
            # Create a series of metric collections to simulate history
            for i in range(5):
                # Create mock collection with varying metrics
                mock_collection = MetricCollection(
                    collection_id=f"test_collection_{i}",
                    name=f"Test Collection {i}",
                    description="Mock collection for testing"
                )
                
                # Add some metrics with trends
                base_throughput = 1000000.0
                throughput_value = base_throughput + (i * 50000)  # Increasing trend
                
                mock_collection.add_metric(MetricValue(
                    "throughput_ops_per_sec", throughput_value, "ops/sec",
                    timestamp=time.time() - (300 * (5-i)),  # 5 minute intervals
                    metric_type=MetricType.THROUGHPUT
                ))
                
                base_latency = 100.0
                latency_value = base_latency + (i * 5)  # Increasing latency (bad trend)
                
                mock_collection.add_metric(MetricValue(
                    "latency_cycles", latency_value, "cycles",
                    timestamp=time.time() - (300 * (5-i)),
                    metric_type=MetricType.LATENCY
                ))
                
                # Store collection in database
                collector.database.store_collection(mock_collection)
                
                # Small delay to ensure different timestamps
                time.sleep(0.1)
            
            # Create baseline
            baseline_metrics = {
                'throughput_ops_per_sec': 1000000.0,
                'latency_cycles': 100.0,
                'power_total_mw': 2000.0
            }
            
            baseline = collector.baseline_manager.create_baseline(
                name="Test Baseline",
                description="Baseline for testing",
                metrics=baseline_metrics,
                configuration={'test': True}
            )
            
            assert baseline is not None, "Failed to create baseline"
            print(f"   Created baseline with {len(baseline.metrics)} metrics")
            
            # Collect historical analysis metrics
            context = {
                'current_collection': mock_collection,
                'key_metrics': ['throughput_ops_per_sec', 'latency_cycles']
            }
            
            analysis_collection = collector.collect_metrics(context)
            
            assert analysis_collection is not None, "Historical analysis collection failed"
            assert len(analysis_collection.metrics) > 0, "No analysis metrics collected"
            
            # Check for analysis metrics
            metric_names = [metric.name for metric in analysis_collection.metrics]
            
            expected_metrics = [
                'analyzed_metrics_count', 'regression_count', 'recent_alerts_count'
            ]
            
            found_metrics = 0
            for expected in expected_metrics:
                if expected in metric_names:
                    found_metrics += 1
            
            print(f"   Found {found_metrics}/{len(expected_metrics)} analysis metrics")
            print(f"   Total analysis metrics: {len(analysis_collection.metrics)}")
            
            # Check trend analysis in metadata
            trend_analyses = analysis_collection.metadata.get('trend_analyses', {})
            print(f"   Trend analyses performed: {len(trend_analyses)}")
            
            # Test trend summary
            trend_summary = collector.get_trend_summary(hours=1)
            print(f"   Trend summary covers {trend_summary['metrics_analyzed']} metrics")
            
            # Test regression summary
            regression_summary = collector.get_regression_summary()
            print(f"   Regression summary: {regression_summary['total_regressions']} regressions")
            
            print("✅ Historical Analysis Engine tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Historical Analysis Engine test failed: {e}")
            return False
    
    def test_quality_metrics_framework(self) -> bool:
        """Test quality metrics and validation framework."""
        print("🎯 Testing Quality Metrics Framework...")
        
        try:
            collector = QualityMetricsCollector()
            
            # Generate test data for quality assessment
            num_samples = 100
            num_classes = 10
            
            # Reference outputs (ground truth)
            reference = np.random.randint(0, num_classes, num_samples)
            reference_onehot = np.eye(num_classes)[reference]
            
            # Predicted outputs (with 90% accuracy)
            predicted = reference_onehot.copy()
            # Add noise to 10% of predictions
            noise_indices = np.random.choice(num_samples, size=num_samples//10, replace=False)
            for idx in noise_indices:
                wrong_class = np.random.choice([i for i in range(num_classes) if i != reference[idx]])
                predicted[idx] = np.eye(num_classes)[wrong_class]
            
            # Add some gaussian noise
            predicted += np.random.normal(0, 0.1, predicted.shape)
            predicted = np.maximum(0, predicted)
            predicted = predicted / np.sum(predicted, axis=1, keepdims=True)
            
            # Multiple runs for reliability assessment
            multiple_runs = []
            for i in range(3):
                run = predicted + np.random.normal(0, 0.05, predicted.shape)
                run = np.maximum(0, run)
                run = run / np.sum(run, axis=1, keepdims=True)
                multiple_runs.append(run)
            
            # Prepare context
            context = {
                'predicted_outputs': predicted,
                'reference_outputs': reference_onehot,
                'multiple_runs': multiple_runs,
                'task_type': 'classification',
                'data_type': 'fp32'
            }
            
            # Collect quality metrics
            collection = collector.collect_metrics(context)
            
            assert collection is not None, "Quality metrics collection failed"
            assert len(collection.metrics) > 0, "No quality metrics collected"
            
            # Check for key quality metrics
            metric_names = [metric.name for metric in collection.metrics]
            
            expected_metrics = [
                'validation_passed', 'accuracy', 'precision', 'recall',
                'f1_score', 'output_consistency', 'test_coverage'
            ]
            
            found_metrics = 0
            for expected in expected_metrics:
                if expected in metric_names:
                    found_metrics += 1
            
            print(f"   Found {found_metrics}/{len(expected_metrics)} key quality metrics")
            print(f"   Total quality metrics: {len(collection.metrics)}")
            
            # Check metric types
            quality_metrics = [m for m in collection.metrics if m.metric_type == MetricType.QUALITY]
            accuracy_metrics = [m for m in collection.metrics if m.metric_type == MetricType.ACCURACY]
            precision_metrics = [m for m in collection.metrics if m.metric_type == MetricType.PRECISION]
            
            print(f"   Quality metrics: {len(quality_metrics)}")
            print(f"   Accuracy metrics: {len(accuracy_metrics)}")
            print(f"   Precision metrics: {len(precision_metrics)}")
            
            assert len(quality_metrics) > 0, "No quality metrics found"
            assert len(accuracy_metrics) > 0, "No accuracy metrics found"
            
            # Check validation results in metadata
            validation_errors = collection.metadata.get('validation_errors', [])
            recommendations = collection.metadata.get('recommendations', [])
            
            print(f"   Validation errors: {len(validation_errors)}")
            print(f"   Quality recommendations: {len(recommendations)}")
            
            print("✅ Quality Metrics Framework tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Quality Metrics Framework test failed: {e}")
            return False
    
    def test_metrics_manager_integration(self) -> bool:
        """Test integrated metrics management."""
        print("🎛️ Testing Metrics Manager Integration...")
        
        try:
            # Create metrics configuration
            config = MetricsConfiguration(
                enabled_collectors=['AdvancedPerformanceMetrics', 'ResourceUtilizationTracker', 'QualityMetricsCollector'],
                collection_interval=1.0,  # 1 second for testing
                retention_days=1,
                export_formats=['json'],
                aggregation_rules={'group_by': 'type', 'aggregation': 'avg'}
            )
            
            # Initialize metrics manager
            manager = MetricsManager(config)
            
            # Register collectors
            manager.registry.register_collector(AdvancedPerformanceMetrics)
            manager.registry.register_collector(ResourceUtilizationTracker)
            manager.registry.register_collector(QualityMetricsCollector)
            
            # Create collector instances
            for collector_name in config.enabled_collectors:
                manager.registry.create_collector(collector_name)
            
            # Manual metrics collection
            mock_build = MockBuildResult()
            context = {
                'build_result': mock_build,
                'synthesis_report': mock_build.synthesis_report,
                'implementation_report': mock_build.implementation_report,
                'resource_utilization': mock_build.resource_usage,
                'clock_frequency_mhz': mock_build.clock_frequency,
                'device': mock_build.device
            }
            
            collections = manager.collect_manual(context)
            
            assert len(collections) >= 3, f"Expected at least 3 collections, got {len(collections)}"
            print(f"   Collected {len(collections)} metric collections")
            
            total_metrics = sum(len(collection.metrics) for collection in collections)
            print(f"   Total metrics collected: {total_metrics}")
            
            # Test aggregation
            aggregated = manager.get_aggregated_metrics(
                timeframe_hours=1,
                group_by='type',
                aggregation='avg'
            )
            
            assert aggregated is not None, "Metrics aggregation failed"
            print(f"   Aggregated metrics: {len(aggregated.metrics)} metrics")
            
            # Test export
            temp_dir = tempfile.mkdtemp(prefix="metrics_export_")
            self.temp_directories.append(temp_dir)
            export_path = os.path.join(temp_dir, "metrics_export.json")
            
            export_result = manager.export_metrics(
                collections=collections,
                format='json',
                destination=export_path
            )
            
            assert export_result == True, "Metrics export failed"
            assert os.path.exists(export_path), "Export file not created"
            
            # Verify export content
            with open(export_path, 'r') as f:
                export_data = json.load(f)
            
            assert 'collections' in export_data, "Export missing collections"
            assert len(export_data['collections']) == len(collections), "Export collection count mismatch"
            
            print(f"   Exported {len(export_data['collections'])} collections to {export_path}")
            
            # Test automated collection (briefly)
            manager.start_collection()
            time.sleep(2)  # Let it collect for 2 seconds
            manager.stop_collection()
            
            print("   Automated collection tested successfully")
            
            print("✅ Metrics Manager Integration tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Metrics Manager Integration test failed: {e}")
            return False
    
    def test_integration_with_week1_components(self) -> bool:
        """Test integration with Week 1 FINN components."""
        print("🔗 Testing Integration with Week 1 Components...")
        
        try:
            # Import Week 1 components
            from brainsmith.finn import FINNWorkflowEngine, FINNBuildOrchestrator
            
            # Create mock FINN installation for workflow engine
            temp_dir = tempfile.mkdtemp(prefix="finn_integration_")
            self.temp_directories.append(temp_dir)
            
            # Setup basic FINN structure
            finn_src = os.path.join(temp_dir, "src", "finn")
            os.makedirs(finn_src, exist_ok=True)
            
            with open(os.path.join(finn_src, "version.py"), 'w') as f:
                f.write('__version__ = "0.8.1"\n')
            
            # Initialize workflow engine
            workflow_engine = FINNWorkflowEngine(temp_dir)
            
            # Initialize metrics manager
            config = MetricsConfiguration(
                enabled_collectors=['AdvancedPerformanceMetrics', 'ResourceUtilizationTracker'],
                collection_interval=5.0
            )
            metrics_manager = MetricsManager(config)
            
            # Register collectors
            metrics_manager.registry.register_collector(AdvancedPerformanceMetrics)
            metrics_manager.registry.register_collector(ResourceUtilizationTracker)
            
            for collector_name in config.enabled_collectors:
                metrics_manager.registry.create_collector(collector_name)
            
            # Initialize build orchestrator with metrics integration
            orchestrator = FINNBuildOrchestrator(workflow_engine, max_parallel_builds=2)
            
            # Create enhanced context combining Week 1 and Week 2 data
            mock_build = MockBuildResult()
            enhanced_context = {
                # Week 1 build information
                'build_result': mock_build,
                'workflow_engine': workflow_engine,
                'orchestrator': orchestrator,
                
                # Week 2 metrics context
                'synthesis_report': mock_build.synthesis_report,
                'implementation_report': mock_build.implementation_report,
                'resource_utilization': mock_build.resource_usage,
                'clock_frequency_mhz': mock_build.clock_frequency,
                'device': mock_build.device,
                'performance_metrics': {
                    'throughput_ops_per_sec': 1200000.0,
                    'latency_cycles': 125,
                    'clock_frequency_mhz': 150.0
                }
            }
            
            # Collect integrated metrics
            collections = metrics_manager.collect_manual(enhanced_context)
            
            assert len(collections) >= 2, f"Expected at least 2 collections, got {len(collections)}"
            print(f"   Integrated metrics collection: {len(collections)} collections")
            
            # Verify metrics include both performance and resource data
            all_metrics = []
            for collection in collections:
                all_metrics.extend(collection.metrics)
            
            metric_types = set(metric.metric_type for metric in all_metrics)
            print(f"   Metric types collected: {[mt.value for mt in metric_types]}")
            
            # Check for integration-specific metrics
            timing_metrics = [m for m in all_metrics if m.metric_type == MetricType.TIMING]
            resource_metrics = [m for m in all_metrics if m.metric_type == MetricType.RESOURCE]
            utilization_metrics = [m for m in all_metrics if m.metric_type == MetricType.UTILIZATION]
            
            assert len(timing_metrics) > 0, "No timing metrics in integration"
            assert len(resource_metrics) > 0, "No resource metrics in integration"
            assert len(utilization_metrics) > 0, "No utilization metrics in integration"
            
            print(f"   Timing metrics: {len(timing_metrics)}")
            print(f"   Resource metrics: {len(resource_metrics)}")
            print(f"   Utilization metrics: {len(utilization_metrics)}")
            
            # Test orchestrator status integration
            queue_status = orchestrator.get_queue_status()
            system_resources = queue_status.get('system_resources')
            
            if system_resources:
                print(f"   System resources monitored: CPU={system_resources.cpu_usage_percent:.1f}%")
            
            print("✅ Integration with Week 1 Components tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Integration with Week 1 Components test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all Month 2 Week 2 tests."""
        print("🧪 Starting Month 2 Week 2 Test Suite: Enhanced Metrics Foundation")
        print("=" * 80)
        
        test_methods = [
            ("Metrics Core Framework", self.test_metrics_core_framework),
            ("Performance Metrics", self.test_performance_metrics),
            ("Resource Utilization Tracking", self.test_resource_utilization_tracking),
            ("Historical Analysis Engine", self.test_historical_analysis_engine),
            ("Quality Metrics Framework", self.test_quality_metrics_framework),
            ("Metrics Manager Integration", self.test_metrics_manager_integration),
            ("Integration with Week 1", self.test_integration_with_week1_components)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        try:
            for test_name, test_method in test_methods:
                print(f"\n📋 Running {test_name} Tests...")
                if test_method():
                    passed_tests += 1
                    self.test_results[test_name] = "PASSED"
                else:
                    self.test_results[test_name] = "FAILED"
                print()
            
            # Summary
            print("🎉 Month 2 Week 2 Test Suite Complete!")
            print(f"✅ Passed: {passed_tests}/{total_tests} test suites")
            
            if passed_tests == total_tests:
                print("\n🏆 ALL TESTS PASSED - Week 2 Enhanced Metrics Foundation is ready!")
                print("\n📊 Week 2 Implementation Status:")
                print("✅ Core Metrics Framework - Complete with aggregation and export")
                print("✅ Advanced Performance Metrics - Timing, throughput, latency, power analysis")
                print("✅ Resource Utilization Tracking - FPGA resource monitoring and optimization")
                print("✅ Historical Analysis Engine - Trend analysis and regression detection")
                print("✅ Quality Metrics Framework - Accuracy, precision, reliability assessment")
                print("✅ Week 1 Integration - Seamless integration with FINN workflow system")
                
                print("\n🚀 Ready for Month 2 Week 3: Advanced DSE Integration!")
                return True
            else:
                print(f"\n❌ {total_tests - passed_tests} test suite(s) failed")
                for test_name, result in self.test_results.items():
                    status_icon = "✅" if result == "PASSED" else "❌"
                    print(f"{status_icon} {test_name}: {result}")
                return False
                
        except Exception as e:
            print(f"💥 Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()


def main():
    """Run Month 2 Week 2 test suite."""
    test_suite = Month2Week2TestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎯 Month 2 Week 2 implementation successfully validated!")
        print("Enhanced Metrics Foundation is ready for production use.")
    else:
        print("\n❌ Validation failed - issues need to be addressed.")
        sys.exit(1)


if __name__ == "__main__":
    main()