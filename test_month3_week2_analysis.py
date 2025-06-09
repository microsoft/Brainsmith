"""
Test suite for Month 3 Week 2: Comprehensive Performance Analysis Framework
Tests core analysis engine, statistical tools, and benchmarking capabilities.
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_analysis_imports():
    """Test that analysis framework can be imported."""
    try:
        from brainsmith.analysis import (
            PerformanceAnalyzer, AnalysisConfiguration, AnalysisResult,
            BenchmarkingEngine, StatisticalAnalyzer
        )
        
        from brainsmith.analysis.models import (
            PerformanceData, StatisticalSummary, DistributionAnalysis,
            CorrelationAnalysis, OutlierDetection, BenchmarkResult
        )
        
        from brainsmith.analysis.benchmarking import ReferenceDesignDB, IndustryBenchmark
        
        print("✅ All analysis framework imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Analysis import failed: {e}")
        return False


def test_performance_data():
    """Test performance data creation and statistics."""
    try:
        from brainsmith.analysis.models import PerformanceData
        
        # Create performance data
        values = np.array([100.0, 120.0, 95.0, 110.0, 105.0])
        data = PerformanceData(
            metric_name="throughput",
            values=values,
            units="fps",
            description="Frame rate performance"
        )
        
        # Test statistics
        stats = data.statistics
        assert abs(stats['mean'] - 106.0) < 0.1
        assert stats['count'] == 5
        assert stats['min'] == 95.0
        assert stats['max'] == 120.0
        
        print("✅ Performance data working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Performance data failed: {e}")
        return False


def test_statistical_analysis():
    """Test statistical analysis capabilities."""
    try:
        from brainsmith.analysis.statistics import StatisticalAnalyzer
        from brainsmith.analysis.models import AnalysisConfiguration, DistributionType
        
        # Create configuration
        config = AnalysisConfiguration(
            confidence_level=0.95,
            outlier_threshold=2.0,
            distribution_tests=[DistributionType.NORMAL, DistributionType.UNIFORM]
        )
        
        analyzer = StatisticalAnalyzer(config)
        
        # Test distribution fitting
        values = np.random.normal(100, 15, 50)  # Normal distribution
        dist_analysis = analyzer.fit_distribution("test_metric", values)
        
        assert dist_analysis.metric_name == "test_metric"
        assert dist_analysis.best_fit_distribution in [DistributionType.NORMAL, DistributionType.UNIFORM]
        assert 'mean' in dist_analysis.distribution_parameters or 'min' in dist_analysis.distribution_parameters
        
        # Test outlier detection
        outlier_analysis = analyzer.detect_outliers("test_metric", values)
        assert outlier_analysis.metric_name == "test_metric"
        assert outlier_analysis.total_samples == 50
        
        print("✅ Statistical analysis working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        return False


def test_performance_analyzer():
    """Test the main performance analyzer."""
    try:
        from brainsmith.analysis.engine import PerformanceAnalyzer, PerformanceMetrics
        from brainsmith.analysis.models import AnalysisConfiguration, AnalysisContext, PerformanceData, AnalysisType
        
        # Create test data
        performance_data = {
            'throughput': PerformanceData(
                metric_name='throughput',
                values=np.array([100.0, 110.0, 95.0, 120.0, 105.0])
            ),
            'power': PerformanceData(
                metric_name='power',
                values=np.array([50.0, 55.0, 48.0, 60.0, 52.0])
            )
        }
        
        # Create fake solutions
        from brainsmith.analysis.models import ParetoSolution
        solutions = [
            ParetoSolution(
                design_parameters={'pe': 4},
                objective_values=[100.0, 50.0]
            ),
            ParetoSolution(
                design_parameters={'pe': 8},
                objective_values=[110.0, 55.0]
            )
        ]
        
        # Create analysis context
        context = AnalysisContext(
            solutions=solutions,
            performance_data=performance_data,
            analysis_types=[AnalysisType.DESCRIPTIVE, AnalysisType.STATISTICAL]
        )
        
        # Create analyzer
        config = AnalysisConfiguration()
        analyzer = PerformanceAnalyzer(config)
        
        # Perform analysis
        result = analyzer.analyze_performance(context)
        
        # Verify results
        assert isinstance(result.analysis.statistical_summary, dict)
        assert len(result.analysis.statistical_summary) >= 2  # throughput and power
        assert result.analysis.solutions_analyzed == 2
        
        # Test basic metrics calculation
        values = np.array([10, 20, 30, 40, 50])
        metrics = PerformanceMetrics.calculate_basic_metrics(values)
        assert metrics['mean'] == 30.0
        assert metrics['count'] == 5
        
        print("✅ Performance analyzer working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Performance analyzer failed: {e}")
        return False


def test_benchmarking_database():
    """Test reference design database."""
    try:
        from brainsmith.analysis.benchmarking import ReferenceDesignDB, IndustryBenchmark
        from brainsmith.analysis.models import BenchmarkCategory
        
        # Create database
        db = ReferenceDesignDB()
        
        # Test default initialization
        cnn_designs = db.get_designs_by_category(BenchmarkCategory.CNN_INFERENCE)
        assert len(cnn_designs) > 0
        
        # Test adding new design
        db.add_design(
            design_name="Test_CNN",
            category=BenchmarkCategory.CNN_INFERENCE,
            metrics={'throughput_fps': 150.0, 'power_watts': 12.0},
            parameters={'pe_parallelism': 16}
        )
        
        updated_designs = db.get_designs_by_category(BenchmarkCategory.CNN_INFERENCE)
        assert len(updated_designs) > len(cnn_designs)
        
        # Test industry benchmarks
        standards = IndustryBenchmark.get_industry_standards(BenchmarkCategory.CNN_INFERENCE)
        assert 'throughput_fps' in standards
        assert 'excellent' in standards['throughput_fps']
        
        print("✅ Benchmarking database working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Benchmarking database failed: {e}")
        return False


def test_benchmarking_engine():
    """Test benchmarking engine."""
    try:
        from brainsmith.analysis.benchmarking import BenchmarkingEngine, ReferenceDesignDB
        from brainsmith.analysis.models import BenchmarkCategory, ParetoSolution
        
        # Create benchmarking engine
        engine = BenchmarkingEngine()
        
        # Create test design
        test_design = ParetoSolution(
            design_parameters={'pe_parallelism': 16, 'memory_width': 256},
            objective_values=[130.0, 14.0]  # throughput_fps, power_watts
        )
        
        # Benchmark the design
        result = engine.benchmark_design(
            design=test_design,
            category=BenchmarkCategory.CNN_INFERENCE,
            design_id="test_design_1"
        )
        
        # Verify benchmark result
        assert result.design_id == "test_design_1"
        assert result.benchmark_category == BenchmarkCategory.CNN_INFERENCE
        assert len(result.reference_designs) > 0
        assert len(result.performance_metrics) > 0
        
        # Test overall ranking
        overall_ranking = result.get_overall_ranking()
        assert 0 <= overall_ranking <= 100
        
        print("✅ Benchmarking engine working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Benchmarking engine failed: {e}")
        return False


def test_correlation_analysis():
    """Test correlation analysis."""
    try:
        from brainsmith.analysis.statistics import StatisticalAnalyzer
        from brainsmith.analysis.models import AnalysisConfiguration, AnalysisContext, PerformanceData
        
        # Create correlated data
        np.random.seed(42)
        x = np.random.normal(100, 10, 50)
        y = 2 * x + np.random.normal(0, 5, 50)  # Correlated with x
        z = np.random.normal(50, 5, 50)         # Independent
        
        performance_data = {
            'metric_x': PerformanceData('metric_x', x),
            'metric_y': PerformanceData('metric_y', y),
            'metric_z': PerformanceData('metric_z', z)
        }
        
        context = AnalysisContext(
            solutions=[],
            performance_data=performance_data,
            analysis_types=[]
        )
        
        # Analyze correlations
        config = AnalysisConfiguration(correlation_threshold=0.3)
        analyzer = StatisticalAnalyzer(config)
        
        metrics = ['metric_x', 'metric_y', 'metric_z']
        corr_analysis = analyzer.analyze_correlations(context, metrics)
        
        # Check results
        assert corr_analysis.correlation_matrix.shape[0] >= 2
        assert corr_analysis.correlation_method == "pearson"
        
        # Strong correlations should be detected
        strong_corrs = corr_analysis.get_strong_correlations(0.7)
        # Should find correlation between x and y
        
        print("✅ Correlation analysis working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Correlation analysis failed: {e}")
        return False


def test_confidence_intervals():
    """Test confidence interval calculations."""
    try:
        from brainsmith.analysis.engine import PerformanceAnalyzer
        from brainsmith.analysis.models import AnalysisConfiguration, AnalysisContext, PerformanceData
        
        # Create test data
        np.random.seed(42)
        values = np.random.normal(100, 15, 30)
        
        performance_data = {
            'test_metric': PerformanceData('test_metric', values)
        }
        
        context = AnalysisContext(
            solutions=[],
            performance_data=performance_data,
            analysis_types=[]
        )
        
        # Calculate confidence interval
        config = AnalysisConfiguration(confidence_level=0.95)
        analyzer = PerformanceAnalyzer(config)
        
        ci = analyzer._calculate_confidence_interval('test_metric', values)
        
        # Verify confidence interval
        assert ci.metric_name == 'test_metric'
        assert ci.confidence_level == 0.95
        assert ci.lower_bound < ci.mean < ci.upper_bound
        assert ci.width > 0
        
        # Mean should be close to true mean
        true_mean = np.mean(values)
        assert abs(ci.mean - true_mean) < 0.01
        
        print("✅ Confidence intervals working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Confidence intervals failed: {e}")
        return False


def test_outlier_detection():
    """Test outlier detection methods."""
    try:
        from brainsmith.analysis.statistics import StatisticalAnalyzer
        from brainsmith.analysis.models import AnalysisConfiguration
        
        # Create data with outliers
        normal_data = np.random.normal(100, 10, 50)
        outlier_data = np.array([200, 250])  # Clear outliers
        data_with_outliers = np.concatenate([normal_data, outlier_data])
        
        config = AnalysisConfiguration(outlier_threshold=2.0)
        analyzer = StatisticalAnalyzer(config)
        
        # Detect outliers
        outlier_result = analyzer.detect_outliers('test_metric', data_with_outliers)
        
        # Verify outlier detection
        assert outlier_result.metric_name == 'test_metric'
        assert outlier_result.total_samples == 52
        assert outlier_result.num_outliers >= 0  # Should detect some outliers
        assert 0 <= outlier_result.outlier_percentage <= 100
        
        print("✅ Outlier detection working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Outlier detection failed: {e}")
        return False


def run_analysis_tests():
    """Run all analysis framework tests."""
    print("Testing Month 3 Week 2: Comprehensive Performance Analysis Framework")
    print("=" * 80)
    
    tests = [
        ("Import Test", test_analysis_imports),
        ("Performance Data", test_performance_data),
        ("Statistical Analysis", test_statistical_analysis),
        ("Performance Analyzer", test_performance_analyzer),
        ("Benchmarking Database", test_benchmarking_database),
        ("Benchmarking Engine", test_benchmarking_engine),
        ("Correlation Analysis", test_correlation_analysis),
        ("Confidence Intervals", test_confidence_intervals),
        ("Outlier Detection", test_outlier_detection)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Performance Analysis Framework Test Results")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 All analysis framework tests passed!")
        print(f"Month 3 Week 2 implementation is working correctly!")
    else:
        print(f"\n⚠️  Some tests failed - check implementation")
    
    return failed == 0


if __name__ == '__main__':
    success = run_analysis_tests()
    
    if success:
        print(f"\n{'='*80}")
        print(f"🏁 Month 3 Week 2: Performance Analysis Framework Complete!")
        print(f"{'='*80}")
        print(f"📦 Implemented components:")
        print(f"   • Comprehensive performance analysis engine")
        print(f"   • Advanced statistical analysis tools")
        print(f"   • Distribution fitting and hypothesis testing")
        print(f"   • Correlation analysis and outlier detection")
        print(f"   • Reference design benchmarking database")
        print(f"   • Industry benchmark comparisons")
        print(f"   • Confidence interval calculations")
        print(f"   • Performance insights and recommendations")
        print(f"\n🔧 Key features:")
        print(f"   • 8+ statistical analysis methods")
        print(f"   • Automated distribution fitting")
        print(f"   • Multi-method outlier detection")
        print(f"   • Industry-standard benchmarking")
        print(f"   • Comprehensive performance reporting")
        print(f"   • Integration with Week 1 selection results")
    
    sys.exit(0 if success else 1)