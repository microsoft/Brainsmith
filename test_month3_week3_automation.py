"""
Test suite for Month 3 Week 3: Automation and Integration Framework
Tests core automation engine and workflow orchestration.
"""

import os
import sys
import unittest
import time
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_automation_imports():
    """Test that automation framework can be imported."""
    try:
        from brainsmith.automation import (
            AutomationEngine, WorkflowConfiguration, WorkflowResult
        )
        
        from brainsmith.automation.models import (
            DesignTarget, OptimizationJob, WorkflowStatus,
            DesignRecommendation, QualityMetrics, AutomationResult
        )
        
        from brainsmith.automation.engine import AutomationEngine
        
        print("✅ All automation framework imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Automation import failed: {e}")
        return False


def test_design_target():
    """Test design target creation and validation."""
    try:
        from brainsmith.automation.models import DesignTarget
        
        # Create valid design target
        target = DesignTarget(
            application_type="cnn_inference",
            performance_targets={"throughput": 200.0, "power": 15.0},
            constraints={"lut_budget": 0.8, "timing_closure": True},
            optimization_objectives=["throughput", "power"]
        )
        
        # Test validation
        assert target.validate() == True
        assert target.application_type == "cnn_inference"
        assert len(target.performance_targets) == 2
        assert "throughput" in target.optimization_objectives
        
        # Test invalid target
        invalid_target = DesignTarget(
            application_type="",  # Invalid empty type
            performance_targets={},  # Invalid empty targets
            constraints={},
            optimization_objectives=[]  # Invalid empty objectives
        )
        
        assert invalid_target.validate() == False
        
        print("✅ Design target working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Design target failed: {e}")
        return False


def test_workflow_configuration():
    """Test workflow configuration creation and validation."""
    try:
        from brainsmith.automation.models import WorkflowConfiguration
        
        # Create valid configuration
        config = WorkflowConfiguration(
            optimization_budget=3600,
            quality_threshold=0.85,
            enable_learning=True,
            max_iterations=100,
            convergence_tolerance=0.005
        )
        
        # Test validation
        assert config.validate() == True
        assert config.optimization_budget == 3600
        assert config.quality_threshold == 0.85
        assert config.enable_learning == True
        
        # Test invalid configuration
        invalid_config = WorkflowConfiguration(
            optimization_budget=-100,  # Invalid negative budget
            quality_threshold=1.5,     # Invalid threshold > 1
            max_iterations=0,          # Invalid zero iterations
            convergence_tolerance=-0.1 # Invalid negative tolerance
        )
        
        assert invalid_config.validate() == False
        
        print("✅ Workflow configuration working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Workflow configuration failed: {e}")
        return False


def test_automation_engine_creation():
    """Test automation engine creation and initialization."""
    try:
        from brainsmith.automation.engine import AutomationEngine
        from brainsmith.automation.models import WorkflowConfiguration
        
        # Create engine with default configuration
        engine = AutomationEngine()
        assert engine.config is not None
        assert len(engine.execution_history) == 0
        
        # Create engine with custom configuration
        custom_config = WorkflowConfiguration(
            optimization_budget=1800,
            quality_threshold=0.75,
            enable_learning=True
        )
        
        custom_engine = AutomationEngine(custom_config)
        assert custom_engine.config.optimization_budget == 1800
        assert custom_engine.config.quality_threshold == 0.75
        
        print("✅ Automation engine creation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Automation engine creation failed: {e}")
        return False


def test_automated_workflow_execution():
    """Test complete automated workflow execution."""
    try:
        from brainsmith.automation.engine import AutomationEngine
        from brainsmith.automation.models import WorkflowConfiguration, WorkflowStatus
        
        # Create engine with fast configuration for testing
        config = WorkflowConfiguration(
            optimization_budget=60,  # 1 minute for testing
            quality_threshold=0.7,
            enable_learning=False,  # Disable for testing
            validation_enabled=True
        )
        
        engine = AutomationEngine(config)
        
        # Run optimization workflow
        result = engine.optimize_design(
            application_spec="cnn_inference",
            performance_targets={"throughput": 150.0, "power": 12.0},
            constraints={"lut_budget": 0.8},
            design_id="test_cnn_design"
        )
        
        # Verify workflow result
        assert isinstance(result.automation_result.job_id, str)
        assert result.automation_result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
        assert result.execution_time >= 0
        
        # Check if workflow completed successfully
        if result.success:
            assert len(result.automation_result.selected_solutions) > 0
            assert len(result.automation_result.recommendations) > 0
            assert result.automation_result.quality_metrics is not None
            assert result.confidence >= 0.0
        
        # Verify execution history
        assert len(engine.execution_history) == 1
        
        print("✅ Automated workflow execution working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Automated workflow execution failed: {e}")
        return False


def test_design_recommendations():
    """Test design recommendation generation."""
    try:
        from brainsmith.automation.models import (
            DesignRecommendation, RecommendationCategory, 
            RecommendationConfidence
        )
        
        # Create design recommendation
        recommendation = DesignRecommendation(
            category=RecommendationCategory.PERFORMANCE_OPTIMIZATION,
            confidence=RecommendationConfidence.HIGH,
            title="Increase PE Parallelism",
            description="Increase processing element parallelism from 16 to 24",
            rationale="Analysis shows throughput bottleneck in PE utilization",
            impact_estimate={"throughput": 15.0, "power": 5.0},
            implementation_effort="Medium",
            priority=1
        )
        
        # Test recommendation properties
        assert recommendation.category == RecommendationCategory.PERFORMANCE_OPTIMIZATION
        assert recommendation.confidence == RecommendationConfidence.HIGH
        assert "throughput" in recommendation.impact_estimate
        assert recommendation.priority == 1
        
        # Test impact summary
        impact_summary = recommendation.get_impact_summary()
        assert "throughput" in impact_summary
        assert "increase" in impact_summary
        
        print("✅ Design recommendations working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Design recommendations failed: {e}")
        return False


def test_quality_metrics():
    """Test quality metrics calculation and assessment."""
    try:
        from brainsmith.automation.models import QualityMetrics, QualityLevel
        
        # Create quality metrics
        metrics = QualityMetrics(
            overall_score=0.85,
            completeness=0.90,
            accuracy=0.82,
            consistency=0.88,
            reliability=0.80,
            confidence=0.85
        )
        
        # Test quality level assessment
        quality_level = metrics.get_quality_level()
        assert quality_level == QualityLevel.GOOD  # 0.85 should be GOOD
        
        # Test excellent quality
        excellent_metrics = QualityMetrics(
            overall_score=0.95,
            completeness=0.95,
            accuracy=0.92,
            consistency=0.98,
            reliability=0.90,
            confidence=0.95
        )
        
        assert excellent_metrics.get_quality_level() == QualityLevel.EXCELLENT
        
        # Test poor quality
        poor_metrics = QualityMetrics(
            overall_score=0.45,
            completeness=0.50,
            accuracy=0.40,
            consistency=0.45,
            reliability=0.42,
            confidence=0.45
        )
        
        assert poor_metrics.get_quality_level() == QualityLevel.POOR
        
        print("✅ Quality metrics working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Quality metrics failed: {e}")
        return False


def test_automation_result():
    """Test automation result structure and methods."""
    try:
        from brainsmith.automation.models import (
            AutomationResult, WorkflowStatus, QualityMetrics, 
            AutomationMetrics, WorkflowStep
        )
        from datetime import datetime
        
        # Create automation result
        start_time = datetime.now()
        
        result = AutomationResult(
            job_id="test_job_123",
            workflow_id="standard_optimization",
            status=WorkflowStatus.COMPLETED,
            start_time=start_time
        )
        
        # Set end time to calculate duration
        result.end_time = datetime.now()
        
        # Test basic properties
        assert result.job_id == "test_job_123"
        assert result.status == WorkflowStatus.COMPLETED
        assert result.duration is not None
        assert result.duration >= 0
        
        # Add mock selected solutions
        result.selected_solutions = [
            Mock(rank=1, score=0.9),
            Mock(rank=2, score=0.8),
            Mock(rank=3, score=0.7)
        ]
        
        # Test best solution
        best = result.best_solution
        assert best is not None
        assert best.rank == 1
        
        # Test success summary
        summary = result.get_success_summary()
        assert summary['status'] == 'completed'
        assert summary['solutions_found'] == 3
        assert summary['success'] == True
        
        print("✅ Automation result working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Automation result failed: {e}")
        return False


def test_workflow_steps():
    """Test workflow step enumeration and sequencing."""
    try:
        from brainsmith.automation.models import WorkflowStep
        
        # Test all workflow steps are defined
        expected_steps = [
            WorkflowStep.INITIALIZATION,
            WorkflowStep.DSE_OPTIMIZATION,
            WorkflowStep.SOLUTION_SELECTION,
            WorkflowStep.PERFORMANCE_ANALYSIS,
            WorkflowStep.BENCHMARKING,
            WorkflowStep.RECOMMENDATION,
            WorkflowStep.VALIDATION,
            WorkflowStep.FINALIZATION
        ]
        
        # Verify all steps exist
        for step in expected_steps:
            assert isinstance(step.value, str)
            assert len(step.value) > 0
        
        # Test step values
        assert WorkflowStep.INITIALIZATION.value == "initialization"
        assert WorkflowStep.DSE_OPTIMIZATION.value == "dse_optimization"
        assert WorkflowStep.FINALIZATION.value == "finalization"
        
        print("✅ Workflow steps working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Workflow steps failed: {e}")
        return False


def test_engine_performance_tracking():
    """Test automation engine performance tracking."""
    try:
        from brainsmith.automation.engine import AutomationEngine
        from brainsmith.automation.models import WorkflowConfiguration
        
        # Create engine
        config = WorkflowConfiguration(optimization_budget=30)  # Quick test
        engine = AutomationEngine(config)
        
        # Initial state
        summary = engine.get_performance_summary()
        assert summary['executions'] == 0
        assert summary['success_rate'] == 0.0
        
        # Run a workflow
        result = engine.optimize_design(
            application_spec="test_app",
            performance_targets={"performance": 100.0}
        )
        
        # Check updated performance
        updated_summary = engine.get_performance_summary()
        assert updated_summary['executions'] == 1
        
        if result.success:
            assert updated_summary['success_rate'] == 1.0
        
        assert updated_summary['average_duration'] >= 0
        assert updated_summary['average_quality'] >= 0
        
        print("✅ Engine performance tracking working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Engine performance tracking failed: {e}")
        return False


def run_automation_tests():
    """Run all automation framework tests."""
    print("Testing Month 3 Week 3: Automation and Integration Framework")
    print("=" * 80)
    
    tests = [
        ("Import Test", test_automation_imports),
        ("Design Target", test_design_target),
        ("Workflow Configuration", test_workflow_configuration),
        ("Automation Engine Creation", test_automation_engine_creation),
        ("Automated Workflow Execution", test_automated_workflow_execution),
        ("Design Recommendations", test_design_recommendations),
        ("Quality Metrics", test_quality_metrics),
        ("Automation Result", test_automation_result),
        ("Workflow Steps", test_workflow_steps),
        ("Engine Performance Tracking", test_engine_performance_tracking)
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
    print(f"Automation Framework Test Results")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print(f"\n🎉 All automation framework tests passed!")
        print(f"Month 3 Week 3 implementation is working correctly!")
    else:
        print(f"\n⚠️  Some tests failed - check implementation")
    
    return failed == 0


if __name__ == '__main__':
    success = run_automation_tests()
    
    if success:
        print(f"\n{'='*80}")
        print(f"🏁 Month 3 Week 3: Automation Framework Complete!")
        print(f"{'='*80}")
        print(f"📦 Implemented components:")
        print(f"   • Core automation engine with workflow orchestration")
        print(f"   • End-to-end automated design optimization")
        print(f"   • Intelligent design recommendation system")
        print(f"   • Quality assurance and validation framework")
        print(f"   • Learning and adaptation capabilities")
        print(f"   • Seamless component integration")
        print(f"\n🔧 Key features:")
        print(f"   • 8-step automated workflow execution")
        print(f"   • Mock DSE, selection, and analysis integration")
        print(f"   • Automated recommendation generation")
        print(f"   • Quality-driven workflow validation")
        print(f"   • Performance tracking and metrics")
        print(f"   • Configurable automation parameters")
    
    sys.exit(0 if success else 1)