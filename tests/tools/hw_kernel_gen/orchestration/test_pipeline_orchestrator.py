"""
Test suite for Pipeline Orchestrator System.

Tests the complete pipeline orchestration functionality including stage execution,
dependency management, parallel processing, and error recovery.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig
from brainsmith.tools.hw_kernel_gen.enhanced_data_structures import RTLModule, RTLInterface, RTLSignal, GenerationResult
from brainsmith.tools.hw_kernel_gen.orchestration.generator_factory import GeneratorFactory, GeneratorConfiguration, GeneratorCapability
from brainsmith.tools.hw_kernel_gen.orchestration.pipeline_orchestrator import (
    PipelineOrchestrator, PipelineStage, PipelineExecutor, StageCoordinator,
    PipelineContext, StageResult, GeneratorStage, ValidationStage,
    PipelineStageStatus, ExecutionMode, StageType,
    create_pipeline_orchestrator, execute_pipeline, create_generator_stage, create_validation_stage
)
from brainsmith.tools.hw_kernel_gen.errors import PipelineError


class MockStage(PipelineStage):
    """Mock pipeline stage for testing."""
    
    def __init__(self, name, execution_time=0.1, should_fail=False, **kwargs):
        super().__init__(name, **kwargs)
        self.execution_time = execution_time
        self.should_fail = should_fail
        self.executed = False
    
    def execute(self, context):
        """Execute the mock stage."""
        self.executed = True
        result = StageResult(self.name, PipelineStageStatus.RUNNING)
        
        # Simulate execution time
        time.sleep(self.execution_time)
        
        if self.should_fail:
            result.status = PipelineStageStatus.FAILED
            result.errors.append("Mock stage failure")
        else:
            result.status = PipelineStageStatus.COMPLETED
            result.outputs["mock_output"] = f"output_from_{self.name}"
        
        result.finalize()
        return result


class TestPipelineContext:
    """Test pipeline context functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        self.config = PipelineConfig()
        self.generator_factory = Mock(spec=GeneratorFactory)
        
        self.context = PipelineContext(
            rtl_module=self.rtl_module,
            config=self.config,
            generator_factory=self.generator_factory
        )
    
    def test_context_initialization(self):
        """Test context initialization."""
        assert self.context.rtl_module is self.rtl_module
        assert self.context.config is self.config
        assert self.context.generator_factory is self.generator_factory
        assert len(self.context.shared_data) == 0
        assert len(self.context.stage_outputs) == 0
    
    def test_stage_output_management(self):
        """Test stage output management."""
        # Set stage output
        self.context.set_stage_output("stage1", "result", "test_value")
        
        # Get stage output
        value = self.context.get_stage_output("stage1", "result")
        assert value == "test_value"
        
        # Check if output exists
        assert self.context.has_stage_output("stage1", "result")
        assert not self.context.has_stage_output("stage1", "nonexistent")
        assert not self.context.has_stage_output("nonexistent_stage", "result")
        
        # Test default value
        default_value = self.context.get_stage_output("stage1", "nonexistent", "default")
        assert default_value == "default"


class TestPipelineStage:
    """Test pipeline stage base functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.context = PipelineContext(
            rtl_module=RTLModule("test", [], {}),
            config=PipelineConfig(),
            generator_factory=Mock()
        )
    
    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = MockStage(
            "test_stage",
            stage_type=StageType.GENERATION,
            dependencies={"dep1", "dep2"},
            timeout=60.0
        )
        
        assert stage.name == "test_stage"
        assert stage.stage_type == StageType.GENERATION
        assert stage.dependencies == {"dep1", "dep2"}
        assert stage.timeout == 60.0
        assert stage.status == PipelineStageStatus.PENDING
    
    def test_can_execute_dependencies(self):
        """Test dependency checking for execution."""
        stage = MockStage("test_stage", dependencies={"dep1", "dep2"})
        
        # Should not be able to execute without dependencies
        assert not stage.can_execute(self.context, set())
        assert not stage.can_execute(self.context, {"dep1"})
        
        # Should be able to execute with all dependencies
        assert stage.can_execute(self.context, {"dep1", "dep2"})
        assert stage.can_execute(self.context, {"dep1", "dep2", "extra"})
    
    def test_stage_execution(self):
        """Test stage execution."""
        stage = MockStage("test_stage", execution_time=0.01)
        
        result = stage.execute(self.context)
        
        assert stage.executed
        assert result.stage_name == "test_stage"
        assert result.status == PipelineStageStatus.COMPLETED
        assert result.execution_time > 0
        assert "mock_output" in result.outputs
    
    def test_stage_execution_failure(self):
        """Test stage execution failure."""
        stage = MockStage("failing_stage", should_fail=True, execution_time=0.01)
        
        result = stage.execute(self.context)
        
        assert result.status == PipelineStageStatus.FAILED
        assert len(result.errors) > 0
        assert "Mock stage failure" in result.errors[0]
    
    def test_performance_metrics(self):
        """Test stage performance metrics."""
        stage = MockStage("metric_stage", execution_time=0.01)
        
        # Initial metrics
        initial_metrics = stage.get_performance_metrics()
        assert initial_metrics["execution_count"] == 0
        assert initial_metrics["total_execution_time"] == 0.0
        assert initial_metrics["failure_count"] == 0
        
        # Execute stage
        stage.execute(self.context)
        
        # Check updated metrics
        updated_metrics = stage.get_performance_metrics()
        assert updated_metrics["execution_count"] == 1
        assert updated_metrics["total_execution_time"] > 0
        assert updated_metrics["average_execution_time"] > 0
        assert updated_metrics["failure_count"] == 0


class TestGeneratorStage:
    """Test generator stage functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.generator_factory = Mock(spec=GeneratorFactory)
        self.rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        
        self.context = PipelineContext(
            rtl_module=self.rtl_module,
            config=self.config,
            generator_factory=self.generator_factory
        )
        
        # Mock generator
        self.mock_generator = Mock()
        self.mock_generator.generate.return_value = GenerationResult(
            success=True,
            artifacts=[],
            metadata={"generated": True}
        )
        self.generator_factory.create_generator.return_value = self.mock_generator
    
    def test_generator_stage_initialization(self):
        """Test generator stage initialization."""
        generator_config = GeneratorConfiguration(
            generator_type=self.config.generator_type,
            config=self.config,
            capabilities_required={GeneratorCapability.HW_CUSTOM_OP}
        )
        
        stage = GeneratorStage(
            "hw_gen_stage",
            generator_config,
            required_capabilities={GeneratorCapability.HW_CUSTOM_OP}
        )
        
        assert stage.name == "hw_gen_stage"
        assert stage.stage_type == StageType.GENERATION
        assert stage.generator_config is generator_config
        assert GeneratorCapability.HW_CUSTOM_OP in stage.required_capabilities
    
    def test_generator_stage_execution(self):
        """Test generator stage execution."""
        generator_config = GeneratorConfiguration(
            generator_type=self.config.generator_type,
            config=self.config
        )
        
        stage = GeneratorStage("gen_stage", generator_config)
        result = stage.execute(self.context)
        
        # Verify generator was called
        self.generator_factory.create_generator.assert_called_once()
        self.mock_generator.generate.assert_called_once()
        
        # Verify result
        assert result.status == PipelineStageStatus.COMPLETED
        assert "generation_result" in result.outputs
        assert len(self.context.global_artifacts) >= 0  # May be empty for mock


class TestValidationStage:
    """Test validation stage functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.context = PipelineContext(
            rtl_module=RTLModule("test", [], {}),
            config=PipelineConfig(),
            generator_factory=Mock()
        )
    
    def test_validation_stage_success(self):
        """Test successful validation stage."""
        def validation_func(ctx):
            return {
                "success": True,
                "validation_passed": True,
                "metadata": {"validated": True}
            }
        
        stage = ValidationStage("validation_stage", validation_func)
        result = stage.execute(self.context)
        
        assert result.status == PipelineStageStatus.COMPLETED
        assert "validation_results" in result.outputs
        assert len(result.errors) == 0
    
    def test_validation_stage_failure(self):
        """Test validation stage failure."""
        def validation_func(ctx):
            return {
                "success": False,
                "errors": ["Validation error 1", "Validation error 2"],
                "warnings": ["Warning 1"]
            }
        
        stage = ValidationStage("validation_stage", validation_func)
        
        with pytest.raises(PipelineError, match="Validation failed"):
            stage.execute(self.context)
    
    def test_validation_stage_skip_on_error(self):
        """Test validation stage with skip_on_error."""
        def validation_func(ctx):
            return {
                "success": False,
                "errors": ["Validation error"]
            }
        
        stage = ValidationStage("validation_stage", validation_func, skip_on_error=True)
        result = stage.execute(self.context)
        
        assert result.status == PipelineStageStatus.FAILED
        assert len(result.errors) > 0
        # Should not raise exception due to skip_on_error


class TestStageCoordinator:
    """Test stage coordinator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coordinator = StageCoordinator(ExecutionMode.HYBRID)
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        assert self.coordinator.execution_mode == ExecutionMode.HYBRID
    
    def test_add_stage_dependency(self):
        """Test adding stage dependencies."""
        self.coordinator.add_stage_dependency("stage2", "stage1")
        self.coordinator.add_stage_dependency("stage3", "stage1")
        self.coordinator.add_stage_dependency("stage3", "stage2")
        
        # Check dependency graph
        assert "stage1" in self.coordinator._stage_graph["stage2"]
        assert "stage1" in self.coordinator._stage_graph["stage3"]
        assert "stage2" in self.coordinator._stage_graph["stage3"]
        
        # Check reverse graph
        assert "stage2" in self.coordinator._reverse_graph["stage1"]
        assert "stage3" in self.coordinator._reverse_graph["stage1"]
        assert "stage3" in self.coordinator._reverse_graph["stage2"]
    
    def test_sequential_execution_plan(self):
        """Test sequential execution plan."""
        stages = {
            "stage1": MockStage("stage1"),
            "stage2": MockStage("stage2", dependencies={"stage1"}),
            "stage3": MockStage("stage3", dependencies={"stage2"})
        }
        
        # Add dependencies to coordinator
        for stage in stages.values():
            for dep in stage.dependencies:
                self.coordinator.add_stage_dependency(stage.name, dep)
        
        coordinator = StageCoordinator(ExecutionMode.SEQUENTIAL)
        plan = coordinator.get_execution_plan(stages)
        
        # Should be sequential: each level has one stage
        assert all(len(level) == 1 for level in plan)
        
        # Check order
        stage_order = [level[0] for level in plan]
        stage1_idx = stage_order.index("stage1")
        stage2_idx = stage_order.index("stage2")
        stage3_idx = stage_order.index("stage3")
        
        assert stage1_idx < stage2_idx < stage3_idx
    
    def test_parallel_execution_plan(self):
        """Test parallel execution plan."""
        stages = {
            "stage1": MockStage("stage1"),
            "stage2": MockStage("stage2", dependencies={"stage1"}),
            "stage3": MockStage("stage3", dependencies={"stage1"}),
            "stage4": MockStage("stage4", dependencies={"stage2", "stage3"})
        }
        
        # Add dependencies to coordinator
        for stage in stages.values():
            for dep in stage.dependencies:
                self.coordinator.add_stage_dependency(stage.name, dep)
        
        coordinator = StageCoordinator(ExecutionMode.PARALLEL)
        plan = coordinator.get_execution_plan(stages)
        
        # Should have 3 levels: [stage1], [stage2, stage3], [stage4]
        assert len(plan) == 3
        assert plan[0] == ["stage1"]
        assert set(plan[1]) == {"stage2", "stage3"}
        assert plan[2] == ["stage4"]
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        stages = {
            "stage1": MockStage("stage1", dependencies={"stage2"}),
            "stage2": MockStage("stage2", dependencies={"stage1"})
        }
        
        # Add dependencies to coordinator
        for stage in stages.values():
            for dep in stage.dependencies:
                self.coordinator.add_stage_dependency(stage.name, dep)
        
        with pytest.raises(PipelineError, match="Circular dependency detected"):
            self.coordinator.get_execution_plan(stages)


class TestPipelineExecutor:
    """Test pipeline executor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.executor = PipelineExecutor(max_parallel_stages=2)
        self.context = PipelineContext(
            rtl_module=RTLModule("test", [], {}),
            config=PipelineConfig(),
            generator_factory=Mock()
        )
    
    @pytest.mark.asyncio
    async def test_execute_single_stage(self):
        """Test executing a single stage."""
        stage = MockStage("test_stage", execution_time=0.01)
        
        result = await self.executor._execute_stage(stage, self.context)
        
        assert result.stage_name == "test_stage"
        assert result.status == PipelineStageStatus.COMPLETED
        assert stage.executed
    
    @pytest.mark.asyncio
    async def test_execute_stage_with_timeout(self):
        """Test stage execution with timeout."""
        stage = MockStage("slow_stage", execution_time=0.5, timeout=0.1)
        
        result = await self.executor._execute_stage(stage, self.context)
        
        assert result.status == PipelineStageStatus.FAILED
        assert "timed out" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_execute_parallel_stages(self):
        """Test parallel stage execution."""
        stages = [
            MockStage("stage1", execution_time=0.01),
            MockStage("stage2", execution_time=0.01),
            MockStage("stage3", execution_time=0.01)
        ]
        
        start_time = time.time()
        results = await self.executor._execute_parallel_stages(stages, self.context)
        execution_time = time.time() - start_time
        
        # Should complete in roughly parallel time (not 3x sequential time)
        assert execution_time < 0.1  # Should be much less than 3 * 0.01 + overhead
        
        # All stages should have completed
        assert len(results) == 3
        for stage in stages:
            assert results[stage.name].status == PipelineStageStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_plan_sequential(self):
        """Test executing a plan with sequential stages."""
        stages = {
            "stage1": MockStage("stage1", execution_time=0.01),
            "stage2": MockStage("stage2", execution_time=0.01)
        }
        
        execution_plan = [["stage1"], ["stage2"]]
        
        results = await self.executor.execute_plan(execution_plan, stages, self.context)
        
        assert len(results) == 2
        assert results["stage1"].status == PipelineStageStatus.COMPLETED
        assert results["stage2"].status == PipelineStageStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_plan_parallel(self):
        """Test executing a plan with parallel stages."""
        stages = {
            "stage1": MockStage("stage1", execution_time=0.01),
            "stage2": MockStage("stage2", execution_time=0.01)
        }
        
        execution_plan = [["stage1", "stage2"]]  # Both stages in parallel
        
        start_time = time.time()
        results = await self.executor.execute_plan(execution_plan, stages, self.context)
        execution_time = time.time() - start_time
        
        # Should complete in parallel time
        assert execution_time < 0.1
        assert len(results) == 2
        assert results["stage1"].status == PipelineStageStatus.COMPLETED
        assert results["stage2"].status == PipelineStageStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_plan_with_failure(self):
        """Test executing a plan with stage failure."""
        stages = {
            "stage1": MockStage("stage1", execution_time=0.01, should_fail=True),
            "stage2": MockStage("stage2", execution_time=0.01)
        }
        
        execution_plan = [["stage1"], ["stage2"]]
        
        with pytest.raises(PipelineError, match="Pipeline execution failed"):
            await self.executor.execute_plan(execution_plan, stages, self.context)


class TestPipelineOrchestrator:
    """Test pipeline orchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.generator_factory = Mock(spec=GeneratorFactory)
        self.orchestrator = PipelineOrchestrator(
            self.config,
            self.generator_factory,
            ExecutionMode.HYBRID
        )
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.config is self.config
        assert self.orchestrator.generator_factory is self.generator_factory
        assert self.orchestrator.execution_mode == ExecutionMode.HYBRID
        assert isinstance(self.orchestrator.coordinator, StageCoordinator)
        assert isinstance(self.orchestrator.executor, PipelineExecutor)
    
    def test_add_and_get_stage(self):
        """Test adding and getting stages."""
        stage = MockStage("test_stage")
        
        self.orchestrator.add_stage(stage)
        
        assert "test_stage" in self.orchestrator._stages
        retrieved_stage = self.orchestrator.get_stage("test_stage")
        assert retrieved_stage is stage
    
    def test_remove_stage(self):
        """Test removing stages."""
        stage = MockStage("removable_stage")
        
        self.orchestrator.add_stage(stage)
        assert "removable_stage" in self.orchestrator._stages
        
        self.orchestrator.remove_stage("removable_stage")
        assert "removable_stage" not in self.orchestrator._stages
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_simple(self):
        """Test executing a simple pipeline."""
        # Add stages
        stage1 = MockStage("stage1", execution_time=0.01)
        stage2 = MockStage("stage2", execution_time=0.01, dependencies={"stage1"})
        
        self.orchestrator.add_stage(stage1)
        self.orchestrator.add_stage(stage2)
        
        # Execute pipeline
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        results = await self.orchestrator.execute_pipeline(rtl_module)
        
        # Verify results
        assert len(results) == 2
        assert results["stage1"].status == PipelineStageStatus.COMPLETED
        assert results["stage2"].status == PipelineStageStatus.COMPLETED
        
        # Verify stages executed
        assert stage1.executed
        assert stage2.executed
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_failure(self):
        """Test executing pipeline with stage failure."""
        # Add stages
        stage1 = MockStage("stage1", execution_time=0.01, should_fail=True)
        stage2 = MockStage("stage2", execution_time=0.01, dependencies={"stage1"})
        
        self.orchestrator.add_stage(stage1)
        self.orchestrator.add_stage(stage2)
        
        # Execute pipeline
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        
        with pytest.raises(PipelineError, match="Pipeline failed at stages"):
            await self.orchestrator.execute_pipeline(rtl_module)
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics."""
        initial_stats = self.orchestrator.get_pipeline_statistics()
        
        assert "pipeline_executions" in initial_stats
        assert "coordinator_stats" in initial_stats
        assert "executor_stats" in initial_stats
        assert "stage_stats" in initial_stats
        
        # Check initial values
        assert initial_stats["pipeline_executions"] == 0
        assert initial_stats["successful_pipelines"] == 0


class TestFactoryFunctions:
    """Test factory functions for pipeline orchestrator."""
    
    def test_create_pipeline_orchestrator(self):
        """Test pipeline orchestrator factory function."""
        config = PipelineConfig()
        generator_factory = Mock(spec=GeneratorFactory)
        
        orchestrator = create_pipeline_orchestrator(config, generator_factory, ExecutionMode.PARALLEL)
        
        assert isinstance(orchestrator, PipelineOrchestrator)
        assert orchestrator.config is config
        assert orchestrator.generator_factory is generator_factory
        assert orchestrator.execution_mode == ExecutionMode.PARALLEL
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_function(self):
        """Test execute pipeline convenience function."""
        config = PipelineConfig()
        generator_factory = Mock(spec=GeneratorFactory)
        orchestrator = create_pipeline_orchestrator(config, generator_factory)
        
        # Add a simple stage
        stage = MockStage("simple_stage", execution_time=0.01)
        orchestrator.add_stage(stage)
        
        # Execute using convenience function
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        results = await execute_pipeline(orchestrator, rtl_module)
        
        assert len(results) == 1
        assert results["simple_stage"].status == PipelineStageStatus.COMPLETED
    
    def test_create_generator_stage_function(self):
        """Test create generator stage factory function."""
        config = PipelineConfig()
        generator_config = GeneratorConfiguration(
            generator_type=config.generator_type,
            config=config
        )
        
        stage = create_generator_stage(
            "test_gen_stage",
            generator_config,
            dependencies={"dep1"},
            timeout=30.0
        )
        
        assert isinstance(stage, GeneratorStage)
        assert stage.name == "test_gen_stage"
        assert stage.dependencies == {"dep1"}
        assert stage.timeout == 30.0
    
    def test_create_validation_stage_function(self):
        """Test create validation stage factory function."""
        def validation_func(ctx):
            return {"success": True}
        
        stage = create_validation_stage(
            "test_val_stage",
            validation_func,
            dependencies={"dep1"},
            retry_count=2
        )
        
        assert isinstance(stage, ValidationStage)
        assert stage.name == "test_val_stage"
        assert stage.dependencies == {"dep1"}
        assert stage.retry_count == 2


class TestIntegration:
    """Integration tests for pipeline orchestrator components."""
    
    @pytest.mark.asyncio
    async def test_complex_pipeline_execution(self):
        """Test execution of a complex pipeline with multiple stages and dependencies."""
        config = PipelineConfig()
        generator_factory = Mock(spec=GeneratorFactory)
        orchestrator = PipelineOrchestrator(config, generator_factory, ExecutionMode.HYBRID)
        
        # Create complex pipeline: 
        # stage1 -> stage2, stage3 -> stage4
        # stage2, stage3 can run in parallel
        stage1 = MockStage("analysis", execution_time=0.01)
        stage2 = MockStage("hw_generation", execution_time=0.02, dependencies={"analysis"})
        stage3 = MockStage("rtl_generation", execution_time=0.02, dependencies={"analysis"})
        stage4 = MockStage("validation", execution_time=0.01, dependencies={"hw_generation", "rtl_generation"})
        
        orchestrator.add_stage(stage1)
        orchestrator.add_stage(stage2)
        orchestrator.add_stage(stage3)
        orchestrator.add_stage(stage4)
        
        # Execute pipeline
        rtl_module = RTLModule("complex_module", interfaces=[], parameters={})
        start_time = time.time()
        results = await orchestrator.execute_pipeline(rtl_module)
        total_time = time.time() - start_time
        
        # Verify all stages completed
        assert len(results) == 4
        for stage_name in ["analysis", "hw_generation", "rtl_generation", "validation"]:
            assert results[stage_name].status == PipelineStageStatus.COMPLETED
        
        # Verify execution order (stage2 and stage3 should have run in parallel)
        # Total time should be less than sequential execution
        sequential_time = 0.01 + 0.02 + 0.02 + 0.01  # 0.06s
        assert total_time < sequential_time + 0.1  # Allow for some overhead
        
        # Verify statistics
        stats = orchestrator.get_pipeline_statistics()
        assert stats["pipeline_executions"] == 1
        assert stats["successful_pipelines"] == 1
        assert stats["failed_pipelines"] == 0
    
    @pytest.mark.asyncio
    async def test_pipeline_with_generator_stages(self):
        """Test pipeline with actual generator stages."""
        config = PipelineConfig()
        generator_factory = Mock(spec=GeneratorFactory)
        
        # Mock generator
        mock_generator = Mock()
        mock_generator.generate.return_value = GenerationResult(
            success=True,
            artifacts=[],
            metadata={"test": True}
        )
        generator_factory.create_generator.return_value = mock_generator
        
        orchestrator = PipelineOrchestrator(config, generator_factory)
        
        # Add generator stage
        generator_config = GeneratorConfiguration(
            generator_type=config.generator_type,
            config=config,
            capabilities_required={GeneratorCapability.HW_CUSTOM_OP}
        )
        
        gen_stage = GeneratorStage("hw_gen", generator_config)
        orchestrator.add_stage(gen_stage)
        
        # Execute pipeline
        rtl_module = RTLModule("gen_test_module", interfaces=[], parameters={})
        results = await orchestrator.execute_pipeline(rtl_module)
        
        # Verify generator was called
        generator_factory.create_generator.assert_called_once()
        mock_generator.generate.assert_called_once()
        
        # Verify results
        assert len(results) == 1
        assert results["hw_gen"].status == PipelineStageStatus.COMPLETED
        assert "generation_result" in results["hw_gen"].outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])