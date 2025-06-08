"""
Pipeline Orchestrator for Hardware Kernel Generator.

This module provides comprehensive pipeline orchestration capabilities for
coordinating multi-stage generation workflows with dependency management,
parallel execution, and error recovery.
"""

import time
import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Union, Tuple
from enum import Enum
from collections import defaultdict, deque
import logging
import weakref

from ..enhanced_config import PipelineConfig
from ..enhanced_data_structures import RTLModule
from ..enhanced_generator_base import GenerationResult, GeneratedArtifact
from ..enhanced_generator_base import GeneratorBase
from ..errors import BrainsmithError, PipelineError
from .generator_factory import GeneratorFactory, GeneratorConfiguration, GeneratorCapability


class PipelineStageStatus(Enum):
    """Status of pipeline stages."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    OPTIMIZED = "optimized"


class StageType(Enum):
    """Types of pipeline stages."""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    FINALIZATION = "finalization"
    CUSTOM = "custom"


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    stage_name: str
    status: PipelineStageStatus
    execution_time: float = 0.0
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[GeneratedArtifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def finalize(self) -> None:
        """Finalize the stage result."""
        self.end_time = time.time()
        if self.start_time:
            self.execution_time = self.end_time - self.start_time


@dataclass
class PipelineContext:
    """Context shared across pipeline stages."""
    rtl_module: RTLModule
    config: PipelineConfig
    generator_factory: GeneratorFactory
    shared_data: Dict[str, Any] = field(default_factory=dict)
    stage_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    global_artifacts: List[GeneratedArtifact] = field(default_factory=list)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set_stage_output(self, stage_name: str, key: str, value: Any) -> None:
        """Set output from a stage."""
        if stage_name not in self.stage_outputs:
            self.stage_outputs[stage_name] = {}
        self.stage_outputs[stage_name][key] = value
    
    def get_stage_output(self, stage_name: str, key: str, default: Any = None) -> Any:
        """Get output from a stage."""
        return self.stage_outputs.get(stage_name, {}).get(key, default)
    
    def has_stage_output(self, stage_name: str, key: str) -> bool:
        """Check if stage has specific output."""
        return stage_name in self.stage_outputs and key in self.stage_outputs[stage_name]


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""
    
    def __init__(
        self,
        name: str,
        stage_type: StageType = StageType.CUSTOM,
        dependencies: Set[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        skip_on_error: bool = False
    ):
        self.name = name
        self.stage_type = stage_type
        self.dependencies = dependencies or set()
        self.execution_mode = execution_mode
        self.timeout = timeout
        self.retry_count = retry_count
        self.skip_on_error = skip_on_error
        self.status = PipelineStageStatus.PENDING
        self._lock = threading.RLock()
        
        # Performance tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._failure_count = 0
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> StageResult:
        """Execute the pipeline stage."""
        pass
    
    def can_execute(self, context: PipelineContext, completed_stages: Set[str]) -> bool:
        """Check if stage can execute based on dependencies."""
        return self.dependencies.issubset(completed_stages)
    
    def prepare(self, context: PipelineContext) -> bool:
        """Prepare stage for execution. Return True if ready."""
        return True
    
    def cleanup(self, context: PipelineContext, result: StageResult) -> None:
        """Cleanup after stage execution."""
        pass
    
    def validate_inputs(self, context: PipelineContext) -> List[str]:
        """Validate stage inputs. Return list of errors."""
        return []
    
    def estimate_execution_time(self, context: PipelineContext) -> float:
        """Estimate execution time for this stage."""
        if self._execution_count > 0:
            return self._total_execution_time / self._execution_count
        return 60.0  # Default estimate
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this stage."""
        avg_time = (
            self._total_execution_time / self._execution_count 
            if self._execution_count > 0 else 0.0
        )
        failure_rate = (
            self._failure_count / self._execution_count 
            if self._execution_count > 0 else 0.0
        )
        
        return {
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_time,
            "failure_count": self._failure_count,
            "failure_rate": failure_rate,
            "current_status": self.status.value
        }
    
    def cleanup_resources(self) -> None:
        """Cleanup stage resources."""
        pass
    
    def _update_metrics(self, execution_time: float, success: bool) -> None:
        """Update performance metrics."""
        with self._lock:
            self._execution_count += 1
            self._total_execution_time += execution_time
            if not success:
                self._failure_count += 1


class GeneratorStage(PipelineStage):
    """Pipeline stage that wraps a generator."""
    
    def __init__(
        self,
        name: str,
        generator_config: GeneratorConfiguration,
        required_capabilities: Set[GeneratorCapability] = None,
        **kwargs
    ):
        super().__init__(name, StageType.GENERATION, **kwargs)
        self.generator_config = generator_config
        self.required_capabilities = required_capabilities or set()
        self._generator_cache: Optional[GeneratorBase] = None
    
    def execute(self, context: PipelineContext) -> StageResult:
        """Execute generator stage."""
        result = StageResult(self.name, PipelineStageStatus.RUNNING)
        
        try:
            # Get or create generator
            generator = self._get_generator(context)
            
            # Prepare generator inputs
            inputs = self._prepare_generator_inputs(context)
            
            # Execute generator
            generation_result = generator.generate(inputs)
            
            # Process results
            result.outputs["generation_result"] = generation_result
            result.artifacts.extend(generation_result.artifacts)
            result.metadata.update(generation_result.metadata)
            
            # Store outputs in context
            context.set_stage_output(self.name, "generation_result", generation_result)
            context.global_artifacts.extend(generation_result.artifacts)
            
            result.status = PipelineStageStatus.COMPLETED
            self._update_metrics(result.execution_time, True)
            
        except Exception as e:
            result.status = PipelineStageStatus.FAILED
            result.errors.append(str(e))
            self._update_metrics(result.execution_time, False)
            
            if not self.skip_on_error:
                raise PipelineError(f"Generator stage {self.name} failed: {e}")
        
        finally:
            result.finalize()
        
        return result
    
    def _get_generator(self, context: PipelineContext) -> GeneratorBase:
        """Get generator instance."""
        if self._generator_cache is None:
            self._generator_cache = context.generator_factory.create_generator(
                self.generator_config,
                context.rtl_module
            )
        return self._generator_cache
    
    def _prepare_generator_inputs(self, context: PipelineContext) -> Dict[str, Any]:
        """Prepare inputs for generator."""
        inputs = {
            "rtl_module": context.rtl_module,
            "config": context.config,
            "context": context
        }
        
        # Add dependency outputs
        for dep_stage in self.dependencies:
            if context.has_stage_output(dep_stage, "generation_result"):
                inputs[f"{dep_stage}_result"] = context.get_stage_output(
                    dep_stage, "generation_result"
                )
        
        return inputs


class ValidationStage(PipelineStage):
    """Pipeline stage for validation."""
    
    def __init__(
        self,
        name: str,
        validation_func: Callable[[PipelineContext], Dict[str, Any]],
        **kwargs
    ):
        super().__init__(name, StageType.VALIDATION, **kwargs)
        self.validation_func = validation_func
    
    def execute(self, context: PipelineContext) -> StageResult:
        """Execute validation stage."""
        result = StageResult(self.name, PipelineStageStatus.RUNNING)
        
        try:
            validation_results = self.validation_func(context)
            
            result.outputs["validation_results"] = validation_results
            result.metadata.update(validation_results.get("metadata", {}))
            
            # Check for validation errors
            errors = validation_results.get("errors", [])
            warnings = validation_results.get("warnings", [])
            
            result.errors.extend(errors)
            result.warnings.extend(warnings)
            
            # Determine status
            if errors:
                result.status = PipelineStageStatus.FAILED
                if not self.skip_on_error:
                    raise PipelineError(f"Validation failed: {errors}")
            else:
                result.status = PipelineStageStatus.COMPLETED
            
            # Store in context
            context.set_stage_output(self.name, "validation_results", validation_results)
            
            self._update_metrics(result.execution_time, result.status == PipelineStageStatus.COMPLETED)
            
        except Exception as e:
            result.status = PipelineStageStatus.FAILED
            result.errors.append(str(e))
            self._update_metrics(result.execution_time, False)
            
            if not self.skip_on_error:
                raise
        
        finally:
            result.finalize()
        
        return result


class StageCoordinator:
    """Coordinates execution of pipeline stages."""
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.HYBRID):
        self.execution_mode = execution_mode
        self._stage_graph: Dict[str, Set[str]] = defaultdict(set)  # dependencies
        self._reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # dependents
        self._coordination_stats = {
            "coordinations": 0,
            "parallel_executions": 0,
            "sequential_executions": 0,
            "optimization_time": 0.0
        }
    
    def add_stage_dependency(self, stage_name: str, dependency: str) -> None:
        """Add dependency relationship between stages."""
        self._stage_graph[stage_name].add(dependency)
        self._reverse_graph[dependency].add(stage_name)
    
    def get_execution_plan(self, stages: Dict[str, PipelineStage]) -> List[List[str]]:
        """Get optimized execution plan for stages."""
        start_time = time.time()
        
        try:
            if self.execution_mode == ExecutionMode.SEQUENTIAL:
                return self._get_sequential_plan(stages)
            elif self.execution_mode == ExecutionMode.PARALLEL:
                return self._get_parallel_plan(stages)
            elif self.execution_mode == ExecutionMode.HYBRID:
                return self._get_hybrid_plan(stages)
            else:  # OPTIMIZED
                return self._get_optimized_plan(stages)
        
        finally:
            self._coordination_stats["optimization_time"] += time.time() - start_time
            self._coordination_stats["coordinations"] += 1
    
    def _get_sequential_plan(self, stages: Dict[str, PipelineStage]) -> List[List[str]]:
        """Get sequential execution plan."""
        # Topological sort
        sorted_stages = self._topological_sort(stages.keys())
        return [[stage] for stage in sorted_stages]
    
    def _get_parallel_plan(self, stages: Dict[str, PipelineStage]) -> List[List[str]]:
        """Get parallel execution plan."""
        # Group stages by dependency levels
        levels = []
        remaining = set(stages.keys())
        
        while remaining:
            # Find stages with no dependencies in remaining set
            current_level = []
            for stage in list(remaining):
                dependencies = self._stage_graph[stage] & remaining
                if not dependencies:
                    current_level.append(stage)
            
            if not current_level:
                # Circular dependency detected
                raise PipelineError("Circular dependency detected in pipeline stages")
            
            levels.append(current_level)
            remaining -= set(current_level)
        
        return levels
    
    def _get_hybrid_plan(self, stages: Dict[str, PipelineStage]) -> List[List[str]]:
        """Get hybrid execution plan."""
        # Start with parallel plan
        parallel_plan = self._get_parallel_plan(stages)
        
        # Optimize by considering execution modes and performance
        optimized_plan = []
        for level in parallel_plan:
            if len(level) == 1:
                optimized_plan.append(level)
            else:
                # Group by execution preferences
                parallel_group = []
                sequential_groups = []
                
                for stage_name in level:
                    stage = stages[stage_name]
                    if stage.execution_mode in [ExecutionMode.PARALLEL, ExecutionMode.HYBRID]:
                        parallel_group.append(stage_name)
                    else:
                        sequential_groups.append([stage_name])
                
                # Add groups to plan
                if parallel_group:
                    optimized_plan.append(parallel_group)
                optimized_plan.extend(sequential_groups)
        
        return optimized_plan
    
    def _get_optimized_plan(self, stages: Dict[str, PipelineStage]) -> List[List[str]]:
        """Get performance-optimized execution plan."""
        # Start with hybrid plan
        base_plan = self._get_hybrid_plan(stages)
        
        # Further optimize based on estimated execution times and dependencies
        optimized_plan = []
        
        for level in base_plan:
            if len(level) <= 1:
                optimized_plan.append(level)
                continue
            
            # Sort by estimated execution time (longest first for better parallelization)
            context_dummy = PipelineContext(
                rtl_module=None,
                config=None,
                generator_factory=None
            )
            
            sorted_level = sorted(
                level,
                key=lambda s: stages[s].estimate_execution_time(context_dummy),
                reverse=True
            )
            optimized_plan.append(sorted_level)
        
        return optimized_plan
    
    def _topological_sort(self, stage_names: Set[str]) -> List[str]:
        """Perform topological sort of stages."""
        in_degree = defaultdict(int)
        for stage in stage_names:
            for dependency in self._stage_graph[stage]:
                if dependency in stage_names:
                    in_degree[stage] += 1
        
        queue = deque([stage for stage in stage_names if in_degree[stage] == 0])
        result = []
        
        while queue:
            stage = queue.popleft()
            result.append(stage)
            
            for dependent in self._reverse_graph[stage]:
                if dependent in stage_names:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        if len(result) != len(stage_names):
            raise PipelineError("Circular dependency detected in pipeline stages")
        
        return result
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return self._coordination_stats.copy()


class PipelineExecutor:
    """Executes pipeline stages according to execution plan."""
    
    def __init__(self, max_parallel_stages: int = 4):
        self.max_parallel_stages = max_parallel_stages
        self._execution_stats = {
            "executions": 0,
            "total_execution_time": 0.0,
            "parallel_stage_count": 0,
            "sequential_stage_count": 0,
            "failure_count": 0
        }
    
    async def execute_plan(
        self,
        execution_plan: List[List[str]],
        stages: Dict[str, PipelineStage],
        context: PipelineContext
    ) -> Dict[str, StageResult]:
        """Execute pipeline according to execution plan."""
        start_time = time.time()
        results = {}
        completed_stages = set()
        
        try:
            for level in execution_plan:
                if len(level) == 1:
                    # Sequential execution
                    stage_name = level[0]
                    result = await self._execute_stage(stages[stage_name], context)
                    results[stage_name] = result
                    
                    if result.status == PipelineStageStatus.COMPLETED:
                        completed_stages.add(stage_name)
                    
                    self._execution_stats["sequential_stage_count"] += 1
                
                else:
                    # Parallel execution
                    level_results = await self._execute_parallel_stages(
                        [stages[name] for name in level],
                        context
                    )
                    
                    for stage_name, result in level_results.items():
                        results[stage_name] = result
                        if result.status == PipelineStageStatus.COMPLETED:
                            completed_stages.add(stage_name)
                    
                    self._execution_stats["parallel_stage_count"] += len(level)
            
            self._execution_stats["executions"] += 1
            
        except Exception as e:
            self._execution_stats["failure_count"] += 1
            raise PipelineError(f"Pipeline execution failed: {e}")
        
        finally:
            self._execution_stats["total_execution_time"] += time.time() - start_time
        
        return results
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        context: PipelineContext
    ) -> StageResult:
        """Execute a single stage."""
        # Validate inputs
        validation_errors = stage.validate_inputs(context)
        if validation_errors:
            result = StageResult(stage.name, PipelineStageStatus.FAILED)
            result.errors.extend(validation_errors)
            result.finalize()
            return result
        
        # Prepare stage
        if not stage.prepare(context):
            result = StageResult(stage.name, PipelineStageStatus.FAILED)
            result.errors.append("Stage preparation failed")
            result.finalize()
            return result
        
        try:
            # Execute with timeout if specified
            if stage.timeout:
                result = await asyncio.wait_for(
                    self._run_stage_async(stage, context),
                    timeout=stage.timeout
                )
            else:
                result = await self._run_stage_async(stage, context)
            
            # Cleanup
            stage.cleanup(context, result)
            
            return result
            
        except asyncio.TimeoutError:
            result = StageResult(stage.name, PipelineStageStatus.FAILED)
            result.errors.append(f"Stage execution timed out after {stage.timeout}s")
            result.finalize()
            return result
        
        except Exception as e:
            result = StageResult(stage.name, PipelineStageStatus.FAILED)
            result.errors.append(str(e))
            result.finalize()
            return result
    
    async def _run_stage_async(
        self,
        stage: PipelineStage,
        context: PipelineContext
    ) -> StageResult:
        """Run stage asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, stage.execute, context)
    
    async def _execute_parallel_stages(
        self,
        stages: List[PipelineStage],
        context: PipelineContext
    ) -> Dict[str, StageResult]:
        """Execute stages in parallel."""
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_parallel_stages)
        
        async def execute_with_semaphore(stage):
            async with semaphore:
                return await self._execute_stage(stage, context)
        
        # Execute all stages concurrently
        tasks = [execute_with_semaphore(stage) for stage in stages]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        stage_results = {}
        for stage, result in zip(stages, results):
            if isinstance(result, Exception):
                error_result = StageResult(stage.name, PipelineStageStatus.FAILED)
                error_result.errors.append(str(result))
                error_result.finalize()
                stage_results[stage.name] = error_result
            else:
                stage_results[stage.name] = result
        
        return stage_results
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self._execution_stats.copy()
        if stats["executions"] > 0:
            stats["average_execution_time"] = (
                stats["total_execution_time"] / stats["executions"]
            )
            stats["failure_rate"] = stats["failure_count"] / stats["executions"]
        else:
            stats["average_execution_time"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats


class PipelineOrchestrator:
    """Main orchestrator for pipeline execution."""
    
    def __init__(
        self,
        config: PipelineConfig,
        generator_factory: GeneratorFactory,
        execution_mode: ExecutionMode = ExecutionMode.HYBRID
    ):
        self.config = config
        self.generator_factory = generator_factory
        self.execution_mode = execution_mode
        
        self.coordinator = StageCoordinator(execution_mode)
        self.executor = PipelineExecutor()
        
        self._stages: Dict[str, PipelineStage] = {}
        self._pipeline_stats = {
            "pipeline_executions": 0,
            "total_pipeline_time": 0.0,
            "successful_pipelines": 0,
            "failed_pipelines": 0
        }
    
    def add_stage(self, stage: PipelineStage) -> None:
        """Add stage to pipeline."""
        self._stages[stage.name] = stage
        
        # Add dependencies to coordinator
        for dependency in stage.dependencies:
            self.coordinator.add_stage_dependency(stage.name, dependency)
    
    def remove_stage(self, stage_name: str) -> None:
        """Remove stage from pipeline."""
        if stage_name in self._stages:
            del self._stages[stage_name]
    
    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """Get stage by name."""
        return self._stages.get(stage_name)
    
    async def execute_pipeline(
        self,
        rtl_module: RTLModule,
        initial_data: Dict[str, Any] = None
    ) -> Dict[str, StageResult]:
        """Execute the complete pipeline."""
        start_time = time.time()
        
        # Create pipeline context
        context = PipelineContext(
            rtl_module=rtl_module,
            config=self.config,
            generator_factory=self.generator_factory,
            shared_data=initial_data or {}
        )
        
        try:
            # Get execution plan
            execution_plan = self.coordinator.get_execution_plan(self._stages)
            
            # Execute pipeline
            results = await self.executor.execute_plan(execution_plan, self._stages, context)
            
            # Update statistics
            self._pipeline_stats["pipeline_executions"] += 1
            
            # Check if pipeline succeeded
            failed_stages = [
                name for name, result in results.items()
                if result.status == PipelineStageStatus.FAILED
            ]
            
            if failed_stages:
                self._pipeline_stats["failed_pipelines"] += 1
                raise PipelineError(f"Pipeline failed at stages: {failed_stages}")
            else:
                self._pipeline_stats["successful_pipelines"] += 1
            
            return results
            
        except Exception as e:
            self._pipeline_stats["failed_pipelines"] += 1
            raise PipelineError(f"Pipeline execution failed: {e}")
        
        finally:
            self._pipeline_stats["total_pipeline_time"] += time.time() - start_time
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = self._pipeline_stats.copy()
        
        # Add component statistics
        stats["coordinator_stats"] = self.coordinator.get_coordination_statistics()
        stats["executor_stats"] = self.executor.get_execution_statistics()
        
        # Add stage statistics
        stage_stats = {}
        for name, stage in self._stages.items():
            stage_stats[name] = stage.get_performance_metrics()
        stats["stage_stats"] = stage_stats
        
        # Calculate derived metrics
        if stats["pipeline_executions"] > 0:
            stats["average_pipeline_time"] = (
                stats["total_pipeline_time"] / stats["pipeline_executions"]
            )
            stats["success_rate"] = (
                stats["successful_pipelines"] / stats["pipeline_executions"]
            )
        else:
            stats["average_pipeline_time"] = 0.0
            stats["success_rate"] = 0.0
        
        return stats


# Factory functions and convenience methods

def create_pipeline_orchestrator(
    config: PipelineConfig,
    generator_factory: GeneratorFactory,
    execution_mode: ExecutionMode = ExecutionMode.HYBRID
) -> PipelineOrchestrator:
    """Create pipeline orchestrator with default configuration."""
    return PipelineOrchestrator(config, generator_factory, execution_mode)


async def execute_pipeline(
    orchestrator: PipelineOrchestrator,
    rtl_module: RTLModule,
    initial_data: Dict[str, Any] = None
) -> Dict[str, StageResult]:
    """Execute pipeline using orchestrator."""
    return await orchestrator.execute_pipeline(rtl_module, initial_data)


def create_generator_stage(
    name: str,
    generator_config: GeneratorConfiguration,
    dependencies: Set[str] = None,
    **kwargs
) -> GeneratorStage:
    """Create generator stage with configuration."""
    return GeneratorStage(
        name=name,
        generator_config=generator_config,
        dependencies=dependencies or set(),
        **kwargs
    )


def create_validation_stage(
    name: str,
    validation_func: Callable[[PipelineContext], Dict[str, Any]],
    dependencies: Set[str] = None,
    **kwargs
) -> ValidationStage:
    """Create validation stage with function."""
    return ValidationStage(
        name=name,
        validation_func=validation_func,
        dependencies=dependencies or set(),
        **kwargs
    )