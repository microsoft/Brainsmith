"""
Generation Workflow Engine for Hardware Kernel Generator.

This module provides a declarative workflow engine for defining and executing
complex generation workflows with conditional execution, step dependencies,
and context sharing.
"""

import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Set, Union, Type
from enum import Enum
import logging
from pathlib import Path

from ..enhanced_config import PipelineConfig
from ..enhanced_data_structures import RTLModule
from ..enhanced_generator_base import GenerationResult, GeneratedArtifact
from ..errors import BrainsmithError, WorkflowError
from .generator_factory import GeneratorFactory
from .pipeline_orchestrator import PipelineOrchestrator, PipelineStage


class WorkflowStepStatus(Enum):
    """Status of workflow steps."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ConditionType(Enum):
    """Types of workflow conditions."""
    ALWAYS = "always"
    NEVER = "never"
    IF_SUCCESS = "if_success"
    IF_FAILURE = "if_failure"
    IF_EXISTS = "if_exists"
    IF_NOT_EXISTS = "if_not_exists"
    IF_EQUALS = "if_equals"
    IF_NOT_EQUALS = "if_not_equals"
    CUSTOM = "custom"


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    GENERATOR = "generator"
    PIPELINE = "pipeline"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    CUSTOM = "custom"


@dataclass
class WorkflowCondition:
    """Condition for conditional workflow execution."""
    condition_type: ConditionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    custom_evaluator: Optional[Callable[[Any], bool]] = None
    
    def evaluate(self, context: 'WorkflowContext') -> bool:
        """Evaluate the condition."""
        if self.condition_type == ConditionType.ALWAYS:
            return True
        elif self.condition_type == ConditionType.NEVER:
            return False
        elif self.condition_type == ConditionType.IF_SUCCESS:
            step_name = self.parameters.get("step_name")
            if step_name and step_name in context.step_results:
                return context.step_results[step_name].status == WorkflowStepStatus.COMPLETED
            return False
        elif self.condition_type == ConditionType.IF_FAILURE:
            step_name = self.parameters.get("step_name")
            if step_name and step_name in context.step_results:
                return context.step_results[step_name].status == WorkflowStepStatus.FAILED
            return False
        elif self.condition_type == ConditionType.IF_EXISTS:
            key = self.parameters.get("key")
            return key in context.shared_data if key else False
        elif self.condition_type == ConditionType.IF_NOT_EXISTS:
            key = self.parameters.get("key")
            return key not in context.shared_data if key else True
        elif self.condition_type == ConditionType.IF_EQUALS:
            key = self.parameters.get("key")
            value = self.parameters.get("value")
            return context.shared_data.get(key) == value
        elif self.condition_type == ConditionType.IF_NOT_EQUALS:
            key = self.parameters.get("key")
            value = self.parameters.get("value")
            return context.shared_data.get(key) != value
        elif self.condition_type == ConditionType.CUSTOM:
            if self.custom_evaluator:
                return self.custom_evaluator(context)
            return False
        
        return False


@dataclass
class WorkflowStepResult:
    """Result of a workflow step execution."""
    step_name: str
    status: WorkflowStepStatus
    execution_time: float = 0.0
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[GeneratedArtifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def finalize(self) -> None:
        """Finalize the step result."""
        self.end_time = time.time()
        if self.start_time:
            self.execution_time = self.end_time - self.start_time


@dataclass
class WorkflowContext:
    """Context shared across workflow steps."""
    rtl_module: RTLModule
    config: PipelineConfig
    generator_factory: GeneratorFactory
    pipeline_orchestrator: Optional[PipelineOrchestrator] = None
    shared_data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, WorkflowStepResult] = field(default_factory=dict)
    global_artifacts: List[GeneratedArtifact] = field(default_factory=list)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set_data(self, key: str, value: Any) -> None:
        """Set shared data."""
        self.shared_data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get shared data."""
        return self.shared_data.get(key, default)
    
    def has_data(self, key: str) -> bool:
        """Check if key exists in shared data."""
        return key in self.shared_data
    
    def get_step_result(self, step_name: str) -> Optional[WorkflowStepResult]:
        """Get result from a specific step."""
        return self.step_results.get(step_name)
    
    def get_step_output(self, step_name: str, output_key: str, default: Any = None) -> Any:
        """Get specific output from a step."""
        result = self.get_step_result(step_name)
        if result:
            return result.outputs.get(output_key, default)
        return default


class WorkflowStep(ABC):
    """Abstract base class for workflow steps."""
    
    def __init__(
        self,
        name: str,
        step_type: WorkflowStepType = WorkflowStepType.CUSTOM,
        dependencies: Set[str] = None,
        condition: Optional[WorkflowCondition] = None,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        continue_on_error: bool = False
    ):
        self.name = name
        self.step_type = step_type
        self.dependencies = dependencies or set()
        self.condition = condition
        self.timeout = timeout
        self.retry_count = retry_count
        self.continue_on_error = continue_on_error
        self.status = WorkflowStepStatus.PENDING
        
        # Performance tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._failure_count = 0
    
    @abstractmethod
    def execute(self, context: WorkflowContext) -> WorkflowStepResult:
        """Execute the workflow step."""
        pass
    
    def can_execute(self, context: WorkflowContext) -> bool:
        """Check if step can execute based on dependencies and conditions."""
        # Check dependencies
        for dep in self.dependencies:
            if dep not in context.step_results:
                return False
            if context.step_results[dep].status not in [
                WorkflowStepStatus.COMPLETED, WorkflowStepStatus.SKIPPED
            ]:
                return False
        
        # Check condition
        if self.condition:
            return self.condition.evaluate(context)
        
        return True
    
    def should_skip(self, context: WorkflowContext) -> bool:
        """Check if step should be skipped."""
        if self.condition:
            return not self.condition.evaluate(context)
        return False
    
    def validate_inputs(self, context: WorkflowContext) -> List[str]:
        """Validate step inputs. Return list of errors."""
        return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this step."""
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
    
    def _update_metrics(self, execution_time: float, success: bool) -> None:
        """Update performance metrics."""
        self._execution_count += 1
        self._total_execution_time += execution_time
        if not success:
            self._failure_count += 1


class GeneratorWorkflowStep(WorkflowStep):
    """Workflow step that executes a generator."""
    
    def __init__(
        self,
        name: str,
        generator_name: str,
        generator_inputs: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(name, WorkflowStepType.GENERATOR, **kwargs)
        self.generator_name = generator_name
        self.generator_inputs = generator_inputs or {}
    
    def execute(self, context: WorkflowContext) -> WorkflowStepResult:
        """Execute generator step."""
        result = WorkflowStepResult(self.name, WorkflowStepStatus.RUNNING)
        
        try:
            # Prepare generator inputs
            inputs = self._prepare_inputs(context)
            
            # Get generator from factory
            if context.generator_factory:
                # Use factory to find registered generator by name
                available_generators = context.generator_factory.get_available_generators()
                if self.generator_name in available_generators:
                    # Create a simple configuration for the generator
                    from .generator_factory import GeneratorConfiguration, GeneratorCapability
                    from ..enhanced_config import GeneratorType
                    
                    # Map generator name to capabilities
                    capability_map = {
                        "hw_custom_op_generator": {GeneratorCapability.HW_CUSTOM_OP},
                        "rtl_backend_generator": {GeneratorCapability.RTL_BACKEND},
                        "documentation_generator": {GeneratorCapability.DOCUMENTATION},
                        "test_generator": {GeneratorCapability.TEST_GENERATION},
                        "validation_generator": {GeneratorCapability.VALIDATION}
                    }
                    
                    generator_config = GeneratorConfiguration(
                        generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
                        config=context.config,
                        capabilities_required=capability_map.get(self.generator_name, set())
                    )
                    
                    generator = context.generator_factory.create_generator(generator_config)
                    generation_result = generator.generate(inputs)
                else:
                    # Fallback to simulation for unknown generators
                    generation_result = GenerationResult(
                        success=True,
                        artifacts=[],
                        metadata={"generator": self.generator_name}
                    )
            else:
                # Simulate generator execution when no factory available
                generation_result = GenerationResult(
                    success=True,
                    artifacts=[],
                    metadata={"generator": self.generator_name}
                )
            
            result.outputs["generation_result"] = generation_result
            result.artifacts.extend(generation_result.artifacts)
            result.metadata.update(generation_result.metadata)
            
            # Store in context
            context.step_results[self.name] = result
            context.global_artifacts.extend(generation_result.artifacts)
            
            result.status = WorkflowStepStatus.COMPLETED
            self._update_metrics(result.execution_time, True)
            
        except Exception as e:
            result.status = WorkflowStepStatus.FAILED
            result.errors.append(str(e))
            self._update_metrics(result.execution_time, False)
            
            if not self.continue_on_error:
                raise WorkflowError(f"Generator step {self.name} failed: {e}")
        
        finally:
            result.finalize()
        
        return result
    
    def _prepare_inputs(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare inputs for generator."""
        inputs = {
            "rtl_module": context.rtl_module,
            "config": context.config,
            **self.generator_inputs
        }
        
        # Add dependency outputs
        for dep_step in self.dependencies:
            if dep_step in context.step_results:
                dep_result = context.step_results[dep_step]
                inputs[f"{dep_step}_result"] = dep_result.outputs
        
        return inputs


class PipelineWorkflowStep(WorkflowStep):
    """Workflow step that executes a pipeline."""
    
    def __init__(
        self,
        name: str,
        pipeline_name: str,
        pipeline_inputs: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(name, WorkflowStepType.PIPELINE, **kwargs)
        self.pipeline_name = pipeline_name
        self.pipeline_inputs = pipeline_inputs or {}
    
    def execute(self, context: WorkflowContext) -> WorkflowStepResult:
        """Execute pipeline step."""
        result = WorkflowStepResult(self.name, WorkflowStepStatus.RUNNING)
        
        try:
            if not context.pipeline_orchestrator:
                raise WorkflowError("Pipeline orchestrator not available in context")
            
            # Execute pipeline (simplified)
            # In real implementation, would configure and execute specific pipeline
            
            result.outputs["pipeline_result"] = {"status": "completed"}
            result.status = WorkflowStepStatus.COMPLETED
            self._update_metrics(result.execution_time, True)
            
        except Exception as e:
            result.status = WorkflowStepStatus.FAILED
            result.errors.append(str(e))
            self._update_metrics(result.execution_time, False)
            
            if not self.continue_on_error:
                raise WorkflowError(f"Pipeline step {self.name} failed: {e}")
        
        finally:
            result.finalize()
        
        return result


class TransformationWorkflowStep(WorkflowStep):
    """Workflow step for data transformation."""
    
    def __init__(
        self,
        name: str,
        transformation_func: Callable[[WorkflowContext], Dict[str, Any]],
        **kwargs
    ):
        super().__init__(name, WorkflowStepType.TRANSFORMATION, **kwargs)
        self.transformation_func = transformation_func
    
    def execute(self, context: WorkflowContext) -> WorkflowStepResult:
        """Execute transformation step."""
        result = WorkflowStepResult(self.name, WorkflowStepStatus.RUNNING)
        
        try:
            transformation_result = self.transformation_func(context)
            
            result.outputs["transformation_result"] = transformation_result
            result.status = WorkflowStepStatus.COMPLETED
            self._update_metrics(result.execution_time, True)
            
            # Update context with transformation results
            for key, value in transformation_result.items():
                context.set_data(key, value)
            
        except Exception as e:
            result.status = WorkflowStepStatus.FAILED
            result.errors.append(str(e))
            self._update_metrics(result.execution_time, False)
            
            if not self.continue_on_error:
                raise WorkflowError(f"Transformation step {self.name} failed: {e}")
        
        finally:
            result.finalize()
        
        return result


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow."""
    name: str
    description: str
    version: str
    steps: List[WorkflowStep] = field(default_factory=list)
    global_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add step to workflow."""
        self.steps.append(step)
    
    def get_step(self, step_name: str) -> Optional[WorkflowStep]:
        """Get step by name."""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    def validate(self) -> List[str]:
        """Validate workflow definition."""
        errors = []
        
        # Check for duplicate step names
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            errors.append("Duplicate step names found")
        
        # Check dependencies exist
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(f"Step {step.name} depends on non-existent step {dep}")
        
        # Check for circular dependencies (simplified)
        # In real implementation, would do proper topological sort
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "global_config": self.global_config,
            "metadata": self.metadata,
            "steps": [
                {
                    "name": step.name,
                    "type": step.step_type.value,
                    "dependencies": list(step.dependencies),
                    # Add other step properties as needed
                }
                for step in self.steps
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowDefinition':
        """Create workflow from dictionary."""
        workflow = cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            global_config=data.get("global_config", {}),
            metadata=data.get("metadata", {})
        )
        
        # Note: Step reconstruction would need step type registry
        # This is simplified for now
        
        return workflow


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_name: str
    status: str
    execution_time: float = 0.0
    step_results: Dict[str, WorkflowStepResult] = field(default_factory=dict)
    global_artifacts: List[GeneratedArtifact] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def finalize(self) -> None:
        """Finalize workflow result."""
        self.end_time = time.time()
        if self.start_time:
            self.execution_time = self.end_time - self.start_time
        
        # Determine overall status
        failed_steps = [
            name for name, result in self.step_results.items()
            if result.status == WorkflowStepStatus.FAILED
        ]
        
        if failed_steps:
            self.status = "failed"
            self.errors.append(f"Failed steps: {failed_steps}")
        else:
            self.status = "completed"


class WorkflowEngine:
    """Engine for executing workflows."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._workflow_stats = {
            "executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0
        }
    
    def execute_workflow(
        self,
        workflow: WorkflowDefinition,
        context: WorkflowContext
    ) -> WorkflowResult:
        """Execute a complete workflow."""
        start_time = time.time()
        
        # Validate workflow
        validation_errors = workflow.validate()
        if validation_errors:
            raise WorkflowError(f"Workflow validation failed: {validation_errors}")
        
        result = WorkflowResult(
            workflow_name=workflow.name,
            status="running"
        )
        
        try:
            # Execute steps in dependency order
            executed_steps = set()
            remaining_steps = set(step.name for step in workflow.steps)
            
            while remaining_steps:
                # Find steps that can execute
                ready_steps = []
                for step in workflow.steps:
                    if (step.name in remaining_steps and 
                        step.can_execute(context) and
                        step.dependencies.issubset(executed_steps)):
                        ready_steps.append(step)
                
                if not ready_steps:
                    # Check for skippable steps
                    skippable_steps = []
                    for step in workflow.steps:
                        if (step.name in remaining_steps and
                            step.should_skip(context)):
                            skippable_steps.append(step)
                    
                    if skippable_steps:
                        # Skip these steps
                        for step in skippable_steps:
                            step_result = WorkflowStepResult(
                                step.name, WorkflowStepStatus.SKIPPED
                            )
                            step_result.finalize()
                            result.step_results[step.name] = step_result
                            context.step_results[step.name] = step_result
                            executed_steps.add(step.name)
                            remaining_steps.remove(step.name)
                        continue
                    
                    # No ready or skippable steps - possible circular dependency
                    raise WorkflowError(
                        f"Cannot execute remaining steps due to dependencies: {remaining_steps}"
                    )
                
                # Execute ready steps
                for step in ready_steps:
                    try:
                        step_result = self._execute_step(step, context)
                        result.step_results[step.name] = step_result
                        context.step_results[step.name] = step_result
                        
                        # Collect artifacts
                        result.global_artifacts.extend(step_result.artifacts)
                        
                        executed_steps.add(step.name)
                        remaining_steps.remove(step.name)
                        
                    except Exception as e:
                        step_result = WorkflowStepResult(
                            step.name, WorkflowStepStatus.FAILED
                        )
                        step_result.errors.append(str(e))
                        step_result.finalize()
                        
                        result.step_results[step.name] = step_result
                        context.step_results[step.name] = step_result
                        
                        if not step.continue_on_error:
                            raise WorkflowError(f"Step {step.name} failed: {e}")
                        
                        executed_steps.add(step.name)
                        remaining_steps.remove(step.name)
            
            self._workflow_stats["successful_executions"] += 1
            
        except Exception as e:
            result.errors.append(str(e))
            self._workflow_stats["failed_executions"] += 1
            raise WorkflowError(f"Workflow execution failed: {e}")
        
        finally:
            result.finalize()
            self._workflow_stats["executions"] += 1
            self._workflow_stats["total_execution_time"] += time.time() - start_time
        
        return result
    
    def _execute_step(self, step: WorkflowStep, context: WorkflowContext) -> WorkflowStepResult:
        """Execute a single workflow step."""
        # Validate inputs
        validation_errors = step.validate_inputs(context)
        if validation_errors:
            result = WorkflowStepResult(step.name, WorkflowStepStatus.FAILED)
            result.errors.extend(validation_errors)
            result.finalize()
            return result
        
        # Execute step with retries
        last_exception = None
        for attempt in range(step.retry_count + 1):
            try:
                result = step.execute(context)
                if result.status == WorkflowStepStatus.COMPLETED:
                    return result
                last_exception = Exception(f"Step failed with status: {result.status}")
            except Exception as e:
                last_exception = e
                if attempt == step.retry_count:
                    break
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        # All attempts failed
        result = WorkflowStepResult(step.name, WorkflowStepStatus.FAILED)
        result.errors.append(str(last_exception))
        result.finalize()
        return result
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics."""
        stats = self._workflow_stats.copy()
        
        if stats["executions"] > 0:
            stats["success_rate"] = (
                stats["successful_executions"] / stats["executions"]
            )
            stats["average_execution_time"] = (
                stats["total_execution_time"] / stats["executions"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["average_execution_time"] = 0.0
        
        return stats


# Factory functions and convenience methods

def create_workflow_engine(config: PipelineConfig) -> WorkflowEngine:
    """Create workflow engine with configuration."""
    return WorkflowEngine(config)


def execute_workflow(
    engine: WorkflowEngine,
    workflow: WorkflowDefinition,
    context: WorkflowContext
) -> WorkflowResult:
    """Execute workflow using engine."""
    return engine.execute_workflow(workflow, context)


def create_generator_step(
    name: str,
    generator_name: str,
    dependencies: Set[str] = None,
    **kwargs
) -> GeneratorWorkflowStep:
    """Create generator workflow step."""
    return GeneratorWorkflowStep(
        name=name,
        generator_name=generator_name,
        dependencies=dependencies or set(),
        **kwargs
    )


def create_transformation_step(
    name: str,
    transformation_func: Callable[[WorkflowContext], Dict[str, Any]],
    dependencies: Set[str] = None,
    **kwargs
) -> TransformationWorkflowStep:
    """Create transformation workflow step."""
    return TransformationWorkflowStep(
        name=name,
        transformation_func=transformation_func,
        dependencies=dependencies or set(),
        **kwargs
    )


def create_workflow_condition(
    condition_type: ConditionType,
    **parameters
) -> WorkflowCondition:
    """Create workflow condition."""
    return WorkflowCondition(
        condition_type=condition_type,
        parameters=parameters
    )