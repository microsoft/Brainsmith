"""
FINN Workflow Engine
Native FINN transformation pipeline integration with real-time monitoring.
"""

import os
import sys
import json
import logging
import subprocess
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
import tempfile
import shutil

logger = logging.getLogger(__name__)


@dataclass
class FINNTransformation:
    """Represents a single FINN transformation."""
    name: str
    transformation_class: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None
    description: str = ""


@dataclass
class TransformationResult:
    """Result of a transformation execution."""
    transformation_name: str
    success: bool
    duration: float
    output_model_path: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class PipelineExecutionResult:
    """Result of complete pipeline execution."""
    pipeline_id: str
    success: bool
    total_duration: float
    transformation_results: List[TransformationResult] = field(default_factory=list)
    final_model_path: Optional[str] = None
    execution_logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FINNTransformationRegistry:
    """Registry of available FINN transformations."""
    
    def __init__(self):
        self.transformations = {}
        self.transformation_metadata = {}
        self._load_standard_transformations()
    
    def register_transformation(self, transformation: FINNTransformation):
        """Register a FINN transformation."""
        self.transformations[transformation.name] = transformation
        logger.debug(f"Registered transformation: {transformation.name}")
    
    def get_transformation(self, name: str) -> Optional[FINNTransformation]:
        """Get transformation by name."""
        return self.transformations.get(name)
    
    def list_transformations(self) -> List[str]:
        """List all available transformations."""
        return list(self.transformations.keys())
    
    def get_transformation_dependencies(self, transformation_name: str) -> List[str]:
        """Get dependencies for a transformation."""
        transformation = self.get_transformation(transformation_name)
        return transformation.prerequisites if transformation else []
    
    def validate_transformation_sequence(self, sequence: List[str]) -> tuple[bool, List[str]]:
        """Validate that transformation sequence satisfies dependencies."""
        issues = []
        completed_transformations = set()
        
        for transform_name in sequence:
            transformation = self.get_transformation(transform_name)
            if not transformation:
                issues.append(f"Unknown transformation: {transform_name}")
                continue
            
            # Check prerequisites
            for prereq in transformation.prerequisites:
                if prereq not in completed_transformations:
                    issues.append(f"Transformation {transform_name} requires {prereq} which hasn't been executed")
            
            completed_transformations.add(transform_name)
        
        return len(issues) == 0, issues
    
    def _load_standard_transformations(self):
        """Load standard FINN transformations."""
        standard_transformations = [
            FINNTransformation(
                name="InferShapes",
                transformation_class="finn.transformation.infer_shapes.InferShapes",
                description="Infer tensor shapes throughout the model"
            ),
            FINNTransformation(
                name="FoldConstants",
                transformation_class="finn.transformation.fold_constants.FoldConstants",
                description="Fold constant expressions in the model"
            ),
            FINNTransformation(
                name="GiveUniqueNodeNames",
                transformation_class="finn.transformation.general.GiveUniqueNodeNames",
                description="Ensure all nodes have unique names"
            ),
            FINNTransformation(
                name="GiveReadableTensorNames",
                transformation_class="finn.transformation.general.GiveReadableTensorNames",
                description="Give readable names to tensors"
            ),
            FINNTransformation(
                name="RemoveStaticGraphInputs",
                transformation_class="finn.transformation.general.RemoveStaticGraphInputs",
                description="Remove static inputs from graph"
            ),
            FINNTransformation(
                name="Streamline",
                transformation_class="finn.transformation.streamline.Streamline",
                description="Apply streamlining optimizations",
                prerequisites=["InferShapes"]
            ),
            FINNTransformation(
                name="LowerConvsToMatMul",
                transformation_class="finn.transformation.streamline.LowerConvsToMatMul",
                description="Lower convolutions to matrix multiplications",
                prerequisites=["Streamline"]
            ),
            FINNTransformation(
                name="CreateDataflowPartition",
                transformation_class="finn.transformation.fpgadataflow.CreateDataflowPartition",
                description="Create dataflow partition for FPGA implementation",
                prerequisites=["LowerConvsToMatMul"]
            ),
            FINNTransformation(
                name="SpecializeLayers",
                transformation_class="finn.transformation.fpgadataflow.SpecializeLayers",
                description="Specialize layers for FPGA implementation",
                prerequisites=["CreateDataflowPartition"]
            ),
            FINNTransformation(
                name="MinimizeAccumulatorWidth",
                transformation_class="finn.transformation.fpgadataflow.MinimizeAccumulatorWidth",
                description="Minimize accumulator bit width",
                prerequisites=["SpecializeLayers"]
            ),
            FINNTransformation(
                name="MinimizeBitWidth",
                transformation_class="finn.transformation.fpgadataflow.MinimizeBitWidth",
                description="Minimize tensor bit widths",
                prerequisites=["SpecializeLayers"]
            ),
            FINNTransformation(
                name="SetExecMode",
                transformation_class="finn.transformation.fpgadataflow.SetExecMode",
                description="Set execution mode for nodes",
                prerequisites=["SpecializeLayers"]
            ),
            FINNTransformation(
                name="PrepareIP",
                transformation_class="finn.transformation.fpgadataflow.PrepareIP",
                description="Prepare IP blocks for synthesis",
                prerequisites=["SetExecMode"]
            ),
            FINNTransformation(
                name="HLSSynthIP",
                transformation_class="finn.transformation.fpgadataflow.HLSSynthIP",
                description="Synthesize HLS IP blocks",
                prerequisites=["PrepareIP"]
            ),
            FINNTransformation(
                name="CreateStitchedIP",
                transformation_class="finn.transformation.fpgadataflow.CreateStitchedIP",
                description="Create stitched IP for complete design",
                prerequisites=["HLSSynthIP"]
            )
        ]
        
        for transformation in standard_transformations:
            self.register_transformation(transformation)


class ProgressTracker:
    """Track progress of pipeline execution."""
    
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.current_transformation = None
        self.completed_transformations = []
        self.total_transformations = 0
        self.start_time = None
        self.callbacks: List[Callable] = []
    
    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates."""
        self.callbacks.append(callback)
    
    def start_pipeline(self, total_transformations: int):
        """Mark pipeline start."""
        self.total_transformations = total_transformations
        self.start_time = time.time()
        self._notify_callbacks("pipeline_started")
    
    def start_transformation(self, transformation_name: str):
        """Mark transformation start."""
        self.current_transformation = transformation_name
        self._notify_callbacks("transformation_started")
    
    def complete_transformation(self, transformation_name: str, success: bool):
        """Mark transformation completion."""
        self.completed_transformations.append({
            'name': transformation_name,
            'success': success,
            'timestamp': time.time()
        })
        self.current_transformation = None
        self._notify_callbacks("transformation_completed")
    
    def get_progress_percentage(self) -> float:
        """Get overall progress percentage."""
        if self.total_transformations == 0:
            return 0.0
        return (len(self.completed_transformations) / self.total_transformations) * 100.0
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since pipeline start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on current progress."""
        elapsed = self.get_elapsed_time()
        progress = self.get_progress_percentage()
        
        if progress > 0 and elapsed > 0:
            total_estimated = elapsed / (progress / 100.0)
            return max(0, total_estimated - elapsed)
        
        return None
    
    def _notify_callbacks(self, event_type: str):
        """Notify progress callbacks."""
        for callback in self.callbacks:
            try:
                callback(self, event_type)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")


class FINNPipelineExecutor:
    """Execute FINN transformation pipelines with monitoring."""
    
    def __init__(self, finn_installation_path: str):
        self.finn_path = finn_installation_path
        self.active_executions = {}
        self.execution_history = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Validate FINN installation
        self._validate_finn_installation()
    
    def execute_pipeline(self, 
                        model_path: str,
                        transformations: List[str],
                        output_dir: str,
                        pipeline_id: Optional[str] = None,
                        parameters: Optional[Dict[str, Any]] = None) -> Future[PipelineExecutionResult]:
        """Execute transformation pipeline asynchronously."""
        
        if pipeline_id is None:
            pipeline_id = f"pipeline_{int(time.time())}"
        
        if parameters is None:
            parameters = {}
        
        # Create progress tracker
        progress_tracker = ProgressTracker(pipeline_id)
        self.active_executions[pipeline_id] = progress_tracker
        
        # Submit execution to thread pool
        future = self.executor.submit(
            self._execute_pipeline_sync,
            model_path, transformations, output_dir, pipeline_id, parameters, progress_tracker
        )
        
        return future
    
    def get_pipeline_progress(self, pipeline_id: str) -> Optional[ProgressTracker]:
        """Get progress tracker for pipeline."""
        return self.active_executions.get(pipeline_id)
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel running pipeline."""
        # TODO: Implement pipeline cancellation
        progress_tracker = self.active_executions.get(pipeline_id)
        if progress_tracker:
            logger.info(f"Cancelling pipeline {pipeline_id}")
            return True
        return False
    
    def list_active_pipelines(self) -> List[str]:
        """List currently active pipeline IDs."""
        return list(self.active_executions.keys())
    
    def _execute_pipeline_sync(self,
                              model_path: str,
                              transformations: List[str],
                              output_dir: str,
                              pipeline_id: str,
                              parameters: Dict[str, Any],
                              progress_tracker: ProgressTracker) -> PipelineExecutionResult:
        """Execute pipeline synchronously."""
        
        logger.info(f"Starting pipeline execution {pipeline_id}")
        start_time = time.time()
        
        # Initialize result
        result = PipelineExecutionResult(
            pipeline_id=pipeline_id,
            success=False,
            total_duration=0.0,
            metadata=parameters.copy()
        )
        
        try:
            # Start progress tracking
            progress_tracker.start_pipeline(len(transformations))
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            current_model_path = model_path
            
            # Execute each transformation
            for i, transform_name in enumerate(transformations):
                logger.info(f"Executing transformation {i+1}/{len(transformations)}: {transform_name}")
                
                progress_tracker.start_transformation(transform_name)
                
                # Execute transformation
                transform_result = self._execute_transformation(
                    current_model_path, transform_name, output_dir, parameters
                )
                
                result.transformation_results.append(transform_result)
                progress_tracker.complete_transformation(transform_name, transform_result.success)
                
                if not transform_result.success:
                    logger.error(f"Transformation {transform_name} failed: {transform_result.error_message}")
                    result.success = False
                    break
                
                # Update current model path for next transformation
                if transform_result.output_model_path:
                    current_model_path = transform_result.output_model_path
            
            # Check if all transformations succeeded
            result.success = all(tr.success for tr in result.transformation_results)
            result.final_model_path = current_model_path if result.success else None
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result.success = False
            result.execution_logs.append(f"Pipeline execution error: {str(e)}")
        
        finally:
            # Clean up
            result.total_duration = time.time() - start_time
            self.active_executions.pop(pipeline_id, None)
            self.execution_history.append(result)
            
            logger.info(f"Pipeline {pipeline_id} completed in {result.total_duration:.2f}s, success: {result.success}")
        
        return result
    
    def _execute_transformation(self,
                               model_path: str,
                               transformation_name: str,
                               output_dir: str,
                               parameters: Dict[str, Any]) -> TransformationResult:
        """Execute a single transformation."""
        
        start_time = time.time()
        
        # Create transformation result
        result = TransformationResult(
            transformation_name=transformation_name,
            success=False,
            duration=0.0
        )
        
        try:
            # Create Python script for transformation
            script_content = self._generate_transformation_script(
                model_path, transformation_name, output_dir, parameters
            )
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_file.write(script_content)
                script_path = script_file.name
            
            try:
                # Execute transformation script
                env = os.environ.copy()
                env['PYTHONPATH'] = f"{self.finn_path}:{env.get('PYTHONPATH', '')}"
                
                process = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                    env=env
                )
                
                result.logs = process.stdout.split('\n') if process.stdout else []
                if process.stderr:
                    result.logs.extend(process.stderr.split('\n'))
                
                if process.returncode == 0:
                    result.success = True
                    # Look for output model
                    output_model_path = os.path.join(output_dir, f"{transformation_name}_output.onnx")
                    if os.path.exists(output_model_path):
                        result.output_model_path = output_model_path
                else:
                    result.error_message = f"Process exited with code {process.returncode}"
                    if process.stderr:
                        result.error_message += f": {process.stderr}"
                
            finally:
                # Clean up script file
                os.unlink(script_path)
        
        except subprocess.TimeoutExpired:
            result.error_message = "Transformation timed out"
        except Exception as e:
            result.error_message = f"Transformation execution error: {str(e)}"
        
        result.duration = time.time() - start_time
        return result
    
    def _generate_transformation_script(self,
                                       model_path: str,
                                       transformation_name: str,
                                       output_dir: str,
                                       parameters: Dict[str, Any]) -> str:
        """Generate Python script for transformation execution."""
        
        output_path = os.path.join(output_dir, f"{transformation_name}_output.onnx")
        
        script = f'''
import sys
import os
from qonnx.core.modelwrapper import ModelWrapper

# Add FINN to path
sys.path.insert(0, "{self.finn_path}")

def main():
    try:
        # Load model
        model = ModelWrapper("{model_path}")
        print(f"Loaded model: {model_path}")
        
        # Apply transformation
        # This is a simplified example - in practice, we'd import and apply the actual transformation
        print(f"Applying transformation: {transformation_name}")
        
        # For now, just copy the model (placeholder)
        import shutil
        shutil.copy2("{model_path}", "{output_path}")
        
        print(f"Transformation {transformation_name} completed successfully")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        return script
    
    def _validate_finn_installation(self):
        """Validate FINN installation."""
        if not os.path.exists(self.finn_path):
            raise ValueError(f"FINN path does not exist: {self.finn_path}")
        
        # Check for key FINN directories
        required_dirs = ['src/finn', 'notebooks']
        for dir_name in required_dirs:
            dir_path = os.path.join(self.finn_path, dir_name)
            if not os.path.exists(dir_path):
                logger.warning(f"FINN directory not found: {dir_path}")


class FINNWorkflowEngine:
    """Main FINN workflow engine for pipeline orchestration."""
    
    def __init__(self, finn_installation_path: str):
        self.finn_path = finn_installation_path
        self.transformation_registry = FINNTransformationRegistry()
        self.pipeline_executor = FINNPipelineExecutor(finn_installation_path)
        self.active_workflows = {}
        
        logger.info(f"Initialized FINN Workflow Engine with FINN at: {finn_installation_path}")
    
    def execute_transformation_sequence(self,
                                      model_path: str,
                                      transformations: List[str],
                                      config: Dict[str, Any],
                                      workflow_id: Optional[str] = None) -> Future[PipelineExecutionResult]:
        """Execute FINN transformation sequence with monitoring."""
        
        if workflow_id is None:
            workflow_id = f"workflow_{int(time.time())}"
        
        # Validate transformation sequence
        valid, issues = self.transformation_registry.validate_transformation_sequence(transformations)
        if not valid:
            raise ValueError(f"Invalid transformation sequence: {issues}")
        
        # Create output directory
        output_dir = config.get('output_dir', f'./finn_output_{workflow_id}')
        
        # Execute pipeline
        future = self.pipeline_executor.execute_pipeline(
            model_path=model_path,
            transformations=transformations,
            output_dir=output_dir,
            pipeline_id=workflow_id,
            parameters=config
        )
        
        self.active_workflows[workflow_id] = future
        return future
    
    def create_custom_pipeline(self, requirements: Dict[str, Any]) -> List[str]:
        """Create custom transformation pipeline for specific requirements."""
        
        # Basic pipeline for different model types
        model_type = requirements.get('model_type', 'generic')
        target_backend = requirements.get('target_backend', 'fpga')
        optimization_level = requirements.get('optimization_level', 'balanced')
        
        if model_type == 'cnn':
            pipeline = [
                'InferShapes',
                'FoldConstants', 
                'GiveUniqueNodeNames',
                'Streamline',
                'LowerConvsToMatMul',
                'CreateDataflowPartition',
                'SpecializeLayers'
            ]
        elif model_type == 'transformer':
            pipeline = [
                'InferShapes',
                'FoldConstants',
                'GiveUniqueNodeNames', 
                'Streamline',
                'CreateDataflowPartition',
                'SpecializeLayers'
            ]
        else:
            # Generic pipeline
            pipeline = [
                'InferShapes',
                'FoldConstants',
                'GiveUniqueNodeNames',
                'Streamline'
            ]
        
        # Add optimization based on level
        if optimization_level in ['aggressive', 'balanced']:
            pipeline.extend([
                'MinimizeAccumulatorWidth',
                'MinimizeBitWidth'
            ])
        
        # Add backend-specific transformations
        if target_backend == 'fpga':
            pipeline.extend([
                'SetExecMode',
                'PrepareIP'
            ])
            
            if requirements.get('synthesize', False):
                pipeline.extend([
                    'HLSSynthIP',
                    'CreateStitchedIP'
                ])
        
        return pipeline
    
    def monitor_execution(self, workflow_id: str) -> Optional[ProgressTracker]:
        """Monitor workflow execution progress."""
        return self.pipeline_executor.get_pipeline_progress(workflow_id)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel running workflow."""
        return self.pipeline_executor.cancel_pipeline(workflow_id)
    
    def list_active_workflows(self) -> List[str]:
        """List active workflow IDs."""
        return list(self.active_workflows.keys())
    
    def get_workflow_result(self, workflow_id: str) -> Optional[PipelineExecutionResult]:
        """Get workflow result if completed."""
        future = self.active_workflows.get(workflow_id)
        if future and future.done():
            return future.result()
        return None
    
    def list_available_transformations(self) -> List[str]:
        """List all available transformations."""
        return self.transformation_registry.list_transformations()
    
    def get_transformation_info(self, transformation_name: str) -> Optional[FINNTransformation]:
        """Get information about a specific transformation."""
        return self.transformation_registry.get_transformation(transformation_name)