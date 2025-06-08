"""
High-level workflow management using existing components.

This module provides workflow coordination and management capabilities
that orchestrate the DesignSpaceOrchestrator for different types of
exploration workflows using existing components only.
"""

from typing import Dict, Any, List, Optional
import logging
from enum import Enum
import time

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Enumeration of available workflow types."""
    FAST = "fast"                    # Roofline analysis only
    STANDARD = "standard"            # Dataflow analysis  
    COMPREHENSIVE = "comprehensive"  # Full generation

class WorkflowStatus(Enum):
    """Enumeration of workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowManager:
    """
    Manages high-level workflows using existing components.
    
    Provides extensible structure for workflow coordination and
    orchestrates the DesignSpaceOrchestrator for different exploration
    scenarios using only existing Brainsmith components.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize workflow manager with orchestrator.
        
        Args:
            orchestrator: DesignSpaceOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.workflow_history = []
        self.current_workflow = None
        
        logger.info("WorkflowManager initialized")
    
    def execute_existing_workflow(self, workflow_type: str, **kwargs) -> Dict[str, Any]:
        """
        Execute workflow using existing components only.
        
        Args:
            workflow_type: Type of workflow ('fast', 'standard', 'comprehensive')
            **kwargs: Additional workflow configuration
            
        Returns:
            Dictionary containing workflow results and metadata
        """
        logger.info(f"Executing workflow type: {workflow_type}")
        
        # Validate workflow type
        try:
            workflow_enum = WorkflowType(workflow_type)
        except ValueError:
            valid_types = [wf.value for wf in WorkflowType]
            raise ValueError(f"Invalid workflow type: {workflow_type}. Must be one of {valid_types}")
        
        # Create workflow record
        workflow_record = {
            'type': workflow_type,
            'start_time': time.time(),
            'status': WorkflowStatus.PENDING.value,
            'blueprint': getattr(self.orchestrator.blueprint, 'name', 'unknown'),
            'config': kwargs
        }
        
        self.current_workflow = workflow_record
        workflow_record['status'] = WorkflowStatus.RUNNING.value
        
        try:
            # Execute appropriate workflow based on type
            if workflow_enum == WorkflowType.FAST:
                result = self._execute_fast_workflow_existing(**kwargs)
            elif workflow_enum == WorkflowType.STANDARD:
                result = self._execute_standard_workflow_existing(**kwargs)
            elif workflow_enum == WorkflowType.COMPREHENSIVE:
                result = self._execute_comprehensive_workflow_existing(**kwargs)
            
            # Mark workflow as completed
            workflow_record['status'] = WorkflowStatus.COMPLETED.value
            workflow_record['end_time'] = time.time()
            workflow_record['duration'] = workflow_record['end_time'] - workflow_record['start_time']
            workflow_record['result_summary'] = self._summarize_workflow_result(result)
            
            logger.info(f"Workflow {workflow_type} completed successfully in {workflow_record['duration']:.2f}s")
            
        except Exception as e:
            # Mark workflow as failed
            workflow_record['status'] = WorkflowStatus.FAILED.value
            workflow_record['end_time'] = time.time()
            workflow_record['duration'] = workflow_record['end_time'] - workflow_record['start_time']
            workflow_record['error'] = str(e)
            
            logger.error(f"Workflow {workflow_type} failed after {workflow_record['duration']:.2f}s: {e}")
            
            # Create error result
            result = {
                'workflow_type': workflow_type,
                'status': 'failed',
                'error': str(e),
                'components_used': 'existing_only'
            }
        
        # Add workflow to history
        self.workflow_history.append(workflow_record.copy())
        self.current_workflow = None
        
        # Return complete workflow result
        return {
            'workflow': workflow_type,
            'result': result,
            'metadata': workflow_record,
            'components_source': 'existing_only'
        }
    
    def _execute_fast_workflow_existing(self, **kwargs) -> Dict[str, Any]:
        """
        Fast workflow using roofline analysis only.
        Provides quick performance bounds using existing analysis tools.
        """
        logger.info("Executing fast workflow (roofline analysis)")
        
        # Use roofline exit point for fastest analysis
        result = self.orchestrator.orchestrate_exploration("roofline")
        
        # Add workflow-specific metadata
        workflow_result = {
            'workflow_type': 'fast',
            'exit_point': 'roofline',
            'analysis_type': 'analytical_bounds',
            'dse_result': result,
            'performance_characteristics': result.analysis.get('roofline_results', {}),
            'execution_method': 'existing_analysis_tools',
            'libraries_used': ['analysis'],
            'estimated_time_saved': 'significant_vs_full_generation'
        }
        
        return workflow_result
    
    def _execute_standard_workflow_existing(self, **kwargs) -> Dict[str, Any]:
        """
        Standard workflow using dataflow analysis.
        Applies existing transforms and provides dataflow-level performance estimation.
        """
        logger.info("Executing standard workflow (dataflow analysis)")
        
        # Use dataflow analysis exit point for balanced exploration
        result = self.orchestrator.orchestrate_exploration("dataflow_analysis")
        
        # Add workflow-specific metadata
        workflow_result = {
            'workflow_type': 'standard',
            'exit_point': 'dataflow_analysis',
            'analysis_type': 'dataflow_estimation',
            'dse_result': result,
            'transformed_model': result.analysis.get('transformed_model', {}),
            'kernel_mapping': result.analysis.get('kernel_mapping', {}),
            'performance_estimates': result.analysis.get('performance_estimates', {}),
            'execution_method': 'existing_transforms_and_estimation',
            'libraries_used': ['transforms', 'kernels', 'analysis'],
            'trade_off': 'detailed_analysis_without_rtl_generation'
        }
        
        return workflow_result
    
    def _execute_comprehensive_workflow_existing(self, **kwargs) -> Dict[str, Any]:
        """
        Comprehensive workflow using full generation.
        Performs complete optimization and RTL/HLS generation using existing FINN flow.
        """
        logger.info("Executing comprehensive workflow (full generation)")
        
        # Use full generation exit point for complete exploration
        result = self.orchestrator.orchestrate_exploration("dataflow_generation")
        
        # Add workflow-specific metadata
        workflow_result = {
            'workflow_type': 'comprehensive',
            'exit_point': 'dataflow_generation',
            'analysis_type': 'complete_generation',
            'dse_result': result,
            'optimization_results': result.analysis.get('optimization_results', {}),
            'generation_results': result.analysis.get('generation_results', {}),
            'execution_method': 'existing_finn_generation_flow',
            'libraries_used': ['transforms', 'kernels', 'hw_optim', 'analysis'],
            'output_artifacts': {
                'rtl_files': result.analysis.get('generation_results', {}).get('rtl_files', []),
                'hls_files': result.analysis.get('generation_results', {}).get('hls_files', []),
                'synthesis_results': result.analysis.get('generation_results', {}).get('synthesis_results', {})
            }
        }
        
        return workflow_result
    
    def _summarize_workflow_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of workflow execution result."""
        return {
            'workflow_type': result.get('workflow_type', 'unknown'),
            'exit_point': result.get('exit_point', 'unknown'),
            'execution_method': result.get('execution_method', 'unknown'),
            'libraries_used': result.get('libraries_used', []),
            'has_dse_result': 'dse_result' in result,
            'components_source': result.get('dse_result', {}).analysis.get('components_source', 'unknown') if result.get('dse_result') else 'unknown'
        }
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all executed workflows.
        
        Returns:
            List of workflow execution records
        """
        return self.workflow_history.copy()
    
    def get_current_workflow_status(self) -> Optional[Dict[str, Any]]:
        """
        Get status of currently running workflow.
        
        Returns:
            Current workflow record or None if no workflow running
        """
        if self.current_workflow:
            current_time = time.time()
            duration = current_time - self.current_workflow['start_time']
            
            return {
                **self.current_workflow,
                'current_duration': duration,
                'estimated_remaining': self._estimate_remaining_time(self.current_workflow['type'], duration)
            }
        
        return None
    
    def _estimate_remaining_time(self, workflow_type: str, elapsed_time: float) -> str:
        """
        Estimate remaining time for workflow execution.
        
        Args:
            workflow_type: Type of workflow being executed
            elapsed_time: Time elapsed so far
            
        Returns:
            Estimated remaining time description
        """
        # Simple time estimation based on workflow type
        estimated_total = {
            'fast': 30.0,        # 30 seconds for roofline
            'standard': 120.0,   # 2 minutes for dataflow analysis
            'comprehensive': 600.0  # 10 minutes for full generation
        }
        
        total_time = estimated_total.get(workflow_type, 300.0)
        remaining = max(0, total_time - elapsed_time)
        
        if remaining < 60:
            return f"~{int(remaining)} seconds"
        else:
            return f"~{int(remaining/60)} minutes"
    
    def cancel_current_workflow(self) -> bool:
        """
        Cancel currently running workflow.
        
        Returns:
            True if workflow was cancelled, False if no workflow running
        """
        if self.current_workflow:
            self.current_workflow['status'] = WorkflowStatus.FAILED.value
            self.current_workflow['end_time'] = time.time()
            self.current_workflow['duration'] = self.current_workflow['end_time'] - self.current_workflow['start_time']
            self.current_workflow['error'] = 'Cancelled by user'
            
            # Add to history
            self.workflow_history.append(self.current_workflow.copy())
            self.current_workflow = None
            
            logger.info("Current workflow cancelled by user")
            return True
        
        return False
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about workflow execution history.
        
        Returns:
            Dictionary with workflow execution statistics
        """
        if not self.workflow_history:
            return {
                'total_workflows': 0,
                'by_type': {},
                'by_status': {},
                'average_duration': 0.0
            }
        
        # Count workflows by type
        by_type = {}
        by_status = {}
        durations = []
        
        for workflow in self.workflow_history:
            wf_type = workflow.get('type', 'unknown')
            wf_status = workflow.get('status', 'unknown')
            
            by_type[wf_type] = by_type.get(wf_type, 0) + 1
            by_status[wf_status] = by_status.get(wf_status, 0) + 1
            
            if 'duration' in workflow:
                durations.append(workflow['duration'])
        
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            'total_workflows': len(self.workflow_history),
            'by_type': by_type,
            'by_status': by_status,
            'average_duration': average_duration,
            'success_rate': by_status.get('completed', 0) / len(self.workflow_history) if self.workflow_history else 0.0
        }
    
    def create_workflow_from_blueprint(self, blueprint, workflow_type: str = "standard") -> Dict[str, Any]:
        """
        Create and execute workflow based on blueprint configuration.
        
        Args:
            blueprint: Blueprint instance with workflow configuration
            workflow_type: Default workflow type if not specified in blueprint
            
        Returns:
            Workflow execution result
        """
        # Extract workflow configuration from blueprint if available
        workflow_config = {}
        if hasattr(blueprint, 'get_workflow_config'):
            workflow_config = blueprint.get_workflow_config()
        
        # Use blueprint-specified workflow type if available
        blueprint_workflow_type = workflow_config.get('type', workflow_type)
        
        logger.info(f"Creating workflow from blueprint: {blueprint.name if hasattr(blueprint, 'name') else 'unknown'}")
        
        return self.execute_existing_workflow(blueprint_workflow_type, **workflow_config)