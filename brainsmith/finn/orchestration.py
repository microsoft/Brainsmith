"""
FINN Build Orchestration
Advanced build orchestration with multi-configuration parallel build management.
"""

import os
import sys
import json
import logging
import threading
import time
import queue
import psutil
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from enum import Enum
import tempfile
import shutil
import multiprocessing

logger = logging.getLogger(__name__)


class BuildStatus(Enum):
    """Build status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BuildPriority(Enum):
    """Build priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BuildRequest:
    """Represents a build request."""
    build_id: str
    model_path: str
    transformations: List[str]
    config: Dict[str, Any]
    priority: BuildPriority = BuildPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class BuildResult:
    """Result of a build execution."""
    build_id: str
    request: BuildRequest
    status: BuildStatus
    success: bool
    duration: float
    output_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemResources:
    """System resource information."""
    cpu_count: int
    memory_gb: float
    available_memory_gb: float
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    load_average: Optional[float] = None


class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = []
        self.max_history = 3600  # Keep 1 hour of history
        self.callbacks = []
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")
    
    def get_current_resources(self) -> SystemResources:
        """Get current system resources."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get disk usage for current directory
        disk_usage = psutil.disk_usage('.')
        
        # Get load average (Unix only)
        load_avg = None
        if hasattr(os, 'getloadavg'):
            load_avg = os.getloadavg()[0]
        
        return SystemResources(
            cpu_count=psutil.cpu_count(),
            memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            disk_usage_percent=(disk_usage.used / disk_usage.total) * 100,
            load_average=load_avg
        )
    
    def add_callback(self, callback: Callable[[SystemResources], None]):
        """Add callback for resource updates."""
        self.callbacks.append(callback)
    
    def get_resource_history(self, minutes: int = 10) -> List[SystemResources]:
        """Get resource history for specified minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [
            (timestamp, resources) for timestamp, resources in self.resource_history
            if timestamp >= cutoff_time
        ]
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                resources = self.get_current_resources()
                timestamp = time.time()
                
                # Add to history
                self.resource_history.append((timestamp, resources))
                
                # Trim history
                if len(self.resource_history) > self.max_history:
                    self.resource_history = self.resource_history[-self.max_history:]
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(resources)
                    except Exception as e:
                        logger.warning(f"Resource monitor callback failed: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)


class BuildResourceManager:
    """Manage resources for build execution."""
    
    def __init__(self, max_parallel_builds: int = None):
        self.max_parallel_builds = max_parallel_builds or min(4, multiprocessing.cpu_count())
        self.resource_monitor = ResourceMonitor()
        self.active_builds = {}
        self.resource_limits = {
            'memory_threshold': 0.8,  # Don't start builds if memory > 80%
            'cpu_threshold': 0.9,     # Don't start builds if CPU > 90%
            'load_threshold': None    # Will be set based on CPU count
        }
        
        # Set load threshold based on CPU count
        self.resource_limits['load_threshold'] = psutil.cpu_count() * 0.8
        
        self.resource_monitor.start_monitoring()
    
    def can_start_build(self, build_request: BuildRequest) -> tuple[bool, str]:
        """Check if build can be started given current resources."""
        
        # Check active build count
        if len(self.active_builds) >= self.max_parallel_builds:
            return False, f"Maximum parallel builds reached ({self.max_parallel_builds})"
        
        # Check system resources
        resources = self.resource_monitor.get_current_resources()
        
        # Memory check
        if resources.memory_usage_percent > self.resource_limits['memory_threshold'] * 100:
            return False, f"Memory usage too high ({resources.memory_usage_percent:.1f}%)"
        
        # CPU check
        if resources.cpu_usage_percent > self.resource_limits['cpu_threshold'] * 100:
            return False, f"CPU usage too high ({resources.cpu_usage_percent:.1f}%)"
        
        # Load average check (Unix only)
        if (resources.load_average is not None and 
            self.resource_limits['load_threshold'] is not None and
            resources.load_average > self.resource_limits['load_threshold']):
            return False, f"Load average too high ({resources.load_average:.1f})"
        
        return True, "Resources available"
    
    def allocate_resources(self, build_id: str, estimated_requirements: Dict[str, Any]):
        """Allocate resources for build."""
        self.active_builds[build_id] = {
            'start_time': time.time(),
            'requirements': estimated_requirements,
            'resources': self.resource_monitor.get_current_resources()
        }
        logger.debug(f"Allocated resources for build {build_id}")
    
    def release_resources(self, build_id: str):
        """Release resources for completed build."""
        if build_id in self.active_builds:
            build_info = self.active_builds.pop(build_id)
            duration = time.time() - build_info['start_time']
            logger.debug(f"Released resources for build {build_id} after {duration:.1f}s")
    
    def get_resource_usage(self, build_id: str) -> Dict[str, float]:
        """Get resource usage for specific build."""
        if build_id not in self.active_builds:
            return {}
        
        build_info = self.active_builds[build_id]
        current_resources = self.resource_monitor.get_current_resources()
        start_resources = build_info['resources']
        
        return {
            'duration': time.time() - build_info['start_time'],
            'memory_delta_gb': start_resources.available_memory_gb - current_resources.available_memory_gb,
            'cpu_usage_percent': current_resources.cpu_usage_percent
        }
    
    def shutdown(self):
        """Shutdown resource manager."""
        self.resource_monitor.stop_monitoring()


class BuildQueue:
    """Priority-based build queue with dependency resolution."""
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.pending_builds = {}  # build_id -> BuildRequest
        self.dependency_graph = {}  # build_id -> List[dependency_build_ids]
        self.completed_builds = set()
        self.failed_builds = set()
        self.lock = threading.Lock()
    
    def enqueue_build(self, build_request: BuildRequest):
        """Add build request to queue."""
        with self.lock:
            # Priority queue uses negative priority for max-heap behavior
            priority = -build_request.priority.value
            self.queue.put((priority, build_request.created_at, build_request))
            self.pending_builds[build_request.build_id] = build_request
            
            # Update dependency graph
            if build_request.dependencies:
                self.dependency_graph[build_request.build_id] = build_request.dependencies.copy()
            
            logger.info(f"Enqueued build {build_request.build_id} with priority {build_request.priority.name}")
    
    def dequeue_ready_build(self) -> Optional[BuildRequest]:
        """Get next ready build (dependencies satisfied)."""
        with self.lock:
            ready_builds = []
            temp_builds = []
            
            # Collect all builds from queue
            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    temp_builds.append(item)
                except queue.Empty:
                    break
            
            # Find builds with satisfied dependencies
            for priority, created_at, build_request in temp_builds:
                if self._dependencies_satisfied(build_request):
                    ready_builds.append((priority, created_at, build_request))
                else:
                    # Put back in queue
                    self.queue.put((priority, created_at, build_request))
            
            # Return highest priority ready build
            if ready_builds:
                ready_builds.sort(key=lambda x: (x[0], x[1]))  # Sort by priority, then time
                _, _, build_request = ready_builds[0]
                
                # Put remaining builds back
                for item in ready_builds[1:]:
                    self.queue.put(item)
                
                # Remove from pending
                del self.pending_builds[build_request.build_id]
                
                return build_request
            
            return None
    
    def mark_build_completed(self, build_id: str, success: bool):
        """Mark build as completed."""
        with self.lock:
            if success:
                self.completed_builds.add(build_id)
            else:
                self.failed_builds.add(build_id)
            
            # Clean up dependency graph
            self.dependency_graph.pop(build_id, None)
            
            logger.debug(f"Marked build {build_id} as {'completed' if success else 'failed'}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information."""
        with self.lock:
            return {
                'pending_builds': len(self.pending_builds),
                'queue_size': self.queue.qsize(),
                'completed_builds': len(self.completed_builds),
                'failed_builds': len(self.failed_builds),
                'pending_build_ids': list(self.pending_builds.keys())
            }
    
    def _dependencies_satisfied(self, build_request: BuildRequest) -> bool:
        """Check if build dependencies are satisfied."""
        dependencies = self.dependency_graph.get(build_request.build_id, [])
        
        for dep_id in dependencies:
            if dep_id not in self.completed_builds:
                return False
        
        return True


class FINNBuildOrchestrator:
    """Main build orchestrator for FINN builds."""
    
    def __init__(self, finn_workflow_engine, max_parallel_builds: int = None):
        self.workflow_engine = finn_workflow_engine
        self.build_queue = BuildQueue()
        self.resource_manager = BuildResourceManager(max_parallel_builds)
        self.result_collector = BuildResultCollector()
        
        self.active_builds = {}  # build_id -> Future
        self.build_results = {}  # build_id -> BuildResult
        
        self.executor = ThreadPoolExecutor(
            max_workers=max_parallel_builds or min(4, multiprocessing.cpu_count())
        )
        
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Start the build scheduler
        self.start_scheduler()
    
    def schedule_build(self, 
                      model_path: str,
                      transformations: List[str],
                      config: Dict[str, Any],
                      build_id: Optional[str] = None,
                      priority: BuildPriority = BuildPriority.NORMAL,
                      dependencies: List[str] = None) -> str:
        """Schedule a FINN build."""
        
        if build_id is None:
            build_id = f"build_{int(time.time() * 1000)}"
        
        if dependencies is None:
            dependencies = []
        
        build_request = BuildRequest(
            build_id=build_id,
            model_path=model_path,
            transformations=transformations,
            config=config.copy(),
            priority=priority,
            dependencies=dependencies
        )
        
        self.build_queue.enqueue_build(build_request)
        logger.info(f"Scheduled build {build_id}")
        
        return build_id
    
    def execute_parallel_builds(self, build_configs: List[Dict[str, Any]]) -> List[str]:
        """Execute multiple builds in parallel."""
        build_ids = []
        
        for i, config in enumerate(build_configs):
            build_id = self.schedule_build(
                model_path=config['model_path'],
                transformations=config['transformations'],
                config=config.get('config', {}),
                build_id=config.get('build_id', f"parallel_build_{i}"),
                priority=BuildPriority(config.get('priority', BuildPriority.NORMAL.value))
            )
            build_ids.append(build_id)
        
        logger.info(f"Scheduled {len(build_ids)} parallel builds")
        return build_ids
    
    def monitor_build_progress(self, build_id: str) -> Optional[Dict[str, Any]]:
        """Monitor individual build progress."""
        
        # Check if build is active
        if build_id in self.active_builds:
            future = self.active_builds[build_id]
            progress = self.workflow_engine.monitor_execution(build_id)
            
            if progress:
                return {
                    'build_id': build_id,
                    'status': BuildStatus.RUNNING,
                    'progress_percent': progress.get_progress_percentage(),
                    'elapsed_time': progress.get_elapsed_time(),
                    'estimated_remaining': progress.estimate_remaining_time(),
                    'current_transformation': progress.current_transformation,
                    'completed_transformations': len(progress.completed_transformations),
                    'resource_usage': self.resource_manager.get_resource_usage(build_id)
                }
        
        # Check if build is completed
        if build_id in self.build_results:
            result = self.build_results[build_id]
            return {
                'build_id': build_id,
                'status': result.status,
                'success': result.success,
                'duration': result.duration,
                'error_message': result.error_message
            }
        
        # Check if build is in queue
        queue_status = self.build_queue.get_queue_status()
        if build_id in queue_status['pending_build_ids']:
            return {
                'build_id': build_id,
                'status': BuildStatus.QUEUED,
                'queue_position': queue_status['pending_build_ids'].index(build_id) + 1,
                'queue_size': queue_status['queue_size']
            }
        
        return None
    
    def cancel_build(self, build_id: str) -> bool:
        """Cancel a build."""
        
        # Cancel if currently running
        if build_id in self.active_builds:
            future = self.active_builds[build_id]
            cancelled = future.cancel()
            
            if cancelled:
                self.active_builds.pop(build_id)
                self.resource_manager.release_resources(build_id)
                
                # Create cancelled result
                result = BuildResult(
                    build_id=build_id,
                    request=None,  # We don't have the original request here
                    status=BuildStatus.CANCELLED,
                    success=False,
                    duration=0.0,
                    error_message="Build cancelled by user"
                )
                self.build_results[build_id] = result
                
                logger.info(f"Cancelled running build {build_id}")
                return True
        
        # Remove from queue if pending
        # Note: This is simplified - a more complete implementation would
        # need to modify the queue to support removal
        
        return False
    
    def get_build_result(self, build_id: str) -> Optional[BuildResult]:
        """Get build result."""
        return self.build_results.get(build_id)
    
    def list_active_builds(self) -> List[str]:
        """List active build IDs."""
        return list(self.active_builds.keys())
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get build queue status."""
        queue_status = self.build_queue.get_queue_status()
        
        queue_status.update({
            'active_builds': len(self.active_builds),
            'max_parallel_builds': self.resource_manager.max_parallel_builds,
            'system_resources': self.resource_manager.resource_monitor.get_current_resources()
        })
        
        return queue_status
    
    def start_scheduler(self):
        """Start the build scheduler."""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Started build scheduler")
    
    def stop_scheduler(self):
        """Stop the build scheduler."""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Stopped build scheduler")
    
    def shutdown(self):
        """Shutdown the orchestrator."""
        self.stop_scheduler()
        self.executor.shutdown(wait=True)
        self.resource_manager.shutdown()
        logger.info("Orchestrator shutdown complete")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.scheduler_running:
            try:
                # Check for completed builds
                self._check_completed_builds()
                
                # Try to start new builds
                self._start_ready_builds()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _check_completed_builds(self):
        """Check for completed builds and update results."""
        completed_builds = []
        
        for build_id, future in self.active_builds.items():
            if future.done():
                completed_builds.append(build_id)
        
        for build_id in completed_builds:
            future = self.active_builds.pop(build_id)
            self.resource_manager.release_resources(build_id)
            
            try:
                pipeline_result = future.result()
                
                # Convert pipeline result to build result
                build_result = BuildResult(
                    build_id=build_id,
                    request=None,  # We'd need to store this separately
                    status=BuildStatus.COMPLETED if pipeline_result.success else BuildStatus.FAILED,
                    success=pipeline_result.success,
                    duration=pipeline_result.total_duration,
                    output_path=pipeline_result.final_model_path,
                    logs=pipeline_result.execution_logs,
                    metrics=pipeline_result.metadata
                )
                
                self.build_results[build_id] = build_result
                self.build_queue.mark_build_completed(build_id, pipeline_result.success)
                
                logger.info(f"Build {build_id} completed: success={pipeline_result.success}")
                
            except Exception as e:
                # Handle build exception
                build_result = BuildResult(
                    build_id=build_id,
                    request=None,
                    status=BuildStatus.FAILED,
                    success=False,
                    duration=0.0,
                    error_message=str(e)
                )
                
                self.build_results[build_id] = build_result
                self.build_queue.mark_build_completed(build_id, False)
                
                logger.error(f"Build {build_id} failed: {e}")
    
    def _start_ready_builds(self):
        """Start ready builds if resources are available."""
        while True:
            # Get next ready build
            build_request = self.build_queue.dequeue_ready_build()
            if not build_request:
                break
            
            # Check if we can start the build
            can_start, reason = self.resource_manager.can_start_build(build_request)
            if not can_start:
                # Put build back in queue
                self.build_queue.enqueue_build(build_request)
                logger.debug(f"Cannot start build {build_request.build_id}: {reason}")
                break
            
            # Start the build
            self._start_build(build_request)
    
    def _start_build(self, build_request: BuildRequest):
        """Start a build execution."""
        try:
            # Allocate resources
            self.resource_manager.allocate_resources(
                build_request.build_id,
                {'estimated_duration': 3600}  # Default 1 hour estimate
            )
            
            # Start workflow execution
            future = self.workflow_engine.execute_transformation_sequence(
                model_path=build_request.model_path,
                transformations=build_request.transformations,
                config=build_request.config,
                workflow_id=build_request.build_id
            )
            
            self.active_builds[build_request.build_id] = future
            build_request.started_at = time.time()
            
            logger.info(f"Started build {build_request.build_id}")
            
        except Exception as e:
            logger.error(f"Failed to start build {build_request.build_id}: {e}")
            
            # Mark as failed
            build_result = BuildResult(
                build_id=build_request.build_id,
                request=build_request,
                status=BuildStatus.FAILED,
                success=False,
                duration=0.0,
                error_message=f"Failed to start build: {e}"
            )
            
            self.build_results[build_request.build_id] = build_result
            self.build_queue.mark_build_completed(build_request.build_id, False)


class BuildResultCollector:
    """Collect and manage build results."""
    
    def __init__(self, results_dir: str = "./build_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def store_result(self, build_result: BuildResult):
        """Store build result to disk."""
        result_file = os.path.join(self.results_dir, f"{build_result.build_id}.json")
        
        # Convert result to JSON-serializable format
        result_data = {
            'build_id': build_result.build_id,
            'status': build_result.status.value,
            'success': build_result.success,
            'duration': build_result.duration,
            'output_path': build_result.output_path,
            'artifacts': build_result.artifacts,
            'logs': build_result.logs,
            'metrics': build_result.metrics,
            'error_message': build_result.error_message,
            'resource_usage': build_result.resource_usage
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    def load_result(self, build_id: str) -> Optional[BuildResult]:
        """Load build result from disk."""
        result_file = os.path.join(self.results_dir, f"{build_id}.json")
        
        if not os.path.exists(result_file):
            return None
        
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # Convert back to BuildResult
            return BuildResult(
                build_id=result_data['build_id'],
                request=None,  # Not stored
                status=BuildStatus(result_data['status']),
                success=result_data['success'],
                duration=result_data['duration'],
                output_path=result_data.get('output_path'),
                artifacts=result_data.get('artifacts', {}),
                logs=result_data.get('logs', []),
                metrics=result_data.get('metrics', {}),
                error_message=result_data.get('error_message'),
                resource_usage=result_data.get('resource_usage', {})
            )
            
        except Exception as e:
            logger.error(f"Failed to load build result {build_id}: {e}")
            return None