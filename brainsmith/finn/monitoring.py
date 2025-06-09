"""
FINN Build Monitoring
Real-time monitoring and progress tracking for FINN builds.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MonitoringEvent(Enum):
    """Types of monitoring events."""
    BUILD_STARTED = "build_started"
    BUILD_COMPLETED = "build_completed"
    BUILD_FAILED = "build_failed"
    TRANSFORMATION_STARTED = "transformation_started"
    TRANSFORMATION_COMPLETED = "transformation_completed"
    RESOURCE_ALERT = "resource_alert"
    PROGRESS_UPDATE = "progress_update"


@dataclass
class MonitoringData:
    """Data structure for monitoring information."""
    build_id: str
    event_type: MonitoringEvent
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None


class FINNBuildMonitor:
    """Monitor FINN build execution with real-time updates."""
    
    def __init__(self):
        self.active_monitors = {}  # build_id -> monitor_data
        self.event_callbacks = []
        self.monitoring_active = True
        self.lock = threading.Lock()
    
    def start_build_monitoring(self, build_id: str, metadata: Dict[str, Any] = None):
        """Start monitoring a build."""
        with self.lock:
            monitor_data = {
                'build_id': build_id,
                'start_time': time.time(),
                'status': 'running',
                'metadata': metadata or {},
                'events': []
            }
            
            self.active_monitors[build_id] = monitor_data
            
            # Notify callbacks
            event = MonitoringData(
                build_id=build_id,
                event_type=MonitoringEvent.BUILD_STARTED,
                timestamp=time.time(),
                data=metadata or {}
            )
            self._notify_callbacks(event)
            
            logger.info(f"Started monitoring build {build_id}")
    
    def update_build_progress(self, build_id: str, progress_data: Dict[str, Any]):
        """Update build progress."""
        with self.lock:
            if build_id not in self.active_monitors:
                return
            
            monitor_data = self.active_monitors[build_id]
            
            # Add progress event
            event = MonitoringData(
                build_id=build_id,
                event_type=MonitoringEvent.PROGRESS_UPDATE,
                timestamp=time.time(),
                data=progress_data
            )
            
            monitor_data['events'].append(event)
            self._notify_callbacks(event)
    
    def complete_build_monitoring(self, build_id: str, success: bool, result_data: Dict[str, Any] = None):
        """Complete monitoring for a build."""
        with self.lock:
            if build_id not in self.active_monitors:
                return
            
            monitor_data = self.active_monitors[build_id]
            monitor_data['status'] = 'completed' if success else 'failed'
            monitor_data['end_time'] = time.time()
            monitor_data['duration'] = monitor_data['end_time'] - monitor_data['start_time']
            
            # Notify callbacks
            event_type = MonitoringEvent.BUILD_COMPLETED if success else MonitoringEvent.BUILD_FAILED
            event = MonitoringData(
                build_id=build_id,
                event_type=event_type,
                timestamp=time.time(),
                data=result_data or {}
            )
            
            monitor_data['events'].append(event)
            self._notify_callbacks(event)
            
            # Remove from active monitors
            del self.active_monitors[build_id]
            
            logger.info(f"Completed monitoring build {build_id} (success={success})")
    
    def get_build_status(self, build_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of monitored build."""
        with self.lock:
            return self.active_monitors.get(build_id)
    
    def list_active_builds(self) -> List[str]:
        """List actively monitored builds."""
        with self.lock:
            return list(self.active_monitors.keys())
    
    def add_event_callback(self, callback: Callable[[MonitoringData], None]):
        """Add callback for monitoring events."""
        self.event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[MonitoringData], None]):
        """Remove event callback."""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    def _notify_callbacks(self, event: MonitoringData):
        """Notify all event callbacks."""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Monitoring callback failed: {e}")


class ProgressTracker:
    """Track detailed progress of operations."""
    
    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        self.start_time = time.time()
        self.phases = []
        self.current_phase = None
        self.progress_callbacks = []
    
    def start_phase(self, phase_name: str, estimated_duration: Optional[float] = None):
        """Start a new phase."""
        if self.current_phase:
            self.complete_phase(self.current_phase['name'])
        
        phase_info = {
            'name': phase_name,
            'start_time': time.time(),
            'estimated_duration': estimated_duration,
            'status': 'running'
        }
        
        self.current_phase = phase_info
        self.phases.append(phase_info)
        
        self._notify_progress()
    
    def complete_phase(self, phase_name: str, success: bool = True):
        """Complete current phase."""
        if self.current_phase and self.current_phase['name'] == phase_name:
            self.current_phase['end_time'] = time.time()
            self.current_phase['duration'] = self.current_phase['end_time'] - self.current_phase['start_time']
            self.current_phase['status'] = 'completed' if success else 'failed'
            self.current_phase = None
            
            self._notify_progress()
    
    def update_phase_progress(self, progress_percent: float, message: Optional[str] = None):
        """Update progress of current phase."""
        if self.current_phase:
            self.current_phase['progress_percent'] = progress_percent
            if message:
                self.current_phase['message'] = message
            
            self._notify_progress()
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress information."""
        completed_phases = [p for p in self.phases if p.get('status') == 'completed']
        total_phases = len(self.phases)
        
        if self.current_phase:
            current_progress = self.current_phase.get('progress_percent', 0) / 100.0
            overall_progress = (len(completed_phases) + current_progress) / max(1, total_phases) * 100
        else:
            overall_progress = len(completed_phases) / max(1, total_phases) * 100
        
        return {
            'operation_id': self.operation_id,
            'overall_progress_percent': min(100, overall_progress),
            'completed_phases': len(completed_phases),
            'total_phases': total_phases,
            'current_phase': self.current_phase,
            'elapsed_time': time.time() - self.start_time,
            'phases': self.phases
        }
    
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add progress update callback."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """Notify progress callbacks."""
        progress_info = self.get_overall_progress()
        
        for callback in self.progress_callbacks:
            try:
                callback(progress_info)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.monitoring_active = False
        self.monitor_thread = None
        self.resource_history = []
        self.alert_callbacks = []
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 95.0
        }
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("Stopped resource monitoring")
    
    def set_threshold(self, resource: str, threshold: float):
        """Set alert threshold for resource."""
        self.thresholds[resource] = threshold
    
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """Add callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'timestamp': time.time()
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_percent': 0.0,
                'timestamp': time.time()
            }
    
    def get_usage_history(self, minutes: int = 10) -> List[Dict[str, float]]:
        """Get resource usage history."""
        cutoff_time = time.time() - (minutes * 60)
        return [usage for usage in self.resource_history if usage['timestamp'] >= cutoff_time]
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                usage = self.get_current_usage()
                self.resource_history.append(usage)
                
                # Trim history (keep last hour)
                cutoff_time = time.time() - 3600
                self.resource_history = [
                    u for u in self.resource_history if u['timestamp'] >= cutoff_time
                ]
                
                # Check thresholds
                self._check_thresholds(usage)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _check_thresholds(self, usage: Dict[str, float]):
        """Check if any thresholds are exceeded."""
        for resource, threshold in self.thresholds.items():
            if resource in usage and usage[resource] > threshold:
                for callback in self.alert_callbacks:
                    try:
                        callback(resource, usage[resource], threshold)
                    except Exception as e:
                        logger.warning(f"Resource alert callback failed: {e}")