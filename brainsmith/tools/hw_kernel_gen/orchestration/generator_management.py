"""
Generator Management System for Hardware Kernel Generator.

This module provides comprehensive generator lifecycle management including
instance pooling, performance monitoring, health checking, and resource optimization.
"""

import time
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable, Type
from enum import Enum
from collections import defaultdict, deque
# import psutil  # Optional dependency for resource monitoring
import logging

from ..enhanced_config import PipelineConfig
from ..enhanced_data_structures import RTLModule
from ..enhanced_generator_base import GenerationResult
from ..enhanced_generator_base import GeneratorBase
from ..errors import BrainsmithError, GeneratorError
from .generator_factory import GeneratorFactory, GeneratorConfiguration, GeneratorCapability


class GeneratorState(Enum):
    """States of generator instances."""
    CREATED = "created"
    INITIALIZED = "initialized"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"


class HealthStatus(Enum):
    """Health status of generators."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class GeneratorInstance:
    """Represents a managed generator instance."""
    generator: GeneratorBase
    instance_id: str
    state: GeneratorState = GeneratorState.CREATED
    created_time: float = field(default_factory=time.time)
    last_used_time: float = field(default_factory=time.time)
    last_health_check: float = 0.0
    usage_count: int = 0
    total_execution_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_usage(self, execution_time: float, success: bool = True) -> None:
        """Update usage statistics."""
        self.last_used_time = time.time()
        self.usage_count += 1
        self.total_execution_time += execution_time
        
        if not success:
            self.error_count += 1
        
        # Update state
        if success:
            self.state = GeneratorState.IDLE
        else:
            self.state = GeneratorState.ERROR
    
    def get_average_execution_time(self) -> float:
        """Get average execution time."""
        if self.usage_count > 0:
            return self.total_execution_time / self.usage_count
        return 0.0
    
    def get_error_rate(self) -> float:
        """Get error rate."""
        if self.usage_count > 0:
            return self.error_count / self.usage_count
        return 0.0
    
    def is_idle(self, idle_threshold: float = 300.0) -> bool:
        """Check if instance has been idle for too long."""
        return (time.time() - self.last_used_time) > idle_threshold
    
    def get_age(self) -> float:
        """Get age of instance in seconds."""
        return time.time() - self.created_time


@dataclass
class PoolConfiguration:
    """Configuration for generator pools."""
    min_instances: int = 1
    max_instances: int = 10
    idle_timeout: float = 300.0  # 5 minutes
    max_age: float = 3600.0  # 1 hour
    health_check_interval: float = 60.0  # 1 minute
    preload_instances: bool = True
    load_balance_strategy: str = "round_robin"  # round_robin, least_used, performance
    enable_metrics: bool = True


class GeneratorPool:
    """Pool for managing generator instances."""
    
    def __init__(
        self,
        generator_name: str,
        generator_config: GeneratorConfiguration,
        pool_config: PoolConfiguration,
        generator_factory: GeneratorFactory
    ):
        self.generator_name = generator_name
        self.generator_config = generator_config
        self.pool_config = pool_config
        self.generator_factory = generator_factory
        
        self._instances: Dict[str, GeneratorInstance] = {}
        self._available_instances: deque = deque()
        self._busy_instances: Set[str] = set()
        self._instance_counter = 0
        self._lock = threading.RLock()
        
        # Pool statistics
        self._pool_stats = {
            "created_instances": 0,
            "destroyed_instances": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "total_requests": 0,
            "current_size": 0,
            "peak_size": 0,
            "health_checks": 0,
            "health_failures": 0
        }
        
        # Health monitoring
        self._last_health_check = 0.0
        self._health_check_thread = None
        self._shutdown = False
        
        # Initialize pool
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the pool with minimum instances."""
        if self.pool_config.preload_instances:
            for _ in range(self.pool_config.min_instances):
                self._create_instance()
    
    def _create_instance(self) -> GeneratorInstance:
        """Create a new generator instance."""
        with self._lock:
            self._instance_counter += 1
            instance_id = f"{self.generator_name}_{self._instance_counter}"
            
            try:
                generator = self.generator_factory.create_generator(
                    self.generator_config,
                    force_new=True
                )
                
                instance = GeneratorInstance(
                    generator=generator,
                    instance_id=instance_id,
                    state=GeneratorState.READY
                )
                
                self._instances[instance_id] = instance
                self._available_instances.append(instance_id)
                
                self._pool_stats["created_instances"] += 1
                self._pool_stats["current_size"] = len(self._instances)
                self._pool_stats["peak_size"] = max(
                    self._pool_stats["peak_size"],
                    self._pool_stats["current_size"]
                )
                
                return instance
                
            except Exception as e:
                raise GeneratorError(f"Failed to create generator instance: {e}")
    
    def acquire_instance(self) -> Optional[GeneratorInstance]:
        """Acquire a generator instance from the pool."""
        with self._lock:
            self._pool_stats["total_requests"] += 1
            
            # Try to get available instance
            if self._available_instances:
                instance_id = self._available_instances.popleft()
                instance = self._instances[instance_id]
                
                # Check if instance is still healthy
                if self._is_instance_healthy(instance):
                    self._busy_instances.add(instance_id)
                    instance.state = GeneratorState.BUSY
                    self._pool_stats["pool_hits"] += 1
                    return instance
                else:
                    # Remove unhealthy instance
                    self._remove_instance(instance_id)
            
            # No available instances, try to create new one
            if len(self._instances) < self.pool_config.max_instances:
                try:
                    instance = self._create_instance()
                    # Remove from available and add to busy
                    self._available_instances.remove(instance.instance_id)
                    self._busy_instances.add(instance.instance_id)
                    instance.state = GeneratorState.BUSY
                    self._pool_stats["pool_misses"] += 1
                    return instance
                except Exception:
                    pass
            
            # Pool is full or creation failed
            self._pool_stats["pool_misses"] += 1
            return None
    
    def release_instance(
        self,
        instance: GeneratorInstance,
        execution_time: float = 0.0,
        success: bool = True
    ) -> None:
        """Release a generator instance back to the pool."""
        with self._lock:
            if instance.instance_id not in self._instances:
                return
            
            # Update instance statistics
            instance.update_usage(execution_time, success)
            
            # Move from busy to available
            self._busy_instances.discard(instance.instance_id)
            
            # Check if instance should be retired
            if (instance.is_idle(self.pool_config.idle_timeout) or
                instance.get_age() > self.pool_config.max_age or
                instance.get_error_rate() > 0.1):  # High error rate
                self._remove_instance(instance.instance_id)
            else:
                instance.state = GeneratorState.IDLE
                self._available_instances.append(instance.instance_id)
    
    def _remove_instance(self, instance_id: str) -> None:
        """Remove instance from pool."""
        if instance_id in self._instances:
            instance = self._instances[instance_id]
            instance.state = GeneratorState.TERMINATED
            
            # Remove from all tracking structures
            del self._instances[instance_id]
            self._busy_instances.discard(instance_id)
            try:
                self._available_instances.remove(instance_id)
            except ValueError:
                pass
            
            self._pool_stats["destroyed_instances"] += 1
            self._pool_stats["current_size"] = len(self._instances)
    
    def _is_instance_healthy(self, instance: GeneratorInstance) -> bool:
        """Check if instance is healthy."""
        try:
            # Basic health checks
            if instance.state == GeneratorState.ERROR:
                return False
            
            if instance.get_error_rate() > 0.2:  # Too many errors
                return False
            
            # Could add more sophisticated health checks here
            return True
            
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the pool."""
        with self._lock:
            self._pool_stats["health_checks"] += 1
            
            healthy_instances = 0
            unhealthy_instances = 0
            
            for instance in self._instances.values():
                if self._is_instance_healthy(instance):
                    healthy_instances += 1
                else:
                    unhealthy_instances += 1
                    self._pool_stats["health_failures"] += 1
            
            health_status = {
                "pool_name": self.generator_name,
                "total_instances": len(self._instances),
                "available_instances": len(self._available_instances),
                "busy_instances": len(self._busy_instances),
                "healthy_instances": healthy_instances,
                "unhealthy_instances": unhealthy_instances,
                "pool_utilization": len(self._busy_instances) / max(len(self._instances), 1),
                "last_health_check": time.time()
            }
            
            self._last_health_check = time.time()
            return health_status
    
    def cleanup(self) -> None:
        """Cleanup pool resources."""
        with self._lock:
            self._shutdown = True
            
            # Remove all instances
            for instance_id in list(self._instances.keys()):
                self._remove_instance(instance_id)
            
            self._available_instances.clear()
            self._busy_instances.clear()
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            stats = self._pool_stats.copy()
            
            # Add computed statistics
            if stats["total_requests"] > 0:
                stats["hit_rate"] = stats["pool_hits"] / stats["total_requests"]
                stats["miss_rate"] = stats["pool_misses"] / stats["total_requests"]
            else:
                stats["hit_rate"] = 0.0
                stats["miss_rate"] = 0.0
            
            # Add instance statistics
            if self._instances:
                total_usage = sum(inst.usage_count for inst in self._instances.values())
                total_errors = sum(inst.error_count for inst in self._instances.values())
                total_exec_time = sum(inst.total_execution_time for inst in self._instances.values())
                
                stats["average_usage_per_instance"] = total_usage / len(self._instances)
                stats["total_error_count"] = total_errors
                stats["total_execution_time"] = total_exec_time
                stats["average_execution_time"] = (
                    total_exec_time / total_usage if total_usage > 0 else 0.0
                )
            
            return stats


class GeneratorLifecycle:
    """Manages the lifecycle of generator instances."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._lifecycle_hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._lifecycle_stats = {
            "creations": 0,
            "initializations": 0,
            "shutdowns": 0,
            "errors": 0
        }
    
    def register_hook(self, event: str, hook: Callable[[GeneratorInstance], None]) -> None:
        """Register lifecycle hook."""
        self._lifecycle_hooks[event].append(hook)
    
    def create_generator(
        self,
        generator_factory: GeneratorFactory,
        generator_config: GeneratorConfiguration
    ) -> GeneratorInstance:
        """Create and initialize generator instance."""
        try:
            # Create generator
            generator = generator_factory.create_generator(generator_config)
            
            instance = GeneratorInstance(
                generator=generator,
                instance_id=f"gen_{int(time.time() * 1000000)}",
                state=GeneratorState.CREATED
            )
            
            self._lifecycle_stats["creations"] += 1
            
            # Run creation hooks
            self._run_hooks("on_create", instance)
            
            # Initialize generator
            self._initialize_generator(instance)
            
            return instance
            
        except Exception as e:
            self._lifecycle_stats["errors"] += 1
            raise GeneratorError(f"Generator lifecycle creation failed: {e}")
    
    def _initialize_generator(self, instance: GeneratorInstance) -> None:
        """Initialize generator instance."""
        try:
            # Run initialization hooks
            self._run_hooks("on_initialize", instance)
            
            instance.state = GeneratorState.READY
            self._lifecycle_stats["initializations"] += 1
            
        except Exception as e:
            instance.state = GeneratorState.ERROR
            instance.last_error = str(e)
            self._lifecycle_stats["errors"] += 1
            raise GeneratorError(f"Generator initialization failed: {e}")
    
    def shutdown_generator(self, instance: GeneratorInstance) -> None:
        """Shutdown generator instance."""
        try:
            # Run shutdown hooks
            self._run_hooks("on_shutdown", instance)
            
            instance.state = GeneratorState.TERMINATED
            self._lifecycle_stats["shutdowns"] += 1
            
        except Exception as e:
            self._lifecycle_stats["errors"] += 1
            logging.warning(f"Error during generator shutdown: {e}")
    
    def _run_hooks(self, event: str, instance: GeneratorInstance) -> None:
        """Run lifecycle hooks for event."""
        for hook in self._lifecycle_hooks.get(event, []):
            try:
                hook(instance)
            except Exception as e:
                logging.warning(f"Lifecycle hook failed for {event}: {e}")
    
    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get lifecycle statistics."""
        return self._lifecycle_stats.copy()


class GeneratorMetrics:
    """Collects and manages generator performance metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "execution_count": 0,
            "total_execution_time": 0.0,
            "error_count": 0,
            "memory_usage_samples": [],
            "cpu_usage_samples": [],
            "last_updated": 0.0
        })
        self._lock = threading.RLock()
    
    def record_execution(
        self,
        generator_name: str,
        execution_time: float,
        success: bool = True,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None
    ) -> None:
        """Record generator execution metrics."""
        with self._lock:
            metrics = self._metrics[generator_name]
            
            metrics["execution_count"] += 1
            metrics["total_execution_time"] += execution_time
            metrics["last_updated"] = time.time()
            
            if not success:
                metrics["error_count"] += 1
            
            if memory_usage is not None:
                metrics["memory_usage_samples"].append(memory_usage)
                # Keep only recent samples
                if len(metrics["memory_usage_samples"]) > 100:
                    metrics["memory_usage_samples"] = metrics["memory_usage_samples"][-100:]
            
            if cpu_usage is not None:
                metrics["cpu_usage_samples"].append(cpu_usage)
                if len(metrics["cpu_usage_samples"]) > 100:
                    metrics["cpu_usage_samples"] = metrics["cpu_usage_samples"][-100:]
    
    def get_metrics(self, generator_name: str) -> Dict[str, Any]:
        """Get metrics for specific generator."""
        with self._lock:
            if generator_name not in self._metrics:
                return {}
            
            metrics = self._metrics[generator_name].copy()
            
            # Calculate derived metrics
            if metrics["execution_count"] > 0:
                metrics["average_execution_time"] = (
                    metrics["total_execution_time"] / metrics["execution_count"]
                )
                metrics["error_rate"] = (
                    metrics["error_count"] / metrics["execution_count"]
                )
            else:
                metrics["average_execution_time"] = 0.0
                metrics["error_rate"] = 0.0
            
            # Calculate resource usage statistics
            if metrics["memory_usage_samples"]:
                samples = metrics["memory_usage_samples"]
                metrics["average_memory_usage"] = sum(samples) / len(samples)
                metrics["peak_memory_usage"] = max(samples)
            
            if metrics["cpu_usage_samples"]:
                samples = metrics["cpu_usage_samples"]
                metrics["average_cpu_usage"] = sum(samples) / len(samples)
                metrics["peak_cpu_usage"] = max(samples)
            
            return metrics
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all generators."""
        with self._lock:
            all_metrics = {}
            for generator_name in self._metrics:
                all_metrics[generator_name] = self.get_metrics(generator_name)
            return all_metrics
    
    def clear_metrics(self, generator_name: Optional[str] = None) -> None:
        """Clear metrics for specific generator or all generators."""
        with self._lock:
            if generator_name:
                if generator_name in self._metrics:
                    del self._metrics[generator_name]
            else:
                self._metrics.clear()


class GeneratorManager:
    """Central manager for all generator instances and pools."""
    
    def __init__(
        self,
        config: PipelineConfig,
        generator_factory: GeneratorFactory
    ):
        self.config = config
        self.generator_factory = generator_factory
        self.lifecycle = GeneratorLifecycle(config)
        self.metrics = GeneratorMetrics()
        
        self._pools: Dict[str, GeneratorPool] = {}
        self._pool_configs: Dict[str, PoolConfiguration] = {}
        self._lock = threading.RLock()
        
        # Manager statistics
        self._manager_stats = {
            "pools_created": 0,
            "total_acquisitions": 0,
            "total_releases": 0,
            "failed_acquisitions": 0
        }
        
        # Health monitoring
        self._health_monitor_thread = None
        self._shutdown = False
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def create_pool(
        self,
        generator_name: str,
        generator_config: GeneratorConfiguration,
        pool_config: Optional[PoolConfiguration] = None
    ) -> GeneratorPool:
        """Create a generator pool."""
        with self._lock:
            if generator_name in self._pools:
                return self._pools[generator_name]
            
            pool_config = pool_config or PoolConfiguration()
            
            pool = GeneratorPool(
                generator_name=generator_name,
                generator_config=generator_config,
                pool_config=pool_config,
                generator_factory=self.generator_factory
            )
            
            self._pools[generator_name] = pool
            self._pool_configs[generator_name] = pool_config
            self._manager_stats["pools_created"] += 1
            
            return pool
    
    def acquire_generator(
        self,
        generator_name: str,
        generator_config: GeneratorConfiguration,
        pool_config: Optional[PoolConfiguration] = None
    ) -> Optional[GeneratorInstance]:
        """Acquire a generator instance."""
        with self._lock:
            self._manager_stats["total_acquisitions"] += 1
            
            # Get or create pool
            if generator_name not in self._pools:
                self.create_pool(generator_name, generator_config, pool_config)
            
            pool = self._pools[generator_name]
            instance = pool.acquire_instance()
            
            if instance is None:
                self._manager_stats["failed_acquisitions"] += 1
            
            return instance
    
    def release_generator(
        self,
        instance: GeneratorInstance,
        execution_time: float = 0.0,
        success: bool = True
    ) -> None:
        """Release a generator instance."""
        with self._lock:
            self._manager_stats["total_releases"] += 1
            
            # Find the pool this instance belongs to
            pool = None
            for pool_name, p in self._pools.items():
                if instance.instance_id in p._instances:
                    pool = p
                    break
            
            if pool:
                pool.release_instance(instance, execution_time, success)
                
                # Record metrics
                self.metrics.record_execution(
                    pool.generator_name,
                    execution_time,
                    success,
                    instance.memory_usage,
                    instance.cpu_usage
                )
    
    def get_pool(self, generator_name: str) -> Optional[GeneratorPool]:
        """Get pool by name."""
        return self._pools.get(generator_name)
    
    def remove_pool(self, generator_name: str) -> None:
        """Remove a generator pool."""
        with self._lock:
            if generator_name in self._pools:
                pool = self._pools[generator_name]
                pool.cleanup()
                del self._pools[generator_name]
                del self._pool_configs[generator_name]
    
    def _start_health_monitoring(self) -> None:
        """Start health monitoring thread."""
        def health_monitor():
            while not self._shutdown:
                try:
                    self._perform_health_checks()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logging.warning(f"Health monitoring error: {e}")
        
        self._health_monitor_thread = threading.Thread(
            target=health_monitor,
            daemon=True
        )
        self._health_monitor_thread.start()
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all pools."""
        with self._lock:
            for pool in self._pools.values():
                try:
                    pool.health_check()
                except Exception as e:
                    logging.warning(f"Health check failed for pool {pool.generator_name}: {e}")
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        with self._lock:
            stats = self._manager_stats.copy()
            
            # Add pool statistics
            pool_stats = {}
            for name, pool in self._pools.items():
                pool_stats[name] = pool.get_pool_statistics()
            stats["pool_statistics"] = pool_stats
            
            # Add lifecycle statistics
            stats["lifecycle_statistics"] = self.lifecycle.get_lifecycle_statistics()
            
            # Add metrics
            stats["generator_metrics"] = self.metrics.get_all_metrics()
            
            # Calculate derived statistics
            if stats["total_acquisitions"] > 0:
                stats["acquisition_failure_rate"] = (
                    stats["failed_acquisitions"] / stats["total_acquisitions"]
                )
            else:
                stats["acquisition_failure_rate"] = 0.0
            
            return stats
    
    def shutdown(self) -> None:
        """Shutdown the generator manager."""
        with self._lock:
            self._shutdown = True
            
            # Cleanup all pools
            for pool in self._pools.values():
                pool.cleanup()
            
            self._pools.clear()
            self._pool_configs.clear()


# Factory functions and convenience methods

def create_generator_manager(
    config: PipelineConfig,
    generator_factory: GeneratorFactory
) -> GeneratorManager:
    """Create generator manager with configuration."""
    return GeneratorManager(config, generator_factory)


def get_generator_statistics(manager: GeneratorManager) -> Dict[str, Any]:
    """Get comprehensive generator statistics."""
    return manager.get_manager_statistics()


def create_pool_configuration(
    min_instances: int = 1,
    max_instances: int = 10,
    **kwargs
) -> PoolConfiguration:
    """Create pool configuration with defaults."""
    return PoolConfiguration(
        min_instances=min_instances,
        max_instances=max_instances,
        **kwargs
    )