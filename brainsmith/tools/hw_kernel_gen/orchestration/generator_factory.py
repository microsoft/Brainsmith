"""
Generator Factory System for Hardware Kernel Generator.

This module provides a comprehensive factory system for creating, managing,
and caching generator instances with dynamic capability-based selection.
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type, Set, Callable, Union
from enum import Enum
from collections import defaultdict, OrderedDict
import weakref
import inspect

from ..enhanced_config import PipelineConfig, GeneratorType
from ..enhanced_data_structures import RTLModule
from ..enhanced_generator_base import GeneratorBase, GenerationResult, GeneratedArtifact
from ..errors import BrainsmithError, GeneratorError, ConfigurationError


class GeneratorCapability(Enum):
    """Capabilities that generators can provide."""
    HW_CUSTOM_OP = "hw_custom_op"
    RTL_BACKEND = "rtl_backend"
    DOCUMENTATION = "documentation"
    TEST_GENERATION = "test_generation"
    WRAPPER_GENERATION = "wrapper_generation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    DATAFLOW_INTEGRATION = "dataflow_integration"
    LEGACY_COMPATIBILITY = "legacy_compatibility"
    ERROR_RECOVERY = "error_recovery"


class GeneratorPriority(Enum):
    """Priority levels for generator selection."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GeneratorMetadata:
    """Metadata about a registered generator."""
    generator_class: Type[GeneratorBase]
    capabilities: Set[GeneratorCapability]
    priority: GeneratorPriority
    version: str
    description: str
    dependencies: Set[str] = field(default_factory=set)
    supported_configs: Set[GeneratorType] = field(default_factory=set)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    usage_count: int = 0
    last_used: float = 0.0


@dataclass
class GeneratorConfiguration:
    """Configuration for generator instantiation."""
    generator_type: GeneratorType
    config: PipelineConfig
    capabilities_required: Set[GeneratorCapability] = field(default_factory=set)
    capabilities_preferred: Set[GeneratorCapability] = field(default_factory=set)
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    allow_fallback: bool = True
    cache_enabled: bool = True
    pool_enabled: bool = True
    max_instances: int = 5
    instance_timeout: float = 300.0  # 5 minutes


class GeneratorSelectionStrategy(Enum):
    """Strategies for selecting generators."""
    CAPABILITY_MATCH = "capability_match"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    COMPATIBILITY_FIRST = "compatibility_first"


class GeneratorRegistry:
    """Registry for managing generator types and their metadata."""
    
    def __init__(self):
        self._generators: Dict[str, GeneratorMetadata] = {}
        self._capability_index: Dict[GeneratorCapability, Set[str]] = defaultdict(set)
        self._priority_index: Dict[GeneratorPriority, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._registry_stats = {
            "registrations": 0,
            "lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def register_generator(
        self,
        name: str,
        generator_class: Type[GeneratorBase],
        capabilities: Set[GeneratorCapability],
        priority: GeneratorPriority = GeneratorPriority.MEDIUM,
        version: str = "1.0.0",
        description: str = "",
        dependencies: Set[str] = None,
        supported_configs: Set[GeneratorType] = None
    ) -> None:
        """Register a generator with the registry."""
        with self._lock:
            if not issubclass(generator_class, GeneratorBase):
                raise ValueError(f"Generator {name} must inherit from GeneratorBase")
            
            metadata = GeneratorMetadata(
                generator_class=generator_class,
                capabilities=capabilities or set(),
                priority=priority,
                version=version,
                description=description,
                dependencies=dependencies or set(),
                supported_configs=supported_configs or set()
            )
            
            # Update indices
            self._generators[name] = metadata
            for capability in capabilities:
                self._capability_index[capability].add(name)
            self._priority_index[priority].add(name)
            
            self._registry_stats["registrations"] += 1
    
    def get_generator_metadata(self, name: str) -> Optional[GeneratorMetadata]:
        """Get metadata for a specific generator."""
        with self._lock:
            self._registry_stats["lookups"] += 1
            return self._generators.get(name)
    
    def find_generators_by_capability(
        self,
        required_capabilities: Set[GeneratorCapability],
        preferred_capabilities: Set[GeneratorCapability] = None
    ) -> List[str]:
        """Find generators that match capability requirements."""
        with self._lock:
            self._registry_stats["lookups"] += 1
            
            # Find generators with all required capabilities
            candidates = None
            for capability in required_capabilities:
                capable_generators = self._capability_index.get(capability, set())
                if candidates is None:
                    candidates = capable_generators.copy()
                else:
                    candidates &= capable_generators
            
            if candidates is None:
                return []
            
            # Score by preferred capabilities
            if preferred_capabilities:
                scored_candidates = []
                for name in candidates:
                    metadata = self._generators[name]
                    score = len(metadata.capabilities & preferred_capabilities)
                    scored_candidates.append((score, name))
                
                # Sort by score (descending) and return names
                scored_candidates.sort(reverse=True)
                return [name for score, name in scored_candidates]
            
            return list(candidates)
    
    def find_generators_by_priority(
        self,
        min_priority: GeneratorPriority = GeneratorPriority.LOW
    ) -> List[str]:
        """Find generators with minimum priority level."""
        with self._lock:
            self._registry_stats["lookups"] += 1
            
            generators = []
            for priority in GeneratorPriority:
                if priority.value >= min_priority.value:
                    generators.extend(self._priority_index[priority])
            
            return generators
    
    def get_all_generators(self) -> Dict[str, GeneratorMetadata]:
        """Get all registered generators."""
        with self._lock:
            return self._generators.copy()
    
    def update_usage_statistics(self, name: str) -> None:
        """Update usage statistics for a generator."""
        with self._lock:
            if name in self._generators:
                metadata = self._generators[name]
                metadata.usage_count += 1
                metadata.last_used = time.time()
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            return {
                **self._registry_stats,
                "total_generators": len(self._generators),
                "capabilities_tracked": len(self._capability_index),
                "priorities_tracked": len(self._priority_index)
            }


class GeneratorCache:
    """Cache for generator instances with LRU eviction."""
    
    def __init__(self, max_size: int = 50, ttl: float = 1800.0):  # 30 minutes
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    def get(self, key: str) -> Optional[GeneratorBase]:
        """Get generator from cache."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Check expiration
                if current_time - self._timestamps[key] > self.ttl:
                    self._remove_expired(key)
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats["hits"] += 1
                return self._cache[key]
            
            self._stats["misses"] += 1
            return None
    
    def put(self, key: str, generator: GeneratorBase) -> None:
        """Put generator in cache."""
        with self._lock:
            current_time = time.time()
            
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
            
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            # Add new entry
            self._cache[key] = generator
            self._timestamps[key] = current_time
    
    def _remove_expired(self, key: str) -> None:
        """Remove expired entry."""
        del self._cache[key]
        del self._timestamps[key]
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
            self._stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self._stats,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "max_size": self.max_size
            }


class GeneratorFactory:
    """Main factory for creating and managing generators."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.registry = GeneratorRegistry()
        self.cache = GeneratorCache()
        self._selection_strategy = GeneratorSelectionStrategy.CAPABILITY_MATCH
        self._factory_stats = {
            "creations": 0,
            "cache_hits": 0,
            "selection_time": 0.0,
            "creation_time": 0.0
        }
        self._lock = threading.RLock()
        
        # Register default generators
        self._register_default_generators()
    
    def _register_default_generators(self) -> None:
        """Register default generator implementations."""
        # This would register actual generator implementations
        # For now, we'll create placeholder registrations
        pass
    
    def set_selection_strategy(self, strategy: GeneratorSelectionStrategy) -> None:
        """Set the generator selection strategy."""
        self._selection_strategy = strategy
    
    def create_generator(
        self,
        generator_config: GeneratorConfiguration,
        rtl_module: Optional[RTLModule] = None,
        force_new: bool = False
    ) -> GeneratorBase:
        """Create a generator instance based on configuration."""
        start_time = time.time()
        
        with self._lock:
            # Create cache key
            cache_key = self._create_cache_key(generator_config, rtl_module)
            
            # Check cache first (unless forced)
            if not force_new and generator_config.cache_enabled:
                cached_generator = self.cache.get(cache_key)
                if cached_generator is not None:
                    self._factory_stats["cache_hits"] += 1
                    return cached_generator
            
            # Select appropriate generator
            selection_start = time.time()
            generator_name = self._select_generator(generator_config)
            self._factory_stats["selection_time"] += time.time() - selection_start
            
            if not generator_name:
                raise GeneratorError(
                    f"No suitable generator found for configuration: {generator_config}"
                )
            
            # Create generator instance
            creation_start = time.time()
            generator = self._create_generator_instance(generator_name, generator_config)
            self._factory_stats["creation_time"] += time.time() - creation_start
            
            # Cache if enabled
            if generator_config.cache_enabled:
                self.cache.put(cache_key, generator)
            
            # Update statistics
            self.registry.update_usage_statistics(generator_name)
            self._factory_stats["creations"] += 1
            
            return generator
    
    def _create_cache_key(
        self,
        generator_config: GeneratorConfiguration,
        rtl_module: Optional[RTLModule]
    ) -> str:
        """Create cache key for generator configuration."""
        key_parts = [
            str(generator_config.generator_type.value),
            str(sorted(cap.value for cap in generator_config.capabilities_required)),
            str(generator_config.config.generator_type.value),
            str(generator_config.config.dataflow.mode.value)
        ]
        
        if rtl_module:
            key_parts.append(str(hash(rtl_module.name)))
        
        return "|".join(key_parts)
    
    def _select_generator(self, generator_config: GeneratorConfiguration) -> Optional[str]:
        """Select appropriate generator based on configuration and strategy."""
        # Find candidates by capabilities
        candidates = self.registry.find_generators_by_capability(
            generator_config.capabilities_required,
            generator_config.capabilities_preferred
        )
        
        if not candidates:
            return None
        
        # Apply selection strategy
        if self._selection_strategy == GeneratorSelectionStrategy.CAPABILITY_MATCH:
            return self._select_by_capability_match(candidates, generator_config)
        elif self._selection_strategy == GeneratorSelectionStrategy.PERFORMANCE_OPTIMIZED:
            return self._select_by_performance(candidates, generator_config)
        elif self._selection_strategy == GeneratorSelectionStrategy.PRIORITY_BASED:
            return self._select_by_priority(candidates)
        elif self._selection_strategy == GeneratorSelectionStrategy.LOAD_BALANCED:
            return self._select_by_load_balance(candidates)
        else:
            # Default to first candidate
            return candidates[0] if candidates else None
    
    def _select_by_capability_match(
        self,
        candidates: List[str],
        generator_config: GeneratorConfiguration
    ) -> Optional[str]:
        """Select generator with best capability match."""
        if not candidates:
            return None
        
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            metadata = self.registry.get_generator_metadata(candidate)
            if not metadata:
                continue
            
            # Score based on capability overlap
            required_match = len(metadata.capabilities & generator_config.capabilities_required)
            preferred_match = len(metadata.capabilities & generator_config.capabilities_preferred)
            total_capabilities = len(metadata.capabilities)
            
            # Calculate score (prefer exact matches and avoid over-capability)
            score = (required_match * 10) + (preferred_match * 5) - (total_capabilities * 0.1)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate
    
    def _select_by_performance(
        self,
        candidates: List[str],
        generator_config: GeneratorConfiguration
    ) -> Optional[str]:
        """Select generator based on performance metrics."""
        if not candidates:
            return None
        
        best_candidate = None
        best_score = float('inf')
        
        for candidate in candidates:
            metadata = self.registry.get_generator_metadata(candidate)
            if not metadata:
                continue
            
            # Calculate performance score
            score = 0.0
            for metric, requirement in generator_config.performance_requirements.items():
                actual = metadata.performance_metrics.get(metric, float('inf'))
                if actual > requirement:
                    score += (actual - requirement) * 100  # Penalty for exceeding requirements
            
            if score < best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate or candidates[0]
    
    def _select_by_priority(self, candidates: List[str]) -> Optional[str]:
        """Select generator with highest priority."""
        if not candidates:
            return None
        
        best_candidate = None
        best_priority = GeneratorPriority.LOW
        
        for candidate in candidates:
            metadata = self.registry.get_generator_metadata(candidate)
            if not metadata:
                continue
            
            if metadata.priority.value > best_priority.value:
                best_priority = metadata.priority
                best_candidate = candidate
        
        return best_candidate or candidates[0]
    
    def _select_by_load_balance(self, candidates: List[str]) -> Optional[str]:
        """Select generator with lowest usage."""
        if not candidates:
            return None
        
        best_candidate = None
        lowest_usage = float('inf')
        
        for candidate in candidates:
            metadata = self.registry.get_generator_metadata(candidate)
            if not metadata:
                continue
            
            if metadata.usage_count < lowest_usage:
                lowest_usage = metadata.usage_count
                best_candidate = candidate
        
        return best_candidate or candidates[0]
    
    def _create_generator_instance(
        self,
        generator_name: str,
        generator_config: GeneratorConfiguration
    ) -> GeneratorBase:
        """Create an instance of the specified generator."""
        metadata = self.registry.get_generator_metadata(generator_name)
        if not metadata:
            raise GeneratorError(f"Generator metadata not found: {generator_name}")
        
        try:
            # Get constructor arguments
            generator_class = metadata.generator_class
            init_signature = inspect.signature(generator_class.__init__)
            init_params = list(init_signature.parameters.keys())[1:]  # Skip 'self'
            
            # Prepare constructor arguments
            kwargs = {}
            if 'config' in init_params:
                kwargs['config'] = generator_config.config
            if 'pipeline_config' in init_params:
                kwargs['pipeline_config'] = generator_config.config
            
            # Create instance
            generator = generator_class(**kwargs)
            
            return generator
            
        except Exception as e:
            raise GeneratorError(f"Failed to create generator {generator_name}: {e}")
    
    def get_available_generators(self) -> Dict[str, GeneratorMetadata]:
        """Get all available generators."""
        return self.registry.get_all_generators()
    
    def get_generator_capabilities(self, generator_name: str) -> Set[GeneratorCapability]:
        """Get capabilities of a specific generator."""
        metadata = self.registry.get_generator_metadata(generator_name)
        return metadata.capabilities if metadata else set()
    
    def clear_cache(self) -> None:
        """Clear the generator cache."""
        self.cache.clear()
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """Get factory statistics."""
        with self._lock:
            return {
                "factory_stats": self._factory_stats.copy(),
                "registry_stats": self.registry.get_registry_statistics(),
                "cache_stats": self.cache.get_stats(),
                "selection_strategy": self._selection_strategy.value
            }


# Factory functions and convenience methods

def create_generator_factory(config: Optional[PipelineConfig] = None) -> GeneratorFactory:
    """Create a generator factory with default configuration."""
    return GeneratorFactory(config)


def register_generator(
    factory: GeneratorFactory,
    name: str,
    generator_class: Type[GeneratorBase],
    capabilities: Set[GeneratorCapability],
    **kwargs
) -> None:
    """Register a generator with the factory."""
    factory.registry.register_generator(name, generator_class, capabilities, **kwargs)


def get_generator_capabilities(
    factory: GeneratorFactory,
    generator_name: str
) -> Set[GeneratorCapability]:
    """Get capabilities of a specific generator."""
    return factory.get_generator_capabilities(generator_name)


def create_generator_configuration(
    generator_type: GeneratorType,
    config: PipelineConfig,
    required_capabilities: Set[GeneratorCapability] = None,
    **kwargs
) -> GeneratorConfiguration:
    """Create generator configuration with sensible defaults."""
    return GeneratorConfiguration(
        generator_type=generator_type,
        config=config,
        capabilities_required=required_capabilities or set(),
        **kwargs
    )