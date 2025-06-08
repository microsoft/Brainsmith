"""
Test suite for Generator Factory System.

Tests the complete generator factory functionality including registry,
cache, factory creation, and capability-based selection.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig, GeneratorType
from brainsmith.tools.hw_kernel_gen.enhanced_data_structures import RTLModule, RTLInterface, RTLSignal
from brainsmith.tools.hw_kernel_gen.enhanced_generator_base import GeneratorBase
from brainsmith.tools.hw_kernel_gen.orchestration.generator_factory import (
    GeneratorFactory, GeneratorRegistry, GeneratorCache, GeneratorConfiguration,
    GeneratorCapability, GeneratorPriority, GeneratorMetadata, GeneratorSelectionStrategy,
    create_generator_factory, register_generator, get_generator_capabilities,
    create_generator_configuration
)
from brainsmith.tools.hw_kernel_gen.errors import GeneratorError


class MockGenerator(GeneratorBase):
    """Mock generator for testing."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.generation_count = 0
    
    def get_template_name(self) -> str:
        """Get the primary template name for this generator."""
        return "mock_template.py.j2"
    
    def get_artifact_type(self) -> str:
        """Get the artifact type produced by this generator."""
        return "mock_artifact"
    
    def generate(self, inputs):
        self.generation_count += 1
        from brainsmith.tools.hw_kernel_gen.enhanced_generator_base import GenerationResult
        return GenerationResult(success=True, artifacts=[], metadata={"mock": True})


class TestGeneratorRegistry:
    """Test generator registry functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = GeneratorRegistry()
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        assert len(self.registry._generators) == 0
        assert len(self.registry._capability_index) == 0
        assert len(self.registry._priority_index) == 0
    
    def test_register_generator(self):
        """Test generator registration."""
        capabilities = {GeneratorCapability.HW_CUSTOM_OP, GeneratorCapability.DATAFLOW_INTEGRATION}
        
        self.registry.register_generator(
            name="test_generator",
            generator_class=MockGenerator,
            capabilities=capabilities,
            priority=GeneratorPriority.HIGH,
            version="1.0.0",
            description="Test generator"
        )
        
        # Check registration
        assert "test_generator" in self.registry._generators
        metadata = self.registry._generators["test_generator"]
        assert metadata.generator_class == MockGenerator
        assert metadata.capabilities == capabilities
        assert metadata.priority == GeneratorPriority.HIGH
        
        # Check indices
        for capability in capabilities:
            assert "test_generator" in self.registry._capability_index[capability]
        assert "test_generator" in self.registry._priority_index[GeneratorPriority.HIGH]
    
    def test_get_generator_metadata(self):
        """Test getting generator metadata."""
        capabilities = {GeneratorCapability.RTL_BACKEND}
        
        self.registry.register_generator(
            "rtl_generator",
            MockGenerator,
            capabilities
        )
        
        metadata = self.registry.get_generator_metadata("rtl_generator")
        assert metadata is not None
        assert metadata.generator_class == MockGenerator
        assert metadata.capabilities == capabilities
        
        # Test non-existent generator
        assert self.registry.get_generator_metadata("nonexistent") is None
    
    def test_find_generators_by_capability(self):
        """Test finding generators by capability."""
        # Register generators with different capabilities
        self.registry.register_generator(
            "hw_gen", MockGenerator, {GeneratorCapability.HW_CUSTOM_OP}
        )
        self.registry.register_generator(
            "rtl_gen", MockGenerator, {GeneratorCapability.RTL_BACKEND}
        )
        self.registry.register_generator(
            "multi_gen", MockGenerator, 
            {GeneratorCapability.HW_CUSTOM_OP, GeneratorCapability.RTL_BACKEND, GeneratorCapability.DOCUMENTATION}
        )
        
        # Test finding by single capability
        hw_generators = self.registry.find_generators_by_capability({GeneratorCapability.HW_CUSTOM_OP})
        assert "hw_gen" in hw_generators
        assert "multi_gen" in hw_generators
        assert "rtl_gen" not in hw_generators
        
        # Test finding by multiple capabilities
        multi_cap_generators = self.registry.find_generators_by_capability(
            {GeneratorCapability.HW_CUSTOM_OP, GeneratorCapability.RTL_BACKEND}
        )
        assert "multi_gen" in multi_cap_generators
        assert "hw_gen" not in multi_cap_generators
        assert "rtl_gen" not in multi_cap_generators
        
        # Test with preferred capabilities
        preferred_generators = self.registry.find_generators_by_capability(
            required_capabilities={GeneratorCapability.HW_CUSTOM_OP},
            preferred_capabilities={GeneratorCapability.DOCUMENTATION}
        )
        # multi_gen should be first because it has the preferred capability
        assert preferred_generators[0] == "multi_gen"
    
    def test_find_generators_by_priority(self):
        """Test finding generators by priority."""
        self.registry.register_generator(
            "low_gen", MockGenerator, {GeneratorCapability.HW_CUSTOM_OP}, GeneratorPriority.LOW
        )
        self.registry.register_generator(
            "high_gen", MockGenerator, {GeneratorCapability.HW_CUSTOM_OP}, GeneratorPriority.HIGH
        )
        self.registry.register_generator(
            "medium_gen", MockGenerator, {GeneratorCapability.HW_CUSTOM_OP}, GeneratorPriority.MEDIUM
        )
        
        # Test minimum priority
        high_priority_gens = self.registry.find_generators_by_priority(GeneratorPriority.HIGH)
        assert "high_gen" in high_priority_gens
        assert "low_gen" not in high_priority_gens
        
        medium_priority_gens = self.registry.find_generators_by_priority(GeneratorPriority.MEDIUM)
        assert "high_gen" in medium_priority_gens
        assert "medium_gen" in medium_priority_gens
        assert "low_gen" not in medium_priority_gens
    
    def test_update_usage_statistics(self):
        """Test usage statistics updates."""
        self.registry.register_generator(
            "stat_gen", MockGenerator, {GeneratorCapability.HW_CUSTOM_OP}
        )
        
        # Initial usage should be 0
        metadata = self.registry.get_generator_metadata("stat_gen")
        assert metadata.usage_count == 0
        assert metadata.last_used == 0.0
        
        # Update usage
        self.registry.update_usage_statistics("stat_gen")
        
        # Check updated statistics
        updated_metadata = self.registry.get_generator_metadata("stat_gen")
        assert updated_metadata.usage_count == 1
        assert updated_metadata.last_used > 0.0
    
    def test_registry_statistics(self):
        """Test registry statistics."""
        initial_stats = self.registry.get_registry_statistics()
        assert initial_stats["total_generators"] == 0
        assert initial_stats["registrations"] == 0
        
        # Register a generator
        self.registry.register_generator(
            "test_gen", MockGenerator, {GeneratorCapability.HW_CUSTOM_OP}
        )
        
        # Look up generator to test lookup stats
        self.registry.get_generator_metadata("test_gen")
        
        stats = self.registry.get_registry_statistics()
        assert stats["total_generators"] == 1
        assert stats["registrations"] == 1
        assert stats["lookups"] == 1


class TestGeneratorCache:
    """Test generator cache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = GeneratorCache(max_size=3, ttl=1.0)  # Small cache for testing
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.max_size == 3
        assert self.cache.ttl == 1.0
        assert len(self.cache._cache) == 0
    
    def test_cache_get_miss(self):
        """Test cache miss."""
        generator = self.cache.get("nonexistent_key")
        assert generator is None
        
        stats = self.cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
    
    def test_cache_put_and_get_hit(self):
        """Test cache put and hit."""
        mock_generator = MockGenerator()
        
        # Put in cache
        self.cache.put("test_key", mock_generator)
        
        # Get from cache (should be hit)
        cached_generator = self.cache.get("test_key")
        assert cached_generator is mock_generator
        
        stats = self.cache.get_stats()
        assert stats["hits"] == 1
        assert stats["cache_size"] == 1
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        mock_generator = MockGenerator()
        
        # Put in cache
        self.cache.put("expiring_key", mock_generator)
        
        # Should be available immediately
        assert self.cache.get("expiring_key") is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        assert self.cache.get("expiring_key") is None
        
        stats = self.cache.get_stats()
        assert stats["expirations"] >= 1
    
    def test_cache_size_limit(self):
        """Test cache size limitation."""
        # Fill cache beyond limit
        for i in range(5):
            self.cache.put(f"key_{i}", MockGenerator())
        
        # Cache should not exceed max size significantly
        stats = self.cache.get_stats()
        assert stats["cache_size"] <= self.cache.max_size
        assert stats["evictions"] > 0
    
    def test_cache_clear(self):
        """Test cache clearing."""
        # Add items to cache
        self.cache.put("key1", MockGenerator())
        self.cache.put("key2", MockGenerator())
        
        assert self.cache.get_stats()["cache_size"] == 2
        
        # Clear cache
        self.cache.clear()
        
        assert self.cache.get_stats()["cache_size"] == 0
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None


class TestGeneratorFactory:
    """Test generator factory functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.factory = GeneratorFactory(self.config)
        
        # Register a mock generator for testing
        self.factory.registry.register_generator(
            name="mock_generator",
            generator_class=MockGenerator,
            capabilities={GeneratorCapability.HW_CUSTOM_OP},
            priority=GeneratorPriority.MEDIUM
        )
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        assert self.factory.config is not None
        assert self.factory.registry is not None
        assert self.factory.cache is not None
    
    def test_create_generator_cache_miss(self):
        """Test generator creation with cache miss."""
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        
        generator_config = GeneratorConfiguration(
            generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
            config=self.config,
            capabilities_required={GeneratorCapability.HW_CUSTOM_OP}
        )
        
        # Create generator
        generator = self.factory.create_generator(generator_config, rtl_module)
        
        assert generator is not None
        assert isinstance(generator, MockGenerator)
        
        # Check statistics
        stats = self.factory.get_factory_statistics()
        assert stats["factory_stats"]["creations"] == 1
    
    def test_create_generator_cache_hit(self):
        """Test generator creation with cache hit."""
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        
        generator_config = GeneratorConfiguration(
            generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
            config=self.config,
            capabilities_required={GeneratorCapability.HW_CUSTOM_OP}
        )
        
        # Create generator twice
        generator1 = self.factory.create_generator(generator_config, rtl_module)
        generator2 = self.factory.create_generator(generator_config, rtl_module)
        
        # Should be the same instance (cached)
        assert generator1 is generator2
        
        # Check cache hit statistics
        stats = self.factory.get_factory_statistics()
        assert stats["factory_stats"]["cache_hits"] >= 1
    
    def test_create_generator_force_new(self):
        """Test generator creation with force_new flag."""
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        
        generator_config = GeneratorConfiguration(
            generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
            config=self.config,
            capabilities_required={GeneratorCapability.HW_CUSTOM_OP}
        )
        
        # Create generator twice with force_new
        generator1 = self.factory.create_generator(generator_config, rtl_module)
        generator2 = self.factory.create_generator(generator_config, rtl_module, force_new=True)
        
        # Should be different instances
        assert generator1 is not generator2
    
    def test_create_generator_no_suitable_generator(self):
        """Test generator creation when no suitable generator exists."""
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        
        generator_config = GeneratorConfiguration(
            generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
            config=self.config,
            capabilities_required={GeneratorCapability.OPTIMIZATION}  # Not available
        )
        
        # Should raise GeneratorError
        with pytest.raises(GeneratorError, match="No suitable generator found"):
            self.factory.create_generator(generator_config, rtl_module)
    
    def test_selection_strategies(self):
        """Test different generator selection strategies."""
        # Register multiple generators with different capabilities
        self.factory.registry.register_generator(
            "basic_gen", MockGenerator, {GeneratorCapability.HW_CUSTOM_OP}, GeneratorPriority.LOW
        )
        self.factory.registry.register_generator(
            "advanced_gen", MockGenerator, 
            {GeneratorCapability.HW_CUSTOM_OP, GeneratorCapability.OPTIMIZATION}, 
            GeneratorPriority.HIGH
        )
        
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        generator_config = GeneratorConfiguration(
            generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
            config=self.config,
            capabilities_required={GeneratorCapability.HW_CUSTOM_OP},
            capabilities_preferred={GeneratorCapability.OPTIMIZATION}
        )
        
        # Test capability match strategy
        self.factory.set_selection_strategy(GeneratorSelectionStrategy.CAPABILITY_MATCH)
        generator = self.factory.create_generator(generator_config, rtl_module, force_new=True)
        assert generator is not None
        
        # Test priority-based strategy
        self.factory.set_selection_strategy(GeneratorSelectionStrategy.PRIORITY_BASED)
        generator = self.factory.create_generator(generator_config, rtl_module, force_new=True)
        assert generator is not None
    
    def test_get_available_generators(self):
        """Test getting available generators."""
        generators = self.factory.get_available_generators()
        assert "mock_generator" in generators
        assert isinstance(generators["mock_generator"], GeneratorMetadata)
    
    def test_get_generator_capabilities(self):
        """Test getting generator capabilities."""
        capabilities = self.factory.get_generator_capabilities("mock_generator")
        assert GeneratorCapability.HW_CUSTOM_OP in capabilities
    
    def test_clear_cache(self):
        """Test clearing factory cache."""
        rtl_module = RTLModule("test_module", interfaces=[], parameters={})
        generator_config = GeneratorConfiguration(
            generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
            config=self.config,
            capabilities_required={GeneratorCapability.HW_CUSTOM_OP}
        )
        
        # Create generator to populate cache
        self.factory.create_generator(generator_config, rtl_module)
        
        # Clear cache
        self.factory.clear_cache()
        
        # Cache should be empty
        cache_stats = self.factory.get_factory_statistics()["cache_stats"]
        assert cache_stats["cache_size"] == 0
    
    def test_factory_statistics(self):
        """Test factory statistics."""
        stats = self.factory.get_factory_statistics()
        
        assert "factory_stats" in stats
        assert "registry_stats" in stats
        assert "cache_stats" in stats
        assert "selection_strategy" in stats
        
        # Check that all required fields are present
        factory_stats = stats["factory_stats"]
        assert "creations" in factory_stats
        assert "cache_hits" in factory_stats
        assert "selection_time" in factory_stats
        assert "creation_time" in factory_stats


class TestFactoryFunctions:
    """Test factory functions and utilities."""
    
    def test_create_generator_factory(self):
        """Test factory creation function."""
        config = PipelineConfig()
        factory = create_generator_factory(config)
        
        assert isinstance(factory, GeneratorFactory)
        assert factory.config is config
    
    def test_register_generator_function(self):
        """Test generator registration function."""
        factory = create_generator_factory()
        capabilities = {GeneratorCapability.DOCUMENTATION}
        
        register_generator(
            factory,
            "doc_generator",
            MockGenerator,
            capabilities,
            priority=GeneratorPriority.LOW
        )
        
        # Check registration
        generators = factory.get_available_generators()
        assert "doc_generator" in generators
        assert generators["doc_generator"].capabilities == capabilities
    
    def test_get_generator_capabilities_function(self):
        """Test get capabilities function."""
        factory = create_generator_factory()
        capabilities = {GeneratorCapability.TEST_GENERATION}
        
        register_generator(factory, "test_gen", MockGenerator, capabilities)
        
        retrieved_capabilities = get_generator_capabilities(factory, "test_gen")
        assert retrieved_capabilities == capabilities
    
    def test_create_generator_configuration_function(self):
        """Test generator configuration creation function."""
        config = PipelineConfig()
        required_caps = {GeneratorCapability.HW_CUSTOM_OP}
        
        gen_config = create_generator_configuration(
            GeneratorType.AUTO_HW_CUSTOM_OP,
            config,
            required_capabilities=required_caps,
            cache_enabled=False
        )
        
        assert gen_config.generator_type == GeneratorType.AUTO_HW_CUSTOM_OP
        assert gen_config.config is config
        assert gen_config.capabilities_required == required_caps
        assert gen_config.cache_enabled == False


class TestIntegration:
    """Integration tests for generator factory components."""
    
    def test_end_to_end_generator_creation(self):
        """Test complete end-to-end generator creation workflow."""
        # Create factory
        config = PipelineConfig()
        factory = GeneratorFactory(config)
        
        # Register multiple generators
        factory.registry.register_generator(
            "hw_gen", MockGenerator, 
            {GeneratorCapability.HW_CUSTOM_OP, GeneratorCapability.DATAFLOW_INTEGRATION},
            GeneratorPriority.HIGH
        )
        factory.registry.register_generator(
            "rtl_gen", MockGenerator,
            {GeneratorCapability.RTL_BACKEND},
            GeneratorPriority.MEDIUM
        )
        
        # Create RTL module
        signals = [RTLSignal("data", "input", 32)]
        interface = RTLInterface("test_interface", "axi_stream", signals)
        rtl_module = RTLModule("test_module", interfaces=[interface], parameters={"WIDTH": 32})
        
        # Create generator configuration
        generator_config = GeneratorConfiguration(
            generator_type=GeneratorType.AUTO_HW_CUSTOM_OP,
            config=config,
            capabilities_required={GeneratorCapability.HW_CUSTOM_OP},
            capabilities_preferred={GeneratorCapability.DATAFLOW_INTEGRATION}
        )
        
        # Create generator
        generator = factory.create_generator(generator_config, rtl_module)
        
        # Verify generator was created and is functional
        assert generator is not None
        assert isinstance(generator, MockGenerator)
        
        # Test generator functionality
        inputs = {"rtl_module": rtl_module, "config": config}
        result = generator.generate(inputs)
        assert result.success == True
        
        # Verify caching works
        generator2 = factory.create_generator(generator_config, rtl_module)
        assert generator is generator2  # Should be cached
        
        # Verify statistics tracking
        stats = factory.get_factory_statistics()
        assert stats["factory_stats"]["creations"] >= 1
        assert stats["factory_stats"]["cache_hits"] >= 1
        
        # Verify registry statistics
        assert stats["registry_stats"]["total_generators"] == 2
        assert stats["registry_stats"]["registrations"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])