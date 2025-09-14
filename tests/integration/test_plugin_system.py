"""Integration tests for the plugin system."""

import pytest

from brainsmith.core.plugins import get_registry, transform, kernel, backend, step
from brainsmith.core.plugins.registry import list_backends_by_kernel, get_default_backend
from tests.fixtures.model_utils import create_simple_model
from tests.utils.plugin_assertions import (
    PluginAssertions,
    EXPECTED_TEST_PLUGINS,
    EXPECTED_KERNEL_BACKENDS,
    MIN_FRAMEWORK_PLUGINS
)


class TestTransformPlugins:
    """Test suite for transform plugins."""
    
    def test_register_transform_plugin(self):
        """Test registering and retrieving a transform plugin."""
        registry = get_registry()
        # Use helper to verify all expected test plugins are available
        PluginAssertions.assert_test_plugins_available(
            registry,
            EXPECTED_TEST_PLUGINS
        )
        
        # Verify specific transform execution capability
        plugin_cls = registry.get("transform", "test_transform")
        assert plugin_cls is not None
        assert plugin_cls.__name__ == "TestTransformPlugin"
        
        # Create instance and test execution
        transform = plugin_cls()
        model = create_simple_model()
        
        # Apply transform
        modified_model = transform.apply(model)
        
        # Verify model was modified
        assert len(modified_model.graph.node) > 0
        node = modified_model.graph.node[0]
        
        # Check that our custom attribute was added
        attr_names = [attr.name for attr in node.attribute]
        assert "test_transform_applied" in attr_names
        
        # Find the attribute and verify its value
        for attr in node.attribute:
            if attr.name == "test_transform_applied":
                assert attr.i == 1
                break
    
    def test_framework_transform_namespacing(self):
        """Test accessing transforms with framework namespacing."""
        registry = get_registry()
        # First ensure plugins are loaded
        all_transforms = registry.all("transform")
        
        # Test FINN transform access - should always be available
        # Try with explicit prefix
        finn_transform = registry.get("transform", "finn:Streamline")
        assert finn_transform is not None
        
        # Try without prefix (should auto-resolve)
        same_transform = registry.get("transform", "Streamline")
        assert same_transform == finn_transform
        
        # Test QONNX transform access - should always be available
        # Try with explicit prefix
        qonnx_transform = registry.get("transform", "qonnx:InferDataTypes")
        assert qonnx_transform is not None
        
        # Try without prefix (should auto-resolve)
        same_transform = registry.get("transform", "InferDataTypes") 
        assert same_transform == qonnx_transform
        
        # Use helper to verify framework transform availability with flexible thresholds
        PluginAssertions.assert_framework_transforms_available(
            registry, 
            MIN_FRAMEWORK_PLUGINS
        )
    
    def test_transform_metadata(self):
        """Test transform with metadata."""
        registry = get_registry()
        # Use helper to verify plugin metadata structure
        PluginAssertions.assert_plugin_execution_capability(
            registry,
            "transform",
            "test_transform_with_metadata"
        )
        
        # Query by metadata
        transforms = registry.find("transform", test_metadata="value")
        assert len(transforms) > 0
        
        # Verify the plugin is in the metadata query results
        plugin_cls = registry.get("transform", "test_transform_with_metadata")
        assert plugin_cls in transforms


class TestKernelPlugins:
    """Test suite for kernel plugins."""
    
    def test_register_kernel_with_backends(self):
        """Test registering kernels with multiple backends."""
        registry = get_registry()
        # Use helper to verify kernel-backend associations
        PluginAssertions.assert_kernel_backend_associations(
            registry,
            EXPECTED_KERNEL_BACKENDS
        )
        
        # Verify kernel execution capability
        PluginAssertions.assert_plugin_execution_capability(
            registry,
            "kernel",
            "TestKernel"
        )
        
        # Verify kernel class details
        kernel_cls = registry.get("kernel", "TestKernel")
        assert kernel_cls.__name__ == "TestKernelPlugin"
        
        # Get backends
        hls_backend = registry.get("backend", "TestKernel_hls")
        rtl_backend = registry.get("backend", "TestKernel_rtl")
        
        assert hls_backend is not None
        assert rtl_backend is not None
        
        # Query backends by kernel name
        backends = list_backends_by_kernel("TestKernel")
        assert len(backends) == 2
        assert "TestKernel_hls" in backends
        assert "TestKernel_rtl" in backends
        
        # Verify backend metadata
        hls_info = registry.find("backend", kernel="TestKernel", language="hls")
        assert len(hls_info) == 1
        assert hls_info[0] == hls_backend
        
        rtl_info = registry.find("backend", kernel="TestKernel", language="rtl")
        assert len(rtl_info) == 1
        assert rtl_info[0] == rtl_backend
    
    def test_kernel_inference_decorator(self):
        """Test kernel inference decorator."""
        registry = get_registry()
        # Get inference transform
        inference_cls = registry.get("transform", "InferTestKernel")
        assert inference_cls is not None
        
        # Verify kernel association in metadata
        inference_transforms = registry.find("transform", kernel_inference=True)
        assert len(inference_transforms) > 0
        assert inference_cls in inference_transforms
        
        # Also verify the kernel parameter was passed
        _, metadata = next((item for item in registry._plugins["transform"].items() if item[1][0] == inference_cls), (None, None))
        assert metadata is not None
        assert metadata[1].get("kernel") == "TestKernel"
        
        # Test execution
        model = create_simple_model()
        inference = inference_cls()
        result = inference.apply(model)
        
        # Verify inference was applied
        if result.graph.node:
            node = result.graph.node[0]
            attr_names = [attr.name for attr in node.attribute]
            assert "kernel_inferred" in attr_names
    
    def test_default_backend_selection(self):
        """Test default backend selection."""
        registry = get_registry()
        # Get default backend (should be first registered)
        default_backend = get_default_backend("TestKernel")
        assert default_backend == "TestKernel_hls"  # HLS was registered first
        
        # Test with kernel that has only one backend
        kernel_cls = registry.get("kernel", "TestKernelWithBackends")
        assert kernel_cls is not None
        
        default = get_default_backend("TestKernelWithBackends")
        assert default == "TestKernelWithBackends_hls"


class TestStepPlugins:
    """Test suite for step plugins."""
    
    def test_register_step_plugin(self):
        """Test registering and executing step plugins."""
        registry = get_registry()
        # Get step
        step_func = registry.get("step", "test_step")
        assert step_func is not None
        assert callable(step_func)
        
        # Execute step
        blueprint = {"name": "test", "clock_ns": 5.0}
        context = {}
        
        step_func(blueprint, context)
        
        # Verify execution
        assert "executed_steps" in context
        assert "test_step" in context["executed_steps"]
    
    # Removed test_step_with_kernel_backends - tested mock step execution


class TestTransformChains:
    """Test transform chain behaviors and dependencies."""
    
    def test_transform_chain_with_dependencies(self):
        """Test transforms that depend on previous transform results."""
        registry = get_registry()
        
        # Create transforms that build on each other using metadata
        @transform(name="chain_step1")
        class Step1Transform:
            def apply(self, model):
                # Add metadata that next transform will read
                metadata = model.metadata_props.add()
                metadata.key = "chain_step1_complete"
                metadata.value = "true"
                return model
        
        @transform(name="chain_step2")
        class Step2Transform:
            def apply(self, model):
                # Check if step1 ran
                step1_complete = False
                for prop in model.metadata_props:
                    if prop.key == "chain_step1_complete" and prop.value == "true":
                        step1_complete = True
                        break
                
                if not step1_complete:
                    raise RuntimeError("Step2 requires Step1 to run first!")
                
                metadata = model.metadata_props.add()
                metadata.key = "chain_step2_complete"
                metadata.value = "true"
                return model
        
        @transform(name="chain_step3")
        class Step3Transform:
            def apply(self, model):
                # Check if both previous steps ran
                step1_complete = False
                step2_complete = False
                
                for prop in model.metadata_props:
                    if prop.key == "chain_step1_complete" and prop.value == "true":
                        step1_complete = True
                    elif prop.key == "chain_step2_complete" and prop.value == "true":
                        step2_complete = True
                
                if not (step1_complete and step2_complete):
                    raise RuntimeError("Step3 requires Step1 and Step2!")
                
                metadata = model.metadata_props.add()
                metadata.key = "chain_complete"
                metadata.value = "true"
                return model
        
        # Create a mock model
        model = create_simple_model()
        
        # Apply transforms in correct order
        for transform_name in ['chain_step1', 'chain_step2', 'chain_step3']:
            t_cls = registry.get('transform', transform_name)
            t_instance = t_cls()
            model = t_instance.apply(model)
        
        # Verify chain executed correctly
        metadata_dict = {}
        for prop in model.metadata_props:
            metadata_dict[prop.key] = prop.value
        
        assert metadata_dict.get("chain_step1_complete") == "true"
        assert metadata_dict.get("chain_step2_complete") == "true"
        assert metadata_dict.get("chain_complete") == "true"
        
        # Test incorrect order fails
        model2 = create_simple_model()
        step2 = registry.get('transform', 'chain_step2')()
        
        with pytest.raises(RuntimeError) as exc_info:
            step2.apply(model2)
        
        assert "Step2 requires Step1" in str(exc_info.value)
    
    def test_transform_chain_failure_recovery(self):
        """Test behavior when a transform in a chain fails."""
        registry = get_registry()
        
        # Create transforms where middle one can fail
        @transform(name="chain_start")
        class StartTransform:
            def apply(self, model):
                metadata = model.metadata_props.add()
                metadata.key = "started"
                metadata.value = "true"
                return model
        
        @transform(name="chain_middle_failing")
        class MiddleFailingTransform:
            def __init__(self, should_fail=True):
                self.should_fail = should_fail
                
            def apply(self, model):
                # Check if start ran
                started = False
                for prop in model.metadata_props:
                    if prop.key == "started" and prop.value == "true":
                        started = True
                        break
                
                if not started:
                    raise RuntimeError("Must run start transform first!")
                    
                if self.should_fail:
                    raise ValueError("Middle transform failed!")
                
                metadata = model.metadata_props.add()
                metadata.key = "middle_complete"
                metadata.value = "true"
                return model
        
        @transform(name="chain_end")
        class EndTransform:
            def apply(self, model):
                # Check if middle completed
                middle_complete = False
                for prop in model.metadata_props:
                    if prop.key == "middle_complete" and prop.value == "true":
                        middle_complete = True
                        break
                
                if not middle_complete:
                    raise RuntimeError("Middle transform must complete first!")
                
                metadata = model.metadata_props.add()
                metadata.key = "end_complete"
                metadata.value = "true"
                return model
        
        # Test failure case
        model = create_simple_model()
        
        # Apply start transform
        start_t = registry.get('transform', 'chain_start')()
        model = start_t.apply(model)
        
        # Verify start ran
        has_started = any(prop.key == "started" and prop.value == "true" 
                         for prop in model.metadata_props)
        assert has_started
        
        # Middle transform should fail
        middle_t = registry.get('transform', 'chain_middle_failing')()
        with pytest.raises(ValueError) as exc_info:
            middle_t.apply(model)
        
        assert "Middle transform failed!" in str(exc_info.value)
        
        # Model should still have state from first transform
        has_started = any(prop.key == "started" and prop.value == "true" 
                         for prop in model.metadata_props)
        assert has_started
        
        has_middle = any(prop.key == "middle_complete" and prop.value == "true" 
                        for prop in model.metadata_props)
        assert not has_middle
        
        # End transform should fail due to missing dependency
        end_t = registry.get('transform', 'chain_end')()
        with pytest.raises(RuntimeError) as exc_info:
            end_t.apply(model)
        
        assert "Middle transform must complete first!" in str(exc_info.value)
        
        # Test success case
        model2 = create_simple_model()
        model2 = start_t.apply(model2)
        
        # Use non-failing version
        middle_success = registry.get('transform', 'chain_middle_failing')(should_fail=False)
        model2 = middle_success.apply(model2)
        model2 = end_t.apply(model2)
        
        # Verify all steps completed
        metadata_dict = {}
        for prop in model2.metadata_props:
            metadata_dict[prop.key] = prop.value
        
        assert metadata_dict.get("started") == "true"
        assert metadata_dict.get("middle_complete") == "true"
        assert metadata_dict.get("end_complete") == "true"
    
    def test_transform_side_effect_isolation(self):
        """Test that transform side effects don't leak between executions."""
        
        # Create a transform with internal state
        @transform(name="stateful_transform")
        class StatefulTransform:
            def __init__(self):
                self.execution_count = 0
                self.processed_models = []
            
            def apply(self, model):
                self.execution_count += 1
                self.processed_models.append(id(model))
                
                # Add execution count to model metadata
                metadata = model.metadata_props.add()
                metadata.key = f"execution_{id(self)}"
                metadata.value = str(self.execution_count)
                return model
        
        registry = get_registry()
        
        # Create two instances
        transform1 = registry.get('transform', 'stateful_transform')()
        transform2 = registry.get('transform', 'stateful_transform')()
        
        # Apply to different models
        model1 = create_simple_model()
        model2 = create_simple_model()
        
        model1 = transform1.apply(model1)
        model2 = transform2.apply(model2)
        
        # Each instance should have independent state
        assert transform1.execution_count == 1
        assert transform2.execution_count == 1
        
        # Check metadata to verify execution numbers
        def get_execution_number(model, transform_instance):
            key = f"execution_{id(transform_instance)}"
            for prop in model.metadata_props:
                if prop.key == key:
                    return int(prop.value)
            return None
        
        assert get_execution_number(model1, transform1) == 1
        assert get_execution_number(model2, transform2) == 1
        
        # Apply transform1 again
        model3 = create_simple_model()
        model3 = transform1.apply(model3)
        
        assert get_execution_number(model3, transform1) == 2
        assert transform1.execution_count == 2
        assert transform2.execution_count == 1  # Should not affect other instance
        
        # Verify each transform tracked different models
        assert len(transform1.processed_models) == 2
        assert len(transform2.processed_models) == 1
        assert transform1.processed_models[0] != transform2.processed_models[0]


class TestPluginStateManagement:
    """Test plugin registry state management."""
    
    def test_registry_singleton_behavior(self):
        """Test that get_registry() returns the same instance."""
        from brainsmith.core.plugins import get_registry
        
        registry1 = get_registry()
        registry2 = get_registry()
        
        # Should be the same object
        assert registry1 is registry2
        
        # Changes to one should affect the other
        @transform(name="singleton_test_transform")
        class SingletonTestTransform:
            def apply(self, model):
                return model
        
        # Should be accessible from both references
        t1 = registry1.get("transform", "singleton_test_transform")
        t2 = registry2.get("transform", "singleton_test_transform")
        assert t1 == t2 == SingletonTestTransform
    
    def test_registry_reset_clears_state(self):
        """Test that registry.reset() properly clears all state."""
        registry = get_registry()
        
        # Add some test plugins
        @transform(name="reset_test_transform")
        class ResetTestTransform:
            def apply(self, model):
                return model
        
        @kernel(name="reset_test_kernel")
        class ResetTestKernel:
            pass
        
        @step(name="reset_test_step")
        def reset_test_step(blueprint, context):
            pass
        
        # Verify they exist
        assert registry.get("transform", "reset_test_transform") == ResetTestTransform
        assert registry.get("kernel", "reset_test_kernel") == ResetTestKernel
        assert registry.get("step", "reset_test_step") == reset_test_step
        
        # Reset the registry
        registry.reset()
        
        # Plugins should no longer be accessible
        with pytest.raises(KeyError):
            registry.get("transform", "reset_test_transform")
        
        with pytest.raises(KeyError):
            registry.get("kernel", "reset_test_kernel")
            
        with pytest.raises(KeyError):
            registry.get("step", "reset_test_step")
        
        # Re-registering should work
        @transform(name="reset_test_transform")
        class NewResetTestTransform:
            def apply(self, model):
                model.new_version = True
                return model
        
        new_transform = registry.get("transform", "reset_test_transform")
        assert new_transform == NewResetTestTransform
        assert new_transform != ResetTestTransform  # Different class
        
        # Test that discovery state is also reset
        # The reset method calls _load_plugins which sets _discovered
        # So we just verify that plugins were reloaded
        registry.reset()
        
        # After reset, plugins should be reloaded (discovered flag is set by reset)
        assert hasattr(registry, '_discovered')
        assert registry._discovered == True
        
        # Framework adapters should be available again
        all_transforms = registry.all("transform")
        assert len(all_transforms) > 0  # Should have framework transforms loaded
    
    def test_plugin_metadata_queries(self):
        """Test complex metadata queries."""
        registry = get_registry()
        
        # Register plugins with various metadata
        @transform(name="metadata_test_1", category="preprocessing", priority=1)
        class MetadataTest1:
            def apply(self, model):
                return model
        
        @transform(name="metadata_test_2", category="preprocessing", priority=2)
        class MetadataTest2:
            def apply(self, model):
                return model
        
        @transform(name="metadata_test_3", category="postprocessing", priority=1)
        class MetadataTest3:
            def apply(self, model):
                return model
        
        @transform(name="metadata_test_4", category="postprocessing", priority=1, experimental=True)
        class MetadataTest4:
            def apply(self, model):
                return model
        
        # Query by single criterion
        preprocessing = registry.find("transform", category="preprocessing")
        assert len(preprocessing) >= 2
        assert MetadataTest1 in preprocessing
        assert MetadataTest2 in preprocessing
        assert MetadataTest3 not in preprocessing
        
        # Query by multiple criteria
        priority1_preprocessing = registry.find("transform", category="preprocessing", priority=1)
        assert MetadataTest1 in priority1_preprocessing
        assert MetadataTest2 not in priority1_preprocessing
        
        # Query for experimental plugins
        experimental = registry.find("transform", experimental=True)
        assert MetadataTest4 in experimental
        assert MetadataTest1 not in experimental
        
        # Query with non-matching criteria should return empty
        no_match = registry.find("transform", category="nonexistent", priority=99)
        assert len(no_match) == 0
        
        # Test that metadata is preserved correctly
        _, metadata = next(
            (item for item in registry._plugins["transform"].items() 
             if item[1][0] == MetadataTest4), 
            (None, None)
        )
        assert metadata is not None
        assert metadata[1]["category"] == "postprocessing"
        assert metadata[1]["priority"] == 1
        assert metadata[1]["experimental"] == True
    
    def test_lazy_loading_behavior(self):
        """Test that plugins are loaded lazily."""
        # Get the registry
        registry = get_registry()
        
        # Since registry is a singleton and reset() calls _load_plugins(),
        # we can't test initial lazy loading. Instead, test that discovery
        # doesn't happen multiple times unnecessarily
        
        # First, ensure plugins are loaded
        if not hasattr(registry, '_discovered'):
            registry.all("transform")  # Force discovery
        
        # Mark the discovery state
        registry._discovered = "test_marker"
        
        # Accessing plugins should not re-discover since already loaded
        try:
            registry.get("transform", "some_transform")
        except KeyError:
            pass  # Expected
        
        # Should still have our marker (no re-discovery)
        assert registry._discovered == "test_marker"
        
        # Even all() should not re-discover
        registry.all("transform")
        assert registry._discovered == "test_marker"
        
        # Reset should force re-discovery
        registry.reset()
        assert registry._discovered == True  # Reset calls _load_plugins()


# TestPluginDiscovery has been removed - tested internal implementation details