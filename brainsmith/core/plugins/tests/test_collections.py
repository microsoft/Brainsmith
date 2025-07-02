"""
Test Natural Access Collections

Tests for collections with direct registry delegation.
"""

import pytest

from brainsmith.core.plugins.decorators import plugin
from brainsmith.core.plugins.registry import get_registry, reset_registry
from brainsmith.core.plugins.collections import create_collections


class TestTransformCollection:
    def setup_method(self):
        reset_registry()
        
        # Register test transforms
        @plugin(type="transform", name="CleanupTransform", stage="cleanup", framework="brainsmith")
        class CleanupTransform:
            def apply(self, model):
                return model, False
        
        @plugin(type="transform", name="QONNXTransform", stage="streamlining", framework="qonnx")
        class QONNXTransform:
            def apply(self, model):
                return model, False
        
        @plugin(type="transform", name="FINNTransform", stage="topology_opt", framework="finn")
        class FINNTransform:
            def apply(self, model):
                return model, False
        
        self.collections = create_collections()
        self.transforms = self.collections['transforms']
    
    def teardown_method(self):
        reset_registry()
    
    def test_direct_transform_access(self):
        """Test direct transform access."""
        # Test direct access
        transform_wrapper = self.transforms.CleanupTransform
        assert transform_wrapper is not None
        assert transform_wrapper.name == "CleanupTransform"
        
        # Test calling - should return instance
        transform_instance = transform_wrapper()
        assert transform_instance is not None
        assert hasattr(transform_instance, 'apply')
    
    def test_framework_accessors(self):
        """Test framework-specific access."""
        # Test QONNX framework accessor
        qonnx_transform = self.transforms.qonnx.QONNXTransform
        assert qonnx_transform is not None
        assert qonnx_transform.name == "QONNXTransform"
        
        # Test FINN framework accessor
        finn_transform = self.transforms.finn.FINNTransform
        assert finn_transform is not None
        assert finn_transform.name == "FINNTransform"
        
        # Test BrainSmith framework accessor
        bs_transform = self.transforms.brainsmith.CleanupTransform
        assert bs_transform is not None
        assert bs_transform.name == "CleanupTransform"
    
    def test_framework_accessor_errors(self):
        """Test helpful error messages from framework accessors."""
        # Test missing transform in specific framework
        with pytest.raises(AttributeError) as exc_info:
            self.transforms.qonnx.NonexistentTransform
        
        assert "not found in qonnx framework" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)
    
    def test_direct_access_errors(self):
        """Test helpful error messages from direct access."""
        with pytest.raises(AttributeError) as exc_info:
            self.transforms.NonexistentTransform
        
        assert "not found" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)
    
    def test_list_by_stage(self):
        """Test listing transforms by stage."""
        cleanup_transforms = self.transforms.list_by_stage("cleanup")
        assert "CleanupTransform" in cleanup_transforms
        
        streamlining_transforms = self.transforms.list_by_stage("streamlining")
        assert "QONNXTransform" in streamlining_transforms
        
        topology_transforms = self.transforms.list_by_stage("topology_opt")
        assert "FINNTransform" in topology_transforms
    
    def test_get_by_stage(self):
        """Test getting wrapped transforms by stage."""
        cleanup_transforms = self.transforms.get_by_stage("cleanup")
        assert "CleanupTransform" in cleanup_transforms
        
        # Should return wrapped instances
        transform_wrapper = cleanup_transforms["CleanupTransform"]
        assert transform_wrapper.name == "CleanupTransform"
        
        # Should be callable
        instance = transform_wrapper()
        assert hasattr(instance, 'apply')
    
    def test_transform_wrapper_repr(self):
        """Test transform wrapper representation."""
        wrapper = self.transforms.CleanupTransform
        repr_str = repr(wrapper)
        
        assert "Transform: CleanupTransform" in repr_str
        assert "stage=cleanup" in repr_str
        assert "framework=brainsmith" in repr_str


class TestKernelCollection:
    def setup_method(self):
        reset_registry()
        
        # Register test kernel and backends
        @plugin(type="kernel", name="TestKernel")
        class TestKernel:
            pass
        
        @plugin(type="backend", name="TestKernelHLS", kernel="TestKernel", backend_type="hls")
        class TestKernelHLS:
            pass
        
        @plugin(type="backend", name="TestKernelRTL", kernel="TestKernel", backend_type="rtl")
        class TestKernelRTL:
            pass
        
        self.collections = create_collections()
        self.kernels = self.collections['kernels']
    
    def teardown_method(self):
        reset_registry()
    
    def test_kernel_access(self):
        """Test kernel accessor functionality."""
        kernel_wrapper = self.kernels.TestKernel
        assert kernel_wrapper is not None
        assert kernel_wrapper.name == "TestKernel"
        
        # Test direct kernel instantiation
        kernel_instance = kernel_wrapper()
        assert kernel_instance is not None
    
    def test_backend_access(self):
        """Test backend access through kernel wrapper."""
        kernel_wrapper = self.kernels.TestKernel
        
        # Test HLS backend access
        hls_backend = kernel_wrapper.hls()
        assert hls_backend is not None
        
        # Test RTL backend access
        rtl_backend = kernel_wrapper.rtl()
        assert rtl_backend is not None
        
        # Test backend with config
        hls_backend_configured = kernel_wrapper.hls(param1="value1")
        assert hls_backend_configured is not None
    
    def test_backend_list(self):
        """Test listing available backends."""
        kernel_wrapper = self.kernels.TestKernel
        backends = kernel_wrapper.list_backends()
        
        assert "hls" in backends
        assert "rtl" in backends
        assert len(backends) == 2
    
    def test_backend_access_errors(self):
        """Test backend access error handling."""
        kernel_wrapper = self.kernels.TestKernel
        
        # Test missing backend
        with pytest.raises(AttributeError) as exc_info:
            kernel_wrapper.nonexistent()
        
        # Should be handled by __getattr__ fallback, but if we add explicit methods:
        # assert "backend not found" in str(exc_info.value)
    
    def test_kernel_access_errors(self):
        """Test kernel access error handling."""
        with pytest.raises(AttributeError) as exc_info:
            self.kernels.NonexistentKernel
        
        assert "not found" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)
    
    def test_kernel_wrapper_repr(self):
        """Test kernel wrapper representation."""
        wrapper = self.kernels.TestKernel
        repr_str = repr(wrapper)
        
        assert "Kernel: TestKernel" in repr_str
        assert "backends:" in repr_str
        assert "hls" in repr_str
        assert "rtl" in repr_str
    
    def test_list_all_kernels(self):
        """Test listing all available kernels."""
        all_kernels = self.kernels.list_all()
        assert "TestKernel" in all_kernels


class TestBackendCollection:
    def setup_method(self):
        reset_registry()
        
        # Register test backend
        @plugin(type="backend", name="TestBackend", kernel="TestKernel", backend_type="hls")
        class TestBackend:
            pass
        
        self.collections = create_collections()
        self.backends = self.collections['backends']
    
    def teardown_method(self):
        reset_registry()
    
    def test_backend_access(self):
        """Test direct backend access."""
        backend_wrapper = self.backends.TestKernel_hls  # Backend key format
        assert backend_wrapper is not None
        assert backend_wrapper.name == "TestKernel_hls"
        
        # Test instantiation
        backend_instance = backend_wrapper()
        assert backend_instance is not None
    
    def test_backend_access_errors(self):
        """Test backend access error handling."""
        with pytest.raises(AttributeError) as exc_info:
            self.backends.NonexistentBackend
        
        assert "not found" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)
    
    def test_backend_wrapper_repr(self):
        """Test backend wrapper representation."""
        wrapper = self.backends.TestKernel_hls
        repr_str = repr(wrapper)
        
        assert "Backend: TestKernel_hls" in repr_str
        assert "kernel=TestKernel" in repr_str
        assert "type=hls" in repr_str
    
    def test_list_all_backends(self):
        """Test listing all available backends."""
        all_backends = self.backends.list_all()
        assert "TestKernel_hls" in all_backends


class TestStepCollection:
    def setup_method(self):
        reset_registry()
        
        # Register test steps
        @plugin(type="step", name="CleanupStep", category="cleanup")
        class CleanupStep:
            def execute(self, model, config):
                return model
        
        @plugin(type="step", name="PreprocessStep", category="preprocessing")
        class PreprocessStep:
            def execute(self, data, config):
                return data
        
        @plugin(type="kernel_inference", name="InferenceStep", kernel="TestKernel", category="inference")
        class InferenceStep:
            def apply(self, model):
                return model, False
        
        self.collections = create_collections()
        self.steps = self.collections['steps']
    
    def teardown_method(self):
        reset_registry()
    
    def test_direct_step_access(self):
        """Test direct step access."""
        step_wrapper = self.steps.CleanupStep
        assert step_wrapper is not None
        assert step_wrapper.name == "CleanupStep"
        
        # Test calling
        # Note: In real usage, this would be called with proper arguments
        # step_result = step_wrapper(model, config)
        step_instance = step_wrapper()
        assert step_instance is not None
    
    def test_category_accessors(self):
        """Test category-specific access."""
        # Test cleanup category
        cleanup_step = self.steps.cleanup.CleanupStep
        assert cleanup_step is not None
        assert cleanup_step.name == "CleanupStep"
        
        # Test preprocessing category
        preprocess_step = self.steps.preprocessing.PreprocessStep
        assert preprocess_step is not None
        assert preprocess_step.name == "PreprocessStep"
        
        # Test inference category
        inference_step = self.steps.inference.InferenceStep
        assert inference_step is not None
        assert inference_step.name == "InferenceStep"
    
    def test_step_access_errors(self):
        """Test step access error handling."""
        with pytest.raises(AttributeError) as exc_info:
            self.steps.NonexistentStep
        
        assert "not found" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)
    
    def test_category_access_errors(self):
        """Test category access error handling."""
        with pytest.raises(AttributeError) as exc_info:
            self.steps.cleanup.NonexistentStep
        
        assert "not found in category 'cleanup'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)
    
    def test_step_wrapper_repr(self):
        """Test step wrapper representation."""
        wrapper = self.steps.CleanupStep
        repr_str = repr(wrapper)
        
        assert "Step: CleanupStep" in repr_str
        assert "category=cleanup" in repr_str


class TestCollectionIntegration:
    def setup_method(self):
        reset_registry()
        
        # Register a variety of plugins for integration testing
        @plugin(type="transform", name="TestTransform", stage="cleanup")
        class TestTransform:
            pass
        
        @plugin(type="kernel", name="TestKernel")
        class TestKernel:
            pass
        
        @plugin(type="backend", name="TestBackend", kernel="TestKernel", backend_type="hls")
        class TestBackend:
            pass
        
        @plugin(type="step", name="TestStep", category="metadata")
        class TestStep:
            pass
        
        self.collections = create_collections()
    
    def teardown_method(self):
        reset_registry()
    
    def test_all_collection_types(self):
        """Test that all collection types are created."""
        assert 'transforms' in self.collections
        assert 'kernels' in self.collections
        assert 'backends' in self.collections
        assert 'steps' in self.collections
        
        # Test each collection has expected methods/attributes
        assert hasattr(self.collections['transforms'], 'qonnx')
        assert hasattr(self.collections['transforms'], 'finn')
        assert hasattr(self.collections['transforms'], 'brainsmith')
        
        assert hasattr(self.collections['kernels'], 'list_all')
        assert hasattr(self.collections['backends'], 'list_all')
        assert hasattr(self.collections['steps'], '__getattr__')
    
    def test_cross_collection_consistency(self):
        """Test consistency across different collection types."""
        # All collections should work with the same registry
        registry = get_registry()
        
        # Test that they all see the same plugins
        assert "TestTransform" in registry.transforms
        assert "TestKernel" in registry.kernels
        assert "TestKernel_hls" in registry.backends
        
        # Test access through collections
        assert hasattr(self.collections['transforms'], 'TestTransform')
        assert hasattr(self.collections['kernels'], 'TestKernel')
        assert hasattr(self.collections['backends'], 'TestKernel_hls')
        assert hasattr(self.collections['steps'], 'TestStep')
    
    def test_collection_independence(self):
        """Test that collections work independently."""
        # Create separate collection instances
        collections1 = create_collections()
        collections2 = create_collections()
        
        # They should work independently but access same registry
        transform1 = collections1['transforms'].TestTransform
        transform2 = collections2['transforms'].TestTransform
        
        # Should both work and reference same underlying class
        assert transform1.name == transform2.name == "TestTransform"