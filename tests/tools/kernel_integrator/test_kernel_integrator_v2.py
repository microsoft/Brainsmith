"""Tests for KernelIntegrator with direct generators feature flag."""
import pytest
from pathlib import Path
from brainsmith.tools.kernel_integrator.kernel_integrator import KernelIntegrator
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata


def test_kernel_integrator_feature_flag():
    """Test KernelIntegrator initialization with feature flag."""
    # Legacy mode
    integrator_legacy = KernelIntegrator(use_direct_generators=False)
    assert integrator_legacy.use_direct_generators is False
    assert hasattr(integrator_legacy, 'generator_manager')
    assert hasattr(integrator_legacy, 'context_generator')
    
    # Direct mode
    integrator_direct = KernelIntegrator(use_direct_generators=True)
    assert integrator_direct.use_direct_generators is True
    assert hasattr(integrator_direct, 'direct_generators')
    assert len(integrator_direct.direct_generators) == 3


def test_list_generators_with_feature_flag():
    """Test generator listing with feature flag."""
    # Direct mode
    integrator = KernelIntegrator(use_direct_generators=True)
    generators = integrator.list_generators()
    
    assert len(generators) == 3
    assert 'hw_custom_op' in generators
    assert 'rtl_backend' in generators
    assert 'rtl_wrapper' in generators


def test_generate_direct():
    """Test direct generation without TemplateContext."""
    integrator = KernelIntegrator(use_direct_generators=True)
    
    # Create test metadata
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=[],
        source_file="test.v"
    )
    
    # Test direct generation
    generated_files = integrator._generate_direct(metadata)
    
    # Should generate 3 files
    assert len(generated_files) == 3
    assert 'test_kernel.py' in generated_files
    assert 'test_kernel_rtl.py' in generated_files
    assert 'test_kernel_wrapper.v' in generated_files
    
    # Verify content was generated
    for filename, content in generated_files.items():
        assert content is not None
        assert len(content) > 0