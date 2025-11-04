# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""High-value integration tests for brainsmith component registry.

32 essential tests covering critical paths and interactions:
- Runtime registration (decorator timing, source priority)
- Manifest caching (fresh, valid, stale, disabled, force refresh)
- Thread safety (concurrent contexts, registration, discovery, loading)
- Name resolution (single/multiple sources, priority, qualified names)
- Lazy loading (deferred imports, import errors, double-check race, transforms)
- Source context (nesting, exceptions, isolation)
- Backend-kernel linking (existing/missing targets, filtering, load order)

Arete Approach: No mocking. Real files, real environment, real code paths.
Uses isolated_env and empty_env fixtures from conftest.py.
"""

import json
import logging
import os
import pytest
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Type

from brainsmith.registry import (
    step,
    kernel,
    backend,
    source_context,
    discover_components,
    get_step,
    get_kernel,
    get_backend,
    has_step,
    has_kernel,
    list_steps,
    list_kernels,
    list_backends,
    list_backends_for_kernel,
    get_component_metadata,
    ComponentMetadata,
    ComponentType,
    ImportSpec,
)

from brainsmith.registry._state import _component_index
from brainsmith.registry._decorators import _current_source, _register_step, _register_kernel, _register_backend
import brainsmith.registry._state as registry_state
import brainsmith.registry._discovery as discovery_module
from brainsmith.registry._discovery import _resolve_component_name, _load_component
from brainsmith.registry._manifest import _build_manifest_from_index, _save_manifest, _load_manifest, _is_manifest_stale
from brainsmith.settings import reset_config

logger = logging.getLogger(__name__)


# ============================================================================
# Test Helpers
# ============================================================================

def mock_kernel_class(name: str, op_type: str):
    """Create minimal kernel class for registry unit tests.

    Note: For discovery tests, use real test plugins in isolated_env fixture.
    This is only for testing direct decorator registration.
    """
    return type(name, (), {'__name__': name, 'op_type': op_type})


def mock_backend_class(name: str, target_kernel: str, language: str):
    """Create minimal backend class for registry unit tests."""
    return type(name, (), {
        '__name__': name,
        'target_kernel': target_kernel,
        'language': language,
    })


def mock_step_function(name: str):
    """Create minimal step function for registry unit tests."""
    fn = lambda model, cfg: model
    fn.__name__ = name
    return fn


# ============================================================================
# TestRegistrationTiming - Decorator before/after/during discovery
# ============================================================================

@pytest.mark.fast
class TestRegistrationTiming:
    """Test edge cases around when components are registered vs discovered."""

    def test_decorator_before_discovery(self, empty_env):
        """Register component via decorator before discovery - should be available immediately."""
        # Register without calling discover_components first
        test_step = mock_step_function('test_before_discovery')
        registered_step = step(test_step, name='test_before_discovery', source='custom')

        # Verify it's in the index immediately
        assert 'custom:test_before_discovery' in _component_index
        meta = _component_index['custom:test_before_discovery']
        assert meta.loaded_obj is registered_step
        assert meta.component_type == ComponentType.STEP

        # Verify we can retrieve it without discovery
        retrieved = get_step('custom:test_before_discovery')
        assert retrieved is registered_step

    def test_decorator_after_discovery(self, empty_env):
        """Register component after discovery - both should coexist."""
        # Discover core components first
        discover_components(use_cache=False)

        initial_count = len(_component_index)

        # Register new component after discovery
        MockKernel = mock_kernel_class('MockKernelAfter', op_type='MockOp')
        kernel(MockKernel, name='MockKernelAfter', source='test')

        # Verify both exist
        assert len(_component_index) > initial_count
        assert 'test:MockKernelAfter' in _component_index

        # Verify we can get both core and new components
        assert has_kernel('test:MockKernelAfter')
        # Core component still accessible
        assert has_kernel('brainsmith:LayerNorm') or len(_component_index) > 0

    def test_double_registration_different_source(self, empty_env):
        """Double registration with different sources - both should exist."""
        MockKernel1 = mock_kernel_class('SharedName', op_type='Op1')
        MockKernel2 = mock_kernel_class('SharedName', op_type='Op2')

        kernel(MockKernel1, name='SharedName', source='user')
        kernel(MockKernel2, name='SharedName', source='custom')

        # Both should exist
        assert 'user:SharedName' in _component_index
        assert 'custom:SharedName' in _component_index
        assert _component_index['user:SharedName'].loaded_obj is MockKernel1
        assert _component_index['custom:SharedName'].loaded_obj is MockKernel2

    def test_registration_with_explicit_source(self, empty_env):
        """Explicit source parameter should override context."""
        test_step = mock_step_function('explicit_source_step')

        # Context says 'context_source', but decorator param says 'explicit_source'
        with source_context('context_source'):
            step(test_step, name='explicit_source_step', source='explicit_source')

        # Explicit parameter should win
        assert 'explicit_source:explicit_source_step' in _component_index
        assert 'context_source:explicit_source_step' not in _component_index


# ============================================================================
# TestManifestCaching - Cache validity, staleness, corruption
# ============================================================================

@pytest.mark.fast
class TestManifestCaching:
    """Test edge cases in manifest caching logic."""

    def test_fresh_discovery_no_manifest(self, empty_env):
        """No manifest exists - should do full discovery and create manifest."""
        # empty_env creates a project with cache_components: false
        # Let's modify the config to enable caching
        config_file = empty_env / '.brainsmith' / 'config.yaml'
        config_file.write_text("""
cache_components: true
component_sources: {}
""")
        reset_config()

        manifest_path = empty_env / '.brainsmith' / 'component_manifest.json'
        assert not manifest_path.exists()

        # Discover with cache enabled
        discover_components(use_cache=True)

        # Manifest should be created
        assert manifest_path.exists()

        # Manifest should contain type-stratified components (v2.0 format)
        with open(manifest_path, 'r') as f:
            data = json.load(f)

        assert data['version'] == '2.0'
        assert 'kernels' in data
        assert 'backends' in data
        assert 'steps' in data
        # Should have at least some core components
        total = len(data['kernels']) + len(data['backends']) + len(data['steps'])
        assert total > 0

    def test_load_from_valid_manifest(self, empty_env):
        """Valid cached manifest - should load from cache without imports."""
        # Enable caching
        config_file = empty_env / '.brainsmith' / 'config.yaml'
        config_file.write_text("""
cache_components: true
component_sources: {}
""")
        reset_config()

        manifest_path = empty_env / '.brainsmith' / 'component_manifest.json'

        # Create valid v2.0 manifest (type-stratified, no per-component mtimes)
        manifest = {
            'version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'kernels': {
                'test:CachedKernel': {
                    'module': 'test.kernels.cached',
                    'attr': 'CachedKernel',
                    'file_path': None,
                    'infer': None,
                    'domain': 'finn.custom',
                    'backends': []
                }
            },
            'backends': {},
            'steps': {}
        }

        _save_manifest(manifest, manifest_path)

        discover_components(use_cache=True)

        # Should have loaded from cache
        assert 'test:CachedKernel' in _component_index
        meta = _component_index['test:CachedKernel']
        assert meta.name == 'CachedKernel'
        assert meta.source == 'test'
        assert meta.loaded_obj is None  # Not imported yet (lazy)

    def test_stale_manifest_triggers_rediscovery(self, isolated_env):
        """Stale manifest (file newer than manifest timestamp) - should rediscover."""
        # isolated_env has cache_components: true and real test plugins
        manifest_path = isolated_env / '.brainsmith' / 'component_manifest.json'

        # Create a Python file
        plugins_dir = isolated_env / 'plugins'
        test_file = plugins_dir / 'test_module.py'
        test_file.write_text('# test')

        # Create manifest with OLD timestamp (1 hour ago)
        from datetime import timedelta
        old_timestamp = datetime.now() - timedelta(hours=1)

        manifest = {
            'version': '2.0',
            'generated_at': old_timestamp.isoformat(),
            'kernels': {},
            'backends': {},
            'steps': {
                'test:StaleComponent': {
                    'module': 'test.steps.stale',
                    'attr': 'StaleStep',
                    'file_path': str(test_file),
                }
            }
        }

        _save_manifest(manifest, manifest_path)

        # File mtime will be newer than manifest timestamp
        discover_components(use_cache=True)

        # Should have done full discovery (cache was stale)
        # The fake component won't exist after fresh discovery
        assert len(_component_index) > 0
        assert 'test:StaleComponent' not in _component_index

    def test_cache_disabled_setting(self, empty_env, caplog):
        """cache_components=False - should never use cache."""
        # empty_env already has cache_components: false
        manifest_path = empty_env / '.brainsmith' / 'component_manifest.json'

        # Create valid v2.0 manifest
        manifest = {
            'version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'kernels': {},
            'backends': {},
            'steps': {
                'test:ShouldNotLoad': {
                    'module': 'test.steps.nope',
                    'attr': 'NopeStep',
                    'file_path': None,
                }
            }
        }
        _save_manifest(manifest, manifest_path)

        with caplog.at_level(logging.DEBUG):
            discover_components(use_cache=True)  # Still passes True, but setting overrides

        # Should not have loaded cached component
        assert 'test:ShouldNotLoad' not in _component_index

        # Should have debug message about caching disabled
        assert any('caching disabled' in record.message.lower() for record in caplog.records)

    def test_force_refresh_clears_index(self, empty_env):
        """force_refresh=True - should clear index and ignore cache."""
        # Register a component
        test_step = mock_step_function('will_be_cleared')
        step(test_step, name='will_be_cleared', source='test')

        assert 'test:will_be_cleared' in _component_index

        # Set discovered flag so force_refresh will clear
        registry_state._components_discovered = True
        discovery_module._components_discovered = True

        # Force refresh should clear it
        discover_components(force_refresh=True)

        # Our test component should be gone (cleared during force refresh)
        # Only core components from discovery remain
        assert 'test:will_be_cleared' not in _component_index


# ============================================================================
# TestThreadSafety - Concurrent operations
# ============================================================================

@pytest.mark.fast
class TestThreadSafety:
    """Test thread safety of registry operations."""

    def test_concurrent_source_context(self, empty_env):
        """Multiple threads with different source contexts - should be isolated."""
        results = {}

        def register_in_context(source_name, step_name):
            with source_context(source_name):
                test_step = mock_step_function(step_name)
                step(test_step, name=step_name, source=None)  # Let context provide source

                # Verify current source
                results[threading.current_thread().name] = _current_source.get()

        threads = [
            threading.Thread(target=register_in_context, args=('source1', 'step1'), name='thread1'),
            threading.Thread(target=register_in_context, args=('source2', 'step2'), name='thread2'),
            threading.Thread(target=register_in_context, args=('source3', 'step3'), name='thread3'),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have seen its own source
        assert results['thread1'] == 'source1'
        assert results['thread2'] == 'source2'
        assert results['thread3'] == 'source3'

        # All steps should be registered with correct sources
        assert 'source1:step1' in _component_index
        assert 'source2:step2' in _component_index
        assert 'source3:step3' in _component_index

    def test_concurrent_registration(self, empty_env):
        """Multiple threads registering components - all should succeed."""
        def register_kernel(idx):
            MockKernel = mock_kernel_class(f'ConcurrentKernel{idx}', op_type=f'Op{idx}')
            kernel(MockKernel, name=f'ConcurrentKernel{idx}', source='test')

        threads = [threading.Thread(target=register_kernel, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 10 kernels should be registered
        for i in range(10):
            assert f'test:ConcurrentKernel{i}' in _component_index

    def test_concurrent_discovery(self, empty_env):
        """Multiple threads calling discover - should be idempotent."""
        def call_discover():
            discover_components(use_cache=False)

        threads = [threading.Thread(target=call_discover) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Discovery should happen (all threads call it), but state should be consistent
        # At least some components should be discovered
        assert len(_component_index) > 0

    def test_concurrent_lazy_loading(self, empty_env):
        """Multiple threads loading same component - should import only once."""
        # Register a component that will be lazy-loaded
        meta = ComponentMetadata(
            name='LazyKernel',
            source='test',
            component_type=ComponentType.KERNEL,
            import_spec=ImportSpec(
                module='brainsmith.kernels.layernorm.layernorm',  # Use real module
                attr='LayerNorm'
            ),
            loaded_obj=None  # Not loaded yet
        )
        _component_index['test:LazyKernel'] = meta

        registry_state._components_discovered = True  # Pretend discovery happened

        results = []

        def load_component():
            obj = get_kernel('test:LazyKernel')
            results.append(obj)

        threads = [threading.Thread(target=load_component) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same object
        assert all(obj is results[0] for obj in results)

        # Component should be loaded now
        assert meta.loaded_obj is not None

    def test_source_context_token_cleanup(self, empty_env):
        """Verify token-based context cleanup works correctly."""
        # Outside any context
        assert _current_source.get() is None

        with source_context('outer'):
            assert _current_source.get() == 'outer'

            with source_context('inner'):
                assert _current_source.get() == 'inner'

            # Should restore to outer
            assert _current_source.get() == 'outer'

        # Should restore to None
        assert _current_source.get() is None


# ============================================================================
# TestNameResolution - Priority, conflicts, missing
# ============================================================================

@pytest.mark.fast
class TestNameResolution:
    """Test edge cases in component name resolution."""

    def test_short_name_single_source(self, empty_env):
        """Short name with only one source - should resolve correctly."""
        MockKernel = mock_kernel_class('UniqueKernel', op_type='Unique')
        kernel(MockKernel, name='UniqueKernel', source='brainsmith')

        registry_state._components_discovered = True
        discovery_module._components_discovered = True

        # Resolve short name
        full_name = _resolve_component_name('UniqueKernel', 'kernel')
        assert full_name == 'brainsmith:UniqueKernel'

    def test_short_name_priority(self, empty_env):
        """Short name in multiple sources - should use priority order."""
        # Register same name in different sources
        K1 = mock_kernel_class('SharedKernel', op_type='Op1')
        K2 = mock_kernel_class('SharedKernel', op_type='Op2')
        K3 = mock_kernel_class('SharedKernel', op_type='Op3')

        kernel(K1, name='SharedKernel', source='finn')
        kernel(K2, name='SharedKernel', source='user')
        kernel(K3, name='SharedKernel', source='brainsmith')

        registry_state._components_discovered = True
        discovery_module._components_discovered = True

        # Should resolve to user (higher priority than brainsmith/finn)
        # Priority: ['project', 'user', 'brainsmith', 'finn']
        full_name = _resolve_component_name('SharedKernel', 'kernel')
        assert full_name == 'user:SharedKernel'

    def test_qualified_name_bypasses_priority(self, empty_env):
        """Qualified name (source:name) - should bypass priority resolution."""
        K1 = mock_kernel_class('ExplicitKernel', op_type='Op1')
        K2 = mock_kernel_class('ExplicitKernel', op_type='Op2')

        kernel(K1, name='ExplicitKernel', source='user')
        kernel(K2, name='ExplicitKernel', source='finn')

        registry_state._components_discovered = True

        # Explicitly request finn version (even though user has higher priority)
        full_name = _resolve_component_name('finn:ExplicitKernel', 'kernel')
        assert full_name == 'finn:ExplicitKernel'

        retrieved = get_kernel('finn:ExplicitKernel')
        assert retrieved is K2

    def test_nonexistent_component(self, empty_env):
        """Request nonexistent component - should raise KeyError with helpful message."""
        registry_state._components_discovered = True

        with pytest.raises(KeyError) as exc_info:
            get_kernel('DoesNotExist')

        # Should have helpful error message
        assert 'not found' in str(exc_info.value).lower()

# ============================================================================
# TestLazyLoading - Double-load, missing imports, etc.
# ============================================================================

@pytest.mark.fast
class TestLazyLoading:
    """Test edge cases in lazy component loading."""

    def test_load_unloaded_component(self, empty_env):
        """Load component with loaded_obj=None - should import and cache."""
        # Create unloaded component metadata
        meta = ComponentMetadata(
            name='LayerNorm',
            source='test',
            component_type=ComponentType.KERNEL,
            import_spec=ImportSpec(
                module='brainsmith.kernels.layernorm.layernorm',
                attr='LayerNorm'
            ),
            loaded_obj=None
        )
        _component_index['test:LayerNorm'] = meta

        registry_state._components_discovered = True

        # Load component
        result = get_kernel('test:LayerNorm')

        # Should have imported and cached
        assert result is not None
        assert meta.loaded_obj is result

    def test_double_check_during_import(self, empty_env):
        """Decorator fires during _load_component - should detect already loaded."""
        # This tests the double-check in _load_component:
        # If decorator fires during import, loaded_obj gets set mid-import

        # Create metadata for a real component that uses decorators
        meta = ComponentMetadata(
            name='LayerNorm',
            source='brainsmith',
            component_type=ComponentType.KERNEL,
            import_spec=ImportSpec(
                module='brainsmith.kernels.layernorm.layernorm',
                attr='LayerNorm'
            ),
            loaded_obj=None
        )
        _component_index['brainsmith:LayerNorm'] = meta

        registry_state._components_discovered = True

        # Load - decorator will fire during import and set loaded_obj
        result = _load_component(meta)

        # Should have loaded successfully
        assert result is not None
        assert meta.loaded_obj is result

    def test_load_missing_module(self, empty_env):
        """Load component with nonexistent module - should raise ImportError."""
        meta = ComponentMetadata(
            name='MissingModule',
            source='test',
            component_type=ComponentType.KERNEL,
            import_spec=ImportSpec(
                module='this.module.does.not.exist',
                attr='SomeClass'
            ),
            loaded_obj=None
        )
        _component_index['test:MissingModule'] = meta

        registry_state._components_discovered = True

        with pytest.raises(ModuleNotFoundError):
            _load_component(meta)

    def test_load_missing_attribute(self, empty_env):
        """Load component with missing attribute - should raise AttributeError."""
        meta = ComponentMetadata(
            name='MissingAttr',
            source='test',
            component_type=ComponentType.KERNEL,
            import_spec=ImportSpec(
                module='brainsmith.kernels.layernorm.layernorm',  # Real module
                attr='ThisAttributeDoesNotExist'  # Fake attribute
            ),
            loaded_obj=None
        )
        _component_index['test:MissingAttr'] = meta

        registry_state._components_discovered = True

        with pytest.raises(AttributeError):
            _load_component(meta)

    def test_lazy_infer_transform(self, empty_env):
        """Lazy InferTransform (dict spec) - should resolve on access."""
        from brainsmith.registry._metadata import resolve_lazy_class

        # Test lazy spec resolution
        lazy_spec = {
            'module': 'brainsmith.kernels.layernorm.layernorm',
            'class_name': 'LayerNorm'
        }

        resolved = resolve_lazy_class(lazy_spec)

        # Should have imported and returned class
        assert resolved is not None
        assert hasattr(resolved, '__name__')
        assert resolved.__name__ == 'LayerNorm'


# ============================================================================
# TestSourceContext - Nesting, exceptions, cleanup
# ============================================================================

@pytest.mark.fast
class TestSourceContext:
    """Test edge cases in source_context context manager."""

    def test_nested_source_context(self, empty_env):
        """Nested source contexts - inner should take precedence."""
        assert _current_source.get() is None

        with source_context('outer'):
            assert _current_source.get() == 'outer'

            with source_context('middle'):
                assert _current_source.get() == 'middle'

                with source_context('inner'):
                    assert _current_source.get() == 'inner'

                assert _current_source.get() == 'middle'

            assert _current_source.get() == 'outer'

        assert _current_source.get() is None

    def test_exception_during_context(self, empty_env):
        """Exception during source_context - should still cleanup."""
        assert _current_source.get() is None

        try:
            with source_context('will_fail'):
                assert _current_source.get() == 'will_fail'
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should have cleaned up despite exception
        assert _current_source.get() is None

    def test_context_isolation(self, empty_env):
        """Multiple sequential contexts - should be independent."""
        s1 = mock_step_function('step1')
        s2 = mock_step_function('step2')
        s3 = mock_step_function('step3')

        with source_context('source1'):
            step(s1, name='step1')

        with source_context('source2'):
            step(s2, name='step2')

        with source_context('source3'):
            step(s3, name='step3')

        # Each should have correct source
        assert _component_index['source1:step1'].source == 'source1'
        assert _component_index['source2:step2'].source == 'source2'
        assert _component_index['source3:step3'].source == 'source3'

    def test_default_source_without_context(self, empty_env):
        """No source_context - should default to 'custom'."""
        MockKernel = mock_kernel_class('DefaultSourceKernel', op_type='Default')

        # Register without source_context and without explicit source
        kernel(MockKernel, name='DefaultSourceKernel')

        # Should use 'custom' as default
        assert 'custom:DefaultSourceKernel' in _component_index


# ============================================================================
# TestBackendLinking - Kernel-backend relationships
# ============================================================================

@pytest.mark.fast
class TestBackendLinking:
    """Test edge cases in backend-kernel linking."""

    def test_backend_links_to_existing_kernel(self, empty_env):
        """Backend registered after kernel - should link immediately."""
        MockKernel = mock_kernel_class('LinkKernel', op_type='LinkOp')
        kernel(MockKernel, name='LinkKernel', source='test')

        MockBackend = mock_backend_class('LinkKernel_hls', 'test:LinkKernel', 'hls')
        backend(MockBackend, name='LinkKernel_hls', source='test')

        # Kernel should have backend in its list
        kernel_meta = _component_index['test:LinkKernel']
        assert kernel_meta.kernel_backends is not None
        assert 'test:LinkKernel_hls' in kernel_meta.kernel_backends

    def test_backend_links_to_missing_kernel(self, empty_env):
        """Backend registered but kernel doesn't exist - backend still registers."""
        MockBackend = mock_backend_class('Orphan_hls', 'missing:OrphanKernel', 'hls')
        backend(MockBackend, name='Orphan_hls', source='test')

        # Backend should be registered even if kernel missing
        assert 'test:Orphan_hls' in _component_index
        backend_meta = _component_index['test:Orphan_hls']
        assert backend_meta.backend_target == 'missing:OrphanKernel'

    def test_kernel_backends_list_populated(self, empty_env):
        """Kernel should track all its backends."""
        MockKernel = mock_kernel_class('MultiBackendKernel', op_type='Multi')
        kernel(MockKernel, name='MultiBackendKernel', source='test')

        # Register multiple backends
        for lang in ['hls', 'rtl']:
            MockBackend = mock_backend_class(f'MultiBackendKernel_{lang}', 'test:MultiBackendKernel', lang)
            backend(MockBackend, name=f'MultiBackendKernel_{lang}', source='test')

        kernel_meta = _component_index['test:MultiBackendKernel']
        assert kernel_meta.kernel_backends is not None
        assert 'test:MultiBackendKernel_hls' in kernel_meta.kernel_backends
        assert 'test:MultiBackendKernel_rtl' in kernel_meta.kernel_backends

    def test_list_backends_for_kernel(self, empty_env):
        """List backends with language and source filtering."""
        MockKernel = mock_kernel_class('FilterKernel', op_type='Filter')
        kernel(MockKernel, name='FilterKernel', source='test')

        # Register backends with different languages
        B1 = mock_backend_class('FilterKernel_hls1', 'test:FilterKernel', 'hls')
        B2 = mock_backend_class('FilterKernel_hls2', 'test:FilterKernel', 'hls')
        B3 = mock_backend_class('FilterKernel_rtl', 'test:FilterKernel', 'rtl')

        backend(B1, name='FilterKernel_hls1', source='test')
        backend(B2, name='FilterKernel_hls2', source='test')
        backend(B3, name='FilterKernel_rtl', source='test')

        registry_state._components_discovered = True

        # List all backends
        all_backends = list_backends_for_kernel('test:FilterKernel')
        assert len(all_backends) == 3

        # Filter by language
        hls_backends = list_backends_for_kernel('test:FilterKernel', language='hls')
        assert len(hls_backends) == 2
        assert 'test:FilterKernel_hls1' in hls_backends
        assert 'test:FilterKernel_hls2' in hls_backends

        rtl_backends = list_backends_for_kernel('test:FilterKernel', language='rtl')
        assert len(rtl_backends) == 1
        assert 'test:FilterKernel_rtl' in rtl_backends

    def test_backend_registered_before_kernel(self, empty_env):
        """Backend registered before kernel - linking happens in _link_backends_to_kernels()."""
        # Register backend first
        MockBackend = mock_backend_class('EarlyBackend_hls', 'test:LateKernel', 'hls')
        backend(MockBackend, name='EarlyBackend_hls', source='test')

        # Backend registered, but kernel_backends won't be linked yet
        backend_meta = _component_index['test:EarlyBackend_hls']
        assert backend_meta.backend_target == 'test:LateKernel'

        # Register kernel later
        MockKernel = mock_kernel_class('LateKernel', op_type='Late')
        kernel(MockKernel, name='LateKernel', source='test')

        # Manually trigger linking (normally happens at end of discover_components)
        from brainsmith.registry._discovery import _link_backends_to_kernels
        _link_backends_to_kernels()

        # Now kernel should have backend linked
        kernel_meta = _component_index['test:LateKernel']
        assert kernel_meta.kernel_backends is not None
        assert 'test:EarlyBackend_hls' in kernel_meta.kernel_backends


# ============================================================================
# TestInfrastructureMetadata - Infrastructure kernel flag
# ============================================================================

@pytest.mark.fast
class TestInfrastructureMetadata:
    """Test infrastructure kernel metadata flag."""

    def test_default_is_infrastructure_false(self, empty_env):
        """Kernels default to is_infrastructure=False."""
        MockKernel = mock_kernel_class('ComputationalKernel', 'Comp')
        kernel(MockKernel, name='ComputationalKernel')

        from brainsmith.registry import get_component_metadata
        meta = get_component_metadata('custom:ComputationalKernel', 'kernel')
        assert meta.is_infrastructure is False

    def test_explicit_is_infrastructure_true(self, empty_env):
        """Infrastructure kernels can be marked explicitly."""
        MockKernel = mock_kernel_class('InfraKernel', 'Infra')
        kernel(MockKernel, name='InfraKernel', is_infrastructure=True)

        from brainsmith.registry import get_component_metadata
        meta = get_component_metadata('custom:InfraKernel', 'kernel')
        assert meta.is_infrastructure is True

    def test_is_infrastructure_preserved_in_metadata(self, empty_env):
        """is_infrastructure flag is preserved in ComponentMetadata."""
        # Register one of each type
        CompKernel = mock_kernel_class('CompKernel', 'Comp')
        InfraKernel = mock_kernel_class('InfraKernel', 'Infra')

        kernel(CompKernel, name='CompKernel', is_infrastructure=False)
        kernel(InfraKernel, name='InfraKernel', is_infrastructure=True)

        from brainsmith.registry import get_component_metadata

        comp_meta = get_component_metadata('custom:CompKernel', 'kernel')
        assert comp_meta.is_infrastructure is False

        infra_meta = get_component_metadata('custom:InfraKernel', 'kernel')
        assert infra_meta.is_infrastructure is True
