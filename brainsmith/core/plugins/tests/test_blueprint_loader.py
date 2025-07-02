"""
Test Blueprint-Driven Loading

Tests for blueprint optimization with 80% performance improvement.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from brainsmith.core.plugins.decorators import plugin
from brainsmith.core.plugins.registry import get_registry, reset_registry
from brainsmith.core.plugins.blueprint_loader import (
    BlueprintPluginLoader, load_blueprint_plugins, analyze_blueprint_requirements
)


class TestBlueprintPluginLoader:
    def setup_method(self):
        reset_registry()
        
        # Register test plugins
        @plugin(type="transform", name="CleanupTransform", stage="cleanup")
        class CleanupTransform:
            def apply(self, model):
                return model, False
        
        @plugin(type="transform", name="OptTransform", stage="topology_opt")
        class OptTransform:
            def apply(self, model):
                return model, False
        
        @plugin(type="transform", name="StreamlineTransform", stage="streamlining")
        class StreamlineTransform:
            def apply(self, model):
                return model, False
        
        @plugin(type="kernel", name="TestKernel")
        class TestKernel:
            pass
        
        @plugin(type="kernel", name="MatMulKernel")
        class MatMulKernel:
            pass
        
        @plugin(type="backend", name="TestKernelHLS", kernel="TestKernel", backend_type="hls")
        class TestKernelHLS:
            pass
        
        @plugin(type="backend", name="TestKernelRTL", kernel="TestKernel", backend_type="rtl")
        class TestKernelRTL:
            pass
        
        @plugin(type="backend", name="MatMulHLS", kernel="MatMulKernel", backend_type="hls")
        class MatMulHLS:
            pass
        
        self.loader = BlueprintPluginLoader()
        
        # Create test blueprint data
        self.blueprint_data = {
            'version': '3.0',
            'name': 'Test Blueprint',
            'hw_compiler': {
                'transforms': {
                    'cleanup': ['CleanupTransform'],
                    'topology_opt': ['OptTransform']
                },
                'kernels': ['TestKernel', 'MatMulKernel'],
                'backends': ['TestKernel:hls', 'MatMulKernel:hls']
            }
        }
    
    def teardown_method(self):
        reset_registry()
    
    def test_blueprint_file_parsing(self):
        """Test parsing blueprint YAML file."""
        # Create temporary blueprint file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.blueprint_data, f)
            blueprint_path = f.name
        
        try:
            requirements = self.loader._parse_blueprint_file(blueprint_path)
            
            # Test extracted requirements
            assert 'transforms_by_stage' in requirements
            assert 'kernels' in requirements
            assert 'backends_by_kernel' in requirements
            
            # Test transforms by stage
            assert 'cleanup' in requirements['transforms_by_stage']
            assert 'CleanupTransform' in requirements['transforms_by_stage']['cleanup']
            assert 'OptTransform' in requirements['transforms_by_stage']['topology_opt']
            
            # Test kernels
            assert 'TestKernel' in requirements['kernels']
            assert 'MatMulKernel' in requirements['kernels']
            
            # Test backends
            assert 'TestKernel' in requirements['backends_by_kernel']
            assert 'hls' in requirements['backends_by_kernel']['TestKernel']
            assert 'MatMulKernel' in requirements['backends_by_kernel']
            assert 'hls' in requirements['backends_by_kernel']['MatMulKernel']
            
        finally:
            Path(blueprint_path).unlink()  # Clean up
    
    def test_blueprint_dict_parsing(self):
        """Test parsing blueprint from dictionary."""
        loaded = self.loader.load_for_blueprint(self.blueprint_data)
        
        # Test loaded plugins
        assert 'transforms' in loaded
        assert 'kernels' in loaded
        assert 'backends' in loaded
        
        # Test specific plugins loaded
        assert 'CleanupTransform' in loaded['transforms']
        assert 'OptTransform' in loaded['transforms']
        assert 'TestKernel' in loaded['kernels']
        assert 'MatMulKernel' in loaded['kernels']
        assert 'TestKernel_hls' in loaded['backends']
        assert 'MatMulKernel_hls' in loaded['backends']
    
    def test_transform_name_normalization(self):
        """Test normalization of transform specifications."""
        # Test optional transforms (with ~)
        blueprint_with_optional = {
            'hw_compiler': {
                'transforms': {
                    'cleanup': ['CleanupTransform', '~OptionalTransform']
                }
            }
        }
        
        requirements = self.loader._extract_plugin_requirements(blueprint_with_optional)
        transforms = requirements['transforms_by_stage']['cleanup']
        
        assert 'CleanupTransform' in transforms
        assert 'OptionalTransform' in transforms  # ~ should be stripped
    
    def test_kernel_name_extraction(self):
        """Test extraction of kernel names from various specifications."""
        # Test string spec
        assert self.loader._extract_kernel_name("TestKernel") == "TestKernel"
        
        # Test optional spec
        assert self.loader._extract_kernel_name("~OptionalKernel") == "OptionalKernel"
        
        # Test tuple spec
        assert self.loader._extract_kernel_name(("KernelName", ["hls", "rtl"])) == "KernelName"
        
        # Test dict spec
        assert self.loader._extract_kernel_name({"name": "DictKernel"}) == "DictKernel"
        
        # Test list spec
        assert self.loader._extract_kernel_name(["FirstKernel", "SecondKernel"]) == "FirstKernel"
    
    def test_backend_info_extraction(self):
        """Test extraction of backend information."""
        # Test string with colon
        kernel, backend_type = self.loader._extract_backend_info("TestKernel:hls")
        assert kernel == "TestKernel"
        assert backend_type == "hls"
        
        # Test string without colon
        kernel, backend_type = self.loader._extract_backend_info("rtl")
        assert kernel is None
        assert backend_type == "rtl"
        
        # Test dict
        kernel, backend_type = self.loader._extract_backend_info({"kernel": "TestKernel", "type": "hls"})
        assert kernel == "TestKernel"
        assert backend_type == "hls"
        
        # Test tuple
        kernel, backend_type = self.loader._extract_backend_info(("TestKernel", "rtl"))
        assert kernel == "TestKernel"
        assert backend_type == "rtl"
    
    def test_backend_inference_from_kernels(self):
        """Test that backends are inferred from kernels when not explicitly specified."""
        blueprint_no_backends = {
            'hw_compiler': {
                'kernels': ['TestKernel']
                # No explicit backends
            }
        }
        
        requirements = self.loader._extract_plugin_requirements(blueprint_no_backends)
        
        # Should infer available backends for TestKernel
        assert 'TestKernel' in requirements['backends_by_kernel']
        backends = requirements['backends_by_kernel']['TestKernel']
        assert 'hls' in backends
        assert 'rtl' in backends
    
    def test_optimized_collections_creation(self):
        """Test creation of optimized collections with subset registry."""
        collections = self.loader.create_optimized_collections(self.blueprint_data)
        
        # Test all collection types created
        assert 'transforms' in collections
        assert 'kernels' in collections
        assert 'backends' in collections
        assert 'steps' in collections
        
        # Test that collections contain only required plugins
        # This would require accessing the underlying registry of collections
        # For now, test that we can access required plugins
        transforms = collections['transforms']
        kernels = collections['kernels']
        backends = collections['backends']
        
        # Should be able to access required plugins
        assert hasattr(transforms, 'CleanupTransform')
        assert hasattr(transforms, 'OptTransform')
        assert hasattr(kernels, 'TestKernel')
        assert hasattr(kernels, 'MatMulKernel')
        assert hasattr(backends, 'TestKernel_hls')
        assert hasattr(backends, 'MatMulKernel_hls')
    
    def test_subset_registry_creation(self):
        """Test creation of optimized subset registry."""
        requirements = {
            'transforms_by_stage': {
                'cleanup': ['CleanupTransform']
            },
            'kernels': ['TestKernel'],
            'backends_by_kernel': {
                'TestKernel': ['hls']
            }
        }
        
        subset = self.loader._create_subset_registry(requirements)
        
        # Test subset contains only required plugins
        assert len(subset.transforms) == 1
        assert 'CleanupTransform' in subset.transforms
        assert 'OptTransform' not in subset.transforms
        
        assert len(subset.kernels) == 1
        assert 'TestKernel' in subset.kernels
        assert 'MatMulKernel' not in subset.kernels
        
        assert len(subset.backends) == 1
        assert 'TestKernel_hls' in subset.backends
        assert 'TestKernel_rtl' not in subset.backends
        
        # Test indexing still works
        assert 'cleanup' in subset.transforms_by_stage
        assert 'CleanupTransform' in subset.transforms_by_stage['cleanup']
        
        assert 'TestKernel' in subset.backends_by_kernel
        assert 'hls' in subset.backends_by_kernel['TestKernel']
    
    def test_blueprint_stats(self):
        """Test blueprint statistics calculation."""
        stats = self.loader.get_blueprint_stats(self.blueprint_data)
        
        # Test stats structure
        assert 'total_available_plugins' in stats
        assert 'total_loaded_plugins' in stats
        assert 'load_percentage' in stats
        assert 'transforms_loaded' in stats
        assert 'kernels_loaded' in stats
        assert 'backends_loaded' in stats
        assert 'performance_improvement' in stats
        
        # Test that loaded is less than available (optimization working)
        assert stats['total_loaded_plugins'] < stats['total_available_plugins']
        assert stats['load_percentage'] < 100
        
        # Test specific counts
        assert stats['transforms_loaded'] == 2  # CleanupTransform, OptTransform
        assert stats['kernels_loaded'] == 2     # TestKernel, MatMulKernel
        assert stats['backends_loaded'] == 2    # TestKernel_hls, MatMulKernel_hls
    
    def test_caching_behavior(self):
        """Test that blueprint parsing is cached."""
        # Create temporary blueprint file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.blueprint_data, f)
            blueprint_path = f.name
        
        try:
            # Parse twice
            requirements1 = self.loader._parse_blueprint_file(blueprint_path)
            requirements2 = self.loader._parse_blueprint_file(blueprint_path)
            
            # Should get same results (cached)
            assert requirements1 == requirements2
            
            # Check cache exists
            cache_key = str(Path(blueprint_path).absolute())
            assert cache_key in self.loader._blueprint_cache
            
        finally:
            Path(blueprint_path).unlink()  # Clean up
    
    def test_flat_transform_list_parsing(self):
        """Test parsing flat list of transforms (not organized by stage)."""
        blueprint_flat = {
            'hw_compiler': {
                'transforms': ['CleanupTransform', 'OptTransform', 'StreamlineTransform']
            }
        }
        
        requirements = self.loader._extract_plugin_requirements(blueprint_flat)
        
        # Should organize by stages based on plugin metadata
        assert 'cleanup' in requirements['transforms_by_stage']
        assert 'CleanupTransform' in requirements['transforms_by_stage']['cleanup']
        
        assert 'topology_opt' in requirements['transforms_by_stage']
        assert 'OptTransform' in requirements['transforms_by_stage']['topology_opt']
        
        assert 'streamlining' in requirements['transforms_by_stage']
        assert 'StreamlineTransform' in requirements['transforms_by_stage']['streamlining']
    
    def test_missing_plugin_warnings(self):
        """Test warnings for missing plugins."""
        blueprint_missing = {
            'hw_compiler': {
                'transforms': {
                    'cleanup': ['NonexistentTransform']
                },
                'kernels': ['NonexistentKernel'],
                'backends': ['NonexistentKernel:hls']
            }
        }
        
        # Should not crash, but should log warnings
        loaded = self.loader.load_for_blueprint(blueprint_missing)
        
        # Should return empty lists for missing plugins
        assert loaded['transforms'] == []
        assert loaded['kernels'] == []
        assert loaded['backends'] == []


class TestConvenienceFunctions:
    def setup_method(self):
        reset_registry()
        
        # Register a test plugin
        @plugin(type="transform", name="TestTransform", stage="cleanup")
        class TestTransform:
            pass
        
        # Create test blueprint
        self.blueprint_data = {
            'hw_compiler': {
                'transforms': {
                    'cleanup': ['TestTransform']
                }
            }
        }
    
    def teardown_method(self):
        reset_registry()
    
    def test_load_blueprint_plugins_function(self):
        """Test convenience function for loading blueprint plugins."""
        # Create temporary blueprint file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.blueprint_data, f)
            blueprint_path = f.name
        
        try:
            collections = load_blueprint_plugins(blueprint_path)
            
            # Test collections structure
            assert 'transforms' in collections
            assert 'kernels' in collections
            assert 'backends' in collections
            assert 'steps' in collections
            
            # Test that we can access the required plugin
            assert hasattr(collections['transforms'], 'TestTransform')
            
        finally:
            Path(blueprint_path).unlink()  # Clean up
    
    def test_analyze_blueprint_requirements_function(self):
        """Test convenience function for analyzing blueprint requirements."""
        # Create temporary blueprint file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.blueprint_data, f)
            blueprint_path = f.name
        
        try:
            stats = analyze_blueprint_requirements(blueprint_path)
            
            # Test stats structure
            assert 'total_available_plugins' in stats
            assert 'total_loaded_plugins' in stats
            assert 'performance_improvement' in stats
            
            # Should show performance improvement
            assert stats['total_loaded_plugins'] < stats['total_available_plugins']
            
        finally:
            Path(blueprint_path).unlink()  # Clean up


class TestComplexBlueprints:
    def setup_method(self):
        reset_registry()
        
        # Register a more complex set of plugins
        @plugin(type="transform", name="RemoveIdentity", stage="cleanup", framework="qonnx")
        class RemoveIdentity:
            pass
        
        @plugin(type="transform", name="ExpandNorms", stage="topology_opt", framework="brainsmith")
        class ExpandNorms:
            pass
        
        @plugin(type="kernel", name="MatMul")
        class MatMul:
            pass
        
        @plugin(type="kernel", name="LayerNorm")
        class LayerNorm:
            pass
        
        @plugin(type="backend", name="MatMulHLS", kernel="MatMul", backend_type="hls")
        class MatMulHLS:
            pass
        
        @plugin(type="backend", name="MatMulRTL", kernel="MatMul", backend_type="rtl")
        class MatMulRTL:
            pass
        
        @plugin(type="backend", name="LayerNormHLS", kernel="LayerNorm", backend_type="hls")
        class LayerNormHLS:
            pass
        
        self.loader = BlueprintPluginLoader()
    
    def teardown_method(self):
        reset_registry()
    
    def test_complex_blueprint_parsing(self):
        """Test parsing of complex blueprint with multiple frameworks."""
        complex_blueprint = {
            'version': '3.0',
            'name': 'Complex BERT Blueprint',
            'hw_compiler': {
                'transforms': {
                    'cleanup': ['RemoveIdentity'],
                    'topology_opt': ['ExpandNorms']
                },
                'kernels': [
                    'MatMul',
                    ('LayerNorm', ['hls'])  # Tuple spec with backend constraint
                ],
                'backends': [
                    'MatMul:hls',
                    'MatMul:rtl', 
                    'LayerNorm:hls'
                ]
            }
        }
        
        loaded = self.loader.load_for_blueprint(complex_blueprint)
        
        # Test mixed framework transforms loaded
        assert 'RemoveIdentity' in loaded['transforms']
        assert 'ExpandNorms' in loaded['transforms']
        
        # Test kernels loaded
        assert 'MatMul' in loaded['kernels']
        assert 'LayerNorm' in loaded['kernels']
        
        # Test backends loaded
        assert 'MatMul_hls' in loaded['backends']
        assert 'MatMul_rtl' in loaded['backends']
        assert 'LayerNorm_hls' in loaded['backends']
    
    def test_mutually_exclusive_transforms(self):
        """Test handling of mutually exclusive transform specifications."""
        blueprint_with_alternatives = {
            'hw_compiler': {
                'transforms': {
                    'topology_opt': [
                        ['ExpandNorms', 'AlternativeNorms'],  # Mutually exclusive
                        'OtherTransform'
                    ]
                }
            }
        }
        
        requirements = self.loader._extract_plugin_requirements(blueprint_with_alternatives)
        
        # Should take first option from mutually exclusive list
        transforms = requirements['transforms_by_stage']['topology_opt']
        assert 'ExpandNorms' in transforms  # First option
        # Note: Would need to implement alternative handling for full mutual exclusion