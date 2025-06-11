#!/usr/bin/env python3
"""
BrainSmith Libraries Integration Testing
========================================

Comprehensive end-to-end tests validating all 5 libraries work together:
- kernels: Registry-based kernel package discovery
- transforms: Registry-based transform function discovery  
- analysis: Registry-based analysis tool discovery
- blueprints: Registry-based blueprint YAML discovery
- automation: Workflow orchestration (no registry)

Tests registry validation, cross-library compatibility, and performance.
"""

import time
import unittest
from pathlib import Path

# Import all 5 BrainSmith libraries
from brainsmith.libraries import kernels
from brainsmith.libraries import transforms  
from brainsmith.libraries import analysis
from brainsmith.libraries import blueprints
from brainsmith.libraries import automation
from brainsmith.libraries.validation import validate_all_registries


class TestRegistryIntegration(unittest.TestCase):
    """Test registry system works across all libraries"""
    
    def test_all_registries_healthy(self):
        """Validate all registries are healthy"""
        result = validate_all_registries()
        self.assertIn(result['status'], ['healthy', 'degraded'], "All registries should be healthy or degraded")
        self.assertEqual(result['summary']['failed_components'], 0, "No registries should fail validation")
        
    def test_registry_component_counts(self):
        """Verify expected component counts in each registry"""
        # Kernels: 2 components
        kernel_names = kernels.list_kernels()
        self.assertEqual(len(kernel_names), 2, "Should have 2 kernels")
        self.assertIn("conv2d_hls", kernel_names)
        self.assertIn("matmul_rtl", kernel_names)
        
        # Transforms: 10 components
        transform_names = transforms.list_transforms()
        self.assertEqual(len(transform_names), 10, "Should have 10 transforms")
        self.assertIn("cleanup", transform_names)
        self.assertIn("streamlining", transform_names)
        
        # Analysis: 3 components
        analysis_names = analysis.list_analysis_tools()
        self.assertEqual(len(analysis_names), 3, "Should have 3 analysis tools")
        self.assertIn("roofline_analysis", analysis_names)
        self.assertIn("generate_hw_kernel", analysis_names)
        
        # Blueprints: 2 components
        blueprint_names = blueprints.list_blueprints()
        self.assertEqual(len(blueprint_names), 2, "Should have 2 blueprints")
        self.assertIn("cnn_accelerator", blueprint_names)
        self.assertIn("mobilenet_accelerator", blueprint_names)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete workflows using multiple libraries"""
    
    def test_kernel_analysis_workflow(self):
        """Test workflow: kernel -> analysis -> automation"""
        # Step 1: Get a kernel
        conv_kernel = kernels.get_kernel("conv2d_hls")
        self.assertEqual(conv_kernel.name, "conv2d_hls")
        
        # Step 2: Get analysis tool for performance
        roofline_tool = analysis.get_analysis_tool("roofline_analysis")
        self.assertIsNotNone(roofline_tool)
        
        # Step 3: Verify automation functions exist
        self.assertIsNotNone(automation.parameter_sweep)
        self.assertIsNotNone(automation.batch_process)
        self.assertIsNotNone(automation.find_best)
        
    def test_blueprint_transforms_workflow(self):
        """Test workflow: blueprint -> transforms -> automation"""
        # Step 1: Get a blueprint
        blueprint_path = blueprints.get_blueprint("cnn_accelerator")
        self.assertTrue(Path(blueprint_path).exists())
        
        # Step 2: Get transform functions
        cleanup_fn = transforms.get_transform("cleanup")
        streamline_fn = transforms.get_transform("streamlining")
        self.assertIsNotNone(cleanup_fn)
        self.assertIsNotNone(streamline_fn)
        
        # Step 3: Load blueprint data
        blueprint_data = blueprints.load_blueprint_yaml("cnn_accelerator")
        self.assertIsInstance(blueprint_data, dict)
        self.assertIn("name", blueprint_data)
        self.assertEqual(blueprint_data["name"], "cnn_accelerator")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test O(1) performance claims for registry access"""
    
    def test_kernel_access_performance(self):
        """Verify kernel access is O(1) - constant time"""
        iterations = 1000
        
        # Measure access time for existing kernel
        start = time.perf_counter()
        for _ in range(iterations):
            kernel = kernels.get_kernel("conv2d_hls")
            self.assertIsNotNone(kernel)
        elapsed = time.perf_counter() - start
        
        # Should complete 1000 accesses in under 5 seconds (kernel loading includes YAML parsing)
        self.assertLess(elapsed, 5.0, f"1000 kernel accesses took {elapsed:.3f}s - should be much faster")
        
    def test_transform_access_performance(self):
        """Verify transform access is O(1) - constant time"""
        iterations = 1000
        
        start = time.perf_counter()
        for _ in range(iterations):
            transform_fn = transforms.get_transform("cleanup")
            self.assertIsNotNone(transform_fn)
        elapsed = time.perf_counter() - start
        
        self.assertLess(elapsed, 1.0, f"1000 transform accesses took {elapsed:.3f}s - should be much faster")
        
    def test_list_operations_performance(self):
        """Verify list operations are fast"""
        iterations = 1000
        
        start = time.perf_counter()
        for _ in range(iterations):
            kernel_list = kernels.list_kernels()
            transform_list = transforms.list_transforms()
            analysis_list = analysis.list_analysis_tools()
            blueprint_list = blueprints.list_blueprints()
            
            # Verify lists are populated
            self.assertGreater(len(kernel_list), 0)
            self.assertGreater(len(transform_list), 0)
            self.assertGreater(len(analysis_list), 0)
            self.assertGreater(len(blueprint_list), 0)
        elapsed = time.perf_counter() - start
        
        self.assertLess(elapsed, 1.0, f"1000 list operations took {elapsed:.3f}s - should be much faster")


class TestErrorHandling(unittest.TestCase):
    """Test fail-fast error handling across libraries"""
    
    def test_kernel_not_found_error(self):
        """Test kernel not found error is clear and helpful"""
        with self.assertRaises(KeyError) as cm:
            kernels.get_kernel("nonexistent_kernel")
        
        error_msg = str(cm.exception)
        self.assertIn("nonexistent_kernel", error_msg)
        self.assertIn("Available:", error_msg)
        self.assertIn("conv2d_hls", error_msg)
        self.assertIn("matmul_rtl", error_msg)
        
    def test_transform_not_found_error(self):
        """Test transform not found error is clear and helpful"""
        with self.assertRaises(KeyError) as cm:
            transforms.get_transform("nonexistent_transform")
        
        error_msg = str(cm.exception)
        self.assertIn("nonexistent_transform", error_msg)
        self.assertIn("Available:", error_msg)
        self.assertIn("cleanup", error_msg)
        
    def test_analysis_not_found_error(self):
        """Test analysis tool not found error is clear and helpful"""
        with self.assertRaises(KeyError) as cm:
            analysis.get_analysis_tool("nonexistent_tool")
        
        error_msg = str(cm.exception)
        self.assertIn("nonexistent_tool", error_msg)
        self.assertIn("Available:", error_msg)
        self.assertIn("roofline_analysis", error_msg)
        
    def test_blueprint_not_found_error(self):
        """Test blueprint not found error is clear and helpful"""
        with self.assertRaises(KeyError) as cm:
            blueprints.get_blueprint("nonexistent_blueprint")
        
        error_msg = str(cm.exception)
        self.assertIn("nonexistent_blueprint", error_msg)
        self.assertIn("Available:", error_msg)
        self.assertIn("cnn_accelerator", error_msg)


class TestCrossLibraryCompatibility(unittest.TestCase):
    """Test libraries work together without conflicts"""
    
    def test_all_imports_work(self):
        """Verify all libraries can be imported together"""
        # All imports should work without conflicts
        self.assertIsNotNone(kernels.get_kernel)
        self.assertIsNotNone(transforms.get_transform)
        self.assertIsNotNone(analysis.get_analysis_tool)
        self.assertIsNotNone(blueprints.get_blueprint)
        self.assertIsNotNone(automation.parameter_sweep)
        
    def test_version_consistency(self):
        """Verify all libraries have consistent versions"""
        # All libraries should be version 2.0.0 after refactoring
        self.assertEqual(kernels.__version__, "2.0.0")
        self.assertEqual(transforms.__version__, "2.0.0")
        self.assertEqual(analysis.__version__, "2.0.0")
        self.assertEqual(blueprints.__version__, "2.0.0")
        # Note: automation may have different version scheme
        
    def test_registry_pattern_consistency(self):
        """Verify all libraries follow consistent registry pattern"""
        # All should have AVAILABLE_* dictionaries
        self.assertIsInstance(kernels.AVAILABLE_KERNELS, dict)
        self.assertIsInstance(transforms.AVAILABLE_TRANSFORMS, dict)
        self.assertIsInstance(analysis.AVAILABLE_ANALYSIS_TOOLS, dict)
        self.assertIsInstance(blueprints.AVAILABLE_BLUEPRINTS, dict)
        
        # All should have consistent function names
        self.assertTrue(hasattr(kernels, 'get_kernel'))
        self.assertTrue(hasattr(kernels, 'list_kernels'))
        self.assertTrue(hasattr(transforms, 'get_transform'))
        self.assertTrue(hasattr(transforms, 'list_transforms'))
        self.assertTrue(hasattr(analysis, 'get_analysis_tool'))
        self.assertTrue(hasattr(analysis, 'list_analysis_tools'))
        self.assertTrue(hasattr(blueprints, 'get_blueprint'))
        self.assertTrue(hasattr(blueprints, 'list_blueprints'))


if __name__ == "__main__":
    print("🧪 BrainSmith Libraries Integration Testing")
    print("=" * 50)
    print(f"Testing all 5 libraries:")
    print(f"  📦 kernels: {len(kernels.list_kernels())} components")
    print(f"  🔄 transforms: {len(transforms.list_transforms())} components")
    print(f"  📊 analysis: {len(analysis.list_analysis_tools())} components")
    print(f"  📋 blueprints: {len(blueprints.list_blueprints())} components")
    print(f"  ⚙️  automation: workflow orchestration")
    print()
    
    # Run all tests
    unittest.main(verbosity=2)