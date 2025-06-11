"""
Tests for the simplified kernels system.
Verifies North Star alignment and functionality.
"""

import unittest
import tempfile
import os
import yaml
from unittest.mock import patch

from brainsmith.kernels import (
    discover_all_kernels,
    load_kernel_package,
    find_compatible_kernels,
    select_optimal_kernel,
    validate_kernel_package,
    generate_finn_config,
    KernelRequirements,
    KernelPackage
)


class TestKernelsSimplification(unittest.TestCase):
    """Test the simplified kernels system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_kernel_data = {
            'name': 'test_kernel',
            'operator_type': 'Convolution',
            'backend': 'HLS',
            'version': '1.0.0',
            'author': 'Test Author',
            'parameters': {
                'pe_range': [1, 32],
                'simd_range': [1, 16],
                'supported_datatypes': ['int8', 'int16']
            },
            'files': {
                'rtl_source': 'test.sv',
                'hw_custom_op': 'test.py'
            },
            'performance': {
                'base_throughput': 1000,
                'base_latency': 10
            },
            'validation': {
                'verified': True
            }
        }
    
    def test_kernel_package_creation(self):
        """Test KernelPackage data structure"""
        kernel = KernelPackage(
            name="test_kernel",
            operator_type="Convolution",
            backend="HLS",
            version="1.0.0",
            parameters={'pe_range': [1, 32], 'simd_range': [1, 16]}
        )
        
        self.assertEqual(kernel.name, "test_kernel")
        self.assertEqual(kernel.operator_type, "Convolution")
        self.assertEqual(kernel.get_pe_range(), (1, 32))
        self.assertEqual(kernel.get_simd_range(), (1, 16))
    
    def test_kernel_compatibility_checking(self):
        """Test kernel compatibility logic"""
        kernel = KernelPackage(
            name="test_kernel",
            operator_type="Convolution",
            backend="HLS",
            version="1.0.0",
            parameters={
                'pe_range': [1, 32],
                'simd_range': [1, 16],
                'supported_datatypes': ['int8', 'int16']
            }
        )
        
        # Compatible requirements
        compatible_reqs = {
            'operator_type': 'Convolution',
            'datatype': 'int8',
            'min_pe': 4,
            'max_pe': 16
        }
        self.assertTrue(kernel.is_compatible_with(compatible_reqs))
        
        # Incompatible operator type
        incompatible_reqs = {
            'operator_type': 'MatMul',
            'datatype': 'int8'
        }
        self.assertFalse(kernel.is_compatible_with(incompatible_reqs))
        
        # Incompatible PE range
        incompatible_pe_reqs = {
            'operator_type': 'Convolution',
            'min_pe': 64  # Exceeds kernel's max PE
        }
        self.assertFalse(kernel.is_compatible_with(incompatible_pe_reqs))
    
    def test_kernel_requirements(self):
        """Test KernelRequirements data structure"""
        requirements = KernelRequirements(
            operator_type="Convolution",
            datatype="int8",
            min_pe=4,
            max_pe=32
        )
        
        self.assertEqual(requirements.operator_type, "Convolution")
        self.assertEqual(requirements.datatype, "int8")
        self.assertEqual(requirements.min_pe, 4)
        self.assertEqual(requirements.max_pe, 32)
    
    def test_load_kernel_package_from_yaml(self):
        """Test loading kernel package from YAML file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test kernel.yaml
            kernel_yaml_path = os.path.join(temp_dir, 'kernel.yaml')
            with open(kernel_yaml_path, 'w') as f:
                yaml.dump(self.test_kernel_data, f)
            
            # Load kernel package
            kernel = load_kernel_package(temp_dir)
            
            self.assertIsNotNone(kernel)
            self.assertEqual(kernel.name, 'test_kernel')
            self.assertEqual(kernel.operator_type, 'Convolution')
            self.assertEqual(kernel.backend, 'HLS')
            self.assertEqual(kernel.get_pe_range(), (1, 32))
    
    def test_validate_kernel_package(self):
        """Test kernel package validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid kernel.yaml
            kernel_yaml_path = os.path.join(temp_dir, 'kernel.yaml')
            with open(kernel_yaml_path, 'w') as f:
                yaml.dump(self.test_kernel_data, f)
            
            # Create referenced files
            for filename in self.test_kernel_data['files'].values():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w') as f:
                    f.write("# Placeholder file")
            
            # Validate package
            result = validate_kernel_package(temp_dir)
            
            self.assertTrue(result.is_valid)
            self.assertEqual(len(result.errors), 0)
    
    def test_validate_invalid_kernel_package(self):
        """Test validation of invalid kernel package"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid kernel.yaml (missing required fields)
            invalid_data = {'name': 'invalid_kernel'}  # Missing required fields
            kernel_yaml_path = os.path.join(temp_dir, 'kernel.yaml')
            with open(kernel_yaml_path, 'w') as f:
                yaml.dump(invalid_data, f)
            
            # Validate package
            result = validate_kernel_package(temp_dir)
            
            self.assertFalse(result.is_valid)
            self.assertGreater(len(result.errors), 0)
    
    def test_find_compatible_kernels(self):
        """Test finding compatible kernels"""
        kernels = {
            'conv_kernel': KernelPackage(
                name="conv_kernel",
                operator_type="Convolution",
                backend="HLS",
                version="1.0.0",
                parameters={'supported_datatypes': ['int8']}
            ),
            'matmul_kernel': KernelPackage(
                name="matmul_kernel", 
                operator_type="MatMul",
                backend="RTL",
                version="1.0.0",
                parameters={'supported_datatypes': ['int8']}
            )
        }
        
        # Find convolution kernels
        conv_reqs = {'operator_type': 'Convolution', 'datatype': 'int8'}
        compatible = find_compatible_kernels(conv_reqs, kernels)
        
        self.assertIn('conv_kernel', compatible)
        self.assertNotIn('matmul_kernel', compatible)
    
    def test_select_optimal_kernel(self):
        """Test optimal kernel selection"""
        kernels = {
            'test_kernel': KernelPackage(
                name="test_kernel",
                operator_type="Convolution",
                backend="HLS", 
                version="1.0.0",
                parameters={
                    'pe_range': [1, 64],
                    'simd_range': [1, 32],
                    'supported_datatypes': ['int8']
                },
                validation={'verified': True}
            )
        }
        
        requirements = KernelRequirements(
            operator_type="Convolution",
            datatype="int8"
        )
        
        selection = select_optimal_kernel(requirements, available_kernels=kernels)
        
        self.assertIsNotNone(selection)
        self.assertEqual(selection.kernel.name, 'test_kernel')
        self.assertGreater(selection.pe_parallelism, 0)
        self.assertGreater(selection.simd_width, 0)
    
    def test_generate_finn_config(self):
        """Test FINN configuration generation"""
        from brainsmith.kernels.types import KernelSelection
        
        kernel = KernelPackage(
            name="test_kernel",
            operator_type="Convolution",
            backend="HLS",
            version="1.0.0"
        )
        
        selection = KernelSelection(
            kernel=kernel,
            pe_parallelism=16,
            simd_width=8,
            memory_mode="internal"
        )
        
        selections = {'layer1': selection}
        finn_config = generate_finn_config(selections)
        
        self.assertIn('folding_config', finn_config)
        self.assertIn('kernels', finn_config)
        self.assertIn('global_settings', finn_config)
        
        # Check layer configuration
        self.assertIn('layer1', finn_config['folding_config'])
        layer_config = finn_config['folding_config']['layer1']
        self.assertEqual(layer_config['PE'], 16)
        self.assertEqual(layer_config['SIMD'], 8)
    
    def test_north_star_alignment(self):
        """Test that the system follows North Star principles"""
        # 1. Simple Functions - check function signatures are simple
        kernels = discover_all_kernels()
        self.assertIsInstance(kernels, dict)
        
        # 2. Pure Functions - test that functions don't have hidden state
        kernels1 = discover_all_kernels()
        kernels2 = discover_all_kernels()
        # Should return same results (no hidden state mutation)
        self.assertEqual(set(kernels1.keys()), set(kernels2.keys()))
        
        # 3. Data Transformations - test that functions transform data
        requirements = KernelRequirements(operator_type="Convolution", datatype="int8")
        compatible = find_compatible_kernels(requirements, kernels1)
        self.assertIsInstance(compatible, list)
        
        # 4. Composable - test that functions can be composed
        if compatible:
            selection = select_optimal_kernel(requirements, available_kernels=kernels1)
            if selection:
                finn_config = generate_finn_config({'test': selection})
                self.assertIsInstance(finn_config, dict)
        
        # 5. Observable - test that data structures are inspectable
        for kernel_name, kernel in kernels1.items():
            self.assertIsInstance(kernel.name, str)
            self.assertIsInstance(kernel.operator_type, str)
            self.assertIsInstance(kernel.parameters, dict)


if __name__ == '__main__':
    unittest.main()