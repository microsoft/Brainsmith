"""
Auto-generated test suite for Vector Dot Product Accelerator
Generated using Brainsmith-2 Hardware Kernel Generator
Source: vector_dot_product.sv
Generated at: 2025-06-08T08:00:00.000000
"""

import pytest
import numpy as np
import os
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Try to import FINN components, fallback if not available
try:
    from finn.core.modelwrapper import ModelWrapper
    from finn.core.onnx_exec import execute_onnx
    FINN_AVAILABLE = True
except ImportError:
    FINN_AVAILABLE = False
    class ModelWrapper: pass

try:
    from qonnx.core.datatype import DataType
    from qonnx.util.basic import gen_finn_dt_tensor
    QONNX_AVAILABLE = True
except ImportError:
    QONNX_AVAILABLE = False
    class DataType: pass
    def gen_finn_dt_tensor(*args, **kwargs): return np.array([])

# Import the generated classes
from .vector_dot_product_hwcustomop import VectorDotProductHWCustomOp

# Import dataflow framework components for validation
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
from brainsmith.dataflow.core.validation import ValidationResult, ValidationSeverity
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp

class TestVectorDotProductOperation:
    """Comprehensive test suite for vector dot product HWCustomOp."""
    
    @pytest.fixture
    def mock_onnx_node(self):
        """Create mock ONNX node for testing."""
        mock_node = Mock()
        mock_node.op_type = "VectorDotProduct"
        mock_node.name = "test_vector_dot_product"
        mock_node.attribute = []
        return mock_node
    
    @pytest.fixture
    def test_node(self, mock_onnx_node):
        """Create test node instance."""
        return VectorDotProductHWCustomOp(mock_onnx_node)
    
    # ===========================================
    # Basic Functionality Tests
    # ===========================================
    
    def test_node_creation(self, test_node):
        """Test basic node creation and initialization."""
        assert isinstance(test_node, VectorDotProductHWCustomOp)
        assert isinstance(test_node, AutoHWCustomOp)
        assert test_node.vector_size == 768
        assert test_node.parallelism == 8
        assert test_node.data_width == 8
        assert test_node.result_width == 32
    
    def test_dataflow_model_creation(self, test_node):
        """Test dataflow model initialization."""
        model = test_node.dataflow_model
        
        # Verify model exists and has correct type
        assert model is not None
        assert model.operation_type == "dot_product"
        
        # Verify interfaces
        interfaces = model.interfaces
        assert len(interfaces) == 4  # 2 inputs, 1 output, 1 config
        
        # Check input interfaces
        input_interfaces = [iface for iface in interfaces if iface.interface_type == "INPUT"]
        assert len(input_interfaces) == 2
        
        for input_iface in input_interfaces:
            assert input_iface.qDim == 768
            assert input_iface.tDim == 96
            assert input_iface.sDim == 8
            assert input_iface.dtype == "INT8"
        
        # Check output interface
        output_interfaces = [iface for iface in interfaces if iface.interface_type == "OUTPUT"]
        assert len(output_interfaces) == 1
        
        output_iface = output_interfaces[0]
        assert output_iface.qDim == 1
        assert output_iface.tDim == 1
        assert output_iface.sDim == 1
        assert output_iface.dtype == "INT32"
    
    # ===========================================
    # Data Type Tests
    # ===========================================
    
    @pytest.mark.parametrize("input_index", [0, 1])
    def test_input_datatypes(self, test_node, input_index):
        """Test input datatype specifications."""
        if QONNX_AVAILABLE:
            input_dtype = test_node.get_input_datatype(input_index)
            assert input_dtype.name in ["INT8", "UINT8"]
            assert input_dtype.bitwidth() == 8
        else:
            # Mock test when QONNX not available
            assert test_node.data_width == 8
    
    def test_output_datatype(self, test_node):
        """Test output datatype specification."""
        if QONNX_AVAILABLE:
            output_dtype = test_node.get_output_datatype(0)
            assert output_dtype.name in ["INT32", "UINT32"]
            assert output_dtype.bitwidth() == 32
        else:
            # Mock test when QONNX not available
            assert test_node.result_width == 32
    
    # ===========================================
    # Resource Estimation Tests
    # ===========================================
    
    def test_resource_estimation(self, test_node):
        """Test resource estimation accuracy."""
        # LUT estimation
        lut_estimate = test_node.lut_estimation()
        assert isinstance(lut_estimate, int)
        assert 2000 <= lut_estimate <= 4000  # Based on metadata hints
        
        # DSP estimation
        dsp_estimate = test_node.dsp_estimation("xczu9eg")
        assert isinstance(dsp_estimate, int)
        assert dsp_estimate == 8  # One per parallel multiplier
        
        # BRAM estimation (should be 0 for streaming operation)
        bram_estimate = test_node.bram_estimation()
        assert isinstance(bram_estimate, int)
        assert bram_estimate == 0
    
    @pytest.mark.parametrize("fpga_part", [
        "xczu9eg", "xczu7ev", "xczu15eg", "xczu19eg"
    ])
    def test_resource_estimation_across_devices(self, test_node, fpga_part):
        """Test resource estimation across different FPGA devices."""
        dsp_estimate = test_node.dsp_estimation(fpga_part)
        lut_estimate = test_node.lut_estimation()
        
        # DSP count should be consistent across devices
        assert dsp_estimate == 8
        
        # LUT estimate should be reasonable
        assert 1000 <= lut_estimate <= 10000
    
    # ===========================================
    # Performance Tests
    # ===========================================
    
    def test_performance_characteristics(self, test_node):
        """Test performance calculations."""
        # Cycle count
        cycles = test_node.get_exp_cycles()
        assert cycles == 96  # Should match metadata specification
        
        # Performance metrics
        metrics = test_node.get_performance_metrics()
        
        assert metrics['latency_cycles'] == 96
        assert metrics['throughput_ops_per_cycle'] == 1
        assert metrics['initiation_interval'] == 96
        assert metrics['parallelism_factor'] == 8
        
        # Bandwidth calculation
        assert 'memory_bandwidth_gbps' in metrics
        assert metrics['memory_bandwidth_gbps'] > 0
        
        # Efficiency metrics
        assert 'compute_efficiency' in metrics
        assert 'resource_efficiency' in metrics
        assert 0 < metrics['compute_efficiency'] <= 1
        assert 0 < metrics['resource_efficiency'] <= 1
    
    def test_initiation_interval_calculation(self, test_node):
        """Test initiation interval calculations."""
        model = test_node.dataflow_model
        
        # Test various parallelism configurations
        test_configs = [
            {"iPar": 1, "wPar": 1},
            {"iPar": 8, "wPar": 1},
            {"iPar": 8, "wPar": 8}
        ]
        
        for config in test_configs:
            ii_result = model.calculate_initiation_intervals(
                config["iPar"], config["wPar"]
            )
            
            assert 'compute_ii' in ii_result
            assert 'memory_ii' in ii_result
            assert 'overall_ii' in ii_result
            
            assert ii_result['compute_ii'] >= 1
            assert ii_result['memory_ii'] >= 1
            assert ii_result['overall_ii'] >= max(ii_result['compute_ii'], ii_result['memory_ii'])
    
    # ===========================================
    # Numerical Accuracy Tests
    # ===========================================
    
    def test_dot_product_numerical_accuracy(self, test_node):
        """Test numerical accuracy of dot product computation."""
        # Generate test vectors
        np.random.seed(42)  # For reproducible tests
        vector_a = np.random.randint(-128, 127, 768, dtype=np.int8)
        vector_b = np.random.randint(-128, 127, 768, dtype=np.int8)
        
        # Reference calculation (high precision)
        reference_result = np.dot(vector_a.astype(np.int64), vector_b.astype(np.int64))
        
        # Hardware simulation
        hardware_result = test_node.compute_dot_product(vector_a, vector_b)
        
        # Should be exact match for integer arithmetic
        assert hardware_result == reference_result
    
    @pytest.mark.parametrize("vector_size", [256, 512, 768, 1024])
    def test_scalability_across_sizes(self, vector_size, mock_onnx_node):
        """Test operation scalability across different vector sizes."""
        # Create node with custom vector size
        test_node = VectorDotProductHWCustomOp(mock_onnx_node)
        test_node.vector_size = vector_size
        
        # Generate appropriately sized test vectors
        vector_a = np.random.randint(-50, 50, vector_size, dtype=np.int8)
        vector_b = np.random.randint(-50, 50, vector_size, dtype=np.int8)
        
        # Test computation
        result = test_node.compute_dot_product(vector_a, vector_b)
        reference = np.dot(vector_a.astype(np.int64), vector_b.astype(np.int64))
        
        assert result == reference
    
    def test_edge_case_vectors(self, test_node):
        """Test edge cases for vector inputs."""
        # Test cases
        test_cases = [
            # All zeros
            (np.zeros(768, dtype=np.int8), np.zeros(768, dtype=np.int8), 0),
            
            # All maximum positive
            (np.full(768, 127, dtype=np.int8), np.full(768, 127, dtype=np.int8), 768 * 127 * 127),
            
            # All maximum negative
            (np.full(768, -128, dtype=np.int8), np.full(768, -128, dtype=np.int8), 768 * 128 * 128),
            
            # Mixed signs
            (np.full(768, 127, dtype=np.int8), np.full(768, -128, dtype=np.int8), 768 * 127 * (-128)),
        ]
        
        for vector_a, vector_b, expected in test_cases:
            result = test_node.compute_dot_product(vector_a, vector_b)
            assert result == expected, f"Failed for case: {vector_a[0]}, {vector_b[0]}"
    
    # ===========================================
    # Attention Mechanism Integration Tests
    # ===========================================
    
    def test_attention_score_calculation(self, test_node):
        """Test attention mechanism integration."""
        # Create typical attention vectors
        query = np.random.randint(-50, 50, 768, dtype=np.int8)
        key = np.random.randint(-50, 50, 768, dtype=np.int8)
        
        # Calculate attention score
        attention_score = test_node.calculate_attention_scores(query, key)
        
        # Verify scaling is applied correctly
        raw_dot_product = test_node.compute_dot_product(query, key)
        expected_score = raw_dot_product / np.sqrt(768)
        
        assert abs(attention_score - expected_score) < 1e-6
    
    def test_bert_optimization(self, test_node):
        """Test BERT-specific optimization."""
        bert_config = {
            'max_position_embeddings': 512,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'batch_size': 1
        }
        
        optimal_config = test_node.optimize_for_bert(bert_config)
        
        # Verify optimization results
        assert isinstance(optimal_config, dict)
        assert 'input_parallelism' in optimal_config
        assert 'compute_parallelism' in optimal_config
        assert 'output_parallelism' in optimal_config
        
        # Verify reasonable parallelism values
        assert 1 <= optimal_config['input_parallelism'] <= 16
        assert 1 <= optimal_config['compute_parallelism'] <= 64
        assert optimal_config['output_parallelism'] == 1
    
    # ===========================================
    # Validation and Verification Tests
    # ===========================================
    
    def test_node_verification(self, test_node):
        """Test comprehensive node verification."""
        # Should not raise any exceptions for valid configuration
        try:
            test_node.verify_node()
        except Exception as e:
            pytest.fail(f"Node verification failed: {e}")
    
    def test_dimensional_consistency_validation(self, test_node):
        """Test dimensional consistency validation."""
        # This should pass for the default configuration
        test_node._verify_dimensional_consistency()
        
        # Test with inconsistent configuration
        interfaces = test_node.dataflow_model.interfaces
        input_interfaces = [iface for iface in interfaces if iface.interface_type == "INPUT"]
        
        if len(input_interfaces) >= 2:
            # Temporarily modify one interface to create inconsistency
            original_sDim = input_interfaces[1].sDim
            input_interfaces[1].sDim = 16  # Different from first interface (8)
            
            with pytest.raises(AssertionError):
                test_node._verify_dimensional_consistency()
            
            # Restore original value
            input_interfaces[1].sDim = original_sDim
    
    def test_performance_constraint_validation(self, test_node):
        """Test performance constraint validation."""
        # Should pass for valid configuration
        test_node._verify_performance_constraints()
        
        # Test with violated constraint
        original_latency = test_node.latency_cycles
        test_node.latency_cycles = 10  # Unreasonably low
        
        with pytest.raises(AssertionError):
            test_node._verify_performance_constraints()
        
        # Restore original value
        test_node.latency_cycles = original_latency
    
    def test_resource_constraint_validation(self, test_node):
        """Test resource constraint validation."""
        # Should pass for valid configuration
        test_node._verify_resource_constraints()
        
        # Test with violated constraint
        original_constraints = test_node.dataflow_model.optimization_config.get('resource_constraints', {})
        test_node.dataflow_model.optimization_config['resource_constraints'] = {'max_luts': 100}  # Too low
        
        with pytest.raises(AssertionError):
            test_node._verify_resource_constraints()
        
        # Restore original constraints
        test_node.dataflow_model.optimization_config['resource_constraints'] = original_constraints
    
    # ===========================================
    # Integration Tests
    # ===========================================
    
    @pytest.mark.skipif(not FINN_AVAILABLE, reason="FINN not available")
    def test_finn_integration(self, test_node):
        """Test integration with FINN framework."""
        # Test that the node can be used in FINN model
        # This is a placeholder - actual FINN integration would require more setup
        assert hasattr(test_node, 'get_input_datatype')
        assert hasattr(test_node, 'get_output_datatype')
        assert hasattr(test_node, 'execute_node')
    
    def test_dataflow_model_optimization(self, test_node):
        """Test dataflow model optimization capabilities."""
        model = test_node.dataflow_model
        
        # Test parallelism bounds
        bounds = model.get_parallelism_bounds()
        assert isinstance(bounds, dict)
        assert 'input' in bounds
        assert 'compute' in bounds
        assert 'output' in bounds
        
        # Test optimization
        constraints = {
            'max_luts': 10000,
            'max_dsps': 100,
            'target_frequency': 200
        }
        
        optimal_config = model.optimize_parallelism(constraints)
        assert isinstance(optimal_config, dict)
        
        # Verify optimized configuration respects constraints
        for param, value in optimal_config.items():
            assert isinstance(value, int)
            assert value >= 1
    
    # ===========================================
    # Performance Regression Tests
    # ===========================================
    
    def test_performance_regression(self, test_node):
        """Test for performance regressions."""
        # Baseline performance metrics
        baseline_metrics = {
            'latency_cycles': 96,
            'lut_estimate': 2500,
            'dsp_estimate': 8,
            'memory_bandwidth': 5.0  # GB/s
        }
        
        # Current performance
        current_cycles = test_node.get_exp_cycles()
        current_luts = test_node.lut_estimation()
        current_dsps = test_node.dsp_estimation("xczu9eg")
        current_metrics = test_node.get_performance_metrics()
        
        # Check for regressions (allow 10% tolerance)
        assert current_cycles <= baseline_metrics['latency_cycles'] * 1.1
        assert current_luts <= baseline_metrics['lut_estimate'] * 1.1
        assert current_dsps <= baseline_metrics['dsp_estimate'] * 1.1
        
        # Bandwidth should be reasonable
        assert current_metrics['memory_bandwidth_gbps'] <= baseline_metrics['memory_bandwidth'] * 1.2
    
    # ===========================================
    # Stress Tests
    # ===========================================
    
    def test_large_vector_stress(self, mock_onnx_node):
        """Stress test with large vectors."""
        # Test with maximum reasonable vector size
        test_node = VectorDotProductHWCustomOp(mock_onnx_node)
        test_node.vector_size = 4096  # Large vector
        
        # Ensure parallelism divides vector size
        assert test_node.vector_size % test_node.parallelism == 0
        
        # Test computation with large vectors
        vector_a = np.random.randint(-10, 10, test_node.vector_size, dtype=np.int8)
        vector_b = np.random.randint(-10, 10, test_node.vector_size, dtype=np.int8)
        
        result = test_node.compute_dot_product(vector_a, vector_b)
        reference = np.dot(vector_a.astype(np.int64), vector_b.astype(np.int64))
        
        assert result == reference
    
    def test_repeated_operations(self, test_node):
        """Test repeated operations for consistency."""
        # Generate test vectors
        vector_a = np.random.randint(-50, 50, 768, dtype=np.int8)
        vector_b = np.random.randint(-50, 50, 768, dtype=np.int8)
        
        # Compute multiple times
        results = []
        for _ in range(10):
            result = test_node.compute_dot_product(vector_a, vector_b)
            results.append(result)
        
        # All results should be identical
        assert all(r == results[0] for r in results)
        
        # Should match reference
        reference = np.dot(vector_a.astype(np.int64), vector_b.astype(np.int64))
        assert results[0] == reference

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])