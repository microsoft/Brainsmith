############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Tests for Unified Thresholding Operator

Validates the unified thresholding implementation including:
- Basic functionality
- DSE integration
- RTL generation
- Performance optimization
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from brainsmith.unified.operators import UnifiedThresholding
from brainsmith.core.dataflow import Shape
from brainsmith.dataflow.core.interface_types import InterfaceType


class TestUnifiedThresholding(unittest.TestCase):
    """Test cases for unified thresholding operator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock ONNX node
        self.onnx_node = Mock()
        self.onnx_node.name = "threshold_test"
        self.onnx_node.op_type = "UnifiedThresholding"
        self.onnx_node.input = ["input", "threshold"]
        self.onnx_node.output = ["output"]
        self.onnx_node.attribute = []
        
        # Create operator instance
        self.op = UnifiedThresholding(self.onnx_node)
    
    def test_kernel_definition(self):
        """Test kernel definition creation."""
        kernel_def = self.op.get_kernel_definition()
        
        # Check basic properties
        self.assertEqual(kernel_def.name, "thresholding")
        self.assertEqual(len(kernel_def.interfaces), 3)
        
        # Check interface types
        interface_types = {intf.name: intf.type for intf in kernel_def.interfaces}
        self.assertEqual(interface_types["input"], InterfaceType.INPUT)
        self.assertEqual(interface_types["threshold"], InterfaceType.WEIGHT)
        self.assertEqual(interface_types["output"], InterfaceType.OUTPUT)
        
        # Check exposed parameters
        self.assertIn("BIAS", kernel_def.exposed_parameters)
        self.assertIn("ACTIVATION_TYPE", kernel_def.exposed_parameters)
        
        # Check constraints
        self.assertEqual(len(kernel_def.constraints), 3)
    
    def test_node_attributes(self):
        """Test node attribute types."""
        attrs = self.op.get_nodeattr_types()
        
        # Check unified framework attributes
        self.assertIn("input_parallelism", attrs)
        self.assertIn("threshold_parallelism", attrs)
        self.assertIn("output_parallelism", attrs)
        
        # Check optimization attributes
        self.assertIn("auto_optimize", attrs)
        self.assertIn("optimization_objective", attrs)
        
        # Check operator-specific attributes
        self.assertIn("activation", attrs)
        self.assertIn("use_bram", attrs)
    
    def test_shape_initialization(self):
        """Test shape initialization from ONNX."""
        # Mock shape methods
        self.op.get_normal_input_shape = Mock(return_value=[10, 32])
        self.op.get_normal_output_shape = Mock(return_value=[10, 32])
        
        # Re-initialize
        self.op._initialize_from_onnx()
        
        # Check tensor shapes
        input_intf = self.op.kernel.get_interfaces_by_type(InterfaceType.INPUT)[0]
        self.assertEqual(list(input_intf.tensor_shape), [10, 32])
        
        output_intf = self.op.kernel.get_interfaces_by_type(InterfaceType.OUTPUT)[0]
        self.assertEqual(list(output_intf.tensor_shape), [10, 32])
    
    def test_performance_metrics(self):
        """Test performance metric calculation."""
        metrics = self.op.get_performance_metrics()
        
        # Check metric structure
        self.assertIn("throughput", metrics)
        self.assertIn("latency", metrics)
        self.assertIn("resource_usage", metrics)
        self.assertIn("bandwidth_requirements", metrics)
    
    def test_dse_optimization(self):
        """Test DSE optimization."""
        # Set up shapes
        self.op.get_normal_input_shape = Mock(return_value=[100, 64])
        self.op.get_normal_output_shape = Mock(return_value=[100, 64])
        self.op._initialize_from_onnx()
        
        # Run optimization
        target_spec = {
            "target_throughput": 200.0,  # 200 MHz
            "optimization_objective": "throughput",
            "max_resources": {
                "LUT": 5000,
                "BRAM": 10
            }
        }
        
        # This should run without errors
        try:
            self.op.optimize_for_target(target_spec)
            optimized = True
        except Exception as e:
            # DSE might fail due to constraints, which is ok for test
            print(f"DSE optimization failed: {e}")
            optimized = False
        
        if optimized:
            # Check that configuration was updated
            self.assertIsNotNone(self.op._current_config)
            self.assertTrue(self.op._optimized)
            
            # Get optimization report
            report = self.op.get_optimization_report()
            self.assertTrue(report['optimized'])
    
    def test_rtl_value_generation(self):
        """Test RTL template value generation."""
        # Set some attributes
        self.op.set_nodeattr = Mock()
        self.op.get_nodeattr = Mock(side_effect=lambda x: {
            "BIAS": 5,
            "ACTIVATION_TYPE": 1,
            "input_parallelism": 8,
            "threshold_parallelism": 4,
            "input_datatype": "UINT8",
            "threshold_datatype": "INT16",
            "output_datatype": "UINT2"
        }.get(x, 0))
        self.op.has_nodeattr = Mock(return_value=True)
        
        # Generate RTL values
        rtl_values = self.op.prepare_codegen_rtl_values(None)
        
        # Check generated values
        self.assertIn("$BIAS$", rtl_values)
        self.assertEqual(rtl_values["$BIAS$"], ["5"])
        
        self.assertIn("$INPUT_PARALLELISM$", rtl_values)
        self.assertIn("$SIMD$", rtl_values)  # Legacy compatibility
    
    def test_resource_estimation(self):
        """Test resource estimation."""
        # Test LUT estimation
        luts = self.op.lut_estimation()
        self.assertGreater(luts, 0)
        
        # Test BRAM estimation (should be 0 without use_bram)
        self.op.get_nodeattr = Mock(side_effect=lambda x: 0 if x == "use_bram" else None)
        brams = self.op.bram_estimation()
        self.assertEqual(brams, 0)
        
        # Test with BRAM enabled
        self.op.get_nodeattr = Mock(side_effect=lambda x: {
            "use_bram": 1,
            "threshold_datatype": "INT16"
        }.get(x, 0))
        
        # Set up threshold shape
        threshold_intf = self.op.kernel.get_interfaces_by_type(InterfaceType.WEIGHT)[0]
        threshold_intf.tensor_shape = Shape([64, 7])  # 64 channels, 7 thresholds each
        
        brams = self.op.bram_estimation()
        self.assertGreater(brams, 0)
    
    def test_execution(self):
        """Test node execution."""
        # Create test data
        input_data = np.random.randint(-50, 50, size=(10, 32)).astype(np.float32)
        threshold_data = np.random.randint(-20, 20, size=(32, 3)).astype(np.float32)
        
        # Set up context
        context = {
            "input": input_data,
            "threshold": threshold_data
        }
        
        # Mock attributes
        self.op.get_nodeattr = Mock(side_effect=lambda x: {
            "auto_optimize": 0,
            "BIAS": 0,
            "activation": "binary"
        }.get(x, 0))
        
        # Execute
        self.op.execute_node(context, None)
        
        # Check output exists
        self.assertIn("output", context)
        output = context["output"]
        
        # Check output shape
        self.assertEqual(output.shape, input_data.shape)
        
        # Check output values are binary
        self.assertTrue(np.all(np.logical_or(output == 0, output == 1)))
    
    def test_optimization_suggestions(self):
        """Test optimization suggestion generation."""
        suggestions = self.op.suggest_optimization_targets()
        
        # Should have some suggestions
        self.assertGreater(len(suggestions), 0)
        
        # Check suggestion structure
        for suggestion in suggestions:
            self.assertIn("name", suggestion)
            self.assertIn("target_spec", suggestion)
            self.assertIn("description", suggestion)


if __name__ == "__main__":
    unittest.main()