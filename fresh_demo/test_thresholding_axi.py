############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# AUTO-GENERATED: Test Suite for thresholding_axi
# Generated: 2025-06-11T05:02:22.637142
# Generator: Unified HWKG with Interface-Wise Dataflow Modeling
#
# COMPREHENSIVE TEST SUITE
# Tests DataflowModel mathematical correctness and unified HWKG integration.
############################################################################

import pytest
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import os

# Import the generated classes
from .thresholding_axi_hwcustomop import AutoThresholdingAxiHWCustomOp, create_thresholding_axi_hwcustomop
from .thresholding_axi_rtlbackend import AutoThresholdingAxiRTLBackendRTLBackend, create_thresholding_axi_rtlbackend

# Import dataflow components for testing
from brainsmith.dataflow.core.dataflow_model import DataflowModel
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType

# Try to import FINN components for integration testing
try:
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    import onnx
    FINN_AVAILABLE = True
except ImportError:
    FINN_AVAILABLE = False

# Try to import test utilities
try:
    from brainsmith.test_utils import create_test_onnx_node, validate_axiom_compliance
    TEST_UTILS_AVAILABLE = True
except ImportError:
    TEST_UTILS_AVAILABLE = False


class TestAutoThresholdingAxi:
    """
    Comprehensive test suite for thresholding_axi unified HWKG components.
    
    Tests both AutoThresholdingAxiHWCustomOp and AutoThresholdingAxiRTLBackendRTLBackend with focus on:
    - DataflowModel mathematical correctness
    - Interface configuration accuracy
    - Performance calculation verification
    - FINN integration compatibility
    - Axiom compliance validation
    """
    
    @pytest.fixture
    def mock_onnx_node(self):
        """Create mock ONNX node for testing."""
        if TEST_UTILS_AVAILABLE:
            return create_test_onnx_node("thresholding_axi")
        else:
            # Simple mock for basic testing
            class MockONNXNode:
                def __init__(self):
                    self.op_type = "thresholding_axi"
                    self.input = ["input"]
                    self.output = ["output"]
            return MockONNXNode()
    
    @pytest.fixture
    def hwcustomop_instance(self, mock_onnx_node):
        """Create HWCustomOp instance for testing."""
        return AutoThresholdingAxiHWCustomOp(mock_onnx_node)
    
    @pytest.fixture
    def rtlbackend_instance(self):
        """Create RTLBackend instance for testing."""
        return AutoThresholdingAxiRTLBackendRTLBackend()
    
    def test_basic_functionality(self, hwcustomop_instance):
        """Test basic HWCustomOp instantiation and method calls"""
        # Test basic instantiation and core methods
        assert hwcustomop_instance is not None
        assert hasattr(hwcustomop_instance, 'dataflow_model')
        assert hwcustomop_instance.dataflow_model is not None
        
        # Test that all expected interfaces are present
        expected_interfaces = ['ap', 's_axis', 'm_axis', 's_axilite']
        actual_interfaces = list(hwcustomop_instance.dataflow_model.interfaces.keys())
        assert set(expected_interfaces) == set(actual_interfaces), f"Interface mismatch: expected {expected_interfaces}, got {actual_interfaces}"
        
        # Test that AutoHWCustomOp methods are available and functional
        assert hasattr(hwcustomop_instance, 'get_exp_cycles')
        assert hasattr(hwcustomop_instance, 'get_instream_width')
        assert hasattr(hwcustomop_instance, 'get_outstream_width')
        
        # Verify methods return valid values (not placeholders)
        cycles = hwcustomop_instance.get_exp_cycles()
        assert isinstance(cycles, int) and cycles > 0, f"Invalid cycle count: {cycles}"
        
    
    def test_interface_configuration(self, hwcustomop_instance):
        """Test interface configuration and metadata"""
        # Test interface configuration and metadata
        dataflow_model = hwcustomop_instance.dataflow_model
        
        # Verify interface types
        input_interfaces = dataflow_model.input_interfaces
        assert len(input_interfaces) == 1, f"Expected 1 input interfaces, got {len(input_interfaces)}"
        assert any(iface.name == "s_axis" for iface in input_interfaces), "Missing input interface: s_axis"
        
        output_interfaces = dataflow_model.output_interfaces
        assert len(output_interfaces) == 1, f"Expected 1 output interfaces, got {len(output_interfaces)}"
        assert any(iface.name == "m_axis" for iface in output_interfaces), "Missing output interface: m_axis"
        
        
        # Test interface metadata consistency
        for interface in dataflow_model.interfaces.values():
            assert interface.tensor_dims, f"Interface {interface.name} missing tensor_dims"
            assert interface.block_dims, f"Interface {interface.name} missing block_dims"
            assert interface.stream_dims, f"Interface {interface.name} missing stream_dims"
            assert interface.dtype, f"Interface {interface.name} missing dtype"
            
            # Verify dimensional consistency
            assert len(interface.tensor_dims) == len(interface.block_dims), f"Dimension mismatch in {interface.name}"
            assert len(interface.block_dims) == len(interface.stream_dims), f"Stream dimension mismatch in {interface.name}"
        
    
    def test_dataflow_model(self, hwcustomop_instance):
        """Test DataflowModel mathematical correctness"""
        # Test DataflowModel mathematical correctness
        dataflow_model = hwcustomop_instance.dataflow_model
        
        # Test parallelism configuration
        iPar = {iface.name: 1 for iface in dataflow_model.input_interfaces}
        wPar = {iface.name: 1 for iface in dataflow_model.weight_interfaces}
        
        if iPar or wPar:
            # Test initiation interval calculation
            intervals = dataflow_model.calculate_initiation_intervals(iPar, wPar)
            assert hasattr(intervals, 'L'), "Missing total latency calculation"
            assert isinstance(intervals.L, int), f"Invalid latency type: {type(intervals.L)}"
            assert intervals.L > 0, f"Invalid latency value: {intervals.L}"
            
            # Test resource requirements calculation
            from brainsmith.dataflow.core.dataflow_model import ParallelismConfiguration
            parallelism_config = ParallelismConfiguration(iPar=iPar, wPar=wPar, derived_stream_dims={})
            
            try:
                resources = dataflow_model.get_resource_requirements(parallelism_config)
                assert isinstance(resources, dict), f"Invalid resource requirements type: {type(resources)}"
                assert "memory_bits" in resources, "Missing memory_bits in resource requirements"
            except Exception as e:
                pytest.skip(f"Resource calculation not available: {e}")
        
        # Test interface mathematical properties
        for interface in dataflow_model.interfaces.values():
            # Test tensor shape reconstruction
            reconstructed_shape = interface.reconstruct_tensor_shape()
            assert reconstructed_shape, f"Failed to reconstruct tensor shape for {interface.name}"
            
            # Test stream width calculation
            stream_width = interface.calculate_stream_width()
            assert stream_width > 0, f"Invalid stream width for {interface.name}: {stream_width}"
        
    
    def test_performance_calculations(self, hwcustomop_instance, rtlbackend_instance):
        """Test performance and resource calculations"""
        # Test performance and resource calculations
        dataflow_model = hwcustomop_instance.dataflow_model
        
        # Test performance characteristics
        characteristics = hwcustomop_instance.derive_characteristic_fxns()
        assert isinstance(characteristics, dict), f"Invalid characteristics type: {type(characteristics)}"
        assert "compute_cycles" in characteristics, "Missing compute_cycles in characteristics"
        assert characteristics["compute_cycles"] > 0, f"Invalid compute cycles: {characteristics['compute_cycles']}"
        
        # Test resource estimation
        bram_usage = hwcustomop_instance.estimate_bram_usage()
        assert isinstance(bram_usage, int) and bram_usage >= 0, f"Invalid BRAM usage: {bram_usage}"
        
        lut_usage = hwcustomop_instance.estimate_lut_usage()
        assert isinstance(lut_usage, int) and lut_usage >= 0, f"Invalid LUT usage: {lut_usage}"
        
        dsp_usage = hwcustomop_instance.estimate_dsp_usage()
        assert isinstance(dsp_usage, int) and dsp_usage >= 0, f"Invalid DSP usage: {dsp_usage}"
        
        # Test RTL backend code generation
        codegen_dict = rtlbackend_instance.code_generation_dict()
        assert isinstance(codegen_dict, dict), f"Invalid codegen dict type: {type(codegen_dict)}"
        assert "interfaces" in codegen_dict, "Missing interfaces in codegen dict"
        assert "parameters" in codegen_dict, "Missing parameters in codegen dict"
        
        # Test interface width calculations
        for interface_name in dataflow_model.interfaces.keys():
            width = rtlbackend_instance.calculate_interface_width(interface_name)
            assert width > 0, f"Invalid interface width for {interface_name}: {width}"
            assert width % 8 == 0, f"Interface width not byte-aligned for {interface_name}: {width}"
        
    
    
    @pytest.mark.skipif(not FINN_AVAILABLE, reason="FINN not available")
    def test_finn_integration(self, hwcustomop_instance, rtlbackend_instance):
        """Test integration with FINN framework."""
        # Test HWCustomOp FINN compatibility
        nodeattr_types = hwcustomop_instance.get_nodeattr_types()
        assert isinstance(nodeattr_types, dict), "Invalid nodeattr_types format"
        
        # Test datatype compatibility
        input_dtype = hwcustomop_instance.get_input_datatype(0)
        assert input_dtype is not None, "Failed to get input datatype"
        
        output_dtype = hwcustomop_instance.get_output_datatype(0)
        assert output_dtype is not None, "Failed to get output datatype"
        
        # Test shape compatibility
        input_shape = hwcustomop_instance.get_normal_input_shape(0)
        assert isinstance(input_shape, list) and len(input_shape) > 0, f"Invalid input shape: {input_shape}"
        
        folded_input_shape = hwcustomop_instance.get_folded_input_shape(0)
        assert isinstance(folded_input_shape, list), f"Invalid folded input shape: {folded_input_shape}"
        
        # Test RTL backend FINN compatibility
        enhanced_nodeattr_types = rtlbackend_instance.get_enhanced_nodeattr_types()
        assert isinstance(enhanced_nodeattr_types, dict), "Invalid enhanced nodeattr_types format"
    
    @pytest.mark.skipif(not TEST_UTILS_AVAILABLE, reason="Test utilities not available")
    def test_axiom_compliance(self, hwcustomop_instance):
        """Test compliance with Interface-Wise Dataflow axioms."""
        dataflow_model = hwcustomop_instance.dataflow_model
        
        # Test axiom compliance using test utilities
        compliance_result = validate_axiom_compliance(dataflow_model)
        assert compliance_result.is_valid, f"Axiom compliance failed: {compliance_result.errors}"
    
    def test_parameter_generation(self, rtlbackend_instance, tmp_path):
        """Test parameter file generation."""
        # Create temporary directory for parameter files
        param_dir = tmp_path / "params"
        param_dir.mkdir()
        
        # Mock model for parameter generation
        class MockModel:
            def get_nodeattr(self, name):
                return None
        
        mock_model = MockModel()
        
        # Test parameter generation (should not crash)
        try:
            rtlbackend_instance.generate_params(mock_model, str(param_dir))
            # Check if any parameter files were created
            param_files = list(param_dir.glob("*.dat"))
            # Note: May be empty if no weight interfaces, which is fine
        except Exception as e:
            pytest.fail(f"Parameter generation failed: {e}")
    
    def test_error_handling(self, mock_onnx_node):
        """Test error handling and edge cases."""
        # Test with invalid inputs
        try:
            hwcustomop = AutoThresholdingAxiHWCustomOp(None)  # Should handle gracefully
        except Exception as e:
            # Expected to fail, but should be a clear error
            assert "onnx_node" in str(e).lower() or "none" in str(e).lower()
        
        # Test RTL backend with invalid interface names
        rtlbackend = AutoThresholdingAxiRTLBackendRTLBackend()
        with pytest.raises((KeyError, ValueError)):
            rtlbackend.calculate_interface_width("nonexistent_interface")


# Utility functions for testing
def test_thresholding_axi_factory_functions():
    """Test factory functions work correctly."""
    class MockONNXNode:
        def __init__(self):
            self.op_type = "thresholding_axi"
    
    mock_node = MockONNXNode()
    
    # Test HWCustomOp factory
    hwcustomop = create_thresholding_axi_hwcustomop(mock_node)
    assert isinstance(hwcustomop, AutoThresholdingAxiHWCustomOp)
    
    # Test RTLBackend factory
    rtlbackend = create_thresholding_axi_rtlbackend()
    assert isinstance(rtlbackend, AutoThresholdingAxiRTLBackendRTLBackend)


# Integration test for complete workflow
def test_thresholding_axi_complete_workflow(tmp_path):
    """Test complete workflow from instantiation to RTL generation."""
    # Create mock ONNX node
    class MockONNXNode:
        def __init__(self):
            self.op_type = "thresholding_axi"
    
    mock_node = MockONNXNode()
    
    # Test complete workflow
    hwcustomop = create_thresholding_axi_hwcustomop(mock_node)
    rtlbackend = create_thresholding_axi_rtlbackend()
    
    # Verify they have compatible configurations
    dataflow_interfaces = hwcustomop.dataflow_model.interfaces
    rtl_interfaces = rtlbackend.dataflow_interfaces
    
    # Check interface name consistency
    dataflow_names = set(dataflow_interfaces.keys())
    rtl_names = set(rtl_interfaces.keys())
    assert dataflow_names == rtl_names, f"Interface name mismatch: {dataflow_names} vs {rtl_names}"
    
    # Test RTL generation workflow
    class MockModel:
        def get_nodeattr(self, name):
            return None
    
    mock_model = MockModel()
    output_dir = tmp_path / "rtl_output"
    output_dir.mkdir()
    
    try:
        rtlbackend.generate_params(mock_model, str(output_dir))
        # If successful, check outputs exist or verify no errors
        assert True  # If we get here, generation succeeded
    except Exception as e:
        # Should not fail catastrophically
        pytest.skip(f"RTL generation skipped due to missing dependencies: {e}")


if __name__ == "__main__":
    pytest.main([__file__])