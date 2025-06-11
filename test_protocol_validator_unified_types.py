#!/usr/bin/env python3
"""
Test protocol validator with unified interface types
"""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-2')

from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, PortGroup, ValidationResult
from brainsmith.tools.hw_kernel_gen.rtl_parser.protocol_validator import ProtocolValidator

def test_dataflow_type_determination():
    """Test _determine_dataflow_type method"""
    
    validator = ProtocolValidator()
    
    # Test INPUT interface
    assert validator._determine_dataflow_type("in0", Direction.INPUT) == InterfaceType.INPUT
    assert validator._determine_dataflow_type("input_data", Direction.INPUT) == InterfaceType.INPUT
    
    # Test OUTPUT interface
    assert validator._determine_dataflow_type("out0", Direction.OUTPUT) == InterfaceType.OUTPUT
    assert validator._determine_dataflow_type("output_data", Direction.OUTPUT) == InterfaceType.OUTPUT
    
    # Test WEIGHT interface (name-based detection)
    assert validator._determine_dataflow_type("weights", Direction.INPUT) == InterfaceType.WEIGHT
    assert validator._determine_dataflow_type("weight_data", Direction.INPUT) == InterfaceType.WEIGHT
    assert validator._determine_dataflow_type("param_buffer", Direction.INPUT) == InterfaceType.WEIGHT
    assert validator._determine_dataflow_type("coeff_stream", Direction.INPUT) == InterfaceType.WEIGHT
    
    print("✅ Dataflow type determination works correctly!")

def test_axi_stream_validation_with_type_determination():
    """Test AXI-Stream validation with dataflow type determination"""
    
    validator = ProtocolValidator()
    
    # Create AXI-Stream input interface
    input_ports = {
        "TDATA": Port("in0_V_data_V", Direction.INPUT, "[7:0]"),
        "TVALID": Port("in0_V_valid", Direction.INPUT, "1"),
        "TREADY": Port("in0_V_ready", Direction.OUTPUT, "1")
    }
    
    input_group = PortGroup(
        interface_type=InterfaceType.INPUT,  # Preliminary type from scanner
        name="in0",
        ports=input_ports
    )
    
    # Validate
    result = validator.validate_axi_stream(input_group)
    
    assert result.valid == True
    assert input_group.interface_type == InterfaceType.INPUT
    assert input_group.metadata['direction'] == Direction.INPUT
    
    print("✅ AXI-Stream input validation works correctly!")

def test_axi_stream_output_validation():
    """Test AXI-Stream output validation"""
    
    validator = ProtocolValidator()
    
    # Create AXI-Stream output interface (inverted directions)
    output_ports = {
        "TDATA": Port("out0_V_data_V", Direction.OUTPUT, "[7:0]"),
        "TVALID": Port("out0_V_valid", Direction.OUTPUT, "1"), 
        "TREADY": Port("out0_V_ready", Direction.INPUT, "1")
    }
    
    output_group = PortGroup(
        interface_type=InterfaceType.INPUT,  # Preliminary type from scanner
        name="out0",
        ports=output_ports
    )
    
    # Validate
    result = validator.validate_axi_stream(output_group)
    
    assert result.valid == True
    assert output_group.interface_type == InterfaceType.OUTPUT  # Should be changed to OUTPUT
    assert output_group.metadata['direction'] == Direction.OUTPUT
    
    print("✅ AXI-Stream output validation works correctly!")

def test_weight_interface_validation():
    """Test weight interface validation"""
    
    validator = ProtocolValidator()
    
    # Create AXI-Stream weight interface
    weight_ports = {
        "TDATA": Port("weights_V_data_V", Direction.INPUT, "[7:0]"),
        "TVALID": Port("weights_V_valid", Direction.INPUT, "1"),
        "TREADY": Port("weights_V_ready", Direction.OUTPUT, "1")
    }
    
    weight_group = PortGroup(
        interface_type=InterfaceType.INPUT,  # Preliminary type from scanner
        name="weights",
        ports=weight_ports
    )
    
    # Validate
    result = validator.validate_axi_stream(weight_group)
    
    assert result.valid == True
    assert weight_group.interface_type == InterfaceType.WEIGHT  # Should be changed to WEIGHT
    assert weight_group.metadata['direction'] == Direction.INPUT
    
    print("✅ Weight interface validation works correctly!")

def test_control_interface_validation():
    """Test global control interface validation"""
    
    validator = ProtocolValidator()
    
    # Create global control interface
    control_ports = {
        "clk": Port("clk", Direction.INPUT, "1"),
        "rst_n": Port("rst_n", Direction.INPUT, "1")
    }
    
    control_group = PortGroup(
        interface_type=InterfaceType.CONTROL,  # From scanner
        name="global",
        ports=control_ports
    )
    
    # Validate
    result = validator.validate_global_control(control_group)
    
    assert result.valid == True
    assert control_group.interface_type == InterfaceType.CONTROL
    
    print("✅ Control interface validation works correctly!")

def test_config_interface_validation():
    """Test AXI-Lite config interface validation"""
    
    validator = ProtocolValidator()
    
    # Create minimal AXI-Lite interface
    config_ports = {
        "AWADDR": Port("s_axi_control_AWADDR", Direction.INPUT, "[31:0]"),
        "AWVALID": Port("s_axi_control_AWVALID", Direction.INPUT, "1"),
        "AWREADY": Port("s_axi_control_AWREADY", Direction.OUTPUT, "1"),
        "WDATA": Port("s_axi_control_WDATA", Direction.INPUT, "[31:0]"),
        "WSTRB": Port("s_axi_control_WSTRB", Direction.INPUT, "[3:0]"),
        "WVALID": Port("s_axi_control_WVALID", Direction.INPUT, "1"),
        "WREADY": Port("s_axi_control_WREADY", Direction.OUTPUT, "1"),
        "BRESP": Port("s_axi_control_BRESP", Direction.OUTPUT, "[1:0]"),
        "BVALID": Port("s_axi_control_BVALID", Direction.OUTPUT, "1"),
        "BREADY": Port("s_axi_control_BREADY", Direction.INPUT, "1")
    }
    
    config_group = PortGroup(
        interface_type=InterfaceType.CONFIG,  # From scanner
        name="s_axi_control",
        ports=config_ports
    )
    
    # Validate
    result = validator.validate_axi_lite(config_group)
    
    assert result.valid == True
    assert config_group.interface_type == InterfaceType.CONFIG
    
    print("✅ Config interface validation works correctly!")

if __name__ == "__main__":
    test_dataflow_type_determination()
    test_axi_stream_validation_with_type_determination()
    test_axi_stream_output_validation()
    test_weight_interface_validation()
    test_control_interface_validation()
    test_config_interface_validation()
    print("🎉 All protocol validator unified type tests passed!")