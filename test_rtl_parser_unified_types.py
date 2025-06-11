#!/usr/bin/env python3
"""
Test RTL parser data structures with unified interface types
"""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-2')

from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Interface, PortGroup, Port, Direction, ValidationResult, HWKernel

def test_interface_with_unified_types():
    """Test Interface class with unified interface types"""
    
    # Test creating interface with unified type
    validation_result = ValidationResult(valid=True, message="Valid interface")
    
    # Create a port
    port = Port(name="data", direction=Direction.INPUT, width="[7:0]")
    
    # Create interface with unified type
    interface = Interface(
        name="input0",
        type=InterfaceType.INPUT,  # Using unified type directly
        ports={"data": port},
        validation_result=validation_result,
        metadata={"direction": Direction.INPUT}
    )
    
    # Test interface properties
    assert interface.type == InterfaceType.INPUT
    assert interface.type.protocol == "axi_stream"
    assert interface.type.is_dataflow == True
    assert interface.type.is_axi_stream == True
    assert interface.type.direction == "input"
    
    print("✅ Interface with unified types works correctly!")

def test_port_group_with_unified_types():
    """Test PortGroup class with unified interface types"""
    
    # Create port group with unified type
    port_group = PortGroup(
        interface_type=InterfaceType.CONFIG,
        name="config",
        metadata={"protocol": "axi_lite"}
    )
    
    # Test properties
    assert port_group.interface_type == InterfaceType.CONFIG
    assert port_group.interface_type.protocol == "axi_lite"
    assert port_group.interface_type.is_configuration == True
    
    print("✅ PortGroup with unified types works correctly!")

def test_hw_kernel_dataflow_interfaces():
    """Test HWKernel dataflow_interfaces property with unified types"""
    
    # Create interfaces with different types
    validation_result = ValidationResult(valid=True)
    
    input_interface = Interface(
        name="input0",
        type=InterfaceType.INPUT,
        ports={},
        validation_result=validation_result
    )
    
    output_interface = Interface(
        name="output0", 
        type=InterfaceType.OUTPUT,
        ports={},
        validation_result=validation_result
    )
    
    config_interface = Interface(
        name="config",
        type=InterfaceType.CONFIG,
        ports={},
        validation_result=validation_result
    )
    
    # Create HWKernel
    hw_kernel = HWKernel(
        name="test_kernel",
        interfaces={
            "input0": input_interface,
            "output0": output_interface,
            "config": config_interface
        }
    )
    
    # Test dataflow_interfaces property
    dataflow_interfaces = hw_kernel.dataflow_interfaces
    assert len(dataflow_interfaces) == 2  # input and output are dataflow
    assert input_interface in dataflow_interfaces
    assert output_interface in dataflow_interfaces
    assert config_interface not in dataflow_interfaces
    
    print("✅ HWKernel dataflow_interfaces property works correctly!")

if __name__ == "__main__":
    test_interface_with_unified_types()
    test_port_group_with_unified_types()
    test_hw_kernel_dataflow_interfaces()
    print("🎉 All RTL parser unified type tests passed!")