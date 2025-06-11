#!/usr/bin/env python3
"""
Unit tests for the unified interface type system
"""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-2')

from brainsmith.dataflow.core.interface_types import InterfaceType

def test_interface_type_properties():
    """Test interface type properties"""
    
    # Test INPUT interface
    assert InterfaceType.INPUT.protocol == "axi_stream"
    assert InterfaceType.INPUT.is_dataflow == True
    assert InterfaceType.INPUT.is_axi_stream == True
    assert InterfaceType.INPUT.is_axi_lite == False
    assert InterfaceType.INPUT.is_configuration == False
    assert InterfaceType.INPUT.direction == "input"
    
    # Test OUTPUT interface
    assert InterfaceType.OUTPUT.protocol == "axi_stream"
    assert InterfaceType.OUTPUT.is_dataflow == True
    assert InterfaceType.OUTPUT.is_axi_stream == True
    assert InterfaceType.OUTPUT.direction == "output"
    
    # Test WEIGHT interface
    assert InterfaceType.WEIGHT.protocol == "axi_stream"
    assert InterfaceType.WEIGHT.is_dataflow == True
    assert InterfaceType.WEIGHT.is_axi_stream == True
    assert InterfaceType.WEIGHT.direction == "input"
    
    # Test CONFIG interface
    assert InterfaceType.CONFIG.protocol == "axi_lite"
    assert InterfaceType.CONFIG.is_dataflow == False
    assert InterfaceType.CONFIG.is_axi_stream == False
    assert InterfaceType.CONFIG.is_axi_lite == True
    assert InterfaceType.CONFIG.is_configuration == True
    assert InterfaceType.CONFIG.direction == "bidirectional"
    
    # Test CONTROL interface
    assert InterfaceType.CONTROL.protocol == "global_control"
    assert InterfaceType.CONTROL.is_dataflow == False
    assert InterfaceType.CONTROL.is_configuration == True
    assert InterfaceType.CONTROL.direction == "input"
    
    # Test UNKNOWN interface
    assert InterfaceType.UNKNOWN.protocol == "unknown"
    assert InterfaceType.UNKNOWN.is_dataflow == False
    
    print("✅ All interface type property tests passed!")

def test_string_representations():
    """Test string representations"""
    assert str(InterfaceType.INPUT) == "InterfaceType.INPUT"
    assert "INPUT" in repr(InterfaceType.INPUT)
    assert "axi_stream" in repr(InterfaceType.INPUT)
    print("✅ String representation tests passed!")

def test_enum_values():
    """Test enum values"""
    assert InterfaceType.INPUT.value == "input"
    assert InterfaceType.OUTPUT.value == "output"
    assert InterfaceType.WEIGHT.value == "weight"
    assert InterfaceType.CONFIG.value == "config"
    assert InterfaceType.CONTROL.value == "control"
    assert InterfaceType.UNKNOWN.value == "unknown"
    print("✅ Enum value tests passed!")

if __name__ == "__main__":
    test_interface_type_properties()
    test_string_representations()
    test_enum_values()
    print("🎉 All tests passed! Interface type system working correctly.")