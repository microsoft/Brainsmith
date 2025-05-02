"""Test RTL parser with thresholding_axi module.

This test demonstrates the RTL parser's capabilities by parsing a complex
SystemVerilog module from the FINN library and verifying the extracted
interface information.
"""

import os
import pytest
from typing import Dict, List

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    Direction,
    Parameter,
    Port,
    HWKernel
)

# Path to test file relative to project root
TEST_FILE = "examples/thresholding/thresholding_axi.sv"

def get_project_root() -> str:
    """Get absolute path to project root."""
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../.."
    ))

def test_parse_thresholding_axi():
    """Test parsing of thresholding_axi.sv module."""
    # Initialize parser
    parser = RTLParser(debug=True)
    
    # Get full path to test file
    file_path = os.path.join(get_project_root(), TEST_FILE)
    assert os.path.exists(file_path), f"Test file not found: {file_path}"
    
    # Parse module
    kernel = parser.parse_file(file_path)
    assert kernel is not None
    assert kernel.name == "thresholding_axi"
    
    # Verify parameters
    expected_params: Dict[str, Dict] = {
        "N": {
            "type": "int unsigned",
            "description": "output precision"
        },
        "WI": {
            "type": "int unsigned",
            "description": "input precision"
        },
        "C": {
            "type": "int unsigned",
            "default": "1",
            "description": "Channels"
        },
        "SIGNED": {
            "type": "bit",
            "default": "1",
            "description": "signed inputs"
        },
        "THRESHOLDS_PATH": {
            "type": "parameter",
            "default": '""'
        }
    }
    
    # Verify each parameter
    for param in kernel.parameters:
        assert param.name in expected_params, f"Unexpected parameter: {param.name}"
        expected = expected_params[param.name]
        
        assert param.param_type == expected["type"], \
            f"Parameter {param.name} type mismatch: {param.param_type} != {expected['type']}"
        
        if "default" in expected:
            assert param.default_value == expected["default"], \
                f"Parameter {param.name} default value mismatch"
        
        if "description" in expected:
            assert param.description == expected["description"], \
                f"Parameter {param.name} description mismatch"
    
    # Verify ports are grouped correctly
    port_groups: Dict[str, List[str]] = {
        "Global Control": ["ap_clk", "ap_rst_n"],
        "AXI Lite": [
            "s_axilite_AWVALID", "s_axilite_AWREADY", "s_axilite_AWADDR",
            "s_axilite_WVALID", "s_axilite_WREADY", "s_axilite_WDATA",
            "s_axilite_WSTRB", "s_axilite_BVALID", "s_axilite_BREADY",
            "s_axilite_BRESP", "s_axilite_ARVALID", "s_axilite_ARREADY",
            "s_axilite_ARADDR", "s_axilite_RVALID", "s_axilite_RREADY",
            "s_axilite_RDATA", "s_axilite_RRESP"
        ],
        "AXI Stream": [
            "s_axis_tready", "s_axis_tvalid", "s_axis_tdata",
            "m_axis_tready", "m_axis_tvalid", "m_axis_tdata"
        ]
    }
    
    # Helper to check port direction
    def is_input_port(name: str) -> bool:
        """Determine if port should be input based on name."""
        if name.startswith(("s_axis_t", "m_axis_t")):
            return name.startswith("s_axis_t") and not name.endswith("ready")
        if name.startswith("s_axilite_"):
            return (
                name.endswith(("VALID", "ADDR", "DATA", "STRB", "READY"))
                and not name.endswith(("RDATA", "RRESP"))
            )
        return name in ["ap_clk", "ap_rst_n"]
    
    # Verify each port
    for port in kernel.ports:
        # Find which group this port belongs to
        found_group = None
        for group, ports in port_groups.items():
            if port.name in ports:
                found_group = group
                break
        assert found_group is not None, f"Port {port.name} not found in any group"
        
        # Check direction
        expected_dir = Direction.INPUT if is_input_port(port.name) else Direction.OUTPUT
        assert port.direction == expected_dir, \
            f"Port {port.name} direction mismatch: {port.direction} != {expected_dir}"
        
        # Check width expressions
        if port.name in ["ap_clk", "ap_rst_n"]:
            assert port.width == "1"
        elif port.name == "s_axis_tdata":
            assert port.width == "((PE*WI+7)/8)*8-1:0"
        elif port.name == "m_axis_tdata":
            assert port.width == "((PE*O_BITS+7)/8)*8-1:0"
        elif port.name.endswith("ADDR"):
            assert port.width == "ADDR_BITS-1:0"
    
    print("\nTest Results:")
    print(f"Module Name: {kernel.name}")
    print(f"Parameter Count: {len(kernel.parameters)}")
    print(f"Port Count: {len(kernel.ports)}")
    print("\nParameters:")
    for param in kernel.parameters:
        print(f"  {param.name}: {param.param_type}" + 
              (f" = {param.default_value}" if param.default_value else ""))
    print("\nPorts by Group:")
    for group, ports in port_groups.items():
        print(f"\n{group}:")
        for port_name in ports:
            port = next(p for p in kernel.ports if p.name == port_name)
            print(f"  {port.direction.value} {port.name}[{port.width}]")

if __name__ == "__main__":
    test_parse_thresholding_axi()