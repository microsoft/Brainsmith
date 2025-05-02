#!/usr/bin/env python3
"""Run RTL parser example on thresholding module.

This script provides a simple way to run the RTL parser on the thresholding
example and see its output. It serves as both a test and a demonstration
of the parser's capabilities.
"""

import os
import sys
from pathlib import Path

# Get project root and add to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def parse_thresholding():
    """Run RTL parser on thresholding module."""
    from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
    
    # Initialize parser
    parser = RTLParser(debug=True)
    
    # Get path to test file
    file_path = os.path.join(project_root, "examples", "thresholding", "thresholding_axi.sv")
    if not os.path.exists(file_path):
        print(f"Error: Test file not found: {file_path}")
        sys.exit(1)
    
    print(f"\nParsing file: {file_path}")
    
    try:
        # Parse module
        kernel = parser.parse_file(file_path)
        
        # Print results
        print("\nParse Results:")
        print("=============")
        print(f"Module Name: {kernel.name}")
        print(f"\nParameters ({len(kernel.parameters)}):")
        for param in kernel.parameters:
            desc = f" // {param.description}" if param.description else ""
            default = f" = {param.default_value}" if param.default_value else ""
            print(f"  {param.name}: {param.param_type}{default}{desc}")
        
        # Group ports by interface
        port_groups = {
            "Global Control": [],
            "AXI Lite": [],
            "AXI Stream": []
        }
        
        # Sort ports into groups
        for port in kernel.ports:
            if port.name in ["ap_clk", "ap_rst_n"]:
                port_groups["Global Control"].append(port)
            elif any(port.name.startswith(p) for p in ["s_axilite_", "config_"]):
                port_groups["AXI Lite"].append(port)
            elif any(port.name.endswith(s) for s in ["TREADY", "TVALID", "TDATA"]):
                port_groups["AXI Stream"].append(port)
        
        print(f"\nPorts by Interface:")
        for group_name, ports in port_groups.items():
            if not ports:
                continue
            print(f"\n{group_name}:")
            for port in sorted(ports, key=lambda p: p.name):
                desc = f" // {port.description}" if port.description else ""
                print(f"  {port.direction.value:6} {port.name}[{port.width}]{desc}")
        
        print("\nSuccess! RTL Parser extracted module interface correctly.")
        
    except Exception as e:
        print(f"Error parsing file: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("Running RTL Parser Example")
    print("=========================")
    
    parse_thresholding()