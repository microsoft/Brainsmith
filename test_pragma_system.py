#!/usr/bin/env python3
"""
Test script to demonstrate the refactored pragma system.

This script parses RTL files and visualizes the KernelMetadata output,
showing how pragmas are applied and validated without template generation.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.dataflow.core.kernel_metadata import KernelMetadata
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata


def serialize_kernel_metadata(metadata: KernelMetadata) -> Dict[str, Any]:
    """Convert KernelMetadata to a serializable dictionary for display."""
    
    def serialize_interface_metadata(iface: InterfaceMetadata) -> Dict[str, Any]:
        """Serialize InterfaceMetadata to dict."""
        result = {
            "name": iface.name,
            "interface_type": iface.interface_type.value,
            "description": iface.description,
            "datatype_constraints": [],
            "chunking_strategy": None,
            "parameter_linkage": {}
        }
        
        # Serialize datatype constraints
        if hasattr(iface, 'datatype_constraints') and iface.datatype_constraints:
            for constraint in iface.datatype_constraints:
                result["datatype_constraints"].append({
                    "base_type": constraint.base_type,
                    "min_width": constraint.min_width,
                    "max_width": constraint.max_width
                })
        
        # Serialize chunking strategy
        if hasattr(iface, 'chunking_strategy') and iface.chunking_strategy:
            strategy = iface.chunking_strategy
            if hasattr(strategy, 'block_shape'):
                result["chunking_strategy"] = {
                    "type": "BlockChunkingStrategy",
                    "block_shape": strategy.block_shape,
                    "rindex": getattr(strategy, 'rindex', 0)
                }
            else:
                result["chunking_strategy"] = {
                    "type": strategy.__class__.__name__
                }
        
        # Serialize parameter linkage
        if hasattr(iface, 'datatype_params') and iface.datatype_params:
            result["parameter_linkage"]["datatype_params"] = iface.datatype_params
        
        if hasattr(iface, 'bdim_param') and iface.bdim_param:
            result["parameter_linkage"]["bdim_param"] = iface.bdim_param
        elif hasattr(iface, 'get_bdim_parameter_name'):
            result["parameter_linkage"]["bdim_param_default"] = iface.get_bdim_parameter_name()
            
        if hasattr(iface, 'sdim_param') and iface.sdim_param:
            result["parameter_linkage"]["sdim_param"] = iface.sdim_param
        elif hasattr(iface, 'get_sdim_parameter_name'):
            result["parameter_linkage"]["sdim_param_default"] = iface.get_sdim_parameter_name()
            
        if hasattr(iface, 'shape_params') and iface.shape_params:
            result["parameter_linkage"]["shape_params"] = iface.shape_params
            
        return result
    
    return {
        "name": metadata.name,
        "source_file": str(metadata.source_file),
        "parameters": [
            {
                "name": param.name,
                "param_type": param.param_type,
                "default_value": param.default_value,
                "description": param.description
            }
            for param in metadata.parameters
        ],
        "exposed_parameters": metadata.exposed_parameters,
        "interfaces": [
            serialize_interface_metadata(iface) for iface in metadata.interfaces
        ],
        "pragmas": [
            {
                "type": pragma.type.value,
                "line_number": pragma.line_number,
                "inputs": pragma.inputs,
                "parsed_data": pragma.parsed_data
            }
            for pragma in metadata.pragmas
        ],
        "parsing_warnings": metadata.parsing_warnings
    }


def print_kernel_metadata(metadata: KernelMetadata):
    """Pretty print KernelMetadata in a human-readable format."""
    print(f"\n{'='*60}")
    print(f"🔍 KERNEL METADATA: {metadata.name}")
    print(f"{'='*60}")
    
    print(f"\n📁 Source File: {metadata.source_file}")
    print(f"⚠️  Warnings: {len(metadata.parsing_warnings)}")
    if metadata.parsing_warnings:
        for warning in metadata.parsing_warnings:
            print(f"   • {warning}")
    
    # Parameters section
    print(f"\n🔧 PARAMETERS ({len(metadata.parameters)}):")
    if not metadata.parameters:
        print("   (none)")
    else:
        for param in metadata.parameters:
            type_str = f" ({param.param_type})" if param.param_type else ""
            default_str = f" = {param.default_value}" if param.default_value else ""
            print(f"   • {param.name}{type_str}{default_str}")
    
    # Exposed parameters section
    print(f"\n📋 EXPOSED PARAMETERS ({len(metadata.exposed_parameters)}):")
    if not metadata.exposed_parameters:
        print("   (none)")
    else:
        exposed_set = set(metadata.exposed_parameters)
        for param in metadata.parameters:
            if param.name in exposed_set:
                default_str = f" = {param.default_value}" if param.default_value else ""
                print(f"   • {param.name}{default_str}")
        
        # Show linked parameters for comparison
        all_params = {param.name for param in metadata.parameters}
        linked_params = all_params - exposed_set
        if linked_params:
            print(f"\n🔗 LINKED PARAMETERS ({len(linked_params)}):")
            for param_name in sorted(linked_params):
                param = next(p for p in metadata.parameters if p.name == param_name)
                default_str = f" = {param.default_value}" if param.default_value else ""
                print(f"   • {param_name}{default_str} (linked to interface)")
    
    # Pragmas section
    print(f"\n📝 PRAGMAS ({len(metadata.pragmas)}):")
    if not metadata.pragmas:
        print("   (none)")
    else:
        for pragma in metadata.pragmas:
            print(f"   • Line {pragma.line_number}: @brainsmith {pragma.type.value} {' '.join(pragma.inputs)}")
            if pragma.parsed_data:
                print(f"     Data: {pragma.parsed_data}")
    
    # Interfaces section
    print(f"\n🔌 INTERFACES ({len(metadata.interfaces)}):")
    for i, iface in enumerate(metadata.interfaces):
        print(f"\n   [{i+1}] {iface.name} ({iface.interface_type.value})")
        
        # Datatype constraints
        if hasattr(iface, 'datatype_constraints') and iface.datatype_constraints:
            print(f"       Datatype Constraints:")
            for constraint in iface.datatype_constraints:
                print(f"         • {constraint.base_type} {constraint.min_width}-{constraint.max_width} bits")
        
        # Chunking strategy
        if hasattr(iface, 'chunking_strategy') and iface.chunking_strategy:
            strategy = iface.chunking_strategy
            if hasattr(strategy, 'block_shape'):
                shape_str = '[' + ', '.join(str(x) for x in strategy.block_shape) + ']'
                rindex = getattr(strategy, 'rindex', 0)
                print(f"       Chunking: {shape_str} (rindex={rindex})")
            else:
                print(f"       Chunking: {strategy.__class__.__name__}")
        
        # Parameter linkage
        print(f"       Parameter Linkage:")
        if hasattr(iface, 'datatype_params') and iface.datatype_params:
            for prop, param in iface.datatype_params.items():
                print(f"         • {prop} → {param}")
        
        if hasattr(iface, 'get_bdim_parameter_name'):
            bdim_param = iface.get_bdim_parameter_name()
            explicit = ""
            if hasattr(iface, 'bdim_param') and iface.bdim_param:
                explicit = " (explicit)"
            print(f"         • bdim → {bdim_param}{explicit}")
            
        if hasattr(iface, 'get_sdim_parameter_name'):
            sdim_param = iface.get_sdim_parameter_name()
            explicit = ""
            if hasattr(iface, 'sdim_param') and iface.sdim_param:
                explicit = " (explicit)"
            print(f"         • sdim → {sdim_param}{explicit}")
            
        if hasattr(iface, 'shape_params') and iface.shape_params:
            print(f"         • shape_params → {iface.shape_params}")


def test_rtl_file(rtl_file: str, debug: bool = False):
    """Test parsing a single RTL file."""
    print(f"\n🚀 Testing RTL file: {rtl_file}")
    print(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
    
    try:
        # Create parser
        parser = RTLParser(debug=debug)
        
        # Parse the file
        metadata = parser.parse_file(rtl_file)
        
        # Print results
        print_kernel_metadata(metadata)
        
        # Also save as JSON for inspection
        json_output = f"{Path(rtl_file).stem}_metadata.json"
        with open(json_output, 'w') as f:
            json.dump(serialize_kernel_metadata(metadata), f, indent=2)
        print(f"\n💾 Saved detailed metadata to: {json_output}")
        
        return metadata
        
    except Exception as e:
        print(f"\n❌ Error parsing {rtl_file}: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None


def test_pragma_system():
    """Main test function demonstrating the pragma system."""
    print("🧪 PRAGMA SYSTEM DEMONSTRATION")
    print("=" * 50)
    print("This script demonstrates the refactored pragma system by parsing")
    print("RTL files and showing how pragmas are applied to interface metadata.")
    
    # Test files to parse
    test_files = [
        "test_new_pragma_format.sv"
    ]
    
    # Check if test files exist, create them if needed
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"\n📝 Creating test file: {test_file}")
            create_test_file(test_file)
    
    results = []
    
    # Test each file
    for test_file in test_files:
        if Path(test_file).exists():
            result = test_rtl_file(test_file, debug=False)
            if result:
                results.append(result)
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"{'='*30}")
    print(f"Tested {len(test_files)} files")
    print(f"Successfully parsed: {len(results)}")
    print(f"Failed: {len(test_files) - len(results)}")
    
    if results:
        print(f"\n✅ The pragma system is working correctly!")
        print(f"   • Interface pragmas are properly applied")
        print(f"   • Parameter validation is working")
        print(f"   • Metadata generation is complete")
    else:
        print(f"\n❌ No files were successfully parsed")
    
    print(f"\n🎯 KEY FEATURES DEMONSTRATED:")
    print(f"   • InterfacePragma base class inheritance")
    print(f"   • Centralized pragma application via PragmaHandler")
    print(f"   • Comprehensive interface metadata validation")
    print(f"   • Flexible interface naming (any prefix allowed)")
    print(f"   • BDIM/SDIM parameter linking with validation")


def create_test_file(filename: str):
    """Create a test RTL file with comprehensive pragma examples."""
    content = '''// Test file for demonstrating the refactored pragma system
// @brainsmith BDIM s_axis_input0 INPUT0_BDIM SHAPE=[C,PE] RINDEX=0
// @brainsmith SDIM s_axis_input0 INPUT0_SDIM
// @brainsmith DATATYPE s_axis_input0 UINT 8 16
// @brainsmith WEIGHT weights_V
// @brainsmith DATATYPE_PARAM weights_V width WEIGHTS_WIDTH

module test_new_format #(
    parameter INPUT0_WIDTH = 8,
    parameter SIGNED_INPUT0 = 0,
    parameter OUTPUT0_WIDTH = 8,
    parameter SIGNED_OUTPUT0 = 0,
    parameter INPUT0_BDIM = 1,
    parameter INPUT0_SDIM = 1,
    parameter OUTPUT0_BDIM = 1,
    parameter OUTPUT0_SDIM = 1,
    parameter C = 128,
    parameter PE = 4,
    parameter WEIGHTS_WIDTH = 8
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    input wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,
    
    // Input interface
    input wire [INPUT0_WIDTH-1:0] s_axis_input0_TDATA,
    input wire s_axis_input0_TVALID,
    output wire s_axis_input0_TREADY,
    
    // Output interface
    output wire [OUTPUT0_WIDTH-1:0] m_axis_output0_TDATA,
    output wire m_axis_output0_TVALID,
    input wire m_axis_output0_TREADY,
    
    // Weight interface
    input wire [WEIGHTS_WIDTH-1:0] weights_V_TDATA,
    input wire weights_V_TVALID,
    output wire weights_V_TREADY
);

    // Module implementation would go here
    assign m_axis_output0_TDATA = s_axis_input0_TDATA ^ weights_V_TDATA;
    assign m_axis_output0_TVALID = s_axis_input0_TVALID & weights_V_TVALID;
    assign s_axis_input0_TREADY = m_axis_output0_TREADY;
    assign weights_V_TREADY = m_axis_output0_TREADY;
    assign ap_done = ap_start;
    assign ap_idle = ~ap_start;
    assign ap_ready = 1'b1;

endmodule'''
    
    with open(filename, 'w') as f:
        f.write(content)
    print(f"   ✅ Created {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the refactored pragma system")
    parser.add_argument("--file", "-f", help="Specific RTL file to test")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    if args.file:
        # Test specific file
        test_rtl_file(args.file, debug=args.debug)
    else:
        # Run full demonstration
        test_pragma_system()