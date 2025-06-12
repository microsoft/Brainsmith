#!/usr/bin/env python3
"""Test InterfaceBuilder with real SystemVerilog file (thresholding_axi.sv)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder
from brainsmith.dataflow.core.interface_types import InterfaceType


def test_thresholding_axi_parsing():
    """Test parsing thresholding_axi.sv and creating InterfaceMetadata."""
    print("Testing with thresholding_axi.sv...")
    
    # Parse the thresholding_axi.sv file
    rtl_file = "examples/thresholding/thresholding_axi.sv"
    
    if not os.path.exists(rtl_file):
        print(f"❌ RTL file not found: {rtl_file}")
        return False
    
    parser = RTLParser(debug=True)
    
    try:
        # Parse the file to get ports and pragmas
        parsed_data = parser.parse_file(rtl_file)
        
        print(f"✅ Successfully parsed {rtl_file}")
        print(f"Module name: {parsed_data.name}")
        print(f"Number of parameters: {len(parsed_data.parameters)}")
        print(f"Number of ports: {len(parsed_data.ports)}")
        print(f"Number of interfaces: {len(parsed_data.interfaces)}")
        print(f"Number of pragmas: {len(parsed_data.pragmas)}")
        
        # Now test the interface builder specifically
        builder = InterfaceBuilder(debug=True)
        
        # Test the new build_interface_metadata method
        metadata_list, unassigned_ports = builder.build_interface_metadata(
            parsed_data.ports, 
            parsed_data.pragmas
        )
        
        print(f"\n--- InterfaceBuilder Results ---")
        print(f"Created {len(metadata_list)} InterfaceMetadata objects:")
        for meta in metadata_list:
            print(f"  - {meta.name}: {meta.interface_type.value}")
            print(f"    Description: {meta.description}")
            print(f"    Allowed datatypes: {len(meta.allowed_datatypes)}")
            for dt in meta.allowed_datatypes:
                print(f"      * {dt.finn_type} ({dt.bit_width} bits, signed={dt.signed})")
            print(f"    Chunking strategy: {type(meta.chunking_strategy).__name__}")
        
        print(f"\nUnassigned ports: {len(unassigned_ports)}")
        for port in unassigned_ports:
            print(f"  - {port.name}: {port.direction.value} {port.width}")
        
        # Validate interface quality
        print(f"\n--- Interface Quality Validation ---")
        
        expected_interface_types = {'control', 'input', 'output'}
        found_types = {meta.interface_type.value for meta in metadata_list}
        
        print(f"Expected interface types: {expected_interface_types}")
        print(f"Found interface types: {found_types}")
        
        metadata_names = {meta.name for meta in metadata_list}
        
        print(f"Found interface names: {metadata_names}")
        
        # Validate that we found reasonable interfaces for thresholding
        if expected_interface_types.issubset(found_types):
            print("✅ All expected interface types found")
        else:
            missing = expected_interface_types - found_types
            print(f"❌ Missing interface types: {missing}")
            return False
        
        # Verify metadata properties
        for meta in metadata_list:
            if not meta.allowed_datatypes:
                print(f"❌ Interface {meta.name} missing datatypes")
                return False
            if not meta.chunking_strategy:
                print(f"❌ Interface {meta.name} missing chunking strategy")
                return False
        
        print("✅ All interface metadata properly configured")
        
        # Test with pragmas if any were found
        if parsed_data.pragmas:
            print(f"\n--- Pragma Application Test ---")
            print(f"Found {len(parsed_data.pragmas)} pragmas:")
            for pragma in parsed_data.pragmas:
                print(f"  - {pragma.type.value}: {pragma.inputs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing {rtl_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = test_thresholding_axi_parsing()
        if success:
            print("\n🎉 Thresholding test passed!")
        else:
            print("\n❌ Thresholding test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)