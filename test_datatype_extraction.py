#!/usr/bin/env python3
"""
Test script to verify datatype parameter extraction from thresholding kernel.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
    from brainsmith.tools.hw_kernel_gen.templates.context_generator import TemplateContextGenerator
    
    # Parse the thresholding kernel
    rtl_file = project_root / "brainsmith/hw_kernels/thresholding/thresholding_axi.sv"
    
    print(f"Parsing RTL file: {rtl_file}")
    parser = RTLParser()
    kernel_metadata = parser.parse_file(str(rtl_file))
    
    print(f"Parse successful!")
    print(f"Found {len(kernel_metadata.interfaces)} interfaces")
    print(f"Found {len(kernel_metadata.parameters)} parameters")
    
    # Print all parsed parameters
    print("\n=== Parsed Parameters ===")
    for param in kernel_metadata.parameters:
        print(f"Parameter: {param.name} (type: {param.param_type}, default: {param.default_value})")
    
    # Test datatype parameter extraction
    print("\n=== Testing Datatype Parameter Extraction ===")
    
    # Generate template context
    context = TemplateContextGenerator.generate_context(kernel_metadata)
    
    # Check if datatype parameter mappings were extracted
    datatype_linked_params = context.get('datatype_linked_params', [])
    datatype_param_mappings = context.get('datatype_param_mappings', {})
    interface_datatype_attributes = context.get('interface_datatype_attributes', [])
    datatype_derivation_methods = context.get('datatype_derivation_methods', {})
    
    print(f"Datatype-linked parameters: {datatype_linked_params}")
    print(f"Datatype parameter mappings: {datatype_param_mappings}")
    print(f"Interface datatype attributes: {len(interface_datatype_attributes)}")
    print(f"Datatype derivation methods: {len(datatype_derivation_methods)}")
    
    # Print derivation methods for debugging  
    if datatype_derivation_methods:
        print("\n=== Datatype Derivation Methods ===")
        for param_name, method_code in datatype_derivation_methods.items():
            print(f"Parameter: {param_name}")
            print(f"Method code (first 100 chars): {method_code[:100]}...")
    
    # Print details for each interface
    print("\n=== Interface Analysis ===")
    for interface in kernel_metadata.interfaces:
        print(f"Interface: {interface.name} ({interface.interface_type.value})")
        if hasattr(interface, 'datatype_params') and interface.datatype_params:
            print(f"  Datatype params: {interface.datatype_params}")
        else:
            print(f"  No datatype params found")
    
    # Test if expected parameters are found
    expected_params = ['WI', 'SIGNED', 'FPARG', 'O_BITS', 'BIAS']
    found_params = set(datatype_linked_params)
    
    print(f"\n=== Validation ===")
    print(f"Expected datatype-linked parameters: {expected_params}")
    print(f"Found datatype-linked parameters: {list(found_params)}")
    
    missing = set(expected_params) - found_params
    extra = found_params - set(expected_params)
    
    if missing:
        print(f"❌ Missing parameters: {list(missing)}")
    if extra:
        print(f"ℹ️  Extra parameters: {list(extra)}")
    if not missing and not extra:
        print("✅ All expected parameters found!")
    
    print("\nDatatype parameter extraction test completed.")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires the full development environment to be set up.")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)