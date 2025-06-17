#!/usr/bin/env python3
"""
Debug script to check what context is being passed to RTL backend template.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
    from brainsmith.tools.hw_kernel_gen.templates.context_generator import TemplateContextGenerator
    from brainsmith.tools.hw_kernel_gen.generators.rtl_backend_generator import RTLBackendGenerator
    
    # Parse the thresholding kernel
    rtl_file = project_root / "brainsmith/hw_kernels/thresholding/thresholding_axi.sv"
    
    print(f"Parsing RTL file: {rtl_file}")
    parser = RTLParser()
    kernel_metadata = parser.parse_file(str(rtl_file))
    
    # Generate template context
    print("Generating template context...")
    context = TemplateContextGenerator.generate_context(kernel_metadata)
    
    # Generate RTL backend context
    print("Generating RTL backend context...")
    rtl_generator = RTLBackendGenerator()
    template_context = TemplateContextGenerator.generate_template_context(kernel_metadata)
    rtl_context = rtl_generator.process_context(template_context)
    
    # Check what's in the RTL context
    print(f"\n=== RTL Context Keys ===")
    for key in sorted(rtl_context.keys()):
        value = rtl_context[key]
        if isinstance(value, dict):
            print(f"{key}: dict with {len(value)} items")
        elif isinstance(value, list):
            print(f"{key}: list with {len(value)} items")
        else:
            print(f"{key}: {type(value).__name__} = {str(value)[:100]}")
    
    # Check datatype derivation methods specifically
    datatype_derivation_methods = rtl_context.get('datatype_derivation_methods', {})
    print(f"\n=== Datatype Derivation Methods ===")
    print(f"Found {len(datatype_derivation_methods)} methods")
    for param_name, method_code in datatype_derivation_methods.items():
        print(f"\nParameter: {param_name}")
        print(f"Method code: {method_code}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)