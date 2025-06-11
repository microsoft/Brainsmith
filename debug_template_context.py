#!/usr/bin/env python3
"""
Debug template context to understand what's being passed to templates.
"""

from pathlib import Path
from brainsmith.tools.hw_kernel_gen.rtl_parser import parse_rtl_file_enhanced
import json

def debug_template_context():
    """Debug template context generation."""
    
    print("🔍 DEBUGGING TEMPLATE CONTEXT")
    print("=" * 50)
    
    # Parse RTL file to enhanced result
    rtl_file = Path("examples/thresholding/thresholding_axi.sv")
    enhanced_result = parse_rtl_file_enhanced(rtl_file)
    
    # Get template context
    context = enhanced_result.get_template_context()
    
    print(f"✅ Enhanced RTL parsing successful")
    print(f"   Module: {enhanced_result.name}")
    print(f"   Raw interfaces: {len(enhanced_result.interfaces)}")
    print(f"   Template context variables: {len(context)}")
    
    # Debug specific interface structures
    print("\n🔍 INTERFACE STRUCTURES:")
    print("-" * 30)
    
    # Raw interfaces from RTL parsing
    print(f"raw interfaces ({len(enhanced_result.interfaces)}):")
    for name, iface in enhanced_result.interfaces.items():
        print(f"  {name}: {type(iface)} - {iface.type}")
    
    # Template interfaces 
    print(f"\ntemplate interfaces ({len(context['interfaces'])}):")
    for i, iface in enumerate(context['interfaces']):
        print(f"  {i}: {iface['name']} - {type(iface['interface_type'])} - {iface['interface_type']}")
        if hasattr(iface['interface_type'], 'value'):
            print(f"      interface_type.value: {iface['interface_type'].value}")
        else:
            print(f"      interface_type is: {type(iface['interface_type'])}")
    
    # Dataflow interfaces
    print(f"\ndataflow_interfaces ({len(context['dataflow_interfaces'])}):")
    for i, iface in enumerate(context['dataflow_interfaces']):
        print(f"  {i}: {iface['name']} - {type(iface.get('interface_type', 'missing'))}")
        if 'interface_type' in iface and hasattr(iface['interface_type'], 'value'):
            print(f"      interface_type.value: {iface['interface_type'].value}")
        else:
            print(f"      interface_type structure: {iface.get('interface_type', 'missing')}")
    
    # Input interfaces
    print(f"\ninput_interfaces ({len(context['input_interfaces'])}):")
    for i, iface in enumerate(context['input_interfaces']):
        print(f"  {i}: {iface['name']} - {type(iface.get('interface_type', 'missing'))}")
    
    return True

if __name__ == "__main__":
    debug_template_context()