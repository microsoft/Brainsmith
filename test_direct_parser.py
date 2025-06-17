#!/usr/bin/env python3
"""
Direct test of RTL parser to check for pragma validation errors.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser

def test_invalid_pragma():
    """Test invalid pragma on OUTPUT interface."""
    print("Testing invalid BDIM pragma on OUTPUT interface...")
    
    try:
        parser = RTLParser(debug=True)
        metadata = parser.parse_file("test_invalid_output_bdim.sv")
        print("✅ Parsing succeeded (this should not happen!)")
        print(f"Exposed parameters: {metadata.exposed_parameters}")
        print(f"Total parameters: {[p.name for p in metadata.parameters]}")
        
    except Exception as e:
        print(f"❌ BDIM validation working: {e}")
        print(f"Error type: {type(e).__name__}")
    
    print("\nTesting invalid SDIM pragma on OUTPUT interface...")
    
    try:
        parser = RTLParser(debug=True)
        metadata = parser.parse_file("test_invalid_output_sdim.sv")
        print("✅ Parsing succeeded (this should not happen!)")
        print(f"Exposed parameters: {metadata.exposed_parameters}")
        print(f"Total parameters: {[p.name for p in metadata.parameters]}")
        
    except Exception as e:
        print(f"❌ SDIM validation working: {e}")
        print(f"Error type: {type(e).__name__}")
    
    print("\nTesting invalid BDIM pragma on CONTROL interface...")
    
    try:
        parser = RTLParser(debug=True)
        metadata = parser.parse_file("test_invalid_control_bdim.sv")
        print("✅ Parsing succeeded (this should not happen!)")
        print(f"Exposed parameters: {metadata.exposed_parameters}")
        print(f"Total parameters: {[p.name for p in metadata.parameters]}")
        
    except Exception as e:
        print(f"❌ CONTROL interface validation working: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_invalid_pragma()