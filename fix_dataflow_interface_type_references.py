#!/usr/bin/env python3
"""
Script to replace all DataflowInterfaceType references with InterfaceType
"""

import os
import re
from pathlib import Path

def fix_file(file_path):
    """Fix DataflowInterfaceType references in a single file"""
    try:
        content = file_path.read_text()
        original_content = content
        
        # Replace import statements
        content = re.sub(
            r'from \.dataflow_interface import ([^,]+,\s*)?DataflowInterfaceType(,\s*[^)]+)?',
            lambda m: f'from .interface_types import InterfaceType\nfrom .dataflow_interface import {m.group(1) or ""}{m.group(2) or ""}',
            content
        )
        
        # Replace import statements with just DataflowInterfaceType
        content = re.sub(
            r'from \.dataflow_interface import DataflowInterfaceType\s*$',
            'from .interface_types import InterfaceType',
            content,
            flags=re.MULTILINE
        )
        
        # Replace from core.dataflow_interface imports
        content = re.sub(
            r'from \.\.core\.dataflow_interface import ([^,]+,\s*)?DataflowInterfaceType(,\s*[^)]+)?',
            lambda m: f'from ..core.interface_types import InterfaceType\nfrom ..core.dataflow_interface import {m.group(1) or ""}{m.group(2) or ""}',
            content
        )
        
        # Replace DataflowInterfaceType usage
        content = re.sub(r'DataflowInterfaceType\.([A-Z_]+)', r'InterfaceType.\1', content)
        content = re.sub(r'DataflowInterfaceType\b', 'InterfaceType', content)
        
        # Clean up double imports
        content = re.sub(r'from \.interface_types import InterfaceType\s*\nfrom \.dataflow_interface import\s*\n', 
                        'from .interface_types import InterfaceType\nfrom .dataflow_interface import ', content)
        
        if content != original_content:
            print(f"Fixed: {file_path}")
            file_path.write_text(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all DataflowInterfaceType references in the dataflow module"""
    dataflow_dir = Path('/home/tafk/dev/brainsmith-2/brainsmith/dataflow')
    
    python_files = list(dataflow_dir.rglob('*.py'))
    
    fixed_count = 0
    for file_path in python_files:
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()