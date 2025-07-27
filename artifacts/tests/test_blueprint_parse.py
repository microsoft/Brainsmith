#!/usr/bin/env python3
"""Debug blueprint parsing."""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from brainsmith.core.blueprint_parser import BlueprintParser

# Parse test blueprint
parser = BlueprintParser()
bp_data = '''
design_space:
  steps:
    - qonnx_to_finn
    - [tidy_up, ~]
    - streamline
  
finn_config:
  board: Pynq-Z1

global_config:
  output_stage: generate_reports  
'''

# Write to temp file
with open('test.yaml', 'w') as f:
    f.write(bp_data)
    
# Parse it
try:
    ds, tree = parser.parse('test.yaml', 'dummy.onnx')
    print(f'Design space steps: {ds.steps}')
    print(f'Number of steps: {len(ds.steps)}')
    print(f'Tree segment steps: {tree.segment_steps}')
    print(f'Tree children: {tree.children}')
    
    # Print full tree
    from brainsmith.core.execution_tree import print_tree
    print("\nFull tree:")
    print_tree(tree)
finally:
    os.unlink('test.yaml')