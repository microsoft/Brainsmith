#!/usr/bin/env python3
"""Test script to verify the step split was successful."""

from brainsmith.steps import list_finn_steps

steps = list_finn_steps()
print(f'Total steps: {len(steps)}')

core_steps = [s for s in steps if s in [
    'cleanup', 'cleanup_advanced', 'fix_dynamic_dimensions',
    'streamlining', 'qonnx_to_finn', 'specialize_layers', 
    'infer_hardware', 'onnx_preprocessing', 'quantization_preprocessing',
    'constrain_folding_and_set_pumped_compute'
]]
print(f'Core steps found: {len(core_steps)}')
print(f'  {core_steps}')

bert_custom_steps = [s for s in steps if s in [
    'remove_head', 'remove_tail', 'shell_metadata_handover', 
    'generate_reference_io'
]]
print(f'BERT custom steps found: {len(bert_custom_steps)}')
print(f'  {bert_custom_steps}')

# Check that all expected steps are present
expected_core = 10
expected_bert = 4

print(f'\nVerification:')
print(f'  Core steps: {len(core_steps)}/{expected_core} {"✓" if len(core_steps) == expected_core else "✗"}')
print(f'  BERT custom steps: {len(bert_custom_steps)}/{expected_bert} {"✓" if len(bert_custom_steps) == expected_bert else "✗"}')