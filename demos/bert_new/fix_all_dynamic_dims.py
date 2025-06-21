#!/usr/bin/env python3
"""Add a transformation to fix ALL dynamic dimensions in the model."""

def create_fix_dynamic_dims_step():
    """Create a transformation step that fixes all dynamic dimensions."""
    
    step_code = '''
def fix_dynamic_dimensions_step(model, cfg):
    """
    Fix all dynamic dimensions in the model to concrete values.
    
    Category: cleanup
    Dependencies: []
    Description: Converts all dynamic dimensions to batch size 1
    """
    import logging
    logger = logging.getLogger(__name__)
    
    changes_made = 0
    
    # Fix graph inputs
    for inp in model.graph.input:
        for i, dim in enumerate(inp.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in input {inp.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    # Fix value_info tensors (intermediate tensors)
    for vi in model.graph.value_info:
        for i, dim in enumerate(vi.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in tensor {vi.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    # Fix graph outputs
    for out in model.graph.output:
        for i, dim in enumerate(out.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                logger.info(f"Fixing dynamic dimension in output {out.name}[{i}]: {dim.dim_param} -> 1")
                dim.dim_value = 1
                dim.ClearField('dim_param')
                changes_made += 1
    
    logger.info(f"Fixed {changes_made} dynamic dimensions")
    return model
'''
    
    return step_code

if __name__ == "__main__":
    code = create_fix_dynamic_dims_step()
    
    # Add to cleanup.py
    with open('../../brainsmith/libraries/transforms/steps/cleanup.py', 'a') as f:
        f.write('\n\n')
        f.write(code)
    
    print("Added fix_dynamic_dimensions_step to cleanup.py")
    print("You can now use it in the pipeline")