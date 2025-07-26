"""
Example implementation of step inheritance for blueprint parser.
This shows how the new syntax would be processed.
"""

from typing import List, Union, Dict, Any, Optional, Literal
from dataclasses import dataclass

# Type definitions
StepDef = Union[str, List[str]]

@dataclass
class StepOperation:
    """Represents a step manipulation operation"""
    op_type: Literal["after", "before", "replace", "remove", "at_start", "at_end"]
    target: Optional[StepDef] = None
    insert: Optional[StepDef] = None
    with_step: Optional[StepDef] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepOperation':
        """Parse operation from YAML dict"""
        if "after" in data:
            return cls(op_type="after", target=data["after"], insert=data.get("insert"))
        elif "before" in data:
            return cls(op_type="before", target=data["before"], insert=data.get("insert"))
        elif "replace" in data:
            return cls(op_type="replace", target=data["replace"], with_step=data.get("with"))
        elif "remove" in data:
            return cls(op_type="remove", target=data["remove"])
        elif "at_start" in data:
            return cls(op_type="at_start", insert=data["at_start"]["insert"])
        elif "at_end" in data:
            return cls(op_type="at_end", insert=data["at_end"]["insert"])
        else:
            raise ValueError(f"Unknown operation: {data}")


def apply_step_inheritance(
    parent_steps: List[StepDef], 
    child_spec: List[Union[StepDef, Dict[str, Any]]],
    has_parent: bool = True
) -> List[StepDef]:
    """
    Apply child step operations to parent steps.
    
    Args:
        parent_steps: List of steps from parent blueprint (or initial steps)
        child_spec: List of steps and/or operations from child blueprint
        has_parent: Whether this blueprint extends another
        
    Returns:
        Resolved list of steps after applying all operations
    """
    # Separate regular steps from operations
    operations: List[StepOperation] = []
    direct_steps: List[StepDef] = []
    
    for item in child_spec:
        if isinstance(item, dict):
            operations.append(StepOperation.from_dict(item))
        else:
            direct_steps.append(item)
    
    # Determine starting point
    if operations and not direct_steps:
        # Only operations: start with parent steps (or empty if no parent)
        result = parent_steps.copy() if has_parent else []
    elif direct_steps and not operations:
        # Only direct steps: legacy behavior
        if has_parent:
            # With parent: child steps replace parent steps
            return direct_steps
        else:
            # No parent: just use the direct steps
            return direct_steps
    else:
        # Mix of steps and operations: direct steps are the base
        result = direct_steps.copy()
    
    # Apply operations in order
    for op in operations:
        result = _apply_single_operation(result, op)
    
    return result


def _apply_single_operation(steps: List[StepDef], op: StepOperation) -> List[StepDef]:
    """Apply a single operation to the step list"""
    
    if op.op_type == "remove":
        return [s for s in steps if not _step_matches(s, op.target)]
    
    elif op.op_type == "replace":
        new_steps = []
        for step in steps:
            if _step_matches(step, op.target):
                new_steps.append(op.with_step)
            else:
                new_steps.append(step)
        return new_steps
    
    elif op.op_type == "after":
        new_steps = []
        for step in steps:
            new_steps.append(step)
            if _step_matches(step, op.target):
                new_steps.append(op.insert)
        return new_steps
    
    elif op.op_type == "before":
        new_steps = []
        for step in steps:
            if _step_matches(step, op.target):
                new_steps.append(op.insert)
            new_steps.append(step)
        return new_steps
    
    elif op.op_type == "at_start":
        return [op.insert] + steps
    
    elif op.op_type == "at_end":
        return steps + [op.insert]
    
    return steps


def _step_matches(step: StepDef, target: StepDef) -> bool:
    """Check if a step matches the target pattern"""
    # Simple string match
    if isinstance(step, str) and isinstance(target, str):
        return step == target
    
    # List match (for branching steps)
    if isinstance(step, list) and isinstance(target, list):
        return set(step) == set(target)
    
    # No match for mismatched types
    return False


# Example usage
if __name__ == "__main__":
    # Parent blueprint steps
    parent_steps = [
        "preprocessing",
        "cleanup", 
        ["optimize_fast", "optimize_thorough"],
        "packaging"
    ]
    
    # Child blueprint with operations (no inherit:all needed)
    child_spec = [
        {"after": "cleanup", "insert": "validation"},
        {"replace": ["optimize_fast", "optimize_thorough"], 
         "with": ["opt_minimal", "opt_fast", "opt_thorough", "opt_extreme"]},
        {"before": "packaging", "insert": ["test", "~"]},
        {"at_end": {"insert": "finalization"}}
    ]
    
    # Apply inheritance
    result = apply_step_inheritance(parent_steps, child_spec)
    
    print("Parent steps:")
    for i, step in enumerate(parent_steps):
        print(f"  {i+1}. {step}")
    
    print("\nChild operations:")
    for op in child_spec:
        if isinstance(op, dict):
            print(f"  - {op}")
    
    print("\nResult steps:")
    for i, step in enumerate(result):
        print(f"  {i+1}. {step}")