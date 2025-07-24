"""Utility functions for segment execution."""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Type, Optional
from brainsmith.core.execution_tree import ExecutionNode
from brainsmith.core.plugins.registry import BrainsmithPluginRegistry
from .types import SegmentResult, TreeExecutionResult


def serialize_tree(root: ExecutionNode) -> str:
    """Serialize execution tree to JSON."""
    def node_to_dict(node: ExecutionNode) -> Dict[str, Any]:
        # Serialize steps, handling transform classes
        serialized_steps = []
        for step in node.segment_steps:
            if isinstance(step, dict):
                # Handle transform stages
                if 'transforms' in step:
                    serialized_step = step.copy()
                    # Convert transform classes to names
                    serialized_step['transforms'] = [
                        t.__name__ if hasattr(t, '__name__') else str(t)
                        for t in step['transforms']
                    ]
                    serialized_steps.append(serialized_step)
                else:
                    serialized_steps.append(step)
            else:
                serialized_steps.append(step)
        
        return {
            "segment_id": node.segment_id,
            "segment_steps": serialized_steps,
            "branch_decision": node.branch_decision,
            "is_branch_point": node.is_branch_point,
            "children": {
                name: node_to_dict(child) 
                for name, child in node.children.items()
            }
        }
    
    return json.dumps(node_to_dict(root), indent=2)


def serialize_results(result: TreeExecutionResult) -> str:
    """Serialize execution results to JSON."""
    return json.dumps({
        "stats": result.stats,
        "total_time": result.total_time,
        "segments": {
            segment_id: {
                "success": r.success,
                "cached": r.cached,
                "error": r.error,
                "execution_time": r.execution_time,
                "output_model": str(r.output_model) if r.output_model else None,
                "output_dir": str(r.output_dir) if r.output_dir else None
            }
            for segment_id, r in result.segment_results.items()
        }
    }, indent=2)


class StageWrapperFactory:
    """Factory for creating cached transform stage wrappers."""
    
    def __init__(self, registry: BrainsmithPluginRegistry):
        self.registry = registry
        self._wrappers: Dict[str, callable] = {}
    
    def create_stage_wrapper(self, stage_name: str, 
                           transform_names: List[str],
                           branch_index: int) -> Tuple[str, callable]:
        """Create wrapper for a transform stage branch.
        
        Args:
            stage_name: Name of the transform stage (e.g., 'cleanup')
            transform_names: List of transform class names
            branch_index: Simple numeric index for this branch
            
        Returns:
            Tuple of (wrapper_name, wrapper_function)
        """
        # Generate simple name
        wrapper_name = self._generate_stage_name(stage_name, branch_index, transform_names)
        
        # Check if already exists
        if wrapper_name in self._wrappers:
            return wrapper_name, self._wrappers[wrapper_name]
        
        # Create new wrapper
        wrapper = self._create_stage_wrapper(stage_name, transform_names)
        wrapper.__name__ = wrapper_name
        
        # Store and return
        self._wrappers[wrapper_name] = wrapper
        return wrapper_name, wrapper
    
    def _generate_stage_name(self, stage: str, branch_index: int, transforms: List[str]) -> str:
        """Generate concise but informative stage step name."""
        # Keep stage identity clear with simple numeric index
        if not transforms:
            return f"{stage}_skip"
        
        # Use simple numeric index for branches
        return f"{stage}_{branch_index}"
    
    def _create_stage_wrapper(self, stage_name: str, transform_names: List[str]) -> callable:
        """Create wrapper that preserves stage context."""
        # Resolve transforms once, filtering out None/skip markers
        transforms = []
        for name in transform_names:
            if name:  # Skip None values
                transform_cls = self.registry.get_transform(name)
                if transform_cls:
                    transforms.append(transform_cls)
        
        def stage_wrapper(model, cfg):
            # Log with stage context
            if transforms:
                print(f"[{stage_name}] Executing {len(transforms)} transforms")
                for i, transform_cls in enumerate(transforms, 1):
                    print(f"  [{i}/{len(transforms)}] {transform_cls.__name__}")
                    model = model.transform(transform_cls())
            else:
                print(f"[{stage_name}] Skipping (no transforms selected)")
            return model
        
        # Attach metadata for debugging
        stage_wrapper._stage_info = {
            "stage": stage_name,
            "transforms": transform_names
        }
        
        return stage_wrapper
    
    def get_all_wrappers(self) -> Dict[str, callable]:
        """Get all created wrappers for FINN registration."""
        return self._wrappers.copy()
    
    def get_stage_info(self, wrapper_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a wrapper by name."""
        # Extract info from wrapper if it exists
        wrapper = self._wrappers.get(wrapper_name)
        if wrapper and hasattr(wrapper, '_stage_info'):
            return wrapper._stage_info
        return None


def share_artifacts_at_branch(
    parent_result: SegmentResult,
    child_segments: List[ExecutionNode],
    base_output_dir: Path
) -> None:
    """
    Copy build artifacts to child segments.
    Uses full directory copies for compatibility.
    """
    if not parent_result.success:
        return
    
    print(f"\n  Sharing artifacts to {len(child_segments)} children...")
    
    for child in child_segments:
        child_dir = base_output_dir / child.segment_id
        # Full copy required for compatibility
        if child_dir.exists():
            shutil.rmtree(child_dir)
        shutil.copytree(parent_result.output_dir, child_dir)