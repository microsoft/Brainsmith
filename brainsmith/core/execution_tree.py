"""
Execution Tree Implementation - Direct Blueprint to Tree Construction

This module implements the execution tree architecture that transforms
blueprints into optimized execution trees with automatic prefix sharing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path

from qonnx.transformation.base import Transformation


@dataclass
class TransformStage:
    """A stage containing multiple transform steps with branching options."""
    name: str
    transform_steps: List[List[Optional[Type[Transformation]]]]
    
    def get_combinations(self) -> List[List[Type[Transformation]]]:
        """Get all valid combinations of transforms for this stage."""
        if not self.transform_steps:
            return [[]]
        
        combinations = [[]]
        
        for step_options in self.transform_steps:
            new_combinations = []
            
            for combo in combinations:
                for option in step_options:
                    if option is None:
                        # Skip option
                        new_combinations.append(combo)
                    else:
                        # Add transform
                        new_combinations.append(combo + [option])
            
            combinations = new_combinations
        
        return combinations


@dataclass
class ExecutionNode:
    """Node in execution tree representing a pipeline step."""
    step_name: str
    config: Dict[str, Any]
    parent: Optional['ExecutionNode'] = None
    children: List['ExecutionNode'] = field(default_factory=list)
    
    # Execution state (for future use)
    status: str = "pending"
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    
    def find_or_create_child(self, step_name: str, config: Dict) -> 'ExecutionNode':
        """Get existing child with matching config or create new one."""
        config_key = self._make_config_key(config)
        
        for child in self.children:
            if child.step_name == step_name and self._make_config_key(child.config) == config_key:
                return child
        
        child = ExecutionNode(step_name, config, parent=self)
        self.children.append(child)
        return child
    
    def _make_config_key(self, config: Dict) -> str:
        """Create comparable key from config for deduplication."""
        items = []
        for k, v in sorted(config.items()):
            if k == "transforms" and isinstance(v, list):
                # Transform class names
                items.append((k, tuple(t.__name__ for t in v)))
            elif k == "kernel_backends" and isinstance(v, list):
                # Backend class names
                items.append((k, tuple((kn, tuple(b.__name__ for b in bc)) for kn, bc in v)))
            else:
                items.append((k, str(v)))
        return str(items)
    
    def get_path_to_root(self) -> List['ExecutionNode']:
        """Get path from this node to root."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path
    
    def count_descendants(self) -> int:
        """Count total number of descendant nodes."""
        count = len(self.children)
        for child in self.children:
            count += child.count_descendants()
        return count


def count_leaves(node: ExecutionNode) -> int:
    """Count leaf nodes in tree."""
    if not node.children:
        return 1
    return sum(count_leaves(child) for child in node.children)


def count_nodes(node: ExecutionNode) -> int:
    """Count all nodes in tree."""
    count = 0 if node.step_name == "root" else 1
    for child in node.children:
        count += count_nodes(child)
    return count


def print_tree(node: ExecutionNode, indent: str = "", last: bool = True):
    """Pretty print the execution tree."""
    if node.step_name != "root":
        prefix = "└── " if last else "├── "
        
        # Format config
        config_str = ""
        if "transforms" in node.config:
            transforms = node.config["transforms"]
            if transforms:
                names = [t.__name__ for t in transforms]
                config_str = f" ({', '.join(names)})"
        elif "kernel_backends" in node.config:
            backend_info = []
            for kernel_name, backend_classes in node.config["kernel_backends"]:
                backend_names = [b.__name__ for b in backend_classes]
                backend_info.append(f"{kernel_name}[{','.join(backend_names)}]")
            config_str = f" ({'; '.join(backend_info)})"
        
        print(f"{indent}{prefix}{node.step_name}{config_str}")
    
    extension = "    " if last else "│   "
    for i, child in enumerate(node.children):
        print_tree(child, indent + extension, i == len(node.children) - 1)


def get_tree_stats(root: ExecutionNode) -> Dict[str, Any]:
    """Get statistics about the execution tree."""
    leaf_count = count_leaves(root)
    node_count = count_nodes(root)
    
    # Calculate depth
    max_depth = 0
    
    def calculate_depth(node: ExecutionNode, depth: int = 0):
        nonlocal max_depth
        if node.step_name != "root":
            max_depth = max(max_depth, depth)
        for child in node.children:
            calculate_depth(child, depth + 1)
    
    calculate_depth(root)
    
    # Estimate sharing factor (how much work we save)
    # Without sharing, we'd have leaf_count * max_depth nodes
    theoretical_nodes = leaf_count * max_depth
    actual_nodes = node_count
    sharing_factor = theoretical_nodes / actual_nodes if actual_nodes > 0 else 1.0
    
    return {
        'total_paths': leaf_count,
        'total_nodes': node_count,
        'max_depth': max_depth,
        'sharing_factor': round(sharing_factor, 2),
        'saved_nodes': theoretical_nodes - actual_nodes
    }