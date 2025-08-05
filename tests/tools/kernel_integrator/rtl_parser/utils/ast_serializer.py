############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""AST Serializer utility for converting tree-sitter AST to text format.

This module provides utilities to serialize tree-sitter AST nodes into
human-readable text formats for testing and debugging purposes.
"""

from typing import Optional, List, Set, TextIO
from tree_sitter import Node, Tree
import json


class ASTSerializer:
    """Serializes tree-sitter AST to various text formats."""
    
    def __init__(self, 
                 max_depth: Optional[int] = None,
                 max_text_length: int = 50,
                 include_positions: bool = True,
                 exclude_types: Optional[Set[str]] = None):
        """Initialize AST serializer with configuration.
        
        Args:
            max_depth: Maximum depth to traverse (None for unlimited).
            max_text_length: Maximum length of node text to display.
            include_positions: Whether to include line:col positions.
            exclude_types: Set of node types to exclude from output.
        """
        self.max_depth = max_depth
        self.max_text_length = max_text_length
        self.include_positions = include_positions
        self.exclude_types = exclude_types or set()
        
    def serialize_tree(self, tree: Tree, format: str = "tree") -> str:
        """Serialize AST tree to string.
        
        Args:
            tree: Tree-sitter Tree object.
            format: Output format ("tree", "json", "compact").
            
        Returns:
            Serialized AST as string.
        """
        if format == "tree":
            return self._serialize_tree_format(tree.root_node)
        elif format == "json":
            return self._serialize_json_format(tree.root_node)
        elif format == "compact":
            return self._serialize_compact_format(tree.root_node)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def serialize_to_file(self, tree: Tree, file_path: str, format: str = "tree") -> None:
        """Serialize AST tree to file.
        
        Args:
            tree: Tree-sitter Tree object.
            file_path: Path to output file.
            format: Output format ("tree", "json", "compact").
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.serialize_tree(tree, format))
    
    def _serialize_tree_format(self, node: Node, depth: int = 0, prefix: str = "", is_last: bool = True) -> str:
        """Serialize node in tree format with visual connectors.
        
        Args:
            node: Current node to serialize.
            depth: Current depth in tree.
            prefix: Prefix string for current line.
            is_last: Whether this is the last child of parent.
            
        Returns:
            Tree-formatted string.
        """
        if self.max_depth is not None and depth > self.max_depth:
            return ""
            
        if node.type in self.exclude_types:
            return ""
        
        lines = []
        
        # Build current node line
        connector = "└── " if is_last else "├── "
        node_line = prefix + connector + self._format_node(node)
        lines.append(node_line)
        
        # Process children
        children = [child for child in node.children if child.type not in self.exclude_types]
        
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            
            # Build prefix for child
            if is_last:
                child_prefix = prefix + "    "
            else:
                child_prefix = prefix + "│   "
            
            child_str = self._serialize_tree_format(child, depth + 1, child_prefix, is_last_child)
            if child_str:
                lines.append(child_str)
        
        return "\n".join(lines)
    
    def _serialize_json_format(self, node: Node, depth: int = 0) -> str:
        """Serialize node in JSON-like format.
        
        Args:
            node: Current node to serialize.
            depth: Current depth in tree.
            
        Returns:
            JSON-formatted string.
        """
        if self.max_depth is not None and depth > self.max_depth:
            return "{...}"
            
        if node.type in self.exclude_types:
            return ""
        
        node_dict = {
            "type": node.type,
            "text": self._get_node_text(node),
        }
        
        if self.include_positions:
            node_dict["start"] = f"{node.start_point[0]}:{node.start_point[1]}"
            node_dict["end"] = f"{node.end_point[0]}:{node.end_point[1]}"
        
        children = [child for child in node.children if child.type not in self.exclude_types]
        if children:
            node_dict["children"] = [
                json.loads(self._serialize_json_format(child, depth + 1))
                for child in children
            ]
        
        return json.dumps(node_dict, indent=2)
    
    def _serialize_compact_format(self, node: Node, depth: int = 0) -> str:
        """Serialize node in compact single-line format.
        
        Args:
            node: Current node to serialize.
            depth: Current depth in tree.
            
        Returns:
            Compact formatted string.
        """
        if self.max_depth is not None and depth > self.max_depth:
            return "..."
            
        if node.type in self.exclude_types:
            return ""
        
        parts = [node.type]
        
        text = self._get_node_text(node)
        if text and text != node.type:
            parts.append(f'"{text}"')
        
        if self.include_positions:
            parts.append(f"[{node.start_point[0]}:{node.start_point[1]}-{node.end_point[0]}:{node.end_point[1]}]")
        
        children = [child for child in node.children if child.type not in self.exclude_types]
        if children:
            child_strs = [self._serialize_compact_format(child, depth + 1) for child in children]
            child_strs = [s for s in child_strs if s]  # Filter empty strings
            if child_strs:
                parts.append("(" + ", ".join(child_strs) + ")")
        
        return " ".join(parts)
    
    def _format_node(self, node: Node) -> str:
        """Format a single node for display.
        
        Args:
            node: Node to format.
            
        Returns:
            Formatted node string.
        """
        parts = [node.type]
        
        # Add node text if meaningful
        text = self._get_node_text(node)
        if text and text != node.type:
            parts.append(f'"{text}"')
        
        # Add position information
        if self.include_positions:
            parts.append(f"[{node.start_point[0]}:{node.start_point[1]}-{node.end_point[0]}:{node.end_point[1]}]")
        
        # Add error indicator
        if node.has_error:
            parts.append("ERROR")
        
        return " ".join(parts)
    
    def _get_node_text(self, node: Node) -> str:
        """Get node text, truncating if necessary.
        
        Args:
            node: Node to get text from.
            
        Returns:
            Node text (possibly truncated).
        """
        if not node.text:
            return ""
            
        text = node.text.decode('utf-8').strip()
        
        # Replace newlines with escape sequence
        text = text.replace('\n', '\\n')
        text = text.replace('\t', '\\t')
        
        # Truncate if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."
        
        return text


class ASTDiffer:
    """Compare two AST trees and report differences."""
    
    def __init__(self, ignore_positions: bool = True, ignore_whitespace: bool = True):
        """Initialize AST differ.
        
        Args:
            ignore_positions: Whether to ignore line:col differences.
            ignore_whitespace: Whether to ignore whitespace differences in text.
        """
        self.ignore_positions = ignore_positions
        self.ignore_whitespace = ignore_whitespace
        
    def compare_trees(self, tree1: Tree, tree2: Tree) -> List[str]:
        """Compare two AST trees and return differences.
        
        Args:
            tree1: First tree to compare.
            tree2: Second tree to compare.
            
        Returns:
            List of difference descriptions.
        """
        differences = []
        self._compare_nodes(tree1.root_node, tree2.root_node, "", differences)
        return differences
    
    def _compare_nodes(self, node1: Optional[Node], node2: Optional[Node], 
                      path: str, differences: List[str]) -> None:
        """Recursively compare two nodes.
        
        Args:
            node1: First node.
            node2: Second node.
            path: Current path in tree.
            differences: List to append differences to.
        """
        if node1 is None and node2 is None:
            return
            
        if node1 is None:
            differences.append(f"{path}: Node missing in first tree (type: {node2.type})")
            return
            
        if node2 is None:
            differences.append(f"{path}: Node missing in second tree (type: {node1.type})")
            return
        
        # Compare node types
        if node1.type != node2.type:
            differences.append(f"{path}: Type mismatch ({node1.type} vs {node2.type})")
            return
        
        # Compare node text
        text1 = self._normalize_text(node1.text.decode('utf-8') if node1.text else "")
        text2 = self._normalize_text(node2.text.decode('utf-8') if node2.text else "")
        
        if text1 != text2:
            differences.append(f"{path}: Text mismatch ('{text1[:50]}...' vs '{text2[:50]}...')")
        
        # Compare positions if not ignored
        if not self.ignore_positions:
            if node1.start_point != node2.start_point or node1.end_point != node2.end_point:
                differences.append(f"{path}: Position mismatch")
        
        # Compare children
        children1 = list(node1.children)
        children2 = list(node2.children)
        
        if len(children1) != len(children2):
            differences.append(f"{path}: Child count mismatch ({len(children1)} vs {len(children2)})")
            # Continue comparing up to min length
            min_len = min(len(children1), len(children2))
            children1 = children1[:min_len]
            children2 = children2[:min_len]
        
        for i, (child1, child2) in enumerate(zip(children1, children2)):
            child_path = f"{path}/{node1.type}[{i}]"
            self._compare_nodes(child1, child2, child_path, differences)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        if self.ignore_whitespace:
            # Collapse multiple whitespace to single space
            import re
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text