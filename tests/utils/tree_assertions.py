"""Tree assertion helpers for DSE execution tests."""

from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass
from tests.utils.test_constants import (
    MIN_CHILDREN_FOR_BRANCH,
    NO_EFFICIENCY,
    EFFICIENCY_DECIMAL_PLACES,
    EFFICIENCY_PERCENTAGE_MULTIPLIER
)


@dataclass
class ExpectedTreeStructure:
    """Expected structure for tree validation."""
    total_nodes: int
    total_leaves: int
    total_paths: int
    total_segments: int
    segment_efficiency: Optional[float] = None



class TreeAssertions:
    """Helper class for DSE tree assertions."""
    
    @staticmethod
    def assert_tree_structure(tree, expected: ExpectedTreeStructure):
        """Assert basic tree structure properties.
        
        Args:
            tree: The DSE tree to validate
            expected: Expected structure properties
        """
        assert tree.count_nodes() == expected.total_nodes, \
            f"Expected {expected.total_nodes} nodes, got {tree.count_nodes()}"
            
        assert tree.count_leaves() == expected.total_leaves, \
            f"Expected {expected.total_leaves} leaves, got {tree.count_leaves()}"
            
        stats = tree.get_statistics()
        assert stats['total_paths'] == expected.total_paths, \
            f"Expected {expected.total_paths} paths, got {stats['total_paths']}"
            
        assert stats['total_segments'] == expected.total_segments, \
            f"Expected {expected.total_segments} segments, got {stats['total_segments']}"
            
        if expected.segment_efficiency is not None:
            assert stats['segment_efficiency'] == expected.segment_efficiency, \
                f"Expected efficiency {expected.segment_efficiency}%, got {stats['segment_efficiency']}%"
    
    @staticmethod
    def assert_execution_order_structure(execution_order, tree):
        """Assert basic execution order properties.
        
        Args:
            execution_order: List of nodes in execution order
            tree: The DSE tree
        """
        # Root should be first
        assert execution_order[0] == tree.root, \
            "Root node should be first in execution order"
            
        assert execution_order[0].segment_id == "root", \
            f"Root segment_id should be 'root', got '{execution_order[0].segment_id}'"
    
    @staticmethod
    def assert_parent_child_relationships(tree):
        """Assert correct parent-child relationships throughout tree.
        
        Args:
            tree: The DSE tree to validate
        """
        def _check_node_relationships(node, expected_parent=None):
            """Recursively check relationships."""
            if expected_parent is not None:
                assert node.parent == expected_parent, \
                    f"Node {node.segment_id} has incorrect parent"
            
            # Check all children
            for child in node.children.values():
                assert child.parent == node, \
                    f"Child {child.segment_id} has incorrect parent"
                _check_node_relationships(child, node)
        
        # Start from root (which has no parent)
        _check_node_relationships(tree.root)
    
    @staticmethod
    def assert_leaf_properties(tree):
        """Assert correct leaf node properties.
        
        Args:
            tree: The DSE tree to validate
        """
        def _check_leaf_consistency(node):
            """Check leaf property consistency."""
            has_children = bool(node.children)
            is_leaf_property = node.is_leaf
            
            # Leaf property should match absence of children
            assert (not has_children) == is_leaf_property, \
                f"Node {node.segment_id} leaf property ({is_leaf_property}) " \
                f"inconsistent with children ({has_children})"
            
            # Recursively check children
            for child in node.children.values():
                _check_leaf_consistency(child)
        
        _check_leaf_consistency(tree.root)
    
    @staticmethod
    def assert_branch_point_properties(tree):
        """Assert correct branch point properties.
        
        Args:
            tree: The DSE tree to validate
        """
        def _check_branch_consistency(node):
            """Check branch point consistency."""
            has_multiple_children = len(node.children) > MIN_CHILDREN_FOR_BRANCH
            is_branch_property = node.is_branch_point
            
            # Branch point property should match multiple children
            assert has_multiple_children == is_branch_property, \
                f"Node {node.segment_id} branch property ({is_branch_property}) " \
                f"inconsistent with children count ({len(node.children)})"
            
            # Recursively check children
            for child in node.children.values():
                _check_branch_consistency(child)
        
        _check_branch_consistency(tree.root)
    
    @staticmethod
    def assert_complete_tree_validation(tree, expected: ExpectedTreeStructure):
        """Perform comprehensive tree validation.
        
        Args:
            tree: The DSE tree to validate
            expected: Expected structure properties
        """
        # Basic structure
        TreeAssertions.assert_tree_structure(tree, expected)
        
        # Relationship consistency
        TreeAssertions.assert_parent_child_relationships(tree)
        
        # Property consistency
        TreeAssertions.assert_leaf_properties(tree)
        TreeAssertions.assert_branch_point_properties(tree)
        
        # Execution order basics
        execution_order = tree.get_execution_order()
        assert len(execution_order) == expected.total_nodes, \
            f"Execution order length {len(execution_order)} != total nodes {expected.total_nodes}"
            
        TreeAssertions.assert_execution_order_structure(execution_order, tree)


def calculate_segment_efficiency(
    total_transforms_with_segments: int,
    total_transforms_without_segments: int
) -> float:
    """Calculate expected segment efficiency.
    
    Args:
        total_transforms_with_segments: Total transforms when using segments
        total_transforms_without_segments: Total transforms without segments
        
    Returns:
        Efficiency percentage rounded to 1 decimal place
    """
    if total_transforms_without_segments == 0:
        return NO_EFFICIENCY
    
    efficiency = EFFICIENCY_PERCENTAGE_MULTIPLIER * (1 - total_transforms_with_segments / total_transforms_without_segments)
    return round(efficiency, EFFICIENCY_DECIMAL_PLACES)