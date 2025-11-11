############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""AST parsing utilities for SystemVerilog RTL parser.

This module handles all tree-sitter AST operations including parsing,
syntax checking, and node traversal utilities. It uses the tree-sitter-systemverilog
package from PyPI for grammar support.

Grammar loading is handled directly in this module as of the migration to
py-tree-sitter >= 0.25.0 with tree-sitter-systemverilog package support.
"""

import logging

from tree_sitter import Language, Node, Parser, Tree

# Import will fail if tree-sitter-systemverilog is not installed
try:
    import tree_sitter_systemverilog as systemverilog
except ImportError as e:
    raise ImportError(
        "tree-sitter-systemverilog package not found. "
        "Please install it with: pip install tree-sitter-systemverilog"
    ) from e

logger = logging.getLogger(__name__)


class ASTParserError(Exception):
    """Base exception for AST parsing errors."""

    pass


class SyntaxError(ASTParserError):
    """Raised when SystemVerilog syntax is invalid."""

    pass


class ASTParser:
    """Handles tree-sitter AST operations for SystemVerilog parsing.

    This class encapsulates all low-level AST operations including:
    - Grammar loading and parser initialization
    - Source code parsing to AST
    - Syntax error detection
    - Node traversal utilities
    """

    def __init__(self, debug: bool = False):
        """Initialize the AST parser with tree-sitter grammar.

        Args:
            debug: Enable debug logging.

        Raises:
            RuntimeError: If grammar loading fails.
        """
        self.debug = debug
        self.parser: Parser | None = None

        try:
            # Create Language object from the systemverilog package
            language = Language(systemverilog.language())
            self.parser = Parser(language)
            logger.info(
                "SystemVerilog grammar loaded successfully from tree-sitter-systemverilog package."
            )
        except Exception as e:
            logger.error(f"Failed to load SystemVerilog grammar: {e}")
            raise RuntimeError(f"Failed to load SystemVerilog grammar: {e}") from e

    def parse_source(self, content: str) -> Tree:
        """Parse SystemVerilog source code into an AST.

        Args:
            content: SystemVerilog source code as string.

        Returns:
            Tree-sitter Tree object representing the AST.

        Raises:
            ASTParserError: If parsing fails.
        """
        if not self.parser:
            raise ASTParserError("Parser not initialized")

        try:
            tree = self.parser.parse(bytes(content, "utf8"))
            return tree
        except Exception as e:
            logger.exception(f"Tree-sitter parsing failed: {e}")
            raise ASTParserError(f"Core parsing failed: {e}")

    def check_syntax_errors(self, tree: Tree) -> SyntaxError | None:
        """Check AST for syntax errors.

        Args:
            tree: Tree-sitter Tree to check.

        Returns:
            SyntaxError if errors found, None otherwise.
        """
        if tree.root_node.has_error:
            error_node = self.find_first_error_node(tree.root_node)
            line = error_node.start_point[0] + 1 if error_node else "unknown"
            col = error_node.start_point[1] + 1 if error_node else "unknown"
            error_msg = f"Invalid SystemVerilog syntax near line {line}, column {col}."
            logger.error(f"Syntax error near line {line}:{col}")
            return SyntaxError(error_msg)
        return None

    def find_modules(self, tree: Tree) -> list[Node]:
        """Find all module declaration nodes in the AST.

        Args:
            tree: Tree-sitter Tree to search.

        Returns:
            List of module_declaration nodes.
        """
        return self._find_nodes_by_type(tree.root_node, "module_declaration")

    def find_first_error_node(self, node: Node) -> Node | None:
        """Find the first AST node marked with an error using BFS.

        Args:
            node: Root node to start search from.

        Returns:
            First error node found, or None.
        """
        queue = [node]
        visited = {node.id}

        while queue:
            current = queue.pop(0)
            if current.has_error or current.is_missing:
                # Try to find a more specific child error first
                for child in current.children:
                    if child.has_error or child.is_missing:
                        return child
                return current

            for child in current.children:
                if child.id not in visited:
                    visited.add(child.id)
                    queue.append(child)

        return None

    def find_child(self, node: Node, types: list[str]) -> Node | None:
        """Find the first direct child node matching any of the given types.

        Args:
            node: Parent node to search.
            types: List of node types to match.

        Returns:
            First matching child node, or None.
        """
        if not node:
            return None
        for child in node.children:
            if child.type in types:
                return child
        return None

    def find_children(self, node: Node, types: list[str]) -> list[Node]:
        """Find all direct child nodes matching any of the given types.

        Args:
            node: Parent node to search.
            types: List of node types to match.

        Returns:
            List of matching child nodes.
        """
        found_nodes = []
        if not node:
            return found_nodes
        for child in node.children:
            if child.type in types:
                found_nodes.append(child)
        return found_nodes

    def debug_node(self, node: Node, prefix: str = "", max_depth: int = 3) -> None:
        """Debug helper to print AST node structure.

        Args:
            node: Node to debug.
            prefix: Prefix for output lines.
            max_depth: Maximum depth to traverse.
        """
        self._debug_node_recursive(node, prefix, max_depth, 0)

    def _debug_node_recursive(
        self, node: Node, prefix: str, max_depth: int, current_depth: int
    ) -> None:
        """Recursive helper for debug_node."""
        if node is None or current_depth > max_depth:
            return

        indent = "  " * current_depth
        node_text_raw = node.text.decode("utf8")
        # Limit displayed text and escape newlines
        node_text_display = node_text_raw.replace("\n", "\\n")[:80]
        if len(node_text_raw) > 80:
            node_text_display += "..."

        logger.debug(
            f"{prefix}{indent}Node type: {node.type}, text: '{node_text_display}' (ID: {node.id})"
        )

        for i, child in enumerate(node.children):
            self._debug_node_recursive(
                child,
                prefix=f"{prefix}Child {i}: ",
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )

    def _find_nodes_by_type(self, root: Node, node_type: str) -> list[Node]:
        """Find all nodes of a specific type in the AST.

        Args:
            root: Root node to start search.
            node_type: Type of nodes to find.

        Returns:
            List of nodes matching the type.
        """
        from collections import deque

        nodes = []
        queue = deque([root])

        while queue:
            node = queue.popleft()
            if node.type == node_type:
                nodes.append(node)
            # Avoid descending into nested modules
            if node != root and node.type == "module_declaration":
                continue
            queue.extend(node.children)

        return nodes
