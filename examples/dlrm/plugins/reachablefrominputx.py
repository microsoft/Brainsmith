import onnx
from onnx import numpy_helper
from typing import Dict, Set, List
from collections import deque
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from onnxscript import ir

class ReachableFromInputTransform(Transformation):
    """
    Analyzes the ONNX model to determine which nodes are reachable from each input.
    This transform does not modify the model, only analyzes reachability.
    Assumes the graph is directed and acyclic (DAG).
    """

    def __init__(self):
        super().__init__()
        self.input_reachability: Dict[str, Set[str]] = {}
        self.ir_model = None

    def _build_subgraph_to_node_map(self, graph):
        """
        Build a mapping from subgraph to the node that contains it.
        Recursively walks through all nodes and their subgraphs.
        """
        subgraph_to_node = {}

        def visit_node(node):
            # Check all attributes for subgraphs
            for attr in node.attributes.values():
                if hasattr(attr, 'value') and isinstance(attr.value, ir.Graph):
                    subgraph = attr.value
                    subgraph_to_node[id(subgraph)] = node
                    # Recursively visit nodes in the subgraph
                    for sub_node in subgraph:
                        visit_node(sub_node)

        # Visit all nodes in the main graph
        for node in graph:
            visit_node(node)

        return subgraph_to_node

    def _get_top_level_node(self, node, main_graph, subgraph_map):
        """
        Get the top-level node in the main graph.
        If node is in a subgraph, return its enclosing parent node.
        Otherwise, return the node itself.
        """
        current = node
        # Walk up the graph hierarchy until we reach the main graph
        while current.graph != main_graph:
            graph_id = id(current.graph)
            if graph_id not in subgraph_map:
                # Couldn't find parent, return current node
                break
            current = subgraph_map[graph_id]
        return current

    def apply(self, model: ModelWrapper) -> ModelWrapper:
        """
        Analyze the model and compute reachability for each input.
        Returns the original model unchanged.
        """
        # Convert ONNX model to IR and store it
        self.ir_model = ir.serde.deserialize_model(model.model)
        graph = self.ir_model.graph

        # Build mapping from subgraphs to their containing nodes
        subgraph_map = self._build_subgraph_to_node_map(graph)

        # Get all model inputs
        model_inputs = list(graph.inputs)

        # Compute reachability from each input using BFS
        self.input_reachability = {}
        for input_value in model_inputs:
            reachable_nodes = set()
            queue = deque()

            # Start by adding all direct consumers of the input
            for consumer in input_value.consumers():
                top_level_node = self._get_top_level_node(consumer, graph, subgraph_map)
                queue.append(consumer)
                reachable_nodes.add(top_level_node)

            # BFS traversal through nodes (no cycle checking needed for DAG)
            while queue:
                current_node = queue.popleft()

                # Add all consumers of this node's outputs to the queue
                for output_value in current_node.outputs:
                    for consumer in output_value.consumers():
                        top_level_node = self._get_top_level_node(consumer, graph, subgraph_map)
                        reachable_nodes.add(top_level_node)
                        queue.append(consumer)

            self.input_reachability[input_value.name] = reachable_nodes

        # Log the results
        for input_name, reachable_nodes in self.input_reachability.items():
            print(f"Input '{input_name}' reaches {len(reachable_nodes)} nodes")

        # Return the original model unchanged
        return model

    def get_reachable_nodes(self, input_name: str) -> Set[str]:
        """
        Get the set of node names reachable from a specific input.
        """
        return self.input_reachability.get(input_name, set())

    def get_all_reachability(self) -> Dict[str, Set[str]]:
        """
        Get the complete reachability mapping for all inputs.
        """
        return self.input_reachability
