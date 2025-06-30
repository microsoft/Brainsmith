"""
Remove Identity Operations Transform

Example transform that removes identity operations from the graph.
"""

from brainsmith.plugin.decorators import transform
from qonnx.transformation.base import Transformation


@transform(
    name="RemoveIdentityOps",
    stage="cleanup",
    description="Remove identity operations from computation graph",
    author="brainsmith-team",
    version="1.0.0"
)
class RemoveIdentityOps(Transformation):
    """Remove identity operations that don't affect computation."""
    
    def apply(self, model):
        """
        Apply transform to remove identity nodes.
        
        Args:
            model: ONNX ModelWrapper
            
        Returns:
            Tuple of (transformed_model, modified_flag)
        """
        graph = model.graph
        graph_modified = False
        
        # Find and remove identity nodes
        nodes_to_remove = []
        for node in graph.node:
            if node.op_type == "Identity":
                # Get input and output names
                input_name = node.input[0]
                output_name = node.output[0]
                
                # Replace all uses of output with input
                for other_node in graph.node:
                    for i, in_name in enumerate(other_node.input):
                        if in_name == output_name:
                            other_node.input[i] = input_name
                
                # Update graph outputs if needed
                for i, out in enumerate(graph.output):
                    if out.name == output_name:
                        graph.output[i].name = input_name
                
                nodes_to_remove.append(node)
                graph_modified = True
        
        # Remove the identity nodes
        for node in nodes_to_remove:
            graph.node.remove(node)
        
        return (model, graph_modified)