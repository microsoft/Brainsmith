"""
ExtractLoopBody Transform

Transform for BERT-specific loop body extraction.
Converted from custom_step_extract_loop_body in old_bert.py.
"""

from qonnx.transformation.base import Transformation
from brainsmith.core.plugins.decorators import transform


@transform(
    name="ExtractLoopBody",
    stage="dataflow_opt",
    framework="brainsmith", 
    description="Extract BERT loop body for hardware implementation"
)
class ExtractLoopBody(Transformation):
    """
    BERT-specific transform for extracting the loop body.
    
    This is a custom transform to extract the loop body from the
    BERT model. It is not a standard transform in the FINN pipeline,
    but it is useful for this model.
    
    Converted from custom_step_extract_loop_body in old_bert.py.
    """
    
    def __init__(self):
        super().__init__()
    
    def apply(self, model):
        """
        Apply ExtractLoopBody transformation for BERT models.
        
        Args:
            model: QONNX ModelWrapper
            
        Returns:
            Tuple[ModelWrapper, bool]: (transformed_model, graph_modified)
        """
        try:
            # Import required dependencies
            from qonnx.transformation.fold_constants import FoldConstants
            import onnxscript
            from onnxscript.utils import graph_view_utils as gvu
            import onnx
            import logging
            
            logger = logging.getLogger(__name__)
            
            # First fold constants
            model = model.transform(FoldConstants())
            
            # Get configuration from model attributes if available
            cfg = getattr(model, '_brainsmith_config', None)
            if cfg is None:
                logger.warning("No BrainSmith config found - using default loop body hierarchy")
                # Use default hierarchy if no config
                loop_body_hierarchy = "encoder.layer.0"
            else:
                loop_body_hierarchy = getattr(cfg, 'loop_body_hierarchy', "encoder.layer.0")
            
            # Get output directory for saving loop body template
            output_dir = getattr(cfg, 'output_dir', './') if cfg else './'
            
            # Deserialize model to IR
            model_ir = onnxscript.ir.serde.deserialize_model(model.model)
            graph = model_ir.graph
            
            # Build hierarchy tree
            P = gvu.PytorchHierarchyNode()
            unadded_nodes = []
            for node in graph._nodes:
                added = P.add_node(node)
                if not added:
                    unadded_nodes.append(node)
            
            P.print_hierarchy()
            logger.info(f"Total nodes: {len(graph._nodes)}")
            logger.info(f"Unadded nodes: {len(unadded_nodes)}")
            
            # Handle unadded Transpose nodes as special case for BERT
            for node in unadded_nodes:
                logger.info(f"Adding metadata for node {node.name}")
                pred_node = node.predecessors()[0]
                node.metadata_props['pkg.torch.onnx.name_scopes'] = pred_node.metadata_props['pkg.torch.onnx.name_scopes']
                node.metadata_props['pkg.torch.onnx.class_hierarchy'] = pred_node.metadata_props['pkg.torch.onnx.class_hierarchy']
                assert(P.add_node(node))
            
            # Extract loop body graph view
            loop_body_graph_view = gvu.bGraphView(f'loop-body', P.get_nodes(loop_body_hierarchy))
            logger.info(f"Loop body graph view: {len(loop_body_graph_view._nodes)} nodes")
            
            # Create loop body model and save template
            loop_body_model = onnxscript.ir.Model(loop_body_graph_view, ir_version=10)
            proto = onnxscript.ir.serde.serialize_model(loop_body_model)
            template_path = output_dir + '/loop-body-template.onnx'
            onnx.save(proto, template_path)
            
            logger.info(f"✅ Saved loop body template to: {template_path}")
            return (model, True)
            
        except ImportError as e:
            # If onnxscript is not available, skip the transformation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"onnxscript not available for ExtractLoopBody: {e}")
            logger.warning("Skipping loop body extraction - install onnxscript if needed")
            return (model, False)
        except Exception as e:
            # Handle any other errors gracefully
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in ExtractLoopBody transform: {e}")
            logger.warning("Continuing without loop body extraction")
            return (model, False)