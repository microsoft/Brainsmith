"""
LoopRolling Transform

Transform for BERT-specific loop rolling operations.
Converted from custom_step_loop_rolling in old_bert.py.
"""

from qonnx.transformation.base import Transformation
from brainsmith.core.plugins.decorators import transform


@transform(
    name="LoopRolling",
    stage="dataflow_opt",
    framework="brainsmith", 
    description="Roll BERT loops for hardware implementation"
)
class LoopRolling(Transformation):
    """
    BERT-specific transform for loop rolling.
    
    This is a custom transform to roll the loops in the BERT model
    to make it easier to work with. It is not a standard transform
    in the FINN pipeline, but it is useful for this model.
    
    Converted from custom_step_loop_rolling in old_bert.py.
    """
    
    def __init__(self):
        super().__init__()
    
    def apply(self, model):
        """
        Apply LoopRolling transformation for BERT models.
        
        Args:
            model: QONNX ModelWrapper
            
        Returns:
            Tuple[ModelWrapper, bool]: (transformed_model, graph_modified)
        """
        try:
            # Import required dependencies
            from qonnx.transformation.fold_constants import FoldConstants
            import onnxscript
            from onnxscript.rewriter import pattern
            from onnxscript.rewriter import pattern_builder_jsm as pb
            from onnxscript.rewriter import rewrite
            import logging
            
            logger = logging.getLogger(__name__)
            
            # Get configuration from model attributes if available
            cfg = getattr(model, '_brainsmith_config', None)
            if cfg is None:
                logger.warning("No BrainSmith config found - using default output directory")
                output_dir = './'
            else:
                output_dir = getattr(cfg, 'output_dir', './')
            
            # Load loop body template
            loop_body_template_path = output_dir + '/loop-body-template.onnx'
            logger.info(f"Loading loop body template from: {loop_body_template_path}")
            
            try:
                LoopBody = pb.LoopBodyTemplate(loop_body_template_path)
            except FileNotFoundError:
                logger.error(f"Loop body template not found at {loop_body_template_path}")
                logger.error("Run ExtractLoopBody transform first to generate the template")
                return (model, False)
            
            # Replace instances of the loop body with function calls
            change_layers_to_function_calls = pattern.RewriteRule(
                LoopBody.pattern,
                LoopBody.function_replace
            )
            logger.info("Replacing layers with function calls...")
            
            # Convert model to IR
            model_proto = model.model
            model_ir = onnxscript.ir.serde.deserialize_model(model_proto)
            
            # Apply pattern rewrite
            model_layers_replaced = rewrite(
                model_ir,
                pattern_rewrite_rules=[change_layers_to_function_calls]
            )
            
            # Add function definition and loop opset
            model_layers_replaced.functions[LoopBody.function.identifier()] = LoopBody.function
            model_layers_replaced.graph.opset_imports['loop'] = 0
            
            # Serialize back to proto
            model_proto = onnxscript.ir.serde.serialize_model(model_layers_replaced)
            model.model = model_proto
            
            # Normalize graph for loop rolling
            normalized_graph = pb.normalize_io_for_loop_rolling(model_layers_replaced.graph, LoopBody)
            logger.info(f"Normalized graph: {normalized_graph is model_layers_replaced.graph}")
            
            # Save intermediate model for debugging
            onnxscript.ir.save(model_layers_replaced, "normalized.onnx")
            
            # Build loop match pattern
            LoopMatchPattern, nodes = LoopBody.build_function_match_pattern(normalized_graph)
            loop_replace_pattern = pb.build_loop_replace_pattern(normalized_graph, LoopBody)
            
            # Change function calls to loops
            change_function_calls_to_loop = pattern.RewriteRule(
                LoopMatchPattern,
                loop_replace_pattern
            )
            rewrite_set = pattern.RewriteRuleSet([change_function_calls_to_loop])
            count = rewrite_set.apply_to_model(model_layers_replaced, verbose=None)
            logger.info(f"Rolled {count} function calls into a loop operator")
            
            # Update model with final result
            model.model = onnxscript.ir.serde.serialize_model(model_layers_replaced)
            
            # Apply final transforms
            model = model.transform(FoldConstants())
            
            # Try to apply InferFinnLoopOp if available
            try:
                from brainsmith.transforms.dataflow_opt.infer_finn_loop_op import InferFinnLoopOp
                infer_transform = InferFinnLoopOp()
                model = model.transform(infer_transform)
            except ImportError:
                logger.warning("InferFinnLoopOp not available - skipping loop inference")
            
            logger.info("✅ Loop rolling completed successfully")
            return (model, True)
            
        except ImportError as e:
            # If onnxscript is not available, skip the transformation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"onnxscript not available for LoopRolling: {e}")
            logger.warning("Skipping loop rolling - install onnxscript if needed")
            return (model, False)
        except Exception as e:
            # Handle any other errors gracefully
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in LoopRolling transform: {e}")
            logger.warning("Continuing without loop rolling")
            return (model, False)