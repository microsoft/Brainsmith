"""
InferFinnLoopOp Transform

Stub transform for FINN's InferFinnLoopOp transformation.
This is a temporary implementation that delegates to the actual FINN transform
until we add it to the FINN manual registry.
"""

from qonnx.transformation.base import Transformation
from brainsmith.core.plugins.decorators import transform


@transform(
    name="InferFinnLoopOp",
    stage="dataflow_opt",
    framework="brainsmith", 
    description="Infer FINN loop operations for hardware implementation (FINN delegate)"
)
class InferFinnLoopOp(Transformation):
    """
    Stub for FINN's InferFinnLoopOp transformation.
    
    This transform infers FINN loop operations to enable hardware loop unrolling
    and optimization for FPGA implementations.
    
    TODO: Replace with actual FINN transform registration in manual registry.
    """
    
    def __init__(self):
        super().__init__()
    
    def apply(self, model):
        """
        Apply InferFinnLoopOp transformation by delegating to FINN implementation.
        
        Args:
            model: QONNX ModelWrapper
            
        Returns:
            Tuple[ModelWrapper, bool]: (transformed_model, graph_modified)
        """
        # Import and delegate to actual FINN transform
        import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
        
        finn_transform = to_hw.InferFinnLoopOp()
        return finn_transform.apply(model)