"""
Transformer-specific hardware inference operations.
"""

from brainsmith.steps import register_step
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import brainsmith.transformation.convert_to_hw_layers as to_bs_hw


@register_step(
    name="transformer.infer_hardware",
    category="transformer",
    description="Infer hardware layers for transformer-specific operations",
    dependencies=["transformer.streamlining"]
)
def infer_hardware_step(model, cfg):
    """ 
    BERT custom step for infer hardware 

    Because we have some custom operations in this plugin module we
    need a custom step for infering the hardware for those operations.

    Such as:
        InferShuffle - to infer the Shuffle operations
        InferQuantSoftmax - to infer the QuantSoftMax

    However, we can also see some extra infer steps that
    are not part of the plugin. Some of these are currently
    not handled by the default steps in FINN and need to be 
    added here, for instace:
        
        InferDuplicateStreamsLayer - is needed because we have
        need to have explicit fork nodes, the hardware gen
        cannot connect to the same stream twice, it needs to be
        explictly duplicated.

    """
    model = model.transform(to_bs_hw.InferLayerNorm())
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(to_bs_hw.InferShuffle())
    #model = model.transform(to_bs_hw.InferQuantSoftmax())
    model = model.transform(to_bs_hw.InferHWSoftmax())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    return model