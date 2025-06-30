"""Hardware layer inference operations."""

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from brainsmith.transforms.kernel_mapping.infer_layernorm import InferLayerNorm
from brainsmith.transforms.kernel_mapping.infer_shuffle import InferShuffle
from brainsmith.transforms.kernel_mapping.infer_hwsoftmax import InferHWSoftmax


def infer_hardware_step(model, cfg):
    """
    Infer hardware layers for operations.
    
    Category: hardware
    Dependencies: [streamlining]
    Description: Infers hardware layers for custom operations

    Custom step for infer hardware because we have some custom operations 
    in this plugin module we need a custom step for infering the hardware 
    for those operations.

    Such as:
        InferShuffle - to infer the Shuffle operations
        InferQuantSoftmax - to infer the QuantSoftMax

    However, we can also see some extra infer steps that
    are not part of the plugin. Some of these are currently
    not handled by the default steps in FINN and need to be 
    added here, for instance:
        
        InferDuplicateStreamsLayer - is needed because we have
        need to have explicit fork nodes, the hardware gen
        cannot connect to the same stream twice, it needs to be
        explictly duplicated.
    """
    model = model.transform(InferLayerNorm())
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(InferShuffle())
    #model = model.transform(InferQuantSoftmax())
    model = model.transform(InferHWSoftmax())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    return model