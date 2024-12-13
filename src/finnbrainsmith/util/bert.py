import onnx
import argparse
from onnxsim import simplify
from qonnx.util.cleanup import cleanup
from qonnx.transformation.general import (
        SortCommutativeInputsInitializerLast, 
        RemoveUnusedTensors, 
        GiveReadableTensorNames,
        GiveUniqueNodeNames,
        ConvertDivToMul 
)
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.remove import remove_node_and_rewire
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
import finn.transformation.streamline as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finnbrainsmith.transformation.convert_to_hw_layers as to_bs_hw
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP


def custom_step_qonnx2finn(model, cfg):
    model = model.transform(ExtractQuantScaleZeroPt())
    model = model.transform(ConvertDivToMul())
    model = model.transform(ConvertQONNXtoFINN())
    return model

def custom_streamlining_step(model, cfg):
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(reorder.MoveScalarMulPastMatMul())
    model = model.transform(reorder.MoveScalarLinearPastInvariants())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    return model

def custom_step_infer_hardware(model, cfg):
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(to_hw.InferLookupLayer())
    model = model.transform(to_bs_hw.InferShuffle())
    model = model.transform(to_bs_hw.InferQuantSoftmax())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    return model

def custom_step_specialise_layers(model, cfg):
    model = model.transform(SpecializeLayers(fpgapart=cfg.fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model

def custom_step_create_ip(model, cfg):
    model = model.transform(PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    return model


def custom_step_remove_head(model, cfg):
    """ Removes all nodes up to the first LayerNormalisation Node and then rewires the input """
    assert len(model.graph.input) == 1, "Error the graph has more inputs than expected"
    tensor_to_node = {output: node for node in model.graph.node for output in node.output}

    to_remove = []

    current_tensor = model.graph.input[0].name
    current_node = model.find_consumer(current_tensor)
    while current_node.op_type != "LayerNormalization":
        to_remove.append(current_node)
        assert len(current_node.output) == 1, "Error expected an linear path to the first LN"
        current_tensor = current_node.output[0]
        current_node = model.find_consumer(current_tensor)

    # Send the global input to the consumers of the layernorm output
    LN_output = current_node.output[0]
    consumers = model.find_consumers(LN_output)

    # Remove nodes
    to_remove.append(current_node)
    for node in to_remove:
        model.graph.node.remove(node)

    in_vi = model.get_tensor_valueinfo(LN_output)
    model.graph.input.pop()
    model.graph.input.append(in_vi)
    model.graph.value_info.remove(in_vi)

    # Reconnect input
    for con in consumers:
        for i,ip in enumerate(con.input):
            if ip == LN_output:
                con.input[i] = model.graph.input[0].name

    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())

    return model


def _recurse_model_tail_removal(model, to_remove, node):
    """ Helper function for recursively walking the BERT graph from the second
    output up to the last LayerNorm to remove it """
    if node is not None:
        if node.op_type != "LayerNormalization":
            to_remove.append(node)
            for tensor in node.input:
                _recurse_model_tail_removal(model, to_remove, model.find_producer(tensor))
    return

def custom_step_remove_tail(model, cfg):
    """ Removes from global_out_1 all the way back to the first LayerNorm """
    out_names = [x.name for x in model.graph.output]
    assert "global_out_1" in out_names, "Error: expected one of the outputs to be called global_out_1, we might need better pattern matching logic here"

    to_remove = []
    current_node = model.find_producer('global_out_1')
    _recurse_model_tail_removal(model, to_remove, current_node)

    for node in to_remove:
        model.graph.node.remove(node)
    del model.graph.output[out_names.index('global_out_1')]

    return model

def custom_step_cleanup(model, cfg):
    """ Some custom cleanup steps for the BERT model """
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model

