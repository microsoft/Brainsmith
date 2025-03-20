############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import onnx
import argparse
import os
import shutil
import json
from onnxsim import simplify
import qonnx.custom_op.registry as registry
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
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.builder.build_dataflow_config import DataflowOutputType
import finn.transformation.streamline as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finnbrainsmith.transformation.convert_to_hw_layers as to_bs_hw
from finnbrainsmith.transformation.expand_norms import ExpandNorms
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

# Included for getting reference IO from model with head/tail removed
import finn.core.onnx_exec as oxe
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.core.datatype import DataType
import numpy as np

#Debugging
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim

# Temporary imports - remove once FloatQuant is available
from qonnx.transformation.base import Transformation

def custom_step_qonnx2finn(model, cfg):
    """
    BERT custom step for converting between QONNX and FINN-ONNX

    The SoftMax custom op requires some extra care here, hence
    the requirement for this plugin step.

    QuantSoftMax makes use of the fact that the output
    of SoftMax is well defined between [0,1] so we can
    specify the output as a fixed-point number with 0
    integer bits and N fractional bits (where N is the
    bitwidth of the output datatype).

    For an INT8 model this means we will have:
        SoftMax -> Quant node (scale=1/255)
    in the ONNX model.

    We then call ExtractQuantScaleZeroPt to pull the
    scale calculation out of the Quant. which gives us

        SoftMax -> Div(1/255) -> Quant (scale=1) -> Mul(1/255)

    Then we convert the Div node to a Mul node with
    ConvertDivToMul :
        
        SoftMax -> Mul(255) -> Quant (scale=1) -> Mul(1/255)

    Then we call ConvertQONNXtoFINN to get:
        
        SoftMax -> Mul(255) -> MultiThreshold -> Mul(1/255)

    By having these steps we can have a scale factor of 1
    in the Quant node, then we can deal with the leftover 
    mul nodes later in the streamlining_step streamlining it into
    a MultiThreshold node. (see custom_streamlining_step below) 

    """
    model = model.transform(ExpandNorms())
    #model = model.transform(ExtractQuantScaleZeroPt())
    model = model.transform(FoldConstants())
    model = model.transform(ConvertDivToMul())
    model = model.transform(ConvertQONNXtoFINN())
    return model

def custom_step_generate_reference_io(model, cfg):
    """
    This step is to generate a reference IO pair for the 
    onnx model where the head and the tail have been 
    chopped off.
    """
    input_m = model.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)
    np.save("input.npy", in_tensor)

    input_t = { input_m.name : in_tensor}
    out_name = model.graph.output[0].name

    y_ref = oxe.execute_onnx(model, input_t, True)
    np.save("expected_output.npy", y_ref[out_name])
    np.savez("expected_context.npz", **y_ref) 
    return model


def custom_streamlining_step(model, cfg):
    """
    BERT custom step for streamlining

    Some additional streamlining steps are required here
    to handle the Mul nodes leftover from the SoftMax
    transformations done in custom_step_qonnx2finn.

    In particular, we need to move the Mul operation
    at the output of the QuantSoftMax lower in the graph
    so that it has the option to be merged into a MultiThreshold 
    node. In particular:

        * MoveScalarMulPastMatMul : moves the Mul past the DynMatMul
        * ModeScalarLinearPartInvariants : moves the Mul over the
          reshape and transpose
        * AbsorbMulIntoMultiThreshold : absorbs the Mul into the MT

    """
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(reorder.MoveOpPastFork(["Mul"]))
    model = model.transform(reorder.MoveScalarMulPastMatMul())
    model = model.transform(reorder.MoveScalarLinearPastInvariants())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(InferDataTypes(allow_scaledint_dtypes=False))
    model = model.transform(GiveUniqueNodeNames())
    return model

def custom_step_infer_hardware(model, cfg):
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

class ExtractShellIntegrationMetadata(Transformation):
    """ Walks the ONNX graph and extracts all relevant metadata for shell integration
    handover. """
    def __init__(self, metadata_file:str):
        super().__init__()
        self.metadata_file:str = metadata_file
        self.md = {}

    def apply(self, model):
        graph = model.graph

        # Extract instream widths
        instreams = {}
        for input_tensor in graph.input:
            consumer = model.find_consumer(input_tensor.name)
            inst = registry.getCustomOp(consumer)
            instream = {}
            instream['width'] = inst.get_instream_width() 
            instreams[input_tensor.name] = instream
            instream['shape'] = inst.get_normal_input_shape() 
        self.md['insteams'] = instreams

        # Extract outstream widths
        outstreams = {}
        for output_tensor in graph.output:
            producer = model.find_producer(output_tensor.name)
            inst = registry.getCustomOp(producer)
            outstream = {}
            outstream['width'] = inst.get_outstream_width() 
            outstreams[output_tensor.name] = outstream
            outstream['shape'] = inst.get_normal_output_shape()
        self.md['outsteams'] = outstreams
    
        static_matmuls = {}
        for node in graph.node:
            if (node.op_type == "MVAU_rtl"):
                inst = registry.getCustomOp(node)
                mm = {}
                mm['MH'] = inst.get_nodeattr("MH")
                mm['MW'] = inst.get_nodeattr("MW")
                mm['SIMD'] = inst.get_nodeattr("SIMD")
                mm['PE'] = inst.get_nodeattr("PE")
                static_matmuls[node.name] = mm
        self.md["static_matmuls"] = static_matmuls

        with open(self.metadata_file, "w") as fp:
            json.dump(self.md, fp, indent=4)

        return(model, False)

def custom_step_shell_metadata_handover(model, cfg):
    """ Extracts the metadata for the shell integration process, such as for the v80.
    This information is stored in a json file that is passed to the build process

    It adds this to the stitched_ip output directory and checks it exists ahead of time
    """
    if DataflowOutputType.STITCHED_IP in cfg.generate_outputs:
        if os.path.isdir(cfg.output_dir + '/stitched_ip'):
            model = model.transform(ExtractShellIntegrationMetadata(cfg.output_dir + "/stitched_ip/shell_handover.json"))
            # copy over the ref IO *.npy files into the stitched_ip for handover
            shutil(cfg.verify_input_npy, cfg.output_dir + '/stitched_ip')
            shutil(cfg.verify_expected_output_npy, cfg.output_dir + '/stitched_ip')
            return model
        else:
            raise RuntimeError(f"Error: could not find stitched IP directory so unable to create metadata. Please ensure this is called after the create_stitched_ip step")

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
    #model = model.transform(QuantizeLayerNormalization(
    #    input_datatype ='INT8',
    #    weight_datatype='FLOAT16',
    #    bias_datatype  ='FLOAT16',
    #    output_datatype='FLOAT16')
    #)
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    return model

class SetPumpedCompute(Transformation):
    """ For all MVAUs and DynMatMuls set the pumped compute attribute """
    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph

        for node in graph.node:
            if (node.op_type == "MVAU_rtl"):
                inst = registry.getCustomOp(node)
                inst.set_nodeattr("pumpedCompute", 1)
        return(model, False)


class TempShuffleFixer(Transformation):
    """ A temporary transformation that ensures that shuffles are sized correctly for the
    initial BERT builds """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph

        for node in graph.node:
            if node.op_type == "Shuffle_hls":
                inst = registry.getCustomOp(node)
                inner_moves = inst.get_nodeattr("inner_moves")
                simd = inst.get_nodeattr("SIMD")
                if (inner_moves == 1) and (simd > 1):
                    print(f"WARNING: as a safety precaution changing the shuffle where the inner dimension moves to SIMD=1 \n{node=}")
                    inst.set_nodeattr("SIMD", 1)
        return (model, False)


def custom_step_constrain_folding_and_set_pumped_compute(model, cfg):
    model = model.transform(TempShuffleFixer())
    model = model.transform(SetPumpedCompute())
    return model



class QuantizeLayerNormalization(Transformation):
    """Add quantization to LayerNormalization nodes in the graph. 
    Temporary implementation pending full quantization support in FINN. """

    def __init__(self, input_datatype=None, weight_datatype=None, bias_datatype=None, output_datatype=None):
        super().__init__()
        self.idt = input_datatype
        self.wdt = weight_datatype
        self.bdt = bias_datatype
        self.odt = output_datatype

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        print('Beginning...')
        for node in graph.node:
            print('Outer')
            node_ind += 1
            print(node.name)
            # Detect LayerNorm
            if node.op_type == "LayerNormalization":
                print('Inner')
                # Get tensors
                act_in = node.input[0]
                act_out = node.output[0]
                scale = node.input[1]
                bias = node.input[2] if len(node.input) > 2 else None
                # Datatype annotations
                model.set_tensor_datatype(act_in, DataType[self.idt])
                model.set_tensor_datatype(scale, DataType[self.wdt])
                model.set_tensor_datatype(act_out, DataType[self.odt])
                if bias:
                    model.set_tensor_datatype(bias, DataType[self.bdt])
                graph_modified = True
        return (model, graph_modified)
