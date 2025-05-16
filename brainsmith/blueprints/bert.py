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
from qonnx.transformation.general import (
        SortCommutativeInputsInitializerLast,
        RemoveUnusedTensors,
        GiveReadableTensorNames,
        GiveUniqueNodeNames,
        ConvertDivToMul
)
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.builder.build_dataflow_config import DataflowOutputType
import finn.transformation.streamline as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import brainsmith.transformation.convert_to_hw_layers as to_bs_hw
from brainsmith.transformation.expand_norms import ExpandNorms

# Included for getting reference IO from model with head/tail removed
import finn.core.onnx_exec as oxe
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.core.datatype import DataType
import numpy as np

# Debugging
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.builder.build_dataflow_steps import (
    step_create_dataflow_partition,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_create_stitched_ip,
    step_measure_rtlsim_performance
)

# Temporary imports - remove once FloatQuant is available
from qonnx.transformation.base import Transformation

import onnxscript
from onnxscript.utils import graph_view_utils as gvu

def custom_step_extract_loop_body(model, cfg):
    """
    BERT custom step for extracting the loop body

    This is a custom step to extract the loop body from the
    BERT model. It is not a standard step in the FINN pipeline,
    but it is useful for this model.
    """
    model = model.transform(FoldConstants())

    model_ir    = onnxscript.ir.serde.deserialize_model(model.model)
    graph       = model_ir.graph

    P = gvu.PytorchHierarchyNode()
    unadded_nodes = []
    for node in graph._nodes:
        added = P.add_node(node)
        if not added:
            unadded_nodes.append(node)
    P.print_hierarchy()
    print(f"Total nodes: {len(graph._nodes)}")
    print(f"Unadded nodes: {len(unadded_nodes)}")

    # Handle the unadded Transpose nodes as a special case for BERT
    # Todo: Make this more robust in the future
    for node in unadded_nodes:
        print(f"added metadata for node {node.name}")
        pred_node = node.predecessors()[0]
        node.metadata_props['pkg.torch.onnx.name_scopes'] = pred_node.metadata_props['pkg.torch.onnx.name_scopes']
        node.metadata_props['pkg.torch.onnx.class_hierarchy'] = pred_node.metadata_props['pkg.torch.onnx.class_hierarchy']
        assert(P.add_node(node))
    loop_body_graph_view = gvu.bGraphView(f'loop-body', P.get_nodes(cfg.loop_body_hierarchy))
    print(f"Layer 0 graph view: {len(loop_body_graph_view._nodes)}")
    loop_body_model = onnxscript.ir.Model(loop_body_graph_view, ir_version=10)
    proto = onnxscript.ir.serde.serialize_model(loop_body_model)
    onnx.save(proto, cfg.output_dir+'/loop-body-template.onnx')
    return model


from onnxscript.rewriter import pattern
from onnxscript.rewriter import pattern_builder_jsm as pb
from onnxscript.rewriter import rewrite

def custom_step_loop_rolling(model, cfg):
    """
    BERT custom step for loop rolling

    This is a custom step to roll the loops in the BERT model
    to make it easier to work with. It is not a standard step
    in the FINN pipeline, but it is useful for this model.
    """

    print("Loading loop body template")
    LoopBody = pb.LoopBodyTemplate(cfg.output_dir+'/loop-body-template.onnx')

    # Replace instances of the loop body with a function call to the loop body
    change_layers_to_function_calls = pattern.RewriteRule(
      LoopBody.pattern,
      LoopBody.function_replace
    )
    print("Replacing layers with function calls")

    model_proto = model.model
    model_ir = onnxscript.ir.serde.deserialize_model(model_proto)

    model_layers_replaced = rewrite(
        model_ir,
        pattern_rewrite_rules = [change_layers_to_function_calls]
    )

    model_layers_replaced.functions[LoopBody.function.identifier()] = LoopBody.function
    model_layers_replaced.graph.opset_imports['loop']=0

    model_proto = onnxscript.ir.serde.serialize_model(model_layers_replaced)

    model.model = model_proto

    normalized_graph = pb.normalize_io_for_loop_rolling(model_layers_replaced.graph, LoopBody)

    print(f"normalized graphs is layer {normalized_graph is model_layers_replaced.graph}")
    onnxscript.ir.save(model_layers_replaced, "normalized.onnx")
    LoopMatchPattern,nodes = LoopBody.build_function_match_pattern(normalized_graph)

    loop_replace_pattern = pb.build_loop_replace_pattern(normalized_graph, LoopBody)

    change_function_calls_to_loop = pattern.RewriteRule(
        LoopMatchPattern,
        loop_replace_pattern
    )
    rewrite_set = pattern.RewriteRuleSet([change_function_calls_to_loop])
    count = rewrite_set.apply_to_model(model_layers_replaced, verbose=None)
    print(f"Rolled {count} function calls into a loop operator")
    model.model = onnxscript.ir.serde.serialize_model(model_layers_replaced)

    model = model.transform(FoldConstants())
    return model


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
    model = model.transform(ExtractQuantScaleZeroPt())
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
    np.save(cfg.output_dir+"/input.npy", in_tensor)

    input_t = { input_m.name : in_tensor}
    out_name = model.graph.output[0].name

    y_ref = oxe.execute_onnx(model, input_t, True)
    np.save(cfg.output_dir+"/expected_output.npy", y_ref[out_name])
    np.savez(cfg.output_dir+"/expected_context.npz", **y_ref)
    return model

from qonnx.transformation.general import SortGraph
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
    model = model.transform(SortGraph())
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
    model = model.transform(to_hw.InferFinnLoopOp())
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
            shutil.copy(cfg.verify_input_npy, cfg.output_dir + '/stitched_ip')
            shutil.copy(cfg.verify_expected_output_npy, cfg.output_dir + '/stitched_ip')
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
        return (model, False)


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

BUILD_BERT_STEPS = [
        # Cleanup and custom graph surgery
        custom_step_cleanup,
        custom_step_remove_head,
        custom_step_remove_tail,
        custom_step_qonnx2finn,

        custom_step_generate_reference_io,
        custom_streamlining_step,
        custom_step_extract_loop_body,
        custom_step_loop_rolling,
        custom_step_infer_hardware,
        step_create_dataflow_partition,
        step_specialize_layers,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_hw_codegen,
        step_hw_ipgen,
        step_measure_rtlsim_performance,
        step_set_fifo_depths,
        step_create_stitched_ip,
        custom_step_shell_metadata_handover,
    ]
