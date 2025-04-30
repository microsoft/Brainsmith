############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import warnings
warnings.simplefilter("ignore")
import onnx
import onnxscript
from onnxscript.utils import graph_view_utils as gvu
import os
import argparse
import torch
import json
from torch import nn
from transformers import BertConfig, BertModel
from transformers.utils.fx import symbolic_trace
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
import brevitas.onnx as bo
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.calibrate import calibration_mode
from brainsmith.core.hw_compiler import forge


def gen_initial_bert_model(
        outfile: str = "bert.onnx",
        hidden_size: int = 384,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 12,
        intermediate_size: int = 1536,
        bitwidth: int = 8,
        seqlen: int = 128
        ) -> None:
    """ Generates the initial BERT model from Brevitas. (Write more here) """

    # Global consts used by Brevitas build step
    dtype = torch.float32

    config = BertConfig(
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      attn_implementation="sdpa",
      hidden_act="relu",
    )
    model = BertModel(config=config)
    model.to(dtype=dtype)
    model.eval()
    vocab_size = model.config.vocab_size
    seq_len = seqlen
    batch_size = 1

    input_ids = torch.randint(vocab_size, (batch_size,seq_len), dtype=torch.int64)
    attention_mask = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.float32)
    token_type_ids = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.int64)
    inp = {
        'input_ids': input_ids,
    }

    input_names = inp.keys()

    onnx_model = torch.onnx.export(model, (input_ids), dynamo=True)
    onnx_model.save("bert-large-dynamo.onnx")
    model_proto = onnx.load("bert-large-dynamo.onnx")
    model_ir    = onnxscript.ir.serde.deserialize_model(model_proto)
    graph       = model_ir.graph
    graph = gvu.add_metadata_to_unannotated_constant_nodes(graph)
    model_proto = onnxscript.ir.serde.serialize_model(model_ir)
    onnx.save(model_proto, "bert-large-dynamo-constants-annotated.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="bert-large-dynamo-constants-annotated.onnx.data")

    P = gvu.PytorchHierarchyNode()
    unadded_nodes = []
    for node in graph._nodes:
        added = P.add_node(node)
        if not added:
            unadded_nodes.append(node)
    P.print_hierarchy()
    print(f"Total nodes: {len(graph._nodes)}")
    print(f"Unadded nodes: {len(unadded_nodes)}")
    for node in unadded_nodes:
        print(f"unadded node: {node}")
    layer_0_graph_view = gvu.bGraphView('bert-large-layer0', P.get_nodes(['','encoder','encoder.layer.0']))
    print(f"Layer 0 graph view: {len(layer_0_graph_view._nodes)}")
    single_layer_0_model = onnxscript.ir.Model(layer_0_graph_view, ir_version=10)
    proto = onnxscript.ir.serde.serialize_model(single_layer_0_model)
    onnx.save(proto, 'bert-large-layer-0.onnx')


    exit()

    model = symbolic_trace(model, input_names)

    pre_output = model(**inp)

    print("Replace SDPA with quantizable variants...")
    model = replace_sdpa_with_quantizable_layers(model)
    print("Replacing done.")

    post_output = model(**inp)

    # Sanity check that the layer replacement worked
    #print(pre_output["pooler_output"].shape)
    #print(pre_output["pooler_output"])
    #print(f"{pre_output['pooler_output'].shape} - {post_output['pooler_output'].shape}")
    #print(pre_output['pooler_output'] - post_output['pooler_output'])

    unsigned_hidden_act = config.hidden_act == 'relu'
    layerwise_compute_layer_map = {}
    layerwise_compute_layer_map[nn.Linear] = (
        qnn.QuantLinear,
        {
            # 'input_quant': Int8ActPerTensorFloat,
            'input_quant': lambda module: Uint8ActPerTensorFloat if module.in_features == config.intermediate_size and unsigned_hidden_act else Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'weight_bit_width': bitwidth,
            'output_quant': None,
            'bias_quant': None,
            'return_quant_tensor': False})
    layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
        qnn.QuantScaledDotProductAttention,
        {
            'softmax_input_quant': Int8ActPerTensorFloat,
            'softmax_input_bit_width': bitwidth,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'attn_output_weights_bit_width': bitwidth,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'q_scaled_bit_width': bitwidth,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'k_transposed_bit_width': bitwidth,
            'v_quant': Int8ActPerTensorFloat,
            'v_bit_width': bitwidth,
            'attn_output_quant': None,
            'return_quant_tensor': False})
    layerwise_compute_layer_map[nn.Tanh] = (
        qnn.QuantTanh,
        {
            'input_quant': None,
            'act_quant': Int8ActPerTensorFloat,
            'act_bit_width': bitwidth,
            'return_quant_tensor': False})

    quant_model = layerwise_quantize(model, compute_layer_map=layerwise_compute_layer_map)
    quant_model.to(dtype=dtype)
    print("calibrating...")
    with torch.no_grad(), calibration_mode(quant_model):
        quant_model(**inp)
    print("exporting...")
    with torch.no_grad():
        bo.export_qonnx(
            quant_model,
            (input_ids),
            outfile,
            do_constant_folding=True,
            input_names=['input_ids'],
            opset_version=17,
        )


def main(args):
    # TODO: Replace this "save and delete" with proper optional saving
    tmp_model_path = os.path.join(os.environ.get("BSMITH_BUILD_DIR"), "initial.onnx")
    # Initial model generation
    gen_initial_bert_model(
        outfile=tmp_model_path,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        bitwidth=args.bitwidth,
        seqlen=args.seqlen
    )
    print("done generating initial model")
    exit()
    model = onnx.load(tmp_model_path)
    if os.path.exists(tmp_model_path):
        os.remove(tmp_model_path)

    # Run Brainsmith bert job on the generated model
    forge('bert', model, args)

    # Extra metadata for handover
    build_dir = os.path.join(os.environ.get("BSMITH_BUILD_DIR"), args.output)
    handover_file = build_dir + '/stitched_ip/shell_handover.json'
    
    if os.path.exists(handover_file):
        with open(handover_file, "r") as fp:
            handover = json.load(fp)
        handover['num_layers'] = args.num_hidden_layers
        with open(handover_file, "w") as fp:
            json.dump(handover, fp, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBERT FINN demo script')
    parser.add_argument('-o', '--output', help='Output build name', required=True)
    parser.add_argument('-z', '--hidden_size', type=int, default=384, help='Sets BERT hidden_size parameter')
    parser.add_argument('-n', '--num_attention_heads', type=int, default=12, help='Sets BERT num_attention_heads parameter')
    parser.add_argument('-l', '--num_hidden_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-i', '--intermediate_size', type=int, default=1536, help='Sets BERT intermediate_size parameter')
    parser.add_argument('-b', '--bitwidth', type=int, default=8, help='The quantization bitwidth (either 4 or 8)')
    parser.add_argument('-f', '--fps', type=int, default=3000, help='The target fps for auto folding')
    parser.add_argument('-c', '--clk', type=float, default=3.33, help='The target clock rate for the hardware')
    parser.add_argument('-s', '--stop_step', type=str, default=None, help='Step to stop at in the build flow')
    parser.add_argument('-p', '--param', type=str, default=None, help='Use a preconfigured file for the folding parameters')
    parser.add_argument('-x', '--fifodepth', type=bool, default=True, help='Skip the FIFO depth stage')
    parser.add_argument('-q', '--seqlen', type=int, default=128, help='Sets the sequence length parameter')
    parser.add_argument('-d', '--dcp', type=bool, default=True, help='Generate a DCP')
    args = parser.parse_args()

    # TODO: Properly parameterize these currently hardcoded values
    args.save_intermediate = True
    args.standalone_thresholds = True
    args.fifosim_n_inferences = 2
    args.board = "V80"
    args.verification_atol = 1e-1
    args.split_large_fifos = True

    main(args)
