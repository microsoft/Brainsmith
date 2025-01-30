############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import os  
import pytest  
import onnx  
from onnxsim import simplify  
from qonnx.util.cleanup import cleanup
from qonnx.core.modelwrapper import ModelWrapper
  
import onnx
import os
import pytest
import shutil
import argparse
import math
import torch
import tempfile
from torch import nn
from transformers import BertConfig, BertModel
from transformers import AutoModel
from transformers.utils.fx import symbolic_trace

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
import brevitas.onnx as bo
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.calibrate import calibration_mode



def gen_initial_bert_model(
        outfile:str="bert.onnx",
        hidden_size:int=384,
        num_attention_heads:int=12,
        intermediate_size:int=1536
        )->None:
    """ Generates the initial BERT model from Brevitas. (Write more here) """
    dtype = torch.float32
    config = BertConfig(
      hidden_size=384,
      num_hidden_layers=1,
      num_attention_heads=12,
      intermediate_size=1536,
      attn_implementation="sdpa",
      hidden_act="relu",
    )
    model = BertModel(config=config)
    model.to(dtype=dtype)
    model.eval()
    vocab_size = model.config.vocab_size
    seq_len = 128
    batch_size = 1
    
    input_ids = torch.randint(vocab_size, (batch_size,seq_len), dtype=torch.int64)
    attention_mask = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.float32)
    token_type_ids = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.int64)
    inp = {
        'input_ids': input_ids,
    }
    
    input_names = inp.keys()
    model = symbolic_trace(model, input_names)
    
    pre_output = model(**inp)
    
    print("Replace SDPA with quantizable variants...")
    model = replace_sdpa_with_quantizable_layers(model)
    print("Replacing done.")
    
    post_output = model(**inp)
    
    unsigned_hidden_act = config.hidden_act == 'relu'
    layerwise_compute_layer_map = {}
    layerwise_compute_layer_map[nn.Linear] = (
        qnn.QuantLinear,
        {
            #'input_quant': Int8ActPerTensorFloat,
            'input_quant': lambda module: Uint8ActPerTensorFloat if module.in_features == config.intermediate_size and unsigned_hidden_act else Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'output_quant': None,
            'bias_quant': None,
            'return_quant_tensor': False})
    layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
        qnn.QuantScaledDotProductAttention,
        {
            'softmax_input_quant': Int8ActPerTensorFloat,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'v_quant': Int8ActPerTensorFloat,
            'attn_output_quant': None,
            'return_quant_tensor': False})
    layerwise_compute_layer_map[nn.Tanh] = (
        qnn.QuantTanh,
        {
            'input_quant': None,
            'act_quant': Int8ActPerTensorFloat,
            'return_quant_tensor': False})

    quant_model = layerwise_quantize(model, compute_layer_map=layerwise_compute_layer_map)
    quant_model.to(dtype=dtype)
    with torch.no_grad(), calibration_mode(quant_model):
        quant_model(**inp)

    with torch.no_grad():
        bo.export_qonnx(
            quant_model,
            (input_ids),
            outfile,
            do_constant_folding=True,
            input_names=['input_ids'],
            opset_version=17,
        )


def create_dynamic_fixtures(step_functions, globals_dict, cfg):  
    for i, step_func in enumerate(step_functions):  
        # Define the fixture function  
        def fixture_func(request, step_func=step_func, prev_fixture_name=step_functions[i-1].__name__ if i > 0 else 'model'):  
            prev_fixture = request.getfixturevalue(prev_fixture_name)  
            return step_func(prev_fixture, cfg)  
  
        # Assign the fixture function to the module scope  
        fixture_func.__name__ = step_func.__name__  
        fixture_func = pytest.fixture(scope='module')(fixture_func)  
  
        # Add the fixture to the provided globals dictionary  
        globals_dict[step_func.__name__] = fixture_func  
  
        # Debugging output  
        print(f"Fixture created: {step_func.__name__}")  

# Fixture for building the initial model  
@pytest.fixture(scope='module')  
def model(  
        hidden_size: int = 384,  
        num_attention_heads: int = 12,  
        intermediate_size: int = 1536,  
        gen_ip: bool = False  
    ):  
    tmp = "./intermediate_models"  
    os.makedirs(tmp, exist_ok=True)  
  
    # Initial model generation  
    gen_initial_bert_model(  
        outfile=f"{tmp}/initial.onnx",  
        hidden_size=hidden_size,  
        num_attention_heads=num_attention_heads,  
        intermediate_size=intermediate_size  
    )  
  
    # Initial model cleanup  
    model = onnx.load(f"{tmp}/initial.onnx")  
    model_simp, check = simplify(model)  
    if check:  
        onnx.save(model_simp, f"{tmp}/simp.onnx")  
    else:  
        raise RuntimeError("Unable to simplify the Brevitas bert model")  
    cleanup(in_file=f"{tmp}/simp.onnx", out_file=f"{tmp}/qonnx_cleanup.onnx")  
  
    return ModelWrapper(f"{tmp}/qonnx_cleanup.onnx")  

