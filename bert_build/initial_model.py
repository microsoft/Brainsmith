import math
import torch
from torch import nn
from transformers import BertConfig, BertModel
from transformers import AutoModel
from transformers.utils.fx import symbolic_trace

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
import brevitas.onnx as bo
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers # Requires installation from `dev`
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.calibrate import calibration_mode

bit_width=8
dtype=torch.float32
smax_val=2**(bit_width-1)-1
umax_val=2**(bit_width)-1

class IntActPerTensorFloatConstScale(Int8ActPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    narrow_range=True
    max_val=smax_val
    min_val=-smax_val

class IntWeightPerTensorFloatConstScale(Int8WeightPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    narrow_range=True
    scaling_const=smax_val

class UintActPerTensorFloatConstScale(Uint8ActPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    max_val=umax_val

class UintActPerTensorFloatConstScale1(Uint8ActPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    max_val=1.0

class IntActTanh(Int8ActPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    narrow_range=True
    max_val=1.0
    min_val=-1.0

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

with torch.no_grad():
    for name, module in model.named_modules():
        if type(module) == nn.Linear:
            module.weight *= (smax_val / (module.weight.abs().max() * math.sqrt(float(module.out_features))))

input_ids = torch.randint(vocab_size, (batch_size,seq_len), dtype=torch.int64)
attention_mask = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.float32)
token_type_ids = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.int64)
inp = {
    'input_ids': input_ids,
#    'attention_mask': attention_mask,
#    'token_type_ids': token_type_ids,
}

input_names = inp.keys()
model = symbolic_trace(model, input_names)

pre_output = model(**inp)

print("Replace SDPA with quantizable variants...")
model = replace_sdpa_with_quantizable_layers(model)
print("Replacing done.")

post_output = model(**inp)

# Old version (some old transformers version)
#print(pre_output.pooler_output.shape)
#print(pre_output.pooler_output)
#print(f"{pre_output.pooler_output.shape} - {post_output.pooler_output.shape}")
#print(pre_output.pooler_output - post_output.pooler_output)

# Sanity check that the layer replacement worked
print(pre_output["pooler_output"].shape)
print(pre_output["pooler_output"])
print(f"{pre_output['pooler_output'].shape} - {post_output['pooler_output'].shape}")
print(pre_output['pooler_output'] - post_output['pooler_output'])

unsigned_hidden_act = config.hidden_act == 'relu'
layerwise_compute_layer_map = {}
layerwise_compute_layer_map[nn.Linear] = (
    qnn.QuantLinear,
    {
        #'input_quant': IntActPerTensorFloatConstScale,
        'input_quant': lambda module: UintActPerTensorFloatConstScale if module.in_features == config.intermediate_size and unsigned_hidden_act else IntActPerTensorFloatConstScale,
        'weight_quant': IntWeightPerTensorFloatConstScale,
        'output_quant': None,
        'bias_quant': None,
        'return_quant_tensor': False})
layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
    qnn.QuantScaledDotProductAttention,
    {
        'softmax_input_quant': IntActPerTensorFloatConstScale,
        'attn_output_weights_quant': UintActPerTensorFloatConstScale1,
        'q_scaled_quant': IntActPerTensorFloatConstScale,
        'k_transposed_quant': IntActPerTensorFloatConstScale,
        'v_quant': IntActPerTensorFloatConstScale,
        'attn_output_quant': None,
        'return_quant_tensor': False})
layerwise_compute_layer_map[nn.Tanh] = (
    qnn.QuantTanh,
    {
        'input_quant': None,
        'act_quant': IntActTanh,
        'return_quant_tensor': False})

quant_model = layerwise_quantize(model, compute_layer_map=layerwise_compute_layer_map)
quant_model.to(dtype=dtype)
with torch.no_grad(), calibration_mode(quant_model):
    quant_model(**inp)

with torch.no_grad():
    bo.export_qonnx(
        quant_model,
        (input_ids),
        'bert-tiny_quant_qonnx.onnx',
        do_constant_folding=True,
        input_names=['input_ids'],
        #dynamic_axes={
        #    'input_ids': {
        #        0: 'batch_size',
        #        1: 'sequence_length',
        #    },  
        #},  
        opset_version=17,
    )
