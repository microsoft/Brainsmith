import torch
from torch import nn
from transformers import BertConfig, BertModel
from transformers import AutoModel
from transformers.utils.fx import symbolic_trace

# Brevitas installed from this PR: https://github.com/Xilinx/brevitas/pull/1090
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
import brevitas.onnx as bo
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.calibrate import calibration_mode


config = BertConfig(
  hidden_size=384,
  num_hidden_layers=1,
  num_attention_heads=12,
  intermediate_size=1536,
  attn_implementation="sdpa",
  hidden_act="relu",
)
model = BertModel(config=config)
model.float()
model.eval()
vocab_size = model.config.vocab_size
seq_len = 128
batch_size = 1 

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

print(model)
print("Replace SDPA with quantizable variants...")
model = replace_sdpa_with_quantizable_layers(model)
print("Replacing done.")
print(model)

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

layerwise_compute_layer_map = {}
layerwise_compute_layer_map[nn.Linear] = (
    qnn.QuantLinear,
    {
        'input_quant': Int8ActPerTensorFloat,
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

quant_model = layerwise_quantize(model, compute_layer_map=layerwise_compute_layer_map)
with torch.no_grad(), calibration_mode(quant_model):
    quant_model(**inp)

with torch.no_grad():
    bo.export_onnx_qcdq(
        quant_model,
        (input_ids),
        'bert-tiny_quant_qcdq.onnx',
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

