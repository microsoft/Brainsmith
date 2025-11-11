############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

from .roofline import roofline_analysis

# DLRM Model definitions
dlrm_params = {  # Parameters common to all BERT models
    "attn_kernel_fusion": True,
    "batch": 1,
    "tile_size": 1,
}
dlrmv2 = {
    "offload": False,
    "arch": "dlrm",
    "num_layers": 1,
    "seq_len": 128,
    "emb_dim": 128,
    "h1": 128,
    "h2": 128,
    "spa_dim": 128,
    "emb_dims": [1000, 5000, 3000],
    "mlp_bottom": [13, 512, 256, 128],
    "mlp_top": [512, 256, 1],
    "interaction_op": "dot",
    "num_heads": 12,
    "head_size": 32,
    "intermediate": 4 * 12 * 32,
}

# BERT Model definitions
bert_params = {  # Parameters common to all BERT models
    "attn_kernel_fusion": True,
    "batch": 1,
    "tile_size": 1,
}
bert_tiny = {
    "offload": False,
    "arch": "bert",
    "num_layers": 3,
    "seq_len": 128,
    "num_heads": 12,
    "head_size": 32,
    "intermediate": 4 * 12 * 32,
}
twin_bert1 = {
    "offload": False,
    "hard_batch": 40,
    "arch": "bert",
    "num_layers": 3,
    "seq_len": 64,
    "num_heads": 12,
    "head_size": 32,
    "intermediate": 4 * 12 * 32,
}
twin_bert2 = {
    "offload": False,
    "arch": "bert",
    "num_layers": 6,
    "seq_len": 216,
    "num_heads": 12,
    "head_size": 32,
    "intermediate": 4 * 12 * 32,
}
twin_bert = {
    "offload": False,
    "arch": "twin_bert",
    "model_1": twin_bert1,
    "model_2": twin_bert2,
    "seq_len": 64,
}
bert_large_64 = {
    "offload": True,
    "arch": "bert",
    "num_layers": 24,
    "seq_len": 64,
    "num_heads": 16,
    "head_size": 64,
    "intermediate": 4 * 16 * 64,
}
bert_large_512 = {
    "offload": True,
    "arch": "bert",
    "num_layers": 24,
    "seq_len": 512,
    "num_heads": 16,
    "head_size": 64,
    "intermediate": 4 * 16 * 64,
}
bert_large_1024 = {
    "offload": True,
    "arch": "bert",
    "num_layers": 24,
    "seq_len": 1024,
    "num_heads": 16,
    "head_size": 64,
    "intermediate": 4 * 16 * 64,
}

# SLM Model definitions
slm_params = {  # Parameters common to all SLM models
    "offload": True,
    "attn_kernel_fusion": True,
    "tile_size": 1,
}
mistral_2k = {
    "num_layers": 32,
    "seq_len": 2048,
    "out_len": 256,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_size": 128,
    "intermediate": 14336,
    "window_size": 4096,
}
mistral_4k = {
    "num_layers": 32,
    "seq_len": 4096,
    "out_len": 256,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_size": 128,
    "intermediate": 14336,
    "window_size": 4096,
}
phi3_mini_4k_1k = {
    "num_layers": 32,
    "seq_len": 1024,
    "out_len": 256,
    "num_heads": 32,
    "head_size": 96,
    "intermediate": 8192,
    "window_size": 2047,
}
phi3_mini_4k_2k = {
    "num_layers": 32,
    "seq_len": 2048,
    "out_len": 256,
    "num_heads": 32,
    "head_size": 96,
    "intermediate": 8192,
    "window_size": 2047,
}

mistral_pp_batch1 = {
    "arch": "slm_pp",
    "batch": 1,
    "spill_size": 128,
    "spill_map": [
        True,  # QKV
        False,  # MHA Score
        False,  # MHA Attn
        True,  # MHA Out
        True,  # MLP Up
        True,  # MLP Down
    ],
    "residuals_in_hbm": True,
}
mistral_tg_batch1 = {
    "arch": "slm_tg",
    "batch": 1,
    "spill_size": 128,
    "spill_map": [
        False,  # QKV
        True,  # MHA Score
        False,  # MHA Attn
        True,  # MHA Out
        True,  # MLP Up
        True,  # MLP Down
    ],
}
mistral_pp_batch32 = {
    "arch": "slm_pp",
    "batch": 32,
    "chunk_size": 64,
    "spill_size": 64,
    "spill_map": [
        True,  # QKV
        False,  # MHA Score
        False,  # MHA Attn
        True,  # MHA Out
        True,  # MLP Up
        True,  # MLP Down
    ],
}
mistral_tg_batch32 = {
    "arch": "slm_tg",
    "batch": 32,
    "spill_size": 64,
    "spill_map": [
        False,  # QKV
        True,  # MHA Score
        False,  # MHA Attn
        True,  # MHA Out
        True,  # MLP Up
        True,  # MLP Down
    ],
}

phi3_pp_batch1 = {
    "arch": "slm_pp",
    "batch": 1,
    "spill_size": 64,
    "spill_map": [
        False,  # QKV
        False,  # MHA Score
        False,  # MHA Attn
        False,  # MHA Out
        True,  # MLP Up
        True,  # MLP Down
    ],
}
phi3_tg_batch1 = {
    "arch": "slm_tg",
    "batch": 1,
    "spill_size": 128,
    "spill_map": [
        True,  # QKV
        False,  # MHA Score
        False,  # MHA Attn
        False,  # MHA Out
        False,  # MLP Up
        True,  # MLP Down
    ],
}
phi3_pp_batch32 = {
    "arch": "slm_pp",
    "batch": 32,
    "chunk_size": 128,
    "spill_size": 64,
    "spill_map": [
        True,  # QKV
        False,  # MHA Score
        False,  # MHA Attn
        True,  # MHA Out
        True,  # MLP Up
        True,  # MLP Down
    ],
}
phi3_tg_batch32 = {
    "arch": "slm_tg",
    "batch": 32,
    "spill_size": 64,
    "spill_map": [
        False,  # QKV
        False,  # MHA Score
        False,  # MHA Attn
        True,  # MHA Out
        True,  # MLP Up
        False,  # MLP Down
    ],
}

# Device parameters
v80 = {
    "luts": 2574208,
    "dsps": 10848,
    "lut_util": 0.6,
    "dsp_util": 0.9,
    "lut_hz": 500e6,
    "dsp_hz": 500e6,
    "hbm_bw": 820 * 8e9,
    "hbm_util": 0.9,
    # TAFK TODO: To check
    "dram_bw": 40 * 8e9,
    "dram_util": 0.9,
    "hbm_bw_slr0": 600 * 8e9,
    "hbm_bw_slr1": 60 * 8e9,
    "hbm_bw_slr2": 60 * 8e9,
    "sram": 84.125 * 8e6,
    "sram_util": 0.9,
    "dsp_4bit": 4,
    "dsp_8bit": 3,
}
u250 = {
    "luts": 1728e3,
    "dsps": 12288,
    "lut_util": 0.6,
    "dsp_util": 0.9,
    "lut_hz": 250e6,
    "dsp_hz": 500e6,
    "hbm_bw": 77 * 8e9,
    "hbm_util": 0.9,
    "sram": 54 * 8e6,
    "sram_util": 0.9,
    "dsp_4bit": 4,
    "dsp_8bit": 2,
}
u55c = {
    "luts": 1304e3,
    "dsps": 9024,
    "lut_util": 0.6,
    "dsp_util": 0.9,
    "lut_hz": 250e6,
    "dsp_hz": 500e6,
    "hbm_bw": 460 * 8e9,
    "hbm_util": 0.9,
    "sram": 3.409e8,
    "sram_util": 0.9,
    "dsp_4bit": 4,
    "dsp_8bit": 2,
}

model_params = {}
model_params.update(bert_params)  # Architecture shared params
model_params.update(bert_large_512)  # Model specific params
# model_params.update(slm_params) # Architecture shared params
# model_params.update(mistral_4k) # Model specific params
# model_params.update(mistral_tg_batch1) # MLO optimizations
hw_params = v80
# hw_params['lut_util'] = 0.0 # Disable LUT compute
dtypes = [8, 4]

roofline_analysis(model_params, hw_params, dtypes)
