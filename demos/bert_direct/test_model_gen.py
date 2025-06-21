#!/usr/bin/env python3
"""Quick test to check if BERT model generation works"""

import warnings
warnings.simplefilter("ignore")
import os
import sys

print("Testing BERT model generation...")

try:
    print("1. Importing torch...")
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
    
    print("2. Importing transformers...")
    from transformers import BertConfig, BertModel
    print("   ✅ Transformers imported")
    
    print("3. Creating small BERT config...")
    config = BertConfig(
        hidden_size=384,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=1536,
        attn_implementation="sdpa",
        hidden_act="relu",
    )
    print("   ✅ Config created")
    
    print("4. Creating BERT model...")
    model = BertModel(config=config)
    print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    print("5. Testing forward pass...")
    input_ids = torch.randint(model.config.vocab_size, (1, 128), dtype=torch.int64)
    with torch.no_grad():
        output = model(input_ids)
    print(f"   ✅ Forward pass successful, output shape: {output.last_hidden_state.shape}")
    
    print("6. Testing quantization imports...")
    import brevitas
    print(f"   ✅ Brevitas version: {brevitas.__version__}")
    
    print("\n✅ All basic imports and model creation work correctly!")
    
except Exception as e:
    print(f"\n❌ Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)