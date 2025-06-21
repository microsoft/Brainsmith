#!/usr/bin/env python3
"""Minimal test to identify where the hang occurs"""

import warnings
warnings.simplefilter("ignore")
import os
import sys

print("Starting minimal BERT generation test...")

# Create minimal build directory
build_dir = "/tmp/bert_test"
os.makedirs(build_dir, exist_ok=True)

print("1. Importing required modules...")
from demos.bert_direct.end2end_bert_direct import generate_bert_model

print("2. Calling generate_bert_model...")
try:
    model_path = generate_bert_model(
        output_dir=build_dir,
        hidden_size=384,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=1536,
        bitwidth=8,
        seqlen=128
    )
    print(f"✅ Model generated successfully: {model_path}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)