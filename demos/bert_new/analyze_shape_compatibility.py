#!/usr/bin/env python3
"""Analyze what folding values would create shape-compatible connections."""

import json

def analyze_compatibility():
    """Analyze BERT model dimensions and suggest compatible folding."""
    
    # BERT model dimensions
    hidden_size = 384
    intermediate_size = 1536
    num_heads = 12
    head_dim = hidden_size // num_heads  # 32
    
    print("BERT Model Dimensions:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print()
    
    print("Shape Compatibility Analysis:")
    print("=" * 60)
    
    # For Thresholding -> MVAU connections
    print("\n1. Thresholding -> MVAU (Query/Key/Value projections)")
    print(f"   Input: {hidden_size} features")
    print(f"   Output: {hidden_size} features")
    
    # For PE=1 Thresholding, output shape is (1, 128, 384, 1)
    # For MVAU with SIMD=12, input needs to be (1, 128, 32, 12)
    # This means we need 384 = 32 * 12
    
    print(f"   Required: {hidden_size} = MW * SIMD")
    print(f"   With SIMD=12: MW = {hidden_size // 12} = 32 ✓")
    print(f"   Thresholding PE must divide {hidden_size} evenly")
    
    # The issue is that Thresholding with PE=1 outputs (B, N, C, 1)
    # But MVAU expects (B, N, MW, SIMD) where MW*SIMD = C
    
    print("\n2. Compatible folding options:")
    print("   Option A: Insert StreamingDataWidthConverter (automatic)")
    print("   Option B: Use Thresholding PE that matches MVAU chunking")
    
    # For option B, Thresholding PE should be a divisor of hidden_size
    # that creates compatible chunking with MVAU SIMD
    
    print(f"\n   For hidden_size={hidden_size}:")
    divisors = [i for i in range(1, hidden_size + 1) if hidden_size % i == 0]
    compatible = []
    for pe in divisors:
        if pe <= 64:  # Reasonable PE values
            chunk_size = hidden_size // pe
            if chunk_size == 12 or 12 % chunk_size == 0 or chunk_size % 12 == 0:
                compatible.append(pe)
    
    print(f"   Compatible Thresholding PE values: {compatible}")
    
    print("\n3. Recommended folding:")
    print("   - Thresholding PE: 12 or 4 (divides evenly into SIMD)")
    print("   - MVAU SIMD: 12, PE: 8")
    print("   - This ensures shape compatibility without converters")

if __name__ == "__main__":
    analyze_compatibility()