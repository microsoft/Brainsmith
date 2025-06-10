"""
Data Layout-Based Chunking Demo

Demonstrates Interface-Wise Dataflow Modeling blocking strategy:
- tensor_dims → blocked into → num_blocks pieces of shape block_dims
- Chunking determined by ONNX tensor layout
- Batch dimension excluded from blocking calculations
"""

import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock

# Mock dependencies
sys.modules['brainsmith.dataflow.core.auto_hw_custom_op'] = Mock()
sys.modules['brainsmith.dataflow.core.interface_metadata'] = Mock()  
sys.modules['brainsmith.dataflow.core.dataflow_interface'] = Mock()
sys.modules['brainsmith.dataflow.core.tensor_blocking'] = Mock()

# Add the generated module to Python path
sys.path.insert(0, str(Path(__file__).parent / "hwkg_demo_final"))

def demonstrate_layout_blocking():
    """Demonstrate the new data layout-based blocking strategy."""
    
    print("=" * 60)
    print("DATA LAYOUT-BASED CHUNKING DEMONSTRATION")
    print("=" * 60)
    print("Showing Interface-Wise Dataflow Modeling blocking strategy")
    print()
    
    # === AXIOM DEMONSTRATION ===
    print("📏 CORE AXIOM: tensor_dims → num_blocks × block_dims")
    print("-" * 40)
    print("• tensor_dims = full tensor shape (no batch dimension)")
    print("• num_blocks = number of blocks")
    print("• block_dims = shape of each block")
    print("• Chunking determined by ONNX layout")
    print()
    
    # === LAYOUT-BASED CHUNKING EXAMPLES ===
    print("🎯 LAYOUT-BASED CHUNKING EXAMPLES")
    print("-" * 40)
    
    examples = [
        # (name, tensor_shape, onnx_layout, description)
        ("CNN Standard", (1, 64, 56, 56), "[N, C, H, W]", "Chunk along channels"),
        ("CNN TensorFlow", (1, 56, 56, 64), "[N, H, W, C]", "Chunk along spatial"),
        ("Transformer", (1, 512, 768), "[N, L, C]", "Chunk along sequence"),
        ("BERT Inverted", (1, 768, 512), "[N, C, L]", "Chunk along features"),
        ("Multi-Head", (1, 512, 12, 64), "[N, L, h, d]", "Chunk along sequence"),
    ]
    
    print(f"{'Model Type':<15} {'Tensor Shape':<18} {'Layout':<12} {'tensor_dims':<20} {'block_dims':<15}")
    print("-" * 85)
    
    for name, shape, layout, description in examples:
        tensor_dims, block_dims = calculate_blocking_from_layout(shape, layout)
        print(f"{name:<15} {str(shape):<18} {layout:<12} {str(tensor_dims):<20} {str(block_dims):<15}")
    
    print()
    
    # === DETAILED CHUNKING CALCULATION ===
    print("🔬 DETAILED CHUNKING CALCULATION")
    print("-" * 40)
    
    # Use CNN example for detailed breakdown
    cnn_shape = (1, 64, 56, 56)
    cnn_layout = "[N, C, H, W]"
    
    print(f"Example: {cnn_shape} with layout {cnn_layout}")
    print()
    
    N, C, H, W = cnn_shape
    tensor_dims = (C, H, W)  # FULL tensor shape excluding batch
    block_dims = (1, H, W)  # Each block is 1 channel × H × W
    max_blocks = C    # Maximum blocks limited by blocking dimension
    
    print(f"Step-by-step calculation:")
    print(f"1. Input tensor: {cnn_shape}")
    print(f"2. Remove batch dimension: {tensor_dims}")
    print(f"3. Layout {cnn_layout} → block along C dimension")
    print(f"4. tensor_dims = {tensor_dims} (FULL tensor shape)")
    print(f"5. block_dims = {block_dims} (shape of each block)")
    print(f"6. Max blocks = {max_blocks} (limited by C dimension)")
    print()
    
    # Show parallelism implications
    print(f"Parallelism implications:")
    print(f"• tensor_dims = {tensor_dims} (full tensor)")
    print(f"• Maximum blocks available: {max_blocks}")
    print(f"• Each block shape: {block_dims}")
    print(f"• 4x parallelism: {max_blocks//4} blocks per unit")
    print(f"• 128x parallelism: wasteful (only {max_blocks} blocks available)")
    print()
    
    # === PARALLELISM BOUNDS DEMONSTRATION ===
    print("⚡ PARALLELISM BOUNDS")
    print("-" * 40)
    
    print("How blocking dimension limits useful parallelism:")
    print()
    print(f"{'Parallelism':<12} {'Chunks/Unit':<12} {'Utilization':<12} {'Efficiency'}")
    print("-" * 50)
    
    for p in [1, 4, 16, 64, 128]:
        if p <= max_blocks:
            blocks_per_unit = max_blocks // p
            remainder = max_blocks % p
            utilization = f"{blocks_per_unit}+{remainder}" if remainder else str(blocks_per_unit)
            efficiency = 100
        else:
            utilization = "N/A"
            efficiency = (max_blocks / p) * 100
        
        print(f"{p}x{'':<10} {utilization:<12} {efficiency:<12.0f}% {'Optimal' if p <= max_blocks and max_blocks % p == 0 else 'Wasteful' if p > max_blocks else 'Good'}")
    
    print()
    
    # === MEMORY SCALING ===
    print("💾 MEMORY SCALING")
    print("-" * 40)
    
    print("Memory scales with block_dims × parallel units:")
    print()
    print(f"{'Parallelism':<12} {'Memory/Unit':<12} {'Total Memory':<14} {'Bandwidth'}")
    print("-" * 50)
    
    bytes_per_element = 2  # 16-bit
    block_dims_elements = np.prod(block_dims)  # Total elements in each block
    
    for p in [1, 4, 16, 64]:
        if p <= max_blocks:
            memory_per_unit = block_dims_elements * bytes_per_element
            total_memory = memory_per_unit * p
            bandwidth_mb_s = total_memory * 100 / (1024 * 1024)  # At 100 MHz
            
            print(f"{p}x{'':<10} {memory_per_unit:<12,} B {total_memory/1024:<14.0f} KB {bandwidth_mb_s:<8.1f} MB/s")
    
    print()
    
    # === LAYOUT COMPARISON ===
    print("🔄 LAYOUT COMPARISON")
    print("-" * 40)
    
    # Same data, different layouts
    tensor_data = (1, 64, 56, 56)  # Same total elements
    layouts = [
        ("[N, C, H, W]", "Standard CNN", 64, 3136),
        ("[N, H, W, C]", "TensorFlow CNN", 3136, 64),
    ]
    
    print("Same tensor, different layouts:")
    print(f"Tensor shape: {tensor_data}")
    print()
    print(f"{'Layout':<15} {'Description':<15} {'tensor_dims':<8} {'block_dims':<8} {'Best For'}")
    print("-" * 60)
    
    for layout, desc, tensor_dims, block_dims in layouts:
        best_for = "Channel parallel" if tensor_dims == 64 else "Spatial parallel"
        print(f"{layout:<15} {desc:<15} {tensor_dims:<8} {block_dims:<8} {best_for}")
    
    print()
    print("Key insight: Layout choice affects parallelism opportunities!")
    print()
    
    # === SUMMARY ===
    print("📋 SUMMARY: Data Layout-Based Chunking")
    print("-" * 40)
    
    print("✅ Core principles:")
    print("1. tensor_dims → num_blocks × block_dims relationship")
    print("2. Batch dimension excluded from blocking")
    print("3. ONNX layout determines blocking strategy")
    print("4. tensor_dims bounds maximum useful parallelism")
    print("5. Memory scales with block_dims × parallel units")
    print()
    print("✅ Benefits:")
    print("• Automatic blocking - no manual strategy selection")
    print("• Predictable performance based on tensor_dims/block_dims")
    print("• Layout-optimized for hardware efficiency")
    print("• Runtime extraction from tensor shapes")
    print()
    
    print("=" * 60)
    print("✅ DEMO COMPLETE!")
    print("Data layout-based blocking provides automatic, efficient tensor processing.")
    print("=" * 60)

def calculate_blocking_from_layout(tensor_shape, onnx_layout):
    """Calculate tensor_dims and block_dims from tensor shape and ONNX layout.
    
    tensor_dims = FULL tensor shape (excluding batch dimension)
    block_dims = shape of each block after blocking
    """
    # tensor_dims is ALWAYS the full tensor shape excluding batch dimension
    tensor_dims = tensor_shape[1:]  # Remove batch dimension
    
    if onnx_layout == "[N, C]":
        N, C = tensor_shape
        block_dims = (C,)  # Single block contains all C elements
    elif onnx_layout == "[N, C, H, W]":
        N, C, H, W = tensor_shape  
        block_dims = (1, H, W)  # Each block is 1 channel × H × W
    elif onnx_layout == "[N, H, W, C]":
        N, H, W, C = tensor_shape
        block_dims = (1, 1, C)  # Each block is 1 spatial location × C
    elif onnx_layout == "[N, L, C]":
        N, L, C = tensor_shape
        block_dims = (1, C)  # Each block is 1 sequence position × C
    elif onnx_layout == "[N, C, L]":
        N, C, L = tensor_shape
        block_dims = (1, L)  # Each block is 1 feature × L
    elif onnx_layout == "[N, L, h, d]":
        N, L, h, d = tensor_shape
        block_dims = (1, h, d)  # Each block is 1 sequence position × h × d
    else:
        # Default fallback - single block
        block_dims = tensor_dims
    
    return tensor_dims, block_dims

if __name__ == "__main__":
    demonstrate_layout_blocking()