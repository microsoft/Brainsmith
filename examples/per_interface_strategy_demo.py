"""
Demo of Per-Interface Chunking Strategy Architecture

This demonstrates the improved architecture where each interface has its own
chunking strategy, eliminating the need for a global override system.
"""

from unittest.mock import Mock
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint, DataflowInterfaceType
from brainsmith.dataflow.core.chunking_strategy import (
    default_chunking, index_chunking, last_dim_chunking, spatial_chunking, no_chunking,
    DefaultChunkingStrategy, IndexBasedChunkingStrategy, FullTensorChunkingStrategy
)
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp


def demo_per_interface_strategies():
    """Demonstrate per-interface chunking strategies."""
    print("=== Per-Interface Chunking Strategy Demo ===\n")
    
    # 1. Create interfaces with different strategies
    print("1. Creating interfaces with different chunking strategies...")
    
    # Default strategy interface
    default_interface = InterfaceMetadata(
        name="input_default",
        interface_type=DataflowInterfaceType.INPUT,
        allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
        chunking_strategy=default_chunking()
    )
    print("   Default strategy: Layout-aware automatic chunking")
    
    # Custom index-based strategy
    custom_interface = InterfaceMetadata(
        name="input_custom", 
        interface_type=DataflowInterfaceType.INPUT,
        allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
        chunking_strategy=index_chunking(-1, [16])
    )
    print("   Custom strategy: Chunk last dimension with size 16")
    
    # Convenience function strategy
    convenience_interface = InterfaceMetadata(
        name="input_convenience",
        interface_type=DataflowInterfaceType.INPUT,
        allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
        chunking_strategy=last_dim_chunking(32)
    )
    print("   Convenience strategy: Last dimension chunking (size 32)")
    
    # Spatial chunking strategy
    spatial_interface = InterfaceMetadata(
        name="input_spatial",
        interface_type=DataflowInterfaceType.INPUT,
        allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
        chunking_strategy=spatial_chunking(16, 16)
    )
    print("   Spatial strategy: 16x16 spatial blocks")
    
    # No chunking strategy
    full_tensor_interface = InterfaceMetadata(
        name="input_full",
        interface_type=DataflowInterfaceType.INPUT,
        allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
        chunking_strategy=no_chunking()
    )
    print("   Full tensor strategy: No chunking")
    
    # 2. Create AutoHWCustomOp with mixed strategies
    print("\n2. Creating AutoHWCustomOp with mixed strategies...")
    mock_node = Mock()
    mock_node.input = ["input_tensor"]
    mock_node.output = ["output_tensor"]
    
    interfaces = [
        default_interface, custom_interface, convenience_interface,
        spatial_interface, full_tensor_interface
    ]
    
    op = AutoHWCustomOp(onnx_node=mock_node, interface_metadata=interfaces)
    print("   AutoHWCustomOp created with 5 different chunking strategies")
    
    # 3. Verify each interface has its own strategy
    print("\n3. Verifying per-interface strategies...")
    for metadata in op.interface_metadata.interfaces:
        strategy_type = type(metadata.chunking_strategy).__name__
        print(f"   {metadata.name}: {strategy_type}")
    
    return op


def demo_strategy_extensibility():
    """Demonstrate how to extend the strategy system."""
    print("\n=== Strategy Extensibility Demo ===\n")
    
    print("1. Built-in strategy types...")
    strategies = [
        ("Default", default_chunking()),
        ("Index-based", index_chunking(-1, [8])),
        ("Last dimension", last_dim_chunking(16)),
        ("Spatial", spatial_chunking(8, 8)),
        ("No chunking", no_chunking())
    ]
    
    for name, strategy in strategies:
        print(f"   {name}: {type(strategy).__name__} (type: {strategy.chunking_type.value})")
    
    print("\n2. Strategy computation examples...")
    tensor_shape = [1, 8, 32, 32]  # NCHW format
    
    for name, strategy in strategies[:3]:  # Test first 3 strategies
        qDim, tDim = strategy.compute_chunking(tensor_shape, "test_interface")
        print(f"   {name}: qDim={qDim}, tDim={tDim}")


def demo_hwkg_integration():
    """Demonstrate how HWKG layer would use per-interface strategies."""
    print("\n=== HWKG Integration Demo ===\n")
    
    # Simulate HWKG parsing RTL pragmas
    rtl_pragmas = {
        "in0_V_data_V": {"type": "index", "start_index": -1, "shape": [16]},
        "weights": {"type": "spatial", "height": 8, "width": 8},
        "bias": {"type": "none"}
    }
    
    print("1. HWKG parses RTL pragmas...")
    for interface, pragma in rtl_pragmas.items():
        print(f"   {interface}: {pragma}")
    
    print("\n2. HWKG creates chunking strategies from pragmas...")
    interfaces = []
    
    for interface_name, pragma in rtl_pragmas.items():
        # HWKG interprets pragma and creates appropriate strategy
        if pragma["type"] == "index":
            strategy = index_chunking(pragma["start_index"], pragma["shape"])
        elif pragma["type"] == "spatial":
            strategy = spatial_chunking(pragma["height"], pragma["width"])
        elif pragma["type"] == "none":
            strategy = no_chunking()
        else:
            strategy = default_chunking()
        
        # Determine interface type from name
        if "weight" in interface_name:
            iface_type = DataflowInterfaceType.WEIGHT
        elif "bias" in interface_name:
            iface_type = DataflowInterfaceType.WEIGHT
        else:
            iface_type = DataflowInterfaceType.INPUT
        
        interface = InterfaceMetadata(
            name=interface_name,
            interface_type=iface_type,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=strategy
        )
        interfaces.append(interface)
        
        print(f"   {interface_name}: {type(strategy).__name__}")
    
    print("\n3. HWKG generates clean AutoHWCustomOp...")
    mock_node = Mock()
    mock_node.input = ["input_tensor", "weight_tensor", "bias_tensor"]
    
    op = AutoHWCustomOp(onnx_node=mock_node, interface_metadata=interfaces)
    print("   AutoHWCustomOp created with pragma-derived strategies")
    print("   No override system needed - each interface has its strategy")
    
    return op


def demo_strategy_patterns():
    """Demonstrate common chunking strategy patterns."""
    print("\n=== Common Strategy Patterns ===\n")
    
    patterns = [
        ("Streaming (last dim)", last_dim_chunking(1), "Process one element at a time"),
        ("Block processing", spatial_chunking(8, 8), "Process 8x8 blocks"),
        ("Channel parallel", index_chunking(1, [1]), "Parallelize across channels"),
        ("Full tensor", no_chunking(), "Process entire tensor"),
        ("Custom chunking", index_chunking(2, [16, 16]), "Custom spatial chunking")
    ]
    
    tensor_shape = [1, 64, 32, 32]
    
    for name, strategy, description in patterns:
        print(f"{name}:")
        print(f"  Description: {description}")
        qDim, tDim = strategy.compute_chunking(tensor_shape, "test")
        print(f"  Result: qDim={qDim}, tDim={tDim}")
        print()


if __name__ == "__main__":
    op1 = demo_per_interface_strategies()
    demo_strategy_extensibility()
    op2 = demo_hwkg_integration()
    demo_strategy_patterns()
    
    print("=" * 60)
    print("Per-Interface Chunking Strategy Architecture Complete")
    print("=" * 60)
    print()
    print("Key Benefits:")
    print("• Each interface has its own chunking strategy")
    print("• No global override system needed")
    print("• Extensible strategy pattern")
    print("• Convenient factory functions")
    print("• Clean HWKG integration")
    print("• Object-oriented design")
    print()
    print("Usage Pattern:")
    print("1. HWKG parses RTL pragmas")
    print("2. HWKG creates appropriate chunking strategies")
    print("3. HWKG passes strategies to interface constructors")
    print("4. Each interface handles its own chunking automatically")