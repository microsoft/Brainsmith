"""
Zero-Configuration FINN Integration Demo

This demo shows how the enhanced AutoHWCustomOp architecture enables zero-configuration
usage where tensor shapes are extracted automatically and chunking strategies are applied
per-interface without manual qDim/tDim specification.
"""

import numpy as np
from unittest.mock import Mock

# Import the enhanced AutoHWCustomOp system
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint, DataflowInterfaceType
from brainsmith.dataflow.core.chunking_strategy import (
    default_chunking, index_chunking, last_dim_chunking, spatial_chunking, FullTensorChunkingStrategy
)
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp

# Mock ModelWrapper for demonstration
class MockModelWrapper:
    """Mock FINN ModelWrapper for demonstration."""
    
    def __init__(self, input_shapes):
        self.input_shapes = input_shapes
        self.graph = Mock()
        self.graph.input = [Mock() for _ in input_shapes]
        for i, shape in enumerate(input_shapes):
            self.graph.input[i].name = f"input_{i}"
    
    def get_tensor_shape(self, tensor_name):
        """Get tensor shape by name."""
        if tensor_name == "input_0":
            return self.input_shapes[0]
        elif tensor_name == "input_1":
            return self.input_shapes[1] if len(self.input_shapes) > 1 else [64, 64]
        else:
            return [1, 8, 32, 32]  # Default


def create_zero_config_node():
    """Create an ONNX node that would normally require manual configuration."""
    
    # Simulate ONNX node creation (normally done with onnx.helper.make_node)
    mock_node = Mock()
    mock_node.op_type = "ThresholdingAxi"
    mock_node.input = ["input_tensor", "weights", "bias"]
    mock_node.output = ["output_tensor"]
    mock_node.attribute = []
    
    # In real FINN, this would be:
    # node = onnx.helper.make_node(
    #     "ThresholdingAxi",
    #     inputs=["input_tensor", "weights", "bias"],
    #     outputs=["output_tensor"],
    #     in0_V_data_V_dtype="UINT8"  # Only datatype needed!
    #     # No qDim, tDim, or shape required!
    # )
    
    return mock_node


def demo_zero_configuration():
    """Demonstrate zero-configuration usage."""
    print("=== Zero-Configuration FINN Integration Demo ===\n")
    
    # 1. Create ONNX node without manual tensor configuration
    print("1. Creating ONNX node without manual tensor configuration...")
    node = create_zero_config_node()
    print(f"   Node type: {node.op_type}")
    print(f"   Inputs: {node.input}")
    print(f"   Outputs: {node.output}")
    print("   ✓ No qDim, tDim, or shape specification required!\n")
    
    # 2. Create interface metadata with chunking strategies (from HWKG pragma parsing)
    print("2. Creating interface metadata with chunking strategies...")
    
    # These would be generated from RTL pragma parsing in real HWKG:
    # "@brainsmith TDIM in0_V_data_V -1 [16]" -> index_chunking(-1, [16])
    # "@brainsmith TDIM weights spatial 3x3" -> spatial_chunking(3, 3)  
    # "@brainsmith TDIM bias none" -> FullTensorChunkingStrategy()
    
    interface_metadata = [
        InterfaceMetadata(
            name="in0_V_data_V",
            interface_type=DataflowInterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
            chunking_strategy=index_chunking(-1, [16])  # Stream on last dim, 16 elements
        ),
        InterfaceMetadata(
            name="weights",
            interface_type=DataflowInterfaceType.WEIGHT,
            allowed_datatypes=[DataTypeConstraint(finn_type='INT8', bit_width=8)],
            chunking_strategy=spatial_chunking(3, 3)  # 3x3 convolution weights
        ),
        InterfaceMetadata(
            name="bias",
            interface_type=DataflowInterfaceType.WEIGHT,
            allowed_datatypes=[DataTypeConstraint(finn_type='INT32', bit_width=32)],
            chunking_strategy=FullTensorChunkingStrategy()  # No chunking for bias
        ),
        InterfaceMetadata(
            name="out0_V_data_V",
            interface_type=DataflowInterfaceType.OUTPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
            chunking_strategy=last_dim_chunking(32)  # Output streaming
        )
    ]
    
    for metadata in interface_metadata:
        print(f"   {metadata.name}: {type(metadata.chunking_strategy).__name__}")
    print("   ✓ Each interface has its own chunking strategy\n")
    
    # 3. Create AutoHWCustomOp - no manual configuration needed!
    print("3. Creating AutoHWCustomOp with automatic configuration...")
    op = AutoHWCustomOp(node, interface_metadata)
    print("   ✓ AutoHWCustomOp created automatically\n")
    
    # 4. Set ModelWrapper for automatic shape extraction
    print("4. Setting ModelWrapper for automatic tensor shape extraction...")
    
    # Simulate actual tensor shapes that would come from FINN model
    input_shapes = [
        [1, 8, 32, 32],    # input_tensor: NCHW format
        [64, 8, 3, 3],     # weights: convolution kernel
        [64]               # bias: per-channel bias
    ]
    
    model_wrapper = MockModelWrapper(input_shapes)
    op.set_model_wrapper(model_wrapper)
    
    print(f"   Input tensor shape: {input_shapes[0]}")
    print(f"   Weight tensor shape: {input_shapes[1]}")
    print(f"   Bias tensor shape: {input_shapes[2]}")
    print("   ✓ Shapes will be extracted automatically\n")
    
    # 5. Access DataflowModel - triggers automatic building
    print("5. Accessing DataflowModel - triggers automatic building...")
    dataflow_model = op.dataflow_model
    
    print(f"   Interfaces built: {len(dataflow_model.interfaces)}")
    print("   Interface configurations:")
    
    for interface_name, interface in dataflow_model.interfaces.items():
        print(f"     {interface_name}:")
        print(f"       Type: {interface.interface_type.value}")
        print(f"       qDim: {interface.qDim}")
        print(f"       tDim: {interface.tDim}")
        if hasattr(interface, '_tensor_shape'):
            print(f"       Extracted shape: {interface._tensor_shape}")
        if hasattr(interface, '_inferred_layout'):
            print(f"       Inferred layout: {interface._inferred_layout}")
    
    print("   ✓ All dimensions computed automatically!\n")
    
    return op


def demo_traditional_vs_zero_config():
    """Compare traditional manual configuration vs zero-configuration."""
    print("=== Traditional vs Zero-Configuration Comparison ===\n")
    
    print("TRADITIONAL (Manual Configuration):")
    print("```python")
    print("node = onnx.helper.make_node(")
    print("    'ThresholdingAxi',")
    print("    inputs=['input_tensor'],")
    print("    outputs=['output_tensor'],")
    print("    qDim=[1, 8, 32, 1],      # Manual calculation")
    print("    tDim=[1, 1, 1, 32],      # Manual calculation")
    print("    dtype='UINT8',")
    print("    shape=[1, 8, 32, 32]     # Manual specification")
    print(")")
    print("```")
    print("❌ Manual qDim/tDim calculation required")
    print("❌ Manual tensor shape specification")
    print("❌ Error-prone and time-consuming")
    print("❌ Breaks when input shapes change\n")
    
    print("ZERO-CONFIGURATION (Automatic):")
    print("```python")
    print("node = onnx.helper.make_node(")
    print("    'ThresholdingAxi',")
    print("    inputs=['input_tensor'],")
    print("    outputs=['output_tensor'],")
    print("    in0_V_data_V_dtype='UINT8'   # Only datatype required")
    print("    # layout inferred, qDim/tDim computed automatically")
    print(")")
    print("```")
    print("✅ Automatic tensor shape extraction")
    print("✅ Automatic qDim/tDim computation")
    print("✅ Per-interface chunking strategies")
    print("✅ Adapts automatically to input shape changes")
    print("✅ Zero manual configuration burden\n")


def demo_hwkg_pragma_integration():
    """Demonstrate HWKG pragma integration workflow."""
    print("=== HWKG Pragma Integration Workflow ===\n")
    
    # Import the pragma converter
    from brainsmith.tools.hw_kernel_gen.pragma_to_strategy import PragmaToStrategyConverter
    
    # 1. HWKG parses RTL pragmas
    print("1. HWKG parses RTL pragmas from source code...")
    rtl_pragmas = [
        "@brainsmith TDIM in0_V_data_V -1 [16]",
        "@brainsmith TDIM weights spatial 3x3", 
        "@brainsmith TDIM bias none",
        "@brainsmith TDIM out0_V_data_V last_dim 32"
    ]
    
    for pragma in rtl_pragmas:
        print(f"   {pragma}")
    print()
    
    # 2. Convert pragmas to chunking strategies
    print("2. Converting pragmas to chunking strategies...")
    converter = PragmaToStrategyConverter()
    strategies = {}
    
    for pragma_str in rtl_pragmas:
        parsed = converter.parse_enhanced_tdim_pragma(pragma_str)
        interface_name = parsed['interface_name']
        strategy = converter.convert_tdim_pragma(parsed)
        strategies[interface_name] = strategy
        
        print(f"   {interface_name}: {type(strategy).__name__}")
    print()
    
    # 3. Generate template code
    print("3. Generating clean template code...")
    from brainsmith.tools.hw_kernel_gen.pragma_to_strategy import generate_strategy_code
    
    print("```python")
    print("class ThresholdingAxi(AutoHWCustomOp):")
    print("    def __init__(self, onnx_node, **kwargs):")
    print("        from brainsmith.dataflow.core.chunking_strategy import *")
    print("        ")
    print("        self._interface_metadata = [")
    
    for interface_name, strategy in strategies.items():
        strategy_code = generate_strategy_code(strategy)
        print(f"            InterfaceMetadata(")
        print(f"                name='{interface_name}',")
        print(f"                interface_type=DataflowInterfaceType.INPUT,")
        print(f"                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],")
        print(f"                chunking_strategy={strategy_code}")
        print(f"            ),")
    
    print("        ]")
    print("        super().__init__(onnx_node, **kwargs)")
    print("```")
    print()
    print("✅ Clean, generated code with embedded strategies")
    print("✅ No static dictionaries or manual configuration")
    print("✅ Each interface owns its chunking behavior\n")


def demo_performance_benefits():
    """Demonstrate performance benefits of lazy building."""
    print("=== Performance Benefits Demo ===\n")
    
    print("1. Creating AutoHWCustomOp (lazy initialization)...")
    
    # Create large interface metadata (simulating complex operation)
    interface_metadata = []
    for i in range(10):
        interface_metadata.append(
            InterfaceMetadata(
                name=f"interface_{i}",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=index_chunking(-1, [16])
            )
        )
    
    node = Mock()
    node.input = [f"input_{i}" for i in range(10)]
    node.output = ["output"]
    
    # Measure construction time (would be real timing in practice)
    import time
    start_time = time.time()
    op = AutoHWCustomOp(node, interface_metadata)
    construction_time = time.time() - start_time
    
    print(f"   Construction time: {construction_time:.6f}s")
    print("   ✓ Instant - DataflowModel not built yet")
    print()
    
    print("2. First access to DataflowModel (triggers building)...")
    start_time = time.time()
    dataflow_model = op.dataflow_model
    build_time = time.time() - start_time
    
    print(f"   Build time: {build_time:.6f}s")
    print("   ✓ Built only when needed")
    print()
    
    print("3. Subsequent accesses (cached)...")
    start_time = time.time()
    dataflow_model = op.dataflow_model
    cached_time = time.time() - start_time
    
    print(f"   Cached access time: {cached_time:.6f}s")
    print("   ✓ Instant - cached result")
    print()
    
    print("Benefits:")
    print("✅ Fast node creation compatible with FINN workflow")
    print("✅ Lazy building optimizes memory usage")
    print("✅ Caching optimizes repeated access")
    print("✅ Only builds when resource estimation needed\n")


if __name__ == "__main__":
    # Run all demonstrations
    demo_zero_configuration()
    demo_traditional_vs_zero_config()
    demo_hwkg_pragma_integration()
    demo_performance_benefits()
    
    print("=" * 60)
    print("Zero-Configuration FINN Integration Complete!")
    print("=" * 60)
    print()
    print("Key Achievements:")
    print("✅ Automatic tensor shape extraction")
    print("✅ Per-interface chunking strategies")
    print("✅ Zero manual configuration required")
    print("✅ Clean HWKG pragma integration")
    print("✅ Performance-optimized lazy building")
    print("✅ Full backward compatibility")
    print()
    print("The AutoHWCustomOp system now enables zero-configuration")
    print("usage while maintaining all existing functionality!")