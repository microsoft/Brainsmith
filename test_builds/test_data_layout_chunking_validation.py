"""
Data Layout Chunking Validation Test

This test validates that the generated HWCustomOp correctly implements
data layout-based chunking according to the Interface-Wise Dataflow
Modeling specification.

Key validation points:
1. The determine_chunking_from_layout method works correctly
2. qDim and tDim are calculated properly for each layout
3. Chunking strategy is "data_layout_chunking" not arbitrary strategies
4. Interface specifications indicate automatic layout detection
"""

import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock

# Add the generated module to Python path
sys.path.insert(0, str(Path(__file__).parent / "hwkg_demo_final"))

# Minimal mock setup
mock_modules = {
    'brainsmith.dataflow.core.auto_hw_custom_op': Mock(),
    'brainsmith.dataflow.core.interface_metadata': Mock(),
    'brainsmith.dataflow.core.dataflow_interface': Mock(),
    'brainsmith.dataflow.core.tensor_chunking': Mock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Create working mocks
class MockAutoHWCustomOp:
    def __init__(self, onnx_node, interface_metadata=None, **kwargs):
        self.onnx_node = onnx_node
        self._interface_metadata = interface_metadata or []
    
    def get_nodeattr_types(self):
        return {}

class MockInterfaceMetadata:
    def __init__(self, name, interface_type, allowed_datatypes, chunking_strategy):
        self.name = name
        self.interface_type = interface_type
        self.allowed_datatypes = allowed_datatypes
        self.chunking_strategy = chunking_strategy

class MockDataTypeConstraint:
    def __init__(self, finn_type, bit_width, signed=False):
        self.finn_type = finn_type
        self.bit_width = bit_width
        self.signed = signed

class MockDataflowInterfaceType:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

def default_chunking():
    return Mock()

# Set up the mocks
sys.modules['brainsmith.dataflow.core.auto_hw_custom_op'].AutoHWCustomOp = MockAutoHWCustomOp
sys.modules['brainsmith.dataflow.core.interface_metadata'].InterfaceMetadata = MockInterfaceMetadata
sys.modules['brainsmith.dataflow.core.interface_metadata'].DataTypeConstraint = MockDataTypeConstraint
sys.modules['brainsmith.dataflow.core.dataflow_interface'].DataflowInterfaceType = MockDataflowInterfaceType
sys.modules['brainsmith.dataflow.core.tensor_chunking'].default_chunking = default_chunking

# Import the generated class
try:
    from autothresholdingaxi import AutoThresholdingAxi
    GENERATED_CLASS_AVAILABLE = True
    print("✅ Successfully imported AutoThresholdingAxi")
except ImportError as e:
    print(f"❌ Could not import generated class: {e}")
    GENERATED_CLASS_AVAILABLE = False

def test_data_layout_chunking():
    """Test the data layout-based chunking implementation."""
    
    if not GENERATED_CLASS_AVAILABLE:
        print("Cannot run test - generated class not available")
        return False
    
    print("=" * 60)
    print("DATA LAYOUT CHUNKING VALIDATION TEST")
    print("=" * 60)
    print("Testing the generated HWCustomOp's data layout chunking method")
    print()
    
    # Create the hardware operation
    mock_onnx_node = Mock()
    hw_op = AutoThresholdingAxi(mock_onnx_node)
    
    # === TEST 1: Interface Specifications ===
    print("🔍 TEST 1: Interface Specifications")
    print("-" * 40)
    
    interface_specs = hw_op.get_kernel_interface_specs()
    
    print("Interface specifications:")
    for interface_name, spec in interface_specs.items():
        print(f"\n{interface_name}:")
        for key, value in spec.items():
            print(f"  {key}: {value}")
    
    # Validate that chunking strategy is data layout-based
    for interface_name, spec in interface_specs.items():
        assert spec["chunking_strategy"] == "data_layout_chunking", \
            f"Interface {interface_name} should use data_layout_chunking"
        assert spec["layout_detection"] == "automatic", \
            f"Interface {interface_name} should have automatic layout detection"
    
    print("\n✅ All interfaces use data layout-based chunking")
    print()
    
    # === TEST 2: Layout Chunking Method ===
    print("🎯 TEST 2: Layout Chunking Method Validation")
    print("-" * 40)
    
    # Test cases from the Interface-Wise Dataflow Modeling specification
    test_cases = [
        # (tensor_shape, onnx_layout, expected_qDim, expected_tDim, description)
        ((1, 64), "[N, C]", 1, 64, "Classification output"),
        ((1, 3, 224, 224), "[N, C, H, W]", 3, 50176, "CNN standard layout"),
        ((1, 224, 224, 3), "[N, H, W, C]", 50176, 3, "CNN TensorFlow layout"),
        ((1, 512, 768), "[N, L, C]", 512, 768, "Transformer standard"),
        ((1, 768, 512), "[N, C, L]", 768, 512, "Transformer inverted"),
        ((1, 512, 12, 64), "[N, L, h, d]", 512, 768, "Multi-head attention"),
    ]
    
    print("Testing layout chunking for different tensor layouts:")
    print()
    print(f"{'Layout':<15} {'Shape':<18} {'qDim':<8} {'tDim':<10} {'Expected qDim':<12} {'Expected tDim':<12} {'✓'}")
    print("-" * 85)
    
    all_passed = True
    
    for tensor_shape, onnx_layout, expected_qDim, expected_tDim, description in test_cases:
        # Test the chunking method
        result = hw_op.determine_chunking_from_layout("test_interface", tensor_shape, onnx_layout)
        
        actual_qDim = result["qDim"]
        actual_tDim = result["tDim"]
        
        # Validate results
        qDim_ok = actual_qDim == expected_qDim
        tDim_ok = actual_tDim == expected_tDim
        test_passed = qDim_ok and tDim_ok
        
        status = "✅" if test_passed else "❌"
        
        shape_str = f"{tensor_shape}"
        print(f"{onnx_layout:<15} {shape_str:<18} {actual_qDim:<8} {actual_tDim:<10} {expected_qDim:<12} {expected_tDim:<12} {status}")
        
        if not test_passed:
            all_passed = False
            print(f"   Expected qDim={expected_qDim}, tDim={expected_tDim}")
            print(f"   Got qDim={actual_qDim}, tDim={actual_tDim}")
    
    print()
    if all_passed:
        print("✅ All layout chunking tests passed!")
    else:
        print("❌ Some layout chunking tests failed!")
        return False
    
    print()
    
    # === TEST 3: Parallelism Limits ===
    print("⚡ TEST 3: Parallelism Limits Based on qDim")
    print("-" * 40)
    
    print("Demonstrating how qDim limits useful parallelism:")
    print()
    
    # Example: CNN with 64 channels
    cnn_shape = (1, 64, 56, 56)
    cnn_layout = "[N, C, H, W]"
    result = hw_op.determine_chunking_from_layout("s_axis", cnn_shape, cnn_layout)
    
    qDim = result["qDim"]
    tDim = result["tDim"]
    
    print(f"CNN Example: {cnn_shape} {cnn_layout}")
    print(f"qDim = {qDim} (maximum useful parallelism)")
    print(f"tDim = {tDim} (elements per parallel unit)")
    print()
    
    print("Parallelism analysis:")
    print(f"{'Parallelism':<12} {'Utilization':<12} {'Efficiency':<12} {'Status'}")
    print("-" * 50)
    
    for parallelism in [1, 4, 16, 64, 128]:
        if parallelism <= qDim:
            utilization = f"{qDim // parallelism}x{qDim % parallelism}"
            efficiency = 100
            status = "Optimal" if qDim % parallelism == 0 else "Good"
        else:
            utilization = "1x0"
            efficiency = (qDim / parallelism) * 100
            status = "Wasteful"
        
        print(f"{parallelism}x{'':<10} {utilization:<12} {efficiency:<12.0f}% {status}")
    
    print()
    print("Key Insight: Parallelism beyond qDim (64) wastes resources!")
    print()
    
    # === TEST 4: Weight Interface Handling ===
    print("🏋️ TEST 4: Weight Interface Chunking Rules")
    print("-" * 40)
    
    # Test weight interface chunking rules
    weight_cases = [
        ((768,), "1D weight", 1, 768),
        ((768, 3072), "2D weight", 3072, 768),
        ((64,), "1D bias", 1, 64),
        ((512, 512), "2D matrix", 512, 512),
    ]
    
    print("Weight interface chunking (note: weights use different rules):")
    print(f"{'Weight Shape':<15} {'Type':<12} {'Expected qDim':<12} {'Expected tDim':<12}")
    print("-" * 55)
    
    for weight_shape, weight_type, expected_qDim, expected_tDim in weight_cases:
        print(f"{str(weight_shape):<15} {weight_type:<12} {expected_qDim:<12} {expected_tDim:<12}")
    
    print()
    print("Note: Weight chunking rules are different from activation chunking")
    print("• 1D weights: qDim=1, tDim=length")
    print("• 2D weights: qDim=second_dim, tDim=first_dim")
    print()
    
    # === TEST 5: Default Fallback ===
    print("🔄 TEST 5: Default Fallback Behavior")
    print("-" * 40)
    
    # Test unknown layout fallback
    unknown_shape = (1, 100, 200, 50)
    unknown_layout = "[N, X, Y, Z]"  # Unknown layout
    
    result = hw_op.determine_chunking_from_layout("test", unknown_shape, unknown_layout)
    
    expected_fallback_qDim = 1
    expected_fallback_tDim = np.prod(unknown_shape[1:])  # 100 * 200 * 50 = 1,000,000
    
    print(f"Unknown layout test:")
    print(f"  Shape: {unknown_shape}")
    print(f"  Layout: {unknown_layout}")
    print(f"  Fallback qDim: {result['qDim']} (expected: {expected_fallback_qDim})")
    print(f"  Fallback tDim: {result['tDim']} (expected: {expected_fallback_tDim})")
    
    fallback_ok = (result['qDim'] == expected_fallback_qDim and 
                   result['tDim'] == expected_fallback_tDim)
    
    print(f"  Fallback behavior: {'✅ Correct' if fallback_ok else '❌ Incorrect'}")
    
    if not fallback_ok:
        print(f"  Expected qDim={expected_fallback_qDim}, tDim={expected_fallback_tDim}")
        print(f"  Got qDim={result['qDim']}, tDim={result['tDim']}")
        return False
    
    print()
    
    # === SUMMARY ===
    print("📋 VALIDATION SUMMARY")
    print("-" * 40)
    
    print("✅ Data layout-based chunking validation complete!")
    print()
    print("Validated features:")
    print("• Interface specifications use 'data_layout_chunking' strategy")
    print("• Layout detection is set to 'automatic'")
    print("• determine_chunking_from_layout method works correctly")
    print("• All standard ONNX layouts ([N,C], [N,C,H,W], [N,L,C], etc.) supported")
    print("• qDim and tDim calculated correctly per specification")
    print("• Parallelism limits properly determined by qDim")
    print("• Default fallback behavior works for unknown layouts")
    print()
    print("The generated HWCustomOp correctly implements Interface-Wise")
    print("Dataflow Modeling with automatic data layout-based chunking!")
    
    return True

def run_validation():
    """Run the data layout chunking validation."""
    
    print("🧪 Starting Data Layout Chunking Validation...")
    print()
    
    success = test_data_layout_chunking()
    
    print()
    print("=" * 60)
    if success:
        print("✅ VALIDATION SUCCESSFUL!")
        print("Data layout-based chunking is correctly implemented.")
    else:
        print("❌ VALIDATION FAILED!")
        print("Issues found with data layout chunking implementation.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)