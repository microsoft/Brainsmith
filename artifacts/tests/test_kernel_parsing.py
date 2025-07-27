"""Test refactored kernel parsing."""

import tempfile
import os
from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core.plugins.registry import get_registry


def test_kernel_parsing():
    """Test that kernel parsing works correctly after refactoring."""
    
    # Ensure plugins are loaded
    get_registry().reset()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: String kernel spec (auto-detect backends)
        yaml1 = """
version: "4.0"
name: "Test1"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  kernels:
    - MVAU
    - VVAU
"""
        
        # Test 2: Dict kernel spec (explicit backends)
        yaml2 = """
version: "4.0" 
name: "Test2"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  kernels:
    - MVAU: MVAU_hls
    - VVAU: [VVAU_hls]
"""
        
        # Test 3: Mixed specs and unknown kernel
        yaml3 = """
version: "4.0"
name: "Test3"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  kernels:
    - MVAU
    - VVAU: VVAU_hls
    - UnknownKernel
"""
        
        model_path = os.path.join(tmpdir, "model.onnx")
        with open(model_path, "wb") as f:
            f.write(b"dummy_model")
        
        parser = BlueprintParser()
        
        # Test 1: Auto-detect backends
        print("Test 1: Auto-detect backends")
        blueprint_path = os.path.join(tmpdir, "test1.yaml")
        with open(blueprint_path, "w") as f:
            f.write(yaml1)
        
        design_space, _ = parser.parse(blueprint_path, model_path)
        kernels = design_space.kernel_backends
        assert len(kernels) >= 1  # At least one kernel with backends
        print(f"  ✓ Found {len(kernels)} kernels with backends")
        
        # Test 2: Explicit backends
        print("\nTest 2: Explicit backends")
        blueprint_path = os.path.join(tmpdir, "test2.yaml")
        with open(blueprint_path, "w") as f:
            f.write(yaml2)
        
        design_space, _ = parser.parse(blueprint_path, model_path)
        kernels = design_space.kernel_backends
        assert len(kernels) >= 1
        print(f"  ✓ Found {len(kernels)} kernels with explicit backends")
        
        # Test 3: Unknown kernel is skipped
        print("\nTest 3: Unknown kernel handling")
        blueprint_path = os.path.join(tmpdir, "test3.yaml")
        with open(blueprint_path, "w") as f:
            f.write(yaml3)
        
        design_space, _ = parser.parse(blueprint_path, model_path)
        kernels = design_space.kernel_backends
        kernel_names = [k[0] for k in kernels]
        assert "UnknownKernel" not in kernel_names
        print(f"  ✓ Unknown kernel skipped, found {len(kernels)} valid kernels")
        
        # Test 4: Invalid spec
        print("\nTest 4: Invalid kernel spec")
        yaml4 = """
version: "4.0"
name: "Test4"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  kernels:
    - {MVAU: MVAU_hls, VVAU: VVAU_hls}  # Invalid: multiple keys
"""
        blueprint_path = os.path.join(tmpdir, "test4.yaml")
        with open(blueprint_path, "w") as f:
            f.write(yaml4)
        
        try:
            design_space, _ = parser.parse(blueprint_path, model_path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid kernel spec" in str(e)
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        print("\n✓ All kernel parsing tests passed!")


if __name__ == "__main__":
    test_kernel_parsing()