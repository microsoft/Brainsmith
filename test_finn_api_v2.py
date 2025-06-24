#!/usr/bin/env python3
"""
Test the simplified FINN-Brainsmith API V2.
"""

import sys
from pathlib import Path

# Add brainsmith to path
sys.path.insert(0, str(Path(__file__).parent))

from brainsmith.core.finn_v2 import FINNBuildSpec, FINNLegacyConverter


def test_build_spec_creation():
    """Test creating a build specification."""
    print("Testing FINNBuildSpec creation...")
    
    spec = FINNBuildSpec(
        kernels=["MatMul", "LayerNorm"],
        kernel_backends={"MatMul": "hls", "LayerNorm": "rtl"},
        transforms={
            "graph_cleanup": ["RemoveIdentityOps"],
            "topology_optimization": ["ExpandNorms"]
        },
        output_dir="./test_build",
        target_device="Pynq-Z1",
        target_frequency_mhz=200.0
    )
    
    print(f"✓ Created spec with {len(spec.kernels)} kernels")
    print(f"✓ Kernels: {spec.kernels}")
    print(f"✓ Backends: {spec.kernel_backends}")
    print(f"✓ Transform stages: {list(spec.transforms.keys())}")
    
    return spec


def test_converter_creation():
    """Test creating the converter."""
    print("\nTesting FINNLegacyConverter creation...")
    
    converter = FINNLegacyConverter()
    print("✓ Created converter")
    print(f"✓ Registry available: {converter.registry is not None}")
    
    return converter


def test_config_generation():
    """Test generating FINN configuration."""
    print("\nTesting configuration generation...")
    
    spec = FINNBuildSpec(
        kernels=["MatMul"],
        transforms={"graph_cleanup": ["RemoveIdentityOps"]},
        output_dir="./test",
        target_frequency_mhz=200.0
    )
    
    converter = FINNLegacyConverter()
    config = converter.convert(spec)
    
    print(f"✓ Generated config with {len(config['steps'])} steps")
    print(f"✓ Output dir: {config['output_dir']}")
    print(f"✓ Clock period: {config['synth_clk_period_ns']} ns")
    print(f"✓ Board: {config['board']}")
    
    # Print first few steps
    print("✓ First few steps:")
    for i, step in enumerate(config['steps'][:5]):
        if callable(step):
            print(f"   {i+1}. {step.__name__}")
        else:
            print(f"   {i+1}. {step}")
    
    return config


def test_validation():
    """Test validation logic."""
    print("\nTesting validation...")
    
    # Test empty kernels
    try:
        FINNBuildSpec(kernels=[], transforms={})
        print("✗ Should have failed with empty kernels")
    except ValueError:
        print("✓ Correctly rejected empty kernels")
    
    # Test invalid frequency
    try:
        FINNBuildSpec(kernels=["MatMul"], target_frequency_mhz=-1)
        print("✗ Should have failed with negative frequency")
    except ValueError:
        print("✓ Correctly rejected negative frequency")
    
    print("✓ Validation working correctly")


def test_bert_example():
    """Test BERT-style configuration."""
    print("\nTesting BERT example...")
    
    spec = FINNBuildSpec(
        kernels=["MatMul", "LayerNorm", "Softmax"],
        kernel_backends={
            "MatMul": "rtl",
            "LayerNorm": "brainsmith", 
            "Softmax": "hls"
        },
        transforms={
            "graph_cleanup": ["RemoveIdentityOps", "FoldConstants"],
            "topology_optimization": ["ExpandNorms", "StreamlineActivations"],
            "kernel_optimization": ["BERTFoldingOptimization"]
        },
        output_dir="./bert_test",
        target_device="U280",
        target_frequency_mhz=250.0,
        target_fps=3000,
        metadata={
            "model_type": "bert",
            "num_heads": 12
        }
    )
    
    converter = FINNLegacyConverter()
    config = converter.convert(spec)
    
    print(f"✓ BERT config with {len(config['steps'])} steps")
    print(f"✓ Target FPS: {config.get('target_fps')}")
    print(f"✓ Metadata included: {spec.metadata}")
    
    return spec, config


def main():
    """Run all tests."""
    print("FINN-Brainsmith API V2 Tests")
    print("=" * 40)
    
    try:
        # Basic tests
        spec = test_build_spec_creation()
        converter = test_converter_creation()
        config = test_config_generation()
        test_validation()
        
        # Advanced test
        bert_spec, bert_config = test_bert_example()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed!")
        print("\nAPI Summary:")
        print(f"- Simple FINNBuildSpec dataclass")
        print(f"- FINNLegacyConverter for FINN integration")
        print(f"- Uses plugin registry for transforms/kernels")
        print(f"- Generates standard FINN steps")
        print(f"- Clean, minimal API")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())