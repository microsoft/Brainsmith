#!/usr/bin/env python3
"""
Example showing how the new FINN API integrates with existing code.
"""

import sys
from pathlib import Path

# Add brainsmith to path
sys.path.insert(0, str(Path(__file__).parent))

from brainsmith.core.finn_v2 import FINNBuildSpec, FINNLegacyConverter


def example_with_existing_hardware_step():
    """
    Example using the existing hardware inference step from:
    brainsmith/libraries/transforms/steps/hardware.py
    """
    
    print("Example: Using existing hardware step")
    print("=" * 40)
    
    # Import the existing hardware step
    try:
        from brainsmith.libraries.transforms.steps.hardware import infer_hardware_step
        print("✓ Found existing hardware step")
    except ImportError as e:
        print(f"⚠ Hardware step not available: {e}")
        return
    
    # Create a build spec that uses the existing step
    spec = FINNBuildSpec(
        kernels=["MatMul", "LayerNorm", "Softmax"],  # These would be inferred by the step
        transforms={
            "graph_cleanup": ["RemoveIdentityOps"],
            "topology_optimization": ["ExpandNorms"], 
            "kernel_mapping": ["infer_hardware_step"]  # Use the existing step
        },
        output_dir="./integration_test",
        target_device="Pynq-Z1",
        target_frequency_mhz=200.0
    )
    
    print(f"✓ Created spec using existing hardware step")
    
    # Convert to FINN format
    converter = FINNLegacyConverter()
    
    # We could modify the converter to handle known step names
    config = converter.convert(spec)
    
    print(f"✓ Generated config with {len(config['steps'])} steps")
    print("✓ This replaces the complex 6-entrypoint mapping!")
    
    return config


def example_simple_vs_old_approach():
    """
    Compare the new simple approach with the old 6-entrypoint system.
    """
    
    print("\nComparison: New vs Old Approach")
    print("=" * 40)
    
    print("OLD (6-entrypoint - confusing):")
    print("  entrypoint_1: ['LayerNorm', 'Softmax']")
    print("  entrypoint_2: ['cleanup', 'streamlining']") 
    print("  entrypoint_3: ['MatMul', 'LayerNorm']")
    print("  entrypoint_4: ['matmul_hls', 'layernorm_rtl']")
    print("  entrypoint_5: ['target_fps_parallelization']")
    print("  entrypoint_6: ['set_fifo_depths']")
    print("  → Complex mapping through multiple layers")
    print("  → Artificial 'entrypoint' concept")
    print("  → Mixed kernels and transforms")
    
    print("\nNEW (direct specification - clear):")
    
    spec = FINNBuildSpec(
        kernels=["MatMul", "LayerNorm"],
        kernel_backends={"MatMul": "hls", "LayerNorm": "rtl"},
        transforms={
            "graph_cleanup": ["RemoveIdentityOps"],
            "topology_optimization": ["ExpandNorms", "StreamlineActivations"],
            "kernel_optimization": ["TargetFPSParallelization"]
        },
        output_dir="./simple_build",
        target_fps=1000
    )
    
    print(f"  kernels: {spec.kernels}")
    print(f"  kernel_backends: {spec.kernel_backends}")
    print(f"  transforms: {spec.transforms}")
    print(f"  target_fps: {spec.target_fps}")
    print("  → Direct specification of what you want")
    print("  → Clear separation of kernels vs transforms")
    print("  → Uses plugin system automatically")
    
    return spec


def example_blueprint_integration():
    """
    Show how this could integrate with Blueprint V2.
    """
    
    print("\nBlueprint V2 Integration")
    print("=" * 40)
    
    # Simulated Blueprint V2 constraints
    blueprint_constraints = {
        "target_frequency_mhz": 250.0,
        "target_throughput_fps": 5000,
        "max_lut_utilization": 0.85,
        "optimization_goal": "performance"
    }
    
    print(f"Blueprint constraints: {blueprint_constraints}")
    
    # Convert Blueprint constraints to FINNBuildSpec
    if blueprint_constraints["optimization_goal"] == "performance":
        # High-performance configuration
        spec = FINNBuildSpec(
            kernels=["MatMul", "LayerNorm", "Softmax"],
            kernel_backends={
                "MatMul": "rtl",  # RTL for performance
                "LayerNorm": "rtl",
                "Softmax": "hls"
            },
            transforms={
                "graph_cleanup": ["RemoveIdentityOps"],
                "topology_optimization": ["ExpandNorms", "StreamlineActivations"],
                "kernel_optimization": ["MaximizeParallelism"]
            },
            target_frequency_mhz=blueprint_constraints["target_frequency_mhz"],
            target_fps=blueprint_constraints["target_throughput_fps"],
            output_dir="./blueprint_build"
        )
    
    print(f"✓ Converted to FINNBuildSpec")
    print(f"  Automatic kernel backend selection based on goal")
    print(f"  Appropriate transforms for performance optimization")
    
    # This could be used in DSE
    converter = FINNLegacyConverter()
    config = converter.convert(spec)
    
    print(f"✓ Ready for FINN execution with {len(config['steps'])} steps")
    
    return spec, config


def main():
    """Run integration examples."""
    
    print("FINN-Brainsmith API V2 Integration Examples")
    print("=" * 50)
    
    # Example with existing hardware step
    example_with_existing_hardware_step()
    
    # Compare new vs old approach
    example_simple_vs_old_approach()
    
    # Blueprint integration
    example_blueprint_integration()
    
    print("\n" + "=" * 50)
    print("✅ Integration examples complete!")
    print("\nKey Benefits:")
    print("- Eliminates confusing 6-entrypoint concept")
    print("- Clear separation of kernels vs transforms") 
    print("- Direct use of plugin system")
    print("- Simple conversion to legacy FINN")
    print("- Easy integration with Blueprint V2")
    print("- No over-engineering!")


if __name__ == "__main__":
    main()