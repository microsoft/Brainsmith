#!/usr/bin/env python3
"""
DEFINITIVE End-to-End Test: Blueprint YAML → Segment Execution

This test demonstrates the complete flow WITHOUT ANY MOCKS:
1. Parse a real blueprint YAML
2. Build an execution tree with segments
3. Execute segments using FINN

If parts are missing, it will show exactly what failed and why.
"""

import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.core.forge import forge
from brainsmith.core.explorer import explore_execution_tree


def main():
    """Run the definitive end-to-end test."""
    print("DEFINITIVE End-to-End Test")
    print("=" * 80)
    
    # Paths
    examples_dir = Path(__file__).parent
    model_path = examples_dir / "test_model.onnx"
    blueprint_path = examples_dir / "finn_steps_blueprint.yaml"
    output_dir = examples_dir / "output" / "definitive_e2e"
    
    # Ensure model exists
    if not model_path.exists():
        print("Creating test model...")
        exec(open(examples_dir / "create_test_model.py").read())
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Blueprint: {blueprint_path}")
    print(f"  Output: {output_dir}")
    
    # Phase 1: Blueprint → Execution Tree
    print("\n" + "-" * 80)
    print("PHASE 1: Blueprint Parsing & Tree Building")
    print("-" * 80)
    
    design_space, tree = forge(str(model_path), str(blueprint_path))
    
    print(f"\n✓ Successfully built execution tree")
    print(f"  Design space: {len(design_space.transform_stages)} transform stages")
    print(f"  Build pipeline: {len(design_space.build_pipeline)} steps")
    
    # Phase 2: Execute Tree
    print("\n" + "-" * 80)
    print("PHASE 2: Segment Execution with FINN")
    print("-" * 80)
    
    # Prepare config
    blueprint_config = {
        "global_config": {
            "fail_fast": False,
            "output_products": design_space.global_config.output_stage.value
        },
        "finn_config": design_space.finn_config
    }
    
    print(f"\nExecuting with config:")
    print(f"  Output stage: {design_space.global_config.output_stage.value}")
    print(f"  FINN board: {design_space.finn_config.get('board')}")
    print(f"  Clock period: {design_space.finn_config.get('synth_clk_period_ns')}ns")
    
    # Execute
    result = explore_execution_tree(tree, model_path, output_dir, blueprint_config)
    
    # Phase 3: Results
    print("\n" + "-" * 80)
    print("PHASE 3: Execution Results")
    print("-" * 80)
    
    print(f"\nExecution Summary:")
    print(f"  Total segments: {result.stats['total']}")
    print(f"  Successful: {result.stats['successful']}")
    print(f"  Failed: {result.stats['failed']}")
    print(f"  Skipped: {result.stats['skipped']}")
    print(f"  Cached: {result.stats['cached']}")
    print(f"  Time: {result.total_time:.2f}s")
    
    # Show segment details
    print(f"\nSegment Details:")
    for seg_id, seg_result in sorted(result.segment_results.items()):
        is_skipped = seg_result.error == "Skipped"
        status = "✓" if seg_result.success else ("⊘" if is_skipped else "✗")
        time_str = f"{seg_result.execution_time:.2f}s" if seg_result.execution_time else "N/A"
        
        if seg_result.success:
            print(f"  {status} {seg_id} - {time_str}")
        elif is_skipped:
            print(f"  {status} {seg_id} - Skipped")
        else:
            print(f"  {status} {seg_id} - Failed: {seg_result.error}")
    
    # Show generated files
    if output_dir.exists():
        all_files = list(output_dir.rglob("*"))
        onnx_files = [f for f in all_files if f.suffix == ".onnx"]
        log_files = [f for f in all_files if f.suffix == ".log"]
        json_files = [f for f in all_files if f.suffix == ".json"]
        
        print(f"\nGenerated Files:")
        print(f"  ONNX models: {len(onnx_files)}")
        print(f"  Log files: {len(log_files)}")
        print(f"  JSON files: {len(json_files)}")
        print(f"  Total files: {len(all_files)}")
        
        # Show tree structure if it exists
        tree_file = output_dir / "tree.json"
        if tree_file.exists():
            print(f"\nExecution Tree Structure:")
            tree_data = json.loads(tree_file.read_text())
            
            def print_tree_node(node, indent=""):
                if node['segment_id'] != 'root':
                    print(f"{indent}├── {node['segment_id']} ({len(node['segment_steps'])} steps)")
                for child in node['children'].values():
                    print_tree_node(child, indent + "│   ")
            
            print_tree_node(tree_data, "  ")
    
    # Conclusion
    print("\n" + "=" * 80)
    if result.stats['successful'] > 0:
        print("✓ END-TO-END TEST COMPLETED WITH SUCCESSFUL SEGMENTS!")
    elif result.stats['failed'] > 0:
        print("✗ END-TO-END TEST COMPLETED WITH FAILURES")
        print("\nThis is expected if FINN is not properly configured or if")
        print("the blueprint contains steps that FINN doesn't recognize.")
    else:
        print("⊘ END-TO-END TEST COMPLETED BUT ALL SEGMENTS WERE SKIPPED")
    
    return 0 if result.stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())