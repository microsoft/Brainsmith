#!/usr/bin/env python3
"""
TRUE end-to-end test from blueprint YAML to segment executor.
NO MOCKS - shows exactly what happens with FINN.
"""

import sys
import os
from pathlib import Path
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    """Run the true end-to-end flow."""
    print("TRUE End-to-End Test: Blueprint → Execution Tree → Segment Executor")
    print("=" * 70)
    
    # Setup paths
    examples_dir = Path(__file__).parent
    model_path = examples_dir / "test_model.onnx"
    blueprint_path = examples_dir / "working_blueprint.yaml"
    output_dir = examples_dir / "output" / "true_e2e_run"
    
    # Ensure model exists
    if not model_path.exists():
        print("\nCreating test model...")
        exec(open(examples_dir / "create_test_model.py").read())
    
    print(f"\nInput model: {model_path}")
    print(f"Blueprint: {blueprint_path}")
    print(f"Output dir: {output_dir}")
    
    # Step 1: Parse blueprint and build tree
    print("\n" + "-" * 70)
    print("STEP 1: Parse Blueprint & Build Execution Tree")
    print("-" * 70)
    
    try:
        from brainsmith.core.forge import forge, get_tree_stats
        
        design_space, tree = forge(str(model_path), str(blueprint_path))
        
        stats = get_tree_stats(tree)
        print(f"\n✓ Successfully created execution tree:")
        print(f"  - Total segments: {stats['total_segments']}")
        print(f"  - Total paths: {stats['total_paths']}")
        print(f"  - Max depth: {stats['max_depth']}")
        
        # Show the tree structure
        print(f"\nTree structure:")
        def show_tree(node, indent=""):
            if node.segment_id != "root":
                steps_info = f" [{len(node.segment_steps)} steps]"
                print(f"{indent}├── {node.branch_decision or 'main'}{steps_info}")
                
                # Show the actual steps
                for step in node.segment_steps:
                    if isinstance(step, dict):
                        if 'stage_name' in step:
                            transforms = step.get('transforms', [])
                            if transforms:
                                transform_names = [t.__name__ if hasattr(t, '__name__') else str(t) 
                                                 for t in transforms]
                                print(f"{indent}│   └── {step['stage_name']}: {', '.join(transform_names)}")
                        else:
                            print(f"{indent}│   └── {step.get('name', step)}")
                    else:
                        print(f"{indent}│   └── {step}")
                        
            # Show children
            for i, (key, child) in enumerate(node.children.items()):
                is_last = i == len(node.children) - 1
                new_indent = indent + ("    " if is_last else "│   ")
                show_tree(child, new_indent)
                
        show_tree(tree)
        
    except Exception as e:
        print(f"\n✗ Failed at blueprint/tree stage:")
        print(f"  Error: {e}")
        traceback.print_exc()
        return 1
    
    # Step 2: Execute with segment executor
    print("\n" + "-" * 70)
    print("STEP 2: Execute Segments with FINN")
    print("-" * 70)
    
    try:
        from brainsmith.core.explorer import explore_execution_tree
        
        # Create config from design space
        blueprint_config = {
            "global_config": {
                "fail_fast": False,
                "output_products": design_space.global_config.output_stage.value
            },
            "finn_config": design_space.finn_config
        }
        
        print(f"\nStarting segment execution...")
        print(f"Config: {blueprint_config}")
        
        # This is where we'll see if FINN actually works
        result = explore_execution_tree(
            tree, 
            str(model_path), 
            str(output_dir), 
            blueprint_config
        )
        
        print(f"\n✓ Execution completed!")
        print(f"\nResults:")
        print(f"  - Total segments: {result.stats['total']}")
        print(f"  - Successful: {result.stats['successful']}")
        print(f"  - Failed: {result.stats['failed']}")
        print(f"  - Skipped: {result.stats['skipped']}")
        print(f"  - Cached: {result.stats['cached']}")
        print(f"  - Total time: {result.total_time:.2f}s")
        
        # Show failures if any
        if result.stats['failed'] > 0:
            print(f"\nFailed segments:")
            for seg_id, seg_result in result.segment_results.items():
                if not seg_result.success:
                    print(f"  - {seg_id}: {seg_result.error}")
        
        # Show generated files
        if output_dir.exists():
            print(f"\nGenerated files:")
            files = list(output_dir.rglob("*"))
            if files:
                for p in sorted(files)[:20]:  # Show first 20
                    if p.is_file():
                        rel_path = p.relative_to(output_dir)
                        print(f"  - {rel_path}")
                if len(files) > 20:
                    print(f"  ... and {len(files) - 20} more files")
            else:
                print("  (no files generated)")
                
    except ImportError as e:
        print(f"\n✗ Missing import:")
        print(f"  {e}")
        print("\nThis likely means FINN is not properly installed or")
        print("there's a missing component in the integration.")
        
    except Exception as e:
        print(f"\n✗ Execution failed:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to give helpful context
        if "build_dataflow_cfg" in str(e):
            print("\n💡 The error mentions build_dataflow_cfg - this is FINN's main entry point")
            print("   that our SegmentExecutor tries to call.")
        elif "ModelWrapper" in str(e):
            print("\n💡 The error mentions ModelWrapper - this is QONNX's model class")
            print("   used to load and transform ONNX models.")
            
        return 1
    
    print("\n" + "=" * 70)
    print("✓ TRUE END-TO-END TEST COMPLETED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())