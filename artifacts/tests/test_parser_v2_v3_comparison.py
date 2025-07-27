#!/usr/bin/env python3
"""
Test comparing v2 and v3 blueprint parsers for functional equivalence.
"""

import sys
import os
import yaml
import tempfile

# Prevent module loading issues
os.environ['BSMITH_PLUGINS_STRICT'] = 'false'
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')


def test_basic_parsing():
    """Test basic blueprint parsing."""
    print("\n1. Testing basic parsing...")
    
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as V2Parser
    from brainsmith.core.blueprint_parser_v3 import parse_blueprint as v3_parse
    
    v2 = V2Parser()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test blueprint
        blueprint_path = os.path.join(tmpdir, "test.yaml")
        model_path = os.path.join(tmpdir, "test.onnx")
        
        # Create dummy model
        with open(model_path, 'wb') as f:
            f.write(b'dummy')
        
        blueprint_data = {
            "description": "Test blueprint",
            "global_config": {
                "output_stage": "generate_reports",
                "working_directory": "test_work",
                "save_intermediate_models": True,
                "max_combinations": 1000
            },
            "finn_config": {
                "board": "pynq_z2",
                "synth_clk_period_ns": 5.0
            },
            "design_space": {
                "steps": ["qonnx_to_finn", "tidy_up", ["streamline", "~"]],
                "kernels": []
            }
        }
        
        with open(blueprint_path, 'w') as f:
            yaml.dump(blueprint_data, f)
        
        try:
            # Parse with both versions
            v2_result = v2.parse(blueprint_path, model_path)
            v3_result = v3_parse(blueprint_path, model_path)
            
            # Compare results
            checks = [
                (v2_result.model_path == v3_result.model_path, "Model path matches"),
                (v2_result.steps == v3_result.steps, "Steps match"),
                (v2_result.kernel_backends == v3_result.kernel_backends, "Kernels match"),
                (v2_result.finn_config == v3_result.finn_config, "FINN config matches"),
                # Compare build config attributes
                (v2_result.global_config.output_stage == v3_result.global_config.output_stage, "Output stage matches"),
                (v2_result.global_config.working_directory == v3_result.global_config.working_directory, "Working dir matches"),
                (v2_result.global_config.save_intermediate_models == v3_result.global_config.save_intermediate_models, "Save intermediate matches"),
                (v2_result.global_config.max_combinations == v3_result.global_config.max_combinations, "Max combinations matches"),
            ]
            
            passed = sum(1 for check, _ in checks if check)
            for check, desc in checks:
                if not check:
                    print(f"  ✗ Failed: {desc}")
            
            print(f"  ✓ Passed {passed}/{len(checks)} checks")
            return passed == len(checks)
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_inheritance():
    """Test YAML inheritance."""
    print("\n2. Testing inheritance...")
    
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as V2Parser
    from brainsmith.core.blueprint_parser_v3 import parse_blueprint as v3_parse
    
    v2 = V2Parser()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test.onnx")
        with open(model_path, 'wb') as f:
            f.write(b'dummy')
        
        # Parent blueprint
        parent_path = os.path.join(tmpdir, "parent.yaml")
        parent_data = {
            "global_config": {
                "output_stage": "generate_reports",
                "working_directory": "parent_work",
                "timeout_minutes": 60
            },
            "finn_config": {
                "board": "pynq_z2",
                "synth_clk_period_ns": 5.0
            },
            "design_space": {
                "steps": ["step1", "step2"],
                "kernels": ["matmul"]
            }
        }
        with open(parent_path, 'w') as f:
            yaml.dump(parent_data, f)
        
        # Child blueprint
        child_path = os.path.join(tmpdir, "child.yaml")
        child_data = {
            "extends": "parent.yaml",
            "global_config": {
                "working_directory": "child_work"
            },
            "design_space": {
                "steps": ["step3", "step4"]  # Override parent steps
            }
        }
        with open(child_path, 'w') as f:
            yaml.dump(child_data, f)
        
        try:
            v2_result = v2.parse(child_path, model_path)
            v3_result = v3_parse(child_path, model_path)
            
            checks = [
                (v2_result.steps == v3_result.steps == ["step3", "step4"], "Child steps override"),
                (v2_result.kernel_backends == v3_result.kernel_backends, "Kernels inherited"),
                (v2_result.global_config.working_directory == v3_result.global_config.working_directory == "child_work", "Child config overrides"),
                (v2_result.global_config.timeout_minutes == v3_result.global_config.timeout_minutes == 60, "Parent config inherited"),
            ]
            
            passed = sum(1 for check, _ in checks if check)
            for check, desc in checks:
                if not check:
                    print(f"  ✗ Failed: {desc}")
            
            print(f"  ✓ Passed {passed}/{len(checks)} checks")
            return passed == len(checks)
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_legacy_mappings():
    """Test legacy parameter mappings."""
    print("\n3. Testing legacy mappings...")
    
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as V2Parser
    from brainsmith.core.blueprint_parser_v3 import parse_blueprint as v3_parse
    
    v2 = V2Parser()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        blueprint_path = os.path.join(tmpdir, "test.yaml")
        model_path = os.path.join(tmpdir, "test.onnx")
        
        with open(model_path, 'wb') as f:
            f.write(b'dummy')
        
        # Use legacy smithy format
        blueprint_data = {
            "platform": "zynq_7000",  # Legacy: maps to board
            "target_clk": "10ns",      # Legacy: maps to synth_clk_period_ns
            "output_stage": "generate_reports",  # Top-level param
            "design_space": {
                "steps": ["step1"],
                "kernels": []
            }
        }
        
        with open(blueprint_path, 'w') as f:
            yaml.dump(blueprint_data, f)
        
        try:
            v2_result = v2.parse(blueprint_path, model_path)
            v3_result = v3_parse(blueprint_path, model_path)
            
            checks = [
                (v2_result.finn_config['board'] == v3_result.finn_config['board'] == "zynq_7000", "Platform mapped to board"),
                (v2_result.finn_config['synth_clk_period_ns'] == v3_result.finn_config['synth_clk_period_ns'] == 10.0, "Target clk mapped"),
                (v2_result.global_config.output_stage.value == v3_result.global_config.output_stage.value == "generate_reports", "Top-level param mapped"),
            ]
            
            passed = sum(1 for check, _ in checks if check)
            for check, desc in checks:
                if not check:
                    print(f"  ✗ Failed: {desc}")
            
            print(f"  ✓ Passed {passed}/{len(checks)} checks")
            return passed == len(checks)
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_time_parsing():
    """Test time unit parsing."""
    print("\n4. Testing time parsing...")
    
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as V2Parser
    from brainsmith.core.blueprint_parser_v3 import parse_blueprint as v3_parse
    
    v2 = V2Parser()
    
    test_cases = [
        ("5ns", 5.0),
        ("5000ps", 5.0),
        ("0.005us", 5.0),
        ("10", 10.0),  # No unit = ns
    ]
    
    all_passed = True
    
    for time_str, expected in test_cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            blueprint_path = os.path.join(tmpdir, "test.yaml")
            model_path = os.path.join(tmpdir, "test.onnx")
            
            with open(model_path, 'wb') as f:
                f.write(b'dummy')
            
            blueprint_data = {
                "target_clk": time_str,
                "platform": "test_board",
                "global_config": {"output_stage": "generate_reports"},
                "design_space": {"steps": ["step1"], "kernels": []}
            }
            
            with open(blueprint_path, 'w') as f:
                yaml.dump(blueprint_data, f)
            
            try:
                v2_result = v2.parse(blueprint_path, model_path)
                v3_result = v3_parse(blueprint_path, model_path)
                
                v2_time = v2_result.finn_config['synth_clk_period_ns']
                v3_time = v3_result.finn_config['synth_clk_period_ns']
                
                if v2_time == v3_time == expected:
                    print(f"  ✓ {time_str} -> {expected}ns")
                else:
                    print(f"  ✗ {time_str}: v2={v2_time}, v3={v3_time}, expected={expected}")
                    all_passed = False
                    
            except Exception as e:
                print(f"  ✗ {time_str}: Exception: {e}")
                all_passed = False
    
    return all_passed


def count_lines():
    """Count lines in each version."""
    print("\n5. Line count comparison...")
    
    files = [
        ("blueprint_parser.py (original)", "/home/tafk/dev/brainsmith-4/brainsmith/core/blueprint_parser.py"),
        ("blueprint_parser_v2.py", "/home/tafk/dev/brainsmith-4/brainsmith/core/blueprint_parser_v2.py"),
        ("blueprint_parser_v3.py", "/home/tafk/dev/brainsmith-4/brainsmith/core/blueprint_parser_v3.py"),
        ("yaml_utils.py", "/home/tafk/dev/brainsmith-4/brainsmith/core/yaml_utils.py"),
        ("time_utils.py", "/home/tafk/dev/brainsmith-4/brainsmith/core/time_utils.py"),
        ("validation.py", "/home/tafk/dev/brainsmith-4/brainsmith/core/validation.py"),
    ]
    
    total_v3 = 0
    for name, path in files:
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = len(f.readlines())
                print(f"  {name}: {lines} lines")
                if "v3" in name or name in ["yaml_utils.py", "time_utils.py", "validation.py"]:
                    total_v3 += lines
    
    print(f"\n  Total v3 + utilities: {total_v3} lines")
    print(f"  Reduction from v2 (511 lines): {511 - total_v3} lines ({(511 - total_v3) / 511 * 100:.1f}%)")


def main():
    """Run all tests."""
    print("=== Blueprint Parser V2 vs V3 Comparison ===")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    results = []
    
    # Run tests
    results.append(("Basic parsing", test_basic_parsing()))
    results.append(("Inheritance", test_inheritance()))
    results.append(("Legacy mappings", test_legacy_mappings()))
    results.append(("Time parsing", test_time_parsing()))
    
    # Line count
    count_lines()
    
    # Summary
    print("\n=== Summary ===")
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for name, result in results:
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}")
    
    print(f"\nTotal: {passed}/{total} test groups passed")
    
    if passed == total:
        print("\n✅ V3 is functionally equivalent to V2!")
        print("✨ And much cleaner for Arete!")
    else:
        print(f"\n❌ {total - passed} test groups failed.")


if __name__ == "__main__":
    main()