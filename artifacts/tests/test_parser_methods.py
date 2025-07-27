#!/usr/bin/env python3
"""
Test individual methods of blueprint parsers for functional equivalence.
"""

import sys
import os
import yaml
import tempfile

# Prevent module loading issues
os.environ['BSMITH_PLUGINS_STRICT'] = 'false'
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

def test_deep_merge():
    """Test _deep_merge method."""
    print("\n1. Testing _deep_merge...")
    
    from brainsmith.core.blueprint_parser import BlueprintParser as Original
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as Refactored
    
    orig = Original()
    refact = Refactored()
    
    # Test cases
    test_cases = [
        # Simple merge
        ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
        # Override
        ({"a": 1}, {"a": 2}, {"a": 2}),
        # Nested merge
        ({"x": {"y": 1}}, {"x": {"z": 2}}, {"x": {"y": 1, "z": 2}}),
        # Deep override
        ({"x": {"y": 1}}, {"x": {"y": 2}}, {"x": {"y": 2}}),
        # List override (not merge)
        ({"a": [1, 2]}, {"a": [3, 4]}, {"a": [3, 4]}),
    ]
    
    passed = 0
    for base, override, expected in test_cases:
        orig_result = orig._deep_merge(base.copy(), override.copy())
        refact_result = refact._deep_merge(base.copy(), override.copy())
        
        if orig_result == refact_result == expected:
            passed += 1
        else:
            print(f"  ✗ Failed: {base} + {override}")
            print(f"    Original: {orig_result}")
            print(f"    Refactored: {refact_result}")
            print(f"    Expected: {expected}")
    
    print(f"  ✓ Passed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def test_parse_time_with_units():
    """Test _parse_time_with_units method."""
    print("\n2. Testing _parse_time_with_units...")
    
    from brainsmith.core.blueprint_parser import BlueprintParser as Original
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as Refactored
    
    orig = Original()
    refact = Refactored()
    
    # Test cases: (input, expected_output_in_ns)
    test_cases = [
        ("5", 5.0),
        ("5ns", 5.0),
        ("5000ps", 5.0),
        ("0.005us", 5.0),
        ("0.000005ms", 5.0),
        (10, 10.0),
        (10.5, 10.5),
        ("2.5ns", 2.5),
        ("1us", 1000.0),
        ("1ms", 1000000.0),
    ]
    
    passed = 0
    for input_val, expected in test_cases:
        try:
            orig_result = orig._parse_time_with_units(input_val)
            refact_result = refact._parse_time_with_units(input_val)
            
            if orig_result == refact_result == expected:
                passed += 1
            else:
                print(f"  ✗ Failed: {input_val}")
                print(f"    Original: {orig_result}")
                print(f"    Refactored: {refact_result}")
                print(f"    Expected: {expected}")
        except Exception as e:
            print(f"  ✗ Exception for {input_val}: {e}")
    
    print(f"  ✓ Passed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def test_step_matches():
    """Test _step_matches method."""
    print("\n3. Testing _step_matches...")
    
    from brainsmith.core.blueprint_parser import BlueprintParser as Original
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as Refactored
    
    orig = Original()
    refact = Refactored()
    
    # Test cases: (step, target, expected)
    test_cases = [
        # String matching
        ("step1", "step1", True),
        ("step1", "step2", False),
        ("", "", True),
        # List matching (order doesn't matter)
        (["a", "b"], ["b", "a"], True),
        (["a", "b"], ["a", "b"], True),
        (["a"], ["a"], True),
        (["a", "b"], ["a", "c"], False),
        # Type mismatches
        ("step", ["step"], False),
        (["step"], "step", False),
        # None handling
        ("step", None, False),
        (None, "step", False),
    ]
    
    passed = 0
    for step, target, expected in test_cases:
        try:
            orig_result = orig._step_matches(step, target)
            refact_result = refact._step_matches(step, target)
            
            if orig_result == refact_result == expected:
                passed += 1
            else:
                print(f"  ✗ Failed: {step} matches {target}")
                print(f"    Original: {orig_result}")
                print(f"    Refactored: {refact_result}")
                print(f"    Expected: {expected}")
        except Exception as e:
            print(f"  ✗ Exception for {step}, {target}: {e}")
    
    print(f"  ✓ Passed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def test_yaml_inheritance():
    """Test YAML inheritance functionality."""
    print("\n4. Testing YAML inheritance...")
    
    from brainsmith.core.blueprint_parser import BlueprintParser as Original
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as Refactored
    
    orig = Original()
    refact = Refactored()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parent YAML
        parent_path = os.path.join(tmpdir, "parent.yaml")
        parent_data = {
            "name": "parent",
            "global_config": {
                "output_stage": "generate_reports",
                "working_directory": "parent_work",
                "save_intermediate_models": True
            },
            "design_space": {
                "steps": ["step1", "step2"],
                "kernels": ["kernel1"]
            }
        }
        with open(parent_path, 'w') as f:
            yaml.dump(parent_data, f)
        
        # Create child YAML
        child_path = os.path.join(tmpdir, "child.yaml")
        child_data = {
            "extends": "parent.yaml",
            "name": "child",
            "global_config": {
                "working_directory": "child_work",
                "max_combinations": 1000
            },
            "design_space": {
                "steps": ["step3", "step4"]
            }
        }
        with open(child_path, 'w') as f:
            yaml.dump(child_data, f)
        
        try:
            # Both parsers use _load_with_inheritance_and_parent
            orig_data, orig_parent = orig._load_with_inheritance_and_parent(child_path)
            refact_data, refact_parent = refact._load_with_inheritance_and_parent(child_path)
            
            # Check merged data
            checks = [
                (orig_data == refact_data, "Merged data matches"),
                (orig_data["name"] == "child", "Child name preserved"),
                (orig_data["global_config"]["working_directory"] == "child_work", "Child config overrides"),
                (orig_data["global_config"]["output_stage"] == "generate_reports", "Parent config inherited"),
                (orig_data["global_config"]["save_intermediate_models"] == True, "Parent config preserved"),
                (orig_data["global_config"]["max_combinations"] == 1000, "Child adds new config"),
                (orig_data["design_space"]["steps"] == ["step3", "step4"], "Steps overridden"),
                (orig_data["design_space"]["kernels"] == ["kernel1"], "Kernels inherited"),
            ]
            
            passed = sum(1 for check, _ in checks if check)
            for check, desc in checks:
                if not check:
                    print(f"  ✗ Failed: {desc}")
            
            print(f"  ✓ Passed {passed}/{len(checks)} checks")
            return passed == len(checks)
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            return False


def test_config_parsing():
    """Test configuration parsing."""
    print("\n5. Testing configuration parsing...")
    
    from brainsmith.core.blueprint_parser import BlueprintParser as Original
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as Refactored
    from brainsmith.core.design_space import OutputStage as OriginalOutputStage
    from brainsmith.core.design_space_v2 import OutputStage as RefactoredOutputStage
    
    orig = Original()
    refact = Refactored()
    
    config_data = {
        "output_stage": "compile_and_package",
        "working_directory": "custom_work",
        "save_intermediate_models": True,
        "max_combinations": 50000,
        "timeout_minutes": 120,
        "fail_fast": True
    }
    
    try:
        orig_config = orig._parse_global_config(config_data)
        refact_config = refact._parse_global_config(config_data)
        
        # Note: Original returns GlobalConfig, refactored returns BuildConfig
        # Compare attributes
        checks = [
            (orig_config.output_stage.value == refact_config.output_stage.value, "Output stage matches"),
            (orig_config.working_directory == refact_config.working_directory, "Working directory matches"),
            (orig_config.save_intermediate_models == refact_config.save_intermediate_models, "Save intermediate matches"),
            (orig_config.max_combinations == refact_config.max_combinations, "Max combinations matches"),
            (orig_config.timeout_minutes == refact_config.timeout_minutes, "Timeout matches"),
            (orig_config.fail_fast == refact_config.fail_fast, "Fail fast matches"),
        ]
        
        passed = sum(1 for check, _ in checks if check)
        for check, desc in checks:
            if not check:
                print(f"  ✗ Failed: {desc}")
        
        print(f"  ✓ Passed {passed}/{len(checks)} checks")
        print(f"  Note: Original uses GlobalConfig, refactored uses BuildConfig")
        return passed == len(checks)
        
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def test_tree_building_separation():
    """Test that tree building was properly separated."""
    print("\n6. Testing tree building separation...")
    
    from brainsmith.core.blueprint_parser import BlueprintParser as Original
    from brainsmith.core.blueprint_parser_v2 import BlueprintParser as Refactored
    from brainsmith.core.tree_builder import TreeBuilder
    
    orig = Original()
    refact = Refactored()
    builder = TreeBuilder()
    
    # Check method presence
    checks = [
        # Original should have tree methods
        (hasattr(orig, '_build_execution_tree'), "Original has _build_execution_tree"),
        (hasattr(orig, '_flush_steps'), "Original has _flush_steps"),
        (hasattr(orig, '_create_branches'), "Original has _create_branches"),
        (hasattr(orig, '_validate_tree_size'), "Original has _validate_tree_size"),
        
        # Refactored should NOT have tree methods
        (not hasattr(refact, '_build_execution_tree'), "Refactored lacks _build_execution_tree"),
        (not hasattr(refact, '_flush_steps'), "Refactored lacks _flush_steps"),
        (not hasattr(refact, '_create_branches'), "Refactored lacks _create_branches"),
        (not hasattr(refact, '_validate_tree_size'), "Refactored lacks _validate_tree_size"),
        
        # TreeBuilder should have these methods
        (hasattr(builder, 'build_tree'), "TreeBuilder has build_tree"),
        (hasattr(builder, '_flush_steps'), "TreeBuilder has _flush_steps"),
        (hasattr(builder, '_create_branches'), "TreeBuilder has _create_branches"),
        (hasattr(builder, '_validate_tree_size'), "TreeBuilder has _validate_tree_size"),
    ]
    
    passed = sum(1 for check, _ in checks if check)
    for check, desc in checks:
        symbol = "✓" if check else "✗"
        print(f"  {symbol} {desc}")
    
    print(f"\n  Total: {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def main():
    """Run all tests."""
    print("=== Blueprint Parser Method Comparison ===")
    
    results = []
    
    # Run tests
    results.append(("Deep merge", test_deep_merge()))
    results.append(("Time parsing", test_parse_time_with_units()))
    results.append(("Step matching", test_step_matches()))
    results.append(("YAML inheritance", test_yaml_inheritance()))
    results.append(("Config parsing", test_config_parsing()))
    results.append(("Tree separation", test_tree_building_separation()))
    
    # Summary
    print("\n=== Summary ===")
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for name, result in results:
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}")
    
    print(f"\nTotal: {passed}/{total} test groups passed")
    
    if passed == total:
        print("\n✅ All tests passed! The refactoring maintains functional equivalence.")
    else:
        print(f"\n❌ {total - passed} test groups failed.")


if __name__ == "__main__":
    main()