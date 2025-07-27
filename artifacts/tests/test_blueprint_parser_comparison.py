#!/usr/bin/env python3
"""
Comprehensive test comparing original and refactored blueprint parsers.
Tests each method to ensure identical behavior.
"""

import sys
import os
import yaml
import tempfile
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

# Import both versions
from brainsmith.core.blueprint_parser import BlueprintParser as OriginalParser
from brainsmith.core.blueprint_parser_v2 import BlueprintParser as RefactoredParser
from brainsmith.core.tree_builder import TreeBuilder


class TestBlueprintParserComparison:
    """Test suite comparing original and refactored parsers."""
    
    def __init__(self):
        self.original = OriginalParser()
        self.refactored = RefactoredParser()
        self.tree_builder = TreeBuilder()
        self.test_results = []
        
    def run_all_tests(self):
        """Run all comparison tests."""
        print("=== Blueprint Parser Comparison Tests ===\n")
        
        # Test private methods that exist in both
        self.test_load_with_inheritance()
        self.test_deep_merge()
        self.test_parse_time_with_units()
        self.test_parse_global_config()
        self.test_validate_step()
        self.test_step_matches()
        self.test_parse_steps()
        self.test_parse_kernels()
        self.test_extract_config_and_mappings()
        
        # Test main parse function
        self.test_parse_complete()
        
        # Print summary
        self.print_summary()
        
    def test_load_with_inheritance(self):
        """Test YAML loading with inheritance."""
        test_name = "_load_with_inheritance"
        
        # Create test YAML files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Parent blueprint
            parent_path = os.path.join(tmpdir, "parent.yaml")
            parent_data = {
                "description": "Parent blueprint",
                "global_config": {
                    "output_stage": "generate_reports",
                    "working_directory": "parent_work"
                },
                "design_space": {
                    "steps": ["step1", "step2"]
                }
            }
            with open(parent_path, 'w') as f:
                yaml.dump(parent_data, f)
            
            # Child blueprint
            child_path = os.path.join(tmpdir, "child.yaml")
            child_data = {
                "extends": "parent.yaml",
                "description": "Child blueprint",
                "global_config": {
                    "working_directory": "child_work"
                }
            }
            with open(child_path, 'w') as f:
                yaml.dump(child_data, f)
            
            # Test both versions
            try:
                # Original uses _load_with_inheritance_and_parent
                orig_result, _ = self.original._load_with_inheritance_and_parent(child_path)
                refactored_result, _ = self.refactored._load_with_inheritance_and_parent(child_path)
                
                # Compare results
                assert orig_result == refactored_result, f"Results differ: {orig_result} vs {refactored_result}"
                assert orig_result["description"] == "Child blueprint"
                assert orig_result["global_config"]["working_directory"] == "child_work"
                assert orig_result["global_config"]["output_stage"] == "generate_reports"  # Inherited
                
                self.test_results.append((test_name, "PASS", "Inheritance works identically"))
            except Exception as e:
                self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_deep_merge(self):
        """Test deep merge functionality."""
        test_name = "_deep_merge"
        
        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [1, 2, 3]
        }
        override = {
            "a": 10,
            "b": {"c": 20, "f": 4},
            "g": 5
        }
        
        try:
            orig_result = self.original._deep_merge(base, override)
            refactored_result = self.refactored._deep_merge(base, override)
            
            assert orig_result == refactored_result, f"Results differ: {orig_result} vs {refactored_result}"
            assert orig_result["a"] == 10
            assert orig_result["b"]["c"] == 20
            assert orig_result["b"]["d"] == 3
            assert orig_result["b"]["f"] == 4
            assert orig_result["g"] == 5
            
            self.test_results.append((test_name, "PASS", "Deep merge works identically"))
        except Exception as e:
            self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_parse_time_with_units(self):
        """Test time parsing with units."""
        test_name = "_parse_time_with_units"
        
        test_cases = [
            ("5", 5.0),
            ("5ns", 5.0),
            ("5000ps", 5.0),
            ("0.005us", 5.0),
            ("0.000005ms", 5.0),
            (10, 10.0),
            (10.5, 10.5)
        ]
        
        try:
            for input_val, expected in test_cases:
                orig_result = self.original._parse_time_with_units(input_val)
                refactored_result = self.refactored._parse_time_with_units(input_val)
                
                assert orig_result == expected, f"Original failed for {input_val}: {orig_result} != {expected}"
                assert refactored_result == expected, f"Refactored failed for {input_val}: {refactored_result} != {expected}"
                assert orig_result == refactored_result
            
            self.test_results.append((test_name, "PASS", "Time parsing works identically"))
        except Exception as e:
            self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_parse_global_config(self):
        """Test global config parsing."""
        test_name = "_parse_global_config"
        
        config_data = {
            "output_stage": "compile_and_package",
            "working_directory": "test_work",
            "save_intermediate_models": True,
            "max_combinations": 50000,
            "timeout_minutes": 120,
            "fail_fast": True
        }
        
        try:
            orig_result = self.original._parse_global_config(config_data)
            refactored_result = self.refactored._parse_global_config(config_data)
            
            # Note: refactored returns BuildConfig, original returns GlobalConfig
            # Compare attributes instead
            assert orig_result.output_stage == refactored_result.output_stage
            assert orig_result.working_directory == refactored_result.working_directory
            assert orig_result.save_intermediate_models == refactored_result.save_intermediate_models
            assert orig_result.max_combinations == refactored_result.max_combinations
            assert orig_result.timeout_minutes == refactored_result.timeout_minutes
            assert orig_result.fail_fast == refactored_result.fail_fast
            
            self.test_results.append((test_name, "PASS", "Config parsing works identically (different class names)"))
        except Exception as e:
            self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_validate_step(self):
        """Test step validation."""
        test_name = "_validate_step"
        
        # Mock registry
        mock_registry = Mock()
        
        with patch('brainsmith.core.plugins.registry.has_step') as mock_has_step:
            mock_has_step.side_effect = lambda x: x in ["valid_step", "another_step"]
            
            try:
                # Test valid steps
                assert self.original._validate_step("valid_step", mock_registry) == "valid_step"
                assert self.refactored._validate_step("valid_step", mock_registry) == "valid_step"
                
                # Test skip indicators
                assert self.original._validate_step("~", mock_registry) == "~"
                assert self.refactored._validate_step("~", mock_registry) == "~"
                assert self.original._validate_step(None, mock_registry) == "~"
                assert self.refactored._validate_step(None, mock_registry) == "~"
                
                # Test invalid step (should raise)
                try:
                    self.original._validate_step("invalid_step", mock_registry)
                    assert False, "Original should have raised ValueError"
                except ValueError:
                    pass
                    
                try:
                    self.refactored._validate_step("invalid_step", mock_registry)
                    assert False, "Refactored should have raised ValueError"
                except ValueError:
                    pass
                
                self.test_results.append((test_name, "PASS", "Step validation works identically"))
            except Exception as e:
                self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_step_matches(self):
        """Test step matching logic."""
        test_name = "_step_matches"
        
        try:
            # String matching
            assert self.original._step_matches("step1", "step1") == True
            assert self.refactored._step_matches("step1", "step1") == True
            assert self.original._step_matches("step1", "step2") == False
            assert self.refactored._step_matches("step1", "step2") == False
            
            # List matching
            assert self.original._step_matches(["a", "b"], ["b", "a"]) == True
            assert self.refactored._step_matches(["a", "b"], ["b", "a"]) == True
            assert self.original._step_matches(["a", "b"], ["a", "c"]) == False
            assert self.refactored._step_matches(["a", "b"], ["a", "c"]) == False
            
            # Mismatched types
            assert self.original._step_matches("step", ["step"]) == False
            assert self.refactored._step_matches("step", ["step"]) == False
            
            self.test_results.append((test_name, "PASS", "Step matching works identically"))
        except Exception as e:
            self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_parse_steps(self):
        """Test step parsing with operations."""
        test_name = "_parse_steps"
        
        # Mock registry
        with patch('brainsmith.core.plugins.registry.has_step') as mock_has_step:
            mock_has_step.return_value = True
            
            # Simple step list
            steps_data = ["step1", "step2", ["opt1", "opt2"]]
            
            try:
                orig_result = self.original._parse_steps(steps_data)
                refactored_result = self.refactored._parse_steps(steps_data)
                
                assert orig_result == refactored_result
                assert len(orig_result) == 3
                assert orig_result[0] == "step1"
                assert orig_result[1] == "step2"
                assert orig_result[2] == ["opt1", "opt2"]
                
                self.test_results.append((test_name, "PASS", "Step parsing works identically"))
            except Exception as e:
                self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_parse_kernels(self):
        """Test kernel parsing."""
        test_name = "_parse_kernels"
        
        # Mock backend class
        MockBackend = type('MockBackend', (), {})
        
        with patch('brainsmith.core.plugins.registry.list_backends_by_kernel') as mock_list_backends:
            with patch('brainsmith.core.plugins.registry.get_backend') as mock_get_backend:
                mock_list_backends.return_value = ["backend1", "backend2"]
                mock_get_backend.return_value = MockBackend
                
                kernels_data = ["kernel1", {"kernel2": ["backend1"]}]
                
                try:
                    orig_result = self.original._parse_kernels(kernels_data)
                    refactored_result = self.refactored._parse_kernels(kernels_data)
                    
                    # Both should return list of (kernel_name, [backend_classes])
                    assert len(orig_result) == len(refactored_result)
                    assert orig_result[0][0] == refactored_result[0][0] == "kernel1"
                    assert orig_result[1][0] == refactored_result[1][0] == "kernel2"
                    
                    self.test_results.append((test_name, "PASS", "Kernel parsing works identically"))
                except Exception as e:
                    self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_extract_config_and_mappings(self):
        """Test config extraction."""
        test_name = "_extract_config_and_mappings"
        
        data = {
            "global_config": {
                "output_stage": "generate_reports",
                "working_directory": "work"
            },
            "finn_config": {
                "board": "pynq_z2"
            },
            "platform": "zynq",
            "target_clk": "5ns"
        }
        
        try:
            orig_global, orig_finn = self.original._extract_config_and_mappings(data)
            refactored_global, refactored_finn = self.refactored._extract_config_and_mappings(data)
            
            # Compare global configs (different class names)
            assert orig_global.output_stage.value == refactored_global.output_stage.value
            assert orig_global.working_directory == refactored_global.working_directory
            
            # Compare FINN configs
            assert orig_finn == refactored_finn
            assert orig_finn["board"] == "zynq"  # Should override from platform
            assert orig_finn["synth_clk_period_ns"] == 5.0
            
            self.test_results.append((test_name, "PASS", "Config extraction works identically"))
        except Exception as e:
            self.test_results.append((test_name, "FAIL", str(e)))
    
    def test_parse_complete(self):
        """Test complete parse function."""
        test_name = "parse (main function)"
        
        # Create test blueprint
        with tempfile.TemporaryDirectory() as tmpdir:
            blueprint_path = os.path.join(tmpdir, "test.yaml")
            model_path = os.path.join(tmpdir, "test.onnx")
            
            # Create dummy model file
            with open(model_path, 'wb') as f:
                f.write(b'dummy')
            
            blueprint_data = {
                "description": "Test blueprint",
                "global_config": {
                    "output_stage": "generate_reports"
                },
                "finn_config": {
                    "board": "pynq_z2",
                    "synth_clk_period_ns": 5.0
                },
                "design_space": {
                    "steps": ["step1", "step2"],
                    "kernels": []
                }
            }
            
            with open(blueprint_path, 'w') as f:
                yaml.dump(blueprint_data, f)
            
            # Mock registry functions
            with patch('brainsmith.core.plugins.registry.has_step') as mock_has_step:
                mock_has_step.return_value = True
                
                try:
                    # Original returns (DesignSpace, ExecutionNode)
                    orig_space, orig_tree = self.original.parse(blueprint_path, model_path)
                    
                    # Refactored returns only DesignSpace
                    refactored_space = self.refactored.parse(blueprint_path, model_path)
                    
                    # Build tree using TreeBuilder for comparison
                    refactored_tree = self.tree_builder.build_tree(refactored_space)
                    
                    # Compare DesignSpaces
                    assert orig_space.model_path == refactored_space.model_path
                    assert orig_space.steps == refactored_space.steps
                    assert orig_space.kernel_backends == refactored_space.kernel_backends
                    assert orig_space.finn_config == refactored_space.finn_config
                    
                    # Compare trees (basic structure)
                    assert type(orig_tree).__name__ == type(refactored_tree).__name__ == "ExecutionNode"
                    assert orig_tree.segment_id == refactored_tree.segment_id
                    assert len(orig_tree.children) == len(refactored_tree.children)
                    
                    self.test_results.append((test_name, "PASS", "Complete parsing works identically"))
                except Exception as e:
                    self.test_results.append((test_name, "FAIL", str(e)))
    
    def print_summary(self):
        """Print test results summary."""
        print("\n=== Test Results Summary ===\n")
        
        passed = sum(1 for _, status, _ in self.test_results if status == "PASS")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAIL")
        
        for test_name, status, message in self.test_results:
            symbol = "✓" if status == "PASS" else "✗"
            print(f"{symbol} {test_name}: {message}")
        
        print(f"\nTotal: {len(self.test_results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed == 0:
            print("\n✅ All tests passed! The refactored parser is functionally identical.")
        else:
            print(f"\n❌ {failed} tests failed. The refactored parser has differences.")


if __name__ == "__main__":
    # Suppress import warnings
    import warnings
    warnings.filterwarnings("ignore", message="Failed to import kernel operators")
    
    tester = TestBlueprintParserComparison()
    tester.run_all_tests()