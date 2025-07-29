# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for unified Executor - testing tree traversal without FINN."""

import pytest
import tempfile
from pathlib import Path
from brainsmith.core.execution_tree import ExecutionSegment
from brainsmith.core.explorer.executor import Executor
from brainsmith.core.explorer.finn_adapter import FINNAdapter
from brainsmith.core.explorer.types import ExecutionError


class TestExecutor:
    """Test executor tree traversal and logic."""
    
    def test_executor_init(self):
        """Test executor initialization and validation."""
        # Create mock adapter (won't be used in these tests)
        adapter = None  # We'll mock this when needed
        
        # Valid config
        finn_config = {"synth_clk_period_ns": 5.0, "board": "Pynq-Z1"}
        global_config = {"fail_fast": True, "output_products": "rtl"}
        
        executor = Executor(adapter, finn_config, global_config)
        
        assert executor.fail_fast is True
        assert executor.output_product == "rtl"
        assert len(executor.output_map["rtl"]) == 3
        
        # Missing required config
        with pytest.raises(ValueError, match="synth_clk_period_ns"):
            Executor(adapter, {}, {})
        
        with pytest.raises(ValueError, match="board"):
            Executor(adapter, {"synth_clk_period_ns": 5.0}, {})
    
    def test_tree_traversal_order(self):
        """Test depth-first traversal order."""
        # Build test tree
        root = ExecutionSegment(segment_steps=[{"name": "step1"}])
        child1 = root.add_child("branch1", [{"name": "step2"}])
        child2 = root.add_child("branch2", [{"name": "step3"}])
        grandchild = child1.add_child("leaf", [{"name": "step4"}])
        
        # Track execution order
        executed = []
        
        class MockAdapter:
            def prepare_model(self, src, dst):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.touch()
            
            def build(self, model, config, output_dir):
                # Track execution
                segment_id = config["output_dir"].split("/")[-1]
                executed.append(segment_id)
                # Return success
                output = output_dir / "intermediate_models" / "out.onnx"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                return output
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.touch()
            output = Path(tmpdir) / "output"
            
            executor = Executor(
                MockAdapter(),
                {"synth_clk_period_ns": 5.0, "board": "test"},
                {"fail_fast": False}
            )
            
            result = executor.execute(root, model, output)
            
            # Verify depth-first order
            assert executed == ["root", "branch1", "leaf", "branch2"]
            assert result.stats["total"] == 4
            assert result.stats["successful"] == 4
    
    def test_fail_fast_mode(self):
        """Test fail-fast stops execution."""
        root = ExecutionSegment(segment_steps=[{"name": "step1"}])
        child1 = root.add_child("child1", [{"name": "step2"}])
        child2 = root.add_child("child2", [{"name": "step3"}])
        
        class FailingAdapter:
            def prepare_model(self, src, dst):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.touch()
            
            def build(self, model, config, output_dir):
                # Success for root
                if "child1" not in str(output_dir) and "child2" not in str(output_dir):
                    output = output_dir / "intermediate_models" / "out.onnx"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.touch()
                    return output
                # Fail on child1
                if "child1" in str(output_dir):
                    raise RuntimeError("Test failure")
                return None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.touch()
            output = Path(tmpdir) / "output"
            
            executor = Executor(
                FailingAdapter(),
                {"synth_clk_period_ns": 5.0, "board": "test"},
                {"fail_fast": True}
            )
            
            with pytest.raises(ExecutionError, match="Segment 'child1' build failed"):
                executor.execute(root, model, output)
    
    def test_skip_on_failure(self):
        """Test descendants are skipped on failure."""
        root = ExecutionSegment(segment_steps=[{"name": "step1"}])
        parent = root.add_child("parent", [{"name": "step2"}])
        child = parent.add_child("child", [{"name": "step3"}])
        
        class SelectiveFailAdapter:
            def prepare_model(self, src, dst):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.touch()
            
            def build(self, model, config, output_dir):
                # Fail on parent
                if "parent" in str(output_dir) and "child" not in str(output_dir):
                    raise RuntimeError("Parent failed")
                # Success for others
                output = output_dir / "intermediate_models" / "out.onnx"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                return output
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.touch()
            output = Path(tmpdir) / "output"
            
            executor = Executor(
                SelectiveFailAdapter(),
                {"synth_clk_period_ns": 5.0, "board": "test"},
                {"fail_fast": False}
            )
            
            result = executor.execute(root, model, output)
            
            # Check results
            assert result.segment_results["root"].success is True
            assert result.segment_results["parent"].success is False
            assert result.segment_results["parent/child"].error == "Skipped"
            
            assert result.stats["successful"] == 1
            assert result.stats["failed"] == 1
            assert result.stats["skipped"] == 1
    
    def test_caching_behavior(self):
        """Test that existing outputs are cached."""
        root = ExecutionSegment(segment_steps=[{"name": "step1"}])
        child = root.add_child("child", [{"name": "step2"}])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.touch()
            output = Path(tmpdir) / "output"
            
            # Pre-create output for root to simulate cache
            root_out = output / "root" / "root_output.onnx"
            root_out.parent.mkdir(parents=True, exist_ok=True)
            root_out.touch()
            
            # Track what gets built
            built = []
            
            class TrackingAdapter:
                def prepare_model(self, src, dst):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.touch()
                
                def build(self, model, config, output_dir):
                    built.append(str(output_dir))
                    output = output_dir / "intermediate_models" / "out.onnx"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.touch()
                    return output
            
            executor = Executor(
                TrackingAdapter(),
                {"synth_clk_period_ns": 5.0, "board": "test"},
                {"fail_fast": False}
            )
            
            result = executor.execute(root, model, output)
            
            # Root should be cached, child should build
            assert len(built) == 1
            assert "child" in built[0]
            
            assert result.segment_results["root"].cached is True
            assert result.segment_results["child"].cached is False
            assert result.stats["cached"] == 1
    
    def test_output_product_mapping(self):
        """Test different output product configurations."""
        adapter = None
        finn_config = {"synth_clk_period_ns": 5.0, "board": "test"}
        
        # Test each output product
        for product, expected_outputs in [
            ("df", 1),      # Just reports
            ("rtl", 3),     # Reports + RTL
            ("dcp", 5),     # Everything
        ]:
            executor = Executor(
                adapter,
                finn_config,
                {"output_products": product}
            )
            assert len(executor.output_map[product]) == expected_outputs