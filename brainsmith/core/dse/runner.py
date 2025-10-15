# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DSE segment runner for executing segment builds."""

import time
from pathlib import Path
from typing import Dict, Any, Set
from brainsmith.core.dse.segment import DSESegment
from brainsmith.core.dse.tree import DSETree
from brainsmith.core.plugins import get_step
from .types import SegmentResult, TreeExecutionResult, ExecutionError
from .finn_adapter import FINNAdapter
from .utils import share_artifacts_at_branch


class SegmentRunner:
    """Runs DSE segments using FINN.

    Handles both tree traversal and individual segment execution
    using FINNAdapter for all FINN interactions.
    """
    
    def __init__(
        self,
        finn_adapter: FINNAdapter,
        base_config: Dict[str, Any],
        kernel_selections: list = None
    ) -> None:
        """Initialize runner.

        Args:
            finn_adapter: Adapter for FINN-specific operations
            base_config: FINN configuration from blueprint
            kernel_selections: Optional list of (kernel, backend) tuples
        """
        self.finn_adapter = finn_adapter
        self.base_config = base_config
        self.kernel_selections = kernel_selections or []
        
        # Extract settings from FINN config
        self.fail_fast = False  # TODO: Add more robust tree exit options
        output_products = base_config.get("output_products", ["estimates"])
        # Take first output product as primary target
        self.output_product = output_products[0] if output_products else "estimates"
        
        # Validate required FINN config fields
        if "synth_clk_period_ns" not in base_config:
            raise ValueError("finn_config must specify synth_clk_period_ns")
        if "board" not in base_config:
            raise ValueError("finn_config must specify board")
        
        # Map output products to FINN types
        self.output_map = {
            "estimates": ["estimate_reports"],
            "rtl_sim": ["estimate_reports", "rtlsim_performance"],
            "ip_gen": ["estimate_reports", "rtlsim_performance", "stitched_ip"],
            "bitfile": [
                "estimate_reports",
                "rtlsim_performance",
                "stitched_ip",
                "bitfile",
                "deployment_package"
            ]
        }
    
    def run_tree(
        self,
        tree: DSETree,
        initial_model: Path,
        output_dir: Path
    ) -> TreeExecutionResult:
        """Run all segments in the DSE tree.
        
        Args:
            tree: DSE tree to execute
            initial_model: Path to initial ONNX model
            output_dir: Base output directory
            
        Returns:
            TreeExecutionResult with all segment results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Executing tree with fail_fast={self.fail_fast}")
        print(f"Output: {output_dir}")
        
        results = {}
        skipped = set()
        start_time = time.time()
        
        # Use a stack for cleaner iteration
        stack = [(tree.root, initial_model, 0)]
        
        while stack:
            segment, input_model, depth = stack.pop()
            indent = "  " * depth
            
            # Skip if parent failed
            if segment.segment_id in skipped:
                print(f"{indent}⊘ Skipped: {segment.segment_id}")
                results[segment.segment_id] = SegmentResult(
                    success=False,
                    segment_id=segment.segment_id,
                    error="Skipped"
                )
                continue
            
            # Execute segment
            print(f"{indent}→ {segment.segment_id}")
            
            # Skip empty segments (e.g., root with immediate branches)
            if not segment.steps:
                print(f"{indent}  (empty segment, passing through)")
                # Create a pass-through result
                results[segment.segment_id] = SegmentResult(
                    success=True,
                    segment_id=segment.segment_id,
                    output_model=input_model,  # Pass input as output
                    output_dir=output_dir / segment.segment_id,
                    execution_time=0
                )
                # Add children to stack
                for child in reversed(list(segment.children.values())):
                    stack.append((child, input_model, depth + 1))
                continue
            
            try:
                result = self.run_segment(segment, input_model, output_dir)
                results[segment.segment_id] = result
            except ExecutionError as e:
                # Handle execution errors properly
                print(f"✗ Failed: {segment.segment_id}")
                print(f"  Error: {str(e)}")
                if self.fail_fast:
                    raise
                
                # Create failure result with actual exception
                results[segment.segment_id] = SegmentResult(
                    success=False,
                    segment_id=segment.segment_id,
                    error=str(e),
                    execution_time=0
                )
                
                # Mark descendants for skipping
                self._mark_descendants_skipped(segment, skipped)
                # Still need to add children to stack so they get marked as skipped
                for child in reversed(list(segment.children.values())):
                    stack.append((child, None, depth + 1))
                continue
            except Exception as e:
                # Catch any unexpected errors
                print(f"✗ Failed: {segment.segment_id}")
                print(f"  Unexpected error: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Create failure result
                results[segment.segment_id] = SegmentResult(
                    success=False,
                    segment_id=segment.segment_id,
                    error=f"{type(e).__name__}: {str(e)}",
                    execution_time=0
                )
                
                # Mark descendants for skipping
                self._mark_descendants_skipped(segment, skipped)
                for child in reversed(list(segment.children.values())):
                    stack.append((child, None, depth + 1))
                continue
            
            # Share artifacts at branch points
            if segment.is_branch_point:
                share_artifacts_at_branch(result, list(segment.children.values()), output_dir)
            
            # Add children to stack (reversed for correct order)
            for child in reversed(list(segment.children.values())):
                stack.append((child, result.output_model, depth + 1))
        
        # Create result and print summary
        total_time = time.time() - start_time
        result = TreeExecutionResult(results, total_time)
        self._print_summary(result)
        
        return result
    
    def run_segment(
        self,
        segment: DSESegment,
        input_model: Path,
        base_output_dir: Path
    ) -> SegmentResult:
        """Run a single DSE segment.
        
        Args:
            segment: Segment to execute
            input_model: Input ONNX model path
            base_output_dir: Base output directory
            
        Returns:
            SegmentResult with execution details
        """
        segment_dir = base_output_dir / segment.segment_id
        # Use safe filename (segment_id may contain slashes)
        safe_name = segment.segment_id.replace("/", "_")
        output_model = segment_dir / f"{safe_name}_output.onnx"
        
        if output_model.exists():
            # Verify it's a valid ONNX file before using cache
            try:
                import onnx
                onnx.load(str(output_model))
                
                print(f"✓ Using cached: {segment.segment_id}")
                return SegmentResult(
                    success=True,
                    segment_id=segment.segment_id,
                    output_model=output_model,
                    output_dir=segment_dir,
                    cached=True
                )
            except Exception:
                # Invalid cache, remove it and rebuild
                output_model.unlink()
        
        print(f"\n→ Executing: {segment.segment_id}")
        
        # Create FINN config
        finn_config = self._make_finn_config(segment, segment_dir)
        
        # Prepare directory and model
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_input = segment_dir / "input.onnx"
        self.finn_adapter.prepare_model(input_model, segment_input)
        
        # Execute build
        start_time = time.time()
        
        try:
            # Use adapter for clean FINN interaction
            final_model = self.finn_adapter.build(segment_input, finn_config, segment_dir)
            
            if final_model:
                # Copy to expected location
                self.finn_adapter.prepare_model(final_model, output_model)
                print(f"✓ Completed: {segment.segment_id}")
                return SegmentResult(
                    success=True,
                    segment_id=segment.segment_id,
                    output_model=output_model,
                    output_dir=segment_dir,
                    execution_time=time.time() - start_time
                )
            else:
                raise RuntimeError("Build succeeded but no output model generated")
                
        except ExecutionError:
            # Re-raise our own errors
            raise
        except Exception as e:
            # Wrap external errors with context but preserve stack trace
            print(f"✗ Failed: {segment.segment_id}")
            raise ExecutionError(
                f"Segment '{segment.segment_id}' build failed: {str(e)}"
            ) from e
    
    def _make_finn_config(self, segment: DSESegment, output_dir: Path) -> Dict[str, Any]:
        """Create FINN configuration for segment.
        
        Args:
            segment: Segment to configure
            output_dir: Output directory for this segment
            
        Returns:
            FINN configuration dictionary
        """
        config = self.base_config.copy()
        config["output_dir"] = str(output_dir)
        config["generate_outputs"] = self.output_map[self.output_product]
        
        # Extract kernel_selections from segment steps if present
        kernel_selections = []
        for step in segment.steps:
            if step.get("name") == "infer_kernels" and "kernel_backends" in step:
                for kernel_name, backend_classes in step["kernel_backends"]:
                    if backend_classes:
                        backend_name = backend_classes[0].__name__.replace('_hls', '').replace('_rtl', '')
                        kernel_selections.append((kernel_name, backend_name))
        
        # Add kernel_selections if found in segment or base config
        if kernel_selections:
            config["kernel_selections"] = kernel_selections
        elif "kernel_selections" in self.base_config:
            config["kernel_selections"] = self.base_config["kernel_selections"]
        
        # Process steps - resolve to callable functions
        steps = []
        for step in segment.steps:
            if "name" in step:
                step_name = step["name"]
                try:
                    # Try to get callable from plugin registry
                    step_fn = get_step(step_name)
                    steps.append(step_fn)
                except KeyError:
                    # Not in registry, pass as string for FINN's internal lookup
                    steps.append(step_name)
            else:
                raise ValueError(f"Step missing name: {step}")
        
        config["steps"] = steps
        return config
    
    def _mark_descendants_skipped(self, segment: DSESegment, skipped: Set[str]) -> None:
        """Mark all descendants as skipped."""
        for child in segment.children.values():
            skipped.add(child.segment_id)
            self._mark_descendants_skipped(child, skipped)
    
    def _print_summary(self, result: TreeExecutionResult) -> None:
        """Print execution summary."""
        stats = result.stats
        print(f"\n{'='*50}")
        print(f"Execution Complete in {result.total_time:.1f}s")
        print(f"  Total:      {stats['total']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed:     {stats['failed']}")
        print(f"  Skipped:    {stats['skipped']}")
        print(f"  Cached:     {stats['cached']}")
        print(f"{'='*50}")