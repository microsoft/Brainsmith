"""Unified executor for segment-based execution trees."""

import time
from pathlib import Path
from typing import Dict, Any, Set
from brainsmith.core.execution_tree import ExecutionNode
from .types import SegmentResult, TreeExecutionResult, ExecutionError
from .finn_adapter import FINNAdapter
from .utils import share_artifacts_at_branch


class Executor:
    """Unified executor handling both tree traversal and segment execution.
    
    Combines the functionality of TreeExecutor and SegmentExecutor into
    a single, streamlined class that uses FINNAdapter for all FINN interactions.
    """
    
    def __init__(
        self,
        finn_adapter: FINNAdapter,
        base_finn_config: Dict[str, Any],
        global_config: Dict[str, Any]
    ) -> None:
        """Initialize executor.
        
        Args:
            finn_adapter: Adapter for FINN-specific operations
            base_finn_config: FINN configuration from blueprint
            global_config: Global configuration including output products
        """
        self.finn_adapter = finn_adapter
        self.base_finn_config = base_finn_config
        self.global_config = global_config
        
        # Extract settings
        self.fail_fast = global_config.get("fail_fast", False)
        self.output_product = global_config.get("output_products", "df")
        
        # Validate required FINN config fields
        if "synth_clk_period_ns" not in base_finn_config:
            raise ValueError("finn_config must specify synth_clk_period_ns")
        if "board" not in base_finn_config:
            raise ValueError("finn_config must specify board")
        
        # Map output products to FINN types
        self.output_map = {
            "df": ["ESTIMATE_REPORTS"],
            "rtl": [
                "ESTIMATE_REPORTS",
                "RTLSIM_PERFORMANCE",
                "STITCHED_IP"
            ],
            "dcp": [
                "ESTIMATE_REPORTS",
                "RTLSIM_PERFORMANCE",
                "STITCHED_IP",
                "BITFILE",
                "DEPLOYMENT_PACKAGE"
            ],
            # Also support OutputStage enum values
            "generate_reports": ["ESTIMATE_REPORTS"],
            "synthesize_bitstream": [
                "ESTIMATE_REPORTS",
                "RTLSIM_PERFORMANCE",
                "STITCHED_IP"
            ],
            "compile_and_package": [
                "ESTIMATE_REPORTS",
                "RTLSIM_PERFORMANCE",
                "STITCHED_IP",
                "BITFILE",
                "DEPLOYMENT_PACKAGE"
            ]
        }
    
    def execute(
        self,
        root: ExecutionNode,
        initial_model: Path,
        output_dir: Path
    ) -> TreeExecutionResult:
        """Execute all segments depth-first.
        
        Args:
            root: Root node of execution tree
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
        stack = [(root, initial_model, 0)]
        
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
            result = self._execute_segment(segment, input_model, output_dir)
            results[segment.segment_id] = result
            
            # Handle failure
            if not result.success:
                if self.fail_fast:
                    raise ExecutionError(f"Failed: {segment.segment_id}")
                # Mark descendants for skipping
                self._mark_descendants_skipped(segment, skipped)
                # Still need to add children to stack so they get marked as skipped
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
    
    def _execute_segment(
        self,
        segment: ExecutionNode,
        input_model: Path,
        base_output_dir: Path
    ) -> SegmentResult:
        """Execute FINN build for one segment.
        
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
        
        # Simple caching - check if output exists
        if output_model.exists():
            print(f"✓ Using cached: {segment.segment_id}")
            return SegmentResult(
                success=True,
                segment_id=segment.segment_id,
                output_model=output_model,
                output_dir=segment_dir,
                cached=True
            )
        
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
                
        except Exception as e:
            print(f"✗ Failed: {segment.segment_id} - {e}")
            return SegmentResult(
                success=False,
                segment_id=segment.segment_id,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _make_finn_config(self, segment: ExecutionNode, output_dir: Path) -> Dict[str, Any]:
        """Create FINN configuration for segment.
        
        Args:
            segment: Segment to configure
            output_dir: Output directory for this segment
            
        Returns:
            FINN configuration dictionary
        """
        config = self.base_finn_config.copy()
        config["output_dir"] = str(output_dir)
        config["generate_outputs"] = self.output_map[self.output_product]
        
        # Use pre-computed wrapper names
        steps = []
        for step in segment.segment_steps:
            if "finn_step_name" in step:
                # Pre-computed wrapper from tree building
                steps.append(step["finn_step_name"])
            elif "name" in step:
                # Regular FINN step
                steps.append(step["name"])
            else:
                raise ValueError(f"Step missing both finn_step_name and name: {step}")
        
        config["steps"] = steps
        return config
    
    def _mark_descendants_skipped(self, segment: ExecutionNode, skipped: Set[str]) -> None:
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