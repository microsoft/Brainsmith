# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DSE segment runner for executing segment builds."""

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Set, List
from brainsmith.dse.segment import DSESegment
from brainsmith.dse.tree import DSETree
from brainsmith.loader import get_step
from brainsmith.dse.types import SegmentResult, TreeExecutionResult, ExecutionError
from brainsmith._internal.finn.adapter import FINNAdapter

logger = logging.getLogger(__name__)


def _share_artifacts_at_branch(
    parent_result: SegmentResult,
    child_segments: List[DSESegment],
    base_output_dir: Path
) -> None:
    """Copy build artifacts to child segments at branch points.

    Uses full directory copies for compatibility.

    Args:
        parent_result: Result from parent segment execution
        child_segments: List of child segments to share artifacts with
        base_output_dir: Base directory for DSE outputs
    """
    if not parent_result.success:
        return

    logger.debug(f"Sharing artifacts to {len(child_segments)} children")

    for child in child_segments:
        child_dir = base_output_dir / child.segment_id
        # Full copy required for compatibility
        if child_dir.exists():
            shutil.rmtree(child_dir)
        shutil.copytree(parent_result.output_dir, child_dir)


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

        # Note: synth_clk_period_ns and board already validated by DSEConfig
        
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

        logger.info(f"Executing tree with fail_fast={self.fail_fast}")
        logger.info(f"Output directory: {output_dir}")
        
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
                logger.warning(f"{indent}Skipped: {segment.segment_id}")
                results[segment.segment_id] = SegmentResult(
                    success=False,
                    segment_id=segment.segment_id,
                    error="Skipped"
                )
                continue

            # Execute segment
            logger.info(f"{indent}Executing: {segment.segment_id}")

            # Skip empty segments (e.g., root with immediate branches)
            if not segment.steps:
                logger.debug(f"{indent}  (empty segment, passing through)")
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
            except KeyboardInterrupt:
                # User cancellation - propagate immediately
                logger.warning("Build cancelled by user")
                raise
            except ExecutionError as e:
                # Handle execution errors properly
                logger.error(f"Segment failed: {segment.segment_id}")
                logger.error(f"  Error: {str(e)}")
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
                # Unexpected errors - log with traceback and wrap as ExecutionError
                logger.exception(f"Unexpected error in segment {segment.segment_id}")

                if self.fail_fast:
                    # Wrap and re-raise with context
                    raise ExecutionError(
                        f"Segment '{segment.segment_id}' failed unexpectedly: {e}"
                    ) from e

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
                _share_artifacts_at_branch(result, list(segment.children.values()), output_dir)
            
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
        # Segment IDs use slashes as path separators, invalid in filenames
        safe_name = segment.segment_id.replace("/", "_")
        output_model = segment_dir / f"{safe_name}_output.onnx"
        
        # Check cache validity
        if output_model.exists():
            try:
                import onnx
                onnx.load(str(output_model))
                # Valid cache - return immediately
                logger.debug(f"Cache hit: {segment.segment_id}")
                return SegmentResult(
                    success=True,
                    segment_id=segment.segment_id,
                    output_model=output_model,
                    output_dir=segment_dir,
                    cached=True
                )
            except (onnx.onnx_cpp2py_export.checker.ValidationError,
                    onnx.onnx_cpp2py_export.shape_inference.InferenceError) as e:
                # Invalid ONNX model - rebuild
                logger.warning(f"Invalid cache for {segment.segment_id}, rebuilding: {e}")
                output_model.unlink()
            except Exception as e:
                # Unexpected error - don't silently swallow it
                logger.error(f"Unexpected error validating cache for {segment.segment_id}: {e}")
                raise

        # Cache miss or invalid - execute build
        logger.info(f"Building segment: {segment.segment_id}")
        
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
                logger.info(f"Completed segment: {segment.segment_id} ({time.time() - start_time:.1f}s)")
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
            logger.error(f"Segment build failed: {segment.segment_id}")
            raise ExecutionError(
                f"Segment '{segment.segment_id}' build failed: {str(e)}"
            ) from e
    
    def _extract_kernel_selections(self, segment: DSESegment) -> List[tuple]:
        """Extract kernel selections from segment steps.

        Searches for 'infer_kernels' steps that contain kernel_backends
        and converts them to FINN's expected format.

        Args:
            segment: Segment to extract kernel selections from

        Returns:
            List of (kernel_name, backend_name) tuples
        """
        kernel_selections = []
        for step in segment.steps:
            if step.get("name") == "infer_kernels" and "kernel_backends" in step:
                for kernel_name, backend_classes in step["kernel_backends"]:
                    if backend_classes:
                        # Convert backend class name to FINN format
                        # e.g., ConvolutionInputGenerator_hls -> ConvolutionInputGenerator
                        backend_name = backend_classes[0].__name__.replace('_hls', '').replace('_rtl', '')
                        kernel_selections.append((kernel_name, backend_name))
        return kernel_selections

    def _resolve_steps(self, segment: DSESegment) -> List:
        """Resolve step names to callable functions.

        Attempts to resolve step names from the plugin registry.
        Falls back to passing strings for FINN's internal lookup.

        Args:
            segment: Segment containing steps to resolve

        Returns:
            List of step callables or strings

        Raises:
            ValueError: If step is missing name field
        """
        steps = []
        for step in segment.steps:
            if "name" not in step:
                raise ValueError(f"Step missing name: {step}")

            step_name = step["name"]
            try:
                # Try to get callable from plugin registry
                step_fn = get_step(step_name)
                steps.append(step_fn)
            except KeyError:
                # Not in registry, pass as string for FINN's internal lookup
                steps.append(step_name)
        return steps

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

        # Extract and set kernel selections
        kernel_selections = self._extract_kernel_selections(segment)
        if kernel_selections:
            config["kernel_selections"] = kernel_selections
        elif "kernel_selections" in self.base_config:
            config["kernel_selections"] = self.base_config["kernel_selections"]

        # Resolve steps to callables
        config["steps"] = self._resolve_steps(segment)

        return config
    
    def _mark_descendants_skipped(self, segment: DSESegment, skipped: Set[str]) -> None:
        """Mark all descendants as skipped."""
        for child in segment.children.values():
            skipped.add(child.segment_id)
            self._mark_descendants_skipped(child, skipped)
    
    def _print_summary(self, result: TreeExecutionResult) -> None:
        """Print execution summary."""
        stats = result.stats
        logger.info(f"{'='*50}")
        logger.info(f"Execution Complete in {result.total_time:.1f}s")
        logger.info(f"  Total:      {stats['total']}")
        logger.info(f"  Successful: {stats['successful']}")
        logger.info(f"  Failed:     {stats['failed']}")
        logger.info(f"  Skipped:    {stats['skipped']}")
        logger.info(f"  Cached:     {stats['cached']}")
        logger.info(f"{'='*50}")
