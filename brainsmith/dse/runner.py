# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DSE segment runner for executing segment builds."""

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Set, List

import onnx
from onnx.onnx_cpp2py_export.checker import ValidationError as OnnxValidationError
from onnx.onnx_cpp2py_export.shape_inference import InferenceError as OnnxInferenceError

from brainsmith.dse.segment import DSESegment
from brainsmith.dse.tree import DSETree
from brainsmith.registry import get_step
from brainsmith.dse.types import SegmentResult, SegmentStatus, TreeExecutionResult, ExecutionError, OutputType
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
    if parent_result.status != SegmentStatus.COMPLETED:
        logger.debug("Skipping artifact sharing: parent segment not completed")
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
        output_product = output_products[0] if output_products else "estimates"
        self.output_type = OutputType.from_finn_product(output_product)

        # Note: synth_clk_period_ns and board already validated by DSEConfig
    
    def _add_segment_context(self, segment_id: str, error: Exception) -> ExecutionError:
        """Add segment context to error if not already present.

        ExecutionErrors are returned as-is (already have context).
        Other errors are wrapped with segment information.

        Args:
            segment_id: ID of segment that failed
            error: The exception that occurred

        Returns:
            ExecutionError with segment context
        """
        if isinstance(error, ExecutionError):
            return error

        # Wrap unexpected error with context
        wrapped = ExecutionError(f"Segment '{segment_id}' failed: {error}")
        wrapped.__cause__ = error
        return wrapped

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
                    segment_id=segment.segment_id,
                    status=SegmentStatus.SKIPPED,
                    error="Parent segment failed"
                )
                continue

            # Execute segment
            logger.info(f"{indent}Executing: {segment.segment_id}")

            # Skip empty segments (e.g., root with immediate branches)
            if not segment.steps:
                logger.debug(f"{indent}  (empty segment, passing through)")
                # Create a pass-through result
                results[segment.segment_id] = SegmentResult(
                    segment_id=segment.segment_id,
                    status=SegmentStatus.COMPLETED,
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
                logger.warning("Build cancelled by user")
                raise
            except Exception as e:
                wrapped_error = self._add_segment_context(segment.segment_id, e)
                logger.error(f"Segment failed: {segment.segment_id}: {wrapped_error}")
                if not isinstance(e, ExecutionError):
                    logger.exception("Unexpected error details:")

                if self.fail_fast:
                    raise wrapped_error

                # Create failure result
                results[segment.segment_id] = SegmentResult(
                    segment_id=segment.segment_id,
                    status=SegmentStatus.FAILED,
                    error=str(wrapped_error),
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
        output_model = segment_dir / "output.onnx"
        
        # Check cache validity
        if output_model.exists():
            try:
                onnx.load(str(output_model))
                # Valid cache - return immediately
                logger.debug(f"Cache hit: {segment.segment_id}")
                return SegmentResult(
                    segment_id=segment.segment_id,
                    status=SegmentStatus.COMPLETED,
                    output_model=output_model,
                    output_dir=segment_dir,
                    cached=True
                )
            except (OnnxValidationError, OnnxInferenceError) as e:
                # Invalid ONNX model - rebuild
                logger.warning(f"Invalid cache for {segment.segment_id}, rebuilding: {e}")
                output_model.unlink()
            except OSError as e:
                # File corruption (rare but possible)
                logger.warning(f"Corrupted cache for {segment.segment_id}, rebuilding: {e}")
                output_model.unlink()

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
                    segment_id=segment.segment_id,
                    status=SegmentStatus.COMPLETED,
                    output_model=output_model,
                    output_dir=segment_dir,
                    execution_time=time.time() - start_time
                )
            else:
                raise RuntimeError("Build succeeded but no output model generated")

        except Exception as e:
            contextualized_error = self._add_segment_context(segment.segment_id, e)
            logger.error(f"Segment failed: {segment.segment_id}: {contextualized_error}")
            if not isinstance(e, ExecutionError):
                logger.exception("Unexpected error details:")
            raise contextualized_error
    
    def _extract_kernel_selections(self, segment: DSESegment) -> List[tuple]:
        """Extract kernel selections from segment steps.

        Searches for 'infer_kernels' steps that contain kernel_backends
        and extracts backend classes for inference.

        Args:
            segment: Segment to extract kernel selections from

        Returns:
            List of (kernel_name, backend_class) tuples

        Note:
            Currently uses first backend per kernel. Future enhancement
            will support selecting specific backends from the list.
        """
        kernel_selections = []
        for step in segment.steps:
            if step.get("name") == "infer_kernels" and "kernel_backends" in step:
                for kernel_name, backend_classes in step["kernel_backends"]:
                    if backend_classes:
                        # TODO: Future - support selecting specific backends
                        # For now, use first registered backend per kernel
                        kernel_selections.append((kernel_name, backend_classes[0]))
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
        config["generate_outputs"] = self.output_type.to_finn_outputs()

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
        stats = result.compute_stats()
        logger.info(f"{'='*50}")
        logger.info(f"Execution Complete in {result.total_time:.1f}s")
        logger.info(f"  Total:      {stats['total']}")
        logger.info(f"  Successful: {stats['successful']}")
        logger.info(f"  Failed:     {stats['failed']}")
        logger.info(f"  Skipped:    {stats['skipped']}")
        logger.info(f"  Cached:     {stats['cached']}")
        logger.info(f"{'='*50}")
