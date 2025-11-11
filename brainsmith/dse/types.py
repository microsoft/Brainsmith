# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainsmith.dse.design_space import GlobalDesignSpace
    from brainsmith.dse.tree import DSETree

logger = logging.getLogger(__name__)


class SegmentStatus(Enum):
    """Execution status for DSE segments.

    Attributes:
        PENDING: Segment not yet executed
        RUNNING: Segment currently executing
        COMPLETED: Segment executed successfully
        FAILED: Segment execution failed
        SKIPPED: Segment skipped due to parent failure
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class OutputType(Enum):
    """Build output types for DSE execution.

    Attributes:
        ESTIMATES: Performance estimates only (fastest)
        RTL: RTL simulation and IP generation
        BITFILE: Full bitstream generation (slowest)
    """

    ESTIMATES = "estimates"
    RTL = "rtl"
    BITFILE = "bitfile"

    def to_finn_products(self) -> list[str]:
        """Convert to FINN output_products configuration."""
        return {
            OutputType.ESTIMATES: ["estimates"],
            OutputType.RTL: ["rtl_sim", "ip_gen"],
            OutputType.BITFILE: ["bitfile"],
        }[self]

    def to_finn_outputs(self) -> list[str]:
        """Convert to FINN generate_outputs configuration."""
        return {
            OutputType.ESTIMATES: ["estimate_reports"],
            OutputType.RTL: ["estimate_reports", "rtlsim_performance", "stitched_ip"],
            OutputType.BITFILE: [
                "estimate_reports",
                "rtlsim_performance",
                "stitched_ip",
                "bitfile",
                "deployment_package",
            ],
        }[self]

    @classmethod
    def from_finn_product(cls, product: str) -> OutputType:
        """Get OutputType from FINN product string.

        Args:
            product: FINN product name (e.g., 'estimates', 'bitfile', 'rtl_sim')

        Returns:
            Matching OutputType

        Raises:
            ValueError: If product is unknown

        Example:
            >>> OutputType.from_finn_product('bitfile')
            <OutputType.BITFILE: 'bitfile'>
            >>> OutputType.from_finn_product('rtl_sim')
            <OutputType.RTL: 'rtl'>
        """
        for output_type in cls:
            if product in output_type.to_finn_products():
                return output_type

        # Product not found - create helpful error message
        all_products = [p for ot in cls for p in ot.to_finn_products()]
        raise ValueError(
            f"Unknown FINN product '{product}'. " f"Valid products: {', '.join(all_products)}"
        )


@dataclass
class SegmentResult:
    """Result from executing a single DSE segment.

    Attributes:
        segment_id: Unique identifier for the segment
        status: Execution status (completed, failed, skipped, etc.)
        output_model: Path to output ONNX model (if successful)
        output_dir: Directory containing build artifacts
        error: Error message (if failed)
        execution_time: Execution time in seconds
        cached: Whether result was retrieved from cache
    """

    segment_id: str
    status: SegmentStatus
    output_model: Path | None = None
    output_dir: Path | None = None
    error: str | None = None
    execution_time: float = 0
    cached: bool = False


@dataclass
class TreeExecutionResult:
    """Results from design space exploration execution.

    Attributes:
        segment_results: Execution results for each DSE segment
        total_time: Total execution time in seconds
        design_space: Original design space (if available)
        dse_tree: Execution tree structure (if available)
    """

    segment_results: dict[str, SegmentResult]
    total_time: float
    design_space: GlobalDesignSpace | None = None
    dse_tree: DSETree | None = None

    def compute_stats(self) -> dict[str, int]:
        """Compute execution statistics.

        Returns:
            Dict with counts: total, successful, failed, cached, skipped
        """
        total = successful = failed = cached = skipped = 0

        for r in self.segment_results.values():
            total += 1
            if r.status == SegmentStatus.COMPLETED:
                if r.cached:
                    cached += 1
                else:
                    successful += 1
            elif r.status == SegmentStatus.FAILED:
                failed += 1
            elif r.status == SegmentStatus.SKIPPED:
                skipped += 1

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "cached": cached,
            "skipped": skipped,
        }

    def validate_success(self, output_dir: Path) -> None:
        """Validate that results contain at least one successful build.

        Args:
            output_dir: Output directory for error messages

        Raises:
            ExecutionError: If no valid builds exist
        """
        stats = self.compute_stats()
        valid_builds = stats["successful"] + stats["cached"]

        if valid_builds == 0:
            raise ExecutionError(
                f"DSE failed: No successful builds\n"
                f"  Failed: {stats['failed']}\n"
                f"  Skipped: {stats['skipped']}\n"
                f"  Check segment logs in: {output_dir}/*/\n"
                f"  Run with --log-level debug for detailed output"
            )

        if stats["successful"] == 0 and stats["cached"] > 0:
            logger.warning(
                f"All builds used cached results ({stats['cached']} cached). "
                f"No new builds were executed."
            )


class ExecutionError(Exception):
    """Exception raised during DSE execution failures."""

    pass
