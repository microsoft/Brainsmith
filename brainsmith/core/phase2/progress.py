"""
Progress tracking for design space exploration.

This module provides utilities for tracking and reporting progress during
the exploration process, including ETA calculation and progress summaries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .data_structures import BuildResult, BuildStatus


@dataclass
class ProgressTracker:
    """
    Track progress during design space exploration.
    
    This class maintains statistics about the exploration progress and
    provides estimates for completion time.
    """
    total_configs: int
    start_time: datetime = field(default_factory=datetime.now)
    
    # Progress counters
    completed: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    
    # Timing statistics
    total_build_time: float = 0.0
    successful_build_time: float = 0.0
    failed_build_time: float = 0.0
    
    def update(self, result: BuildResult):
        """
        Update progress with a new build result.
        
        Args:
            result: The build result to add to progress
        """
        self.completed += 1
        
        # Update status counters
        if result.status == BuildStatus.SUCCESS:
            self.successful += 1
            self.successful_build_time += result.duration_seconds
        elif result.status == BuildStatus.FAILED:
            self.failed += 1
            self.failed_build_time += result.duration_seconds
        elif result.status == BuildStatus.SKIPPED:
            self.skipped += 1
        
        self.total_build_time += result.duration_seconds
    
    def get_eta(self) -> Optional[datetime]:
        """
        Calculate estimated time of completion.
        
        Returns:
            Estimated completion time, or None if no data yet
        """
        if self.completed == 0:
            return None
        
        # Calculate average time per build
        avg_time_per_build = self.total_build_time / self.completed
        
        # Estimate remaining time
        remaining_configs = self.total_configs - self.completed
        estimated_seconds = remaining_configs * avg_time_per_build
        
        # Return ETA
        return datetime.now() + timedelta(seconds=estimated_seconds)
    
    def get_summary(self) -> str:
        """
        Get a human-readable progress summary.
        
        Returns:
            Progress summary string
        """
        # Calculate percentages
        progress_pct = (self.completed / self.total_configs * 100) if self.total_configs > 0 else 0
        success_rate = (self.successful / self.completed * 100) if self.completed > 0 else 0
        
        # Calculate timing
        elapsed = (datetime.now() - self.start_time).total_seconds()
        avg_build_time = self.total_build_time / self.completed if self.completed > 0 else 0
        
        # Build summary
        lines = [
            f"Progress: {self.completed}/{self.total_configs} ({progress_pct:.1f}%)",
            f"Success: {self.successful} ({success_rate:.1f}%) | Failed: {self.failed} | Skipped: {self.skipped}",
            f"Elapsed: {self._format_duration(elapsed)} | Avg build: {avg_build_time:.1f}s",
        ]
        
        # Add ETA if available
        eta = self.get_eta()
        if eta:
            remaining = (eta - datetime.now()).total_seconds()
            lines.append(f"ETA: {self._format_duration(remaining)} ({eta.strftime('%H:%M:%S')})")
        
        return " | ".join(lines)
    
    def get_detailed_summary(self) -> str:
        """
        Get a detailed progress summary with additional statistics.
        
        Returns:
            Detailed progress summary string
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate rates
        builds_per_minute = (self.completed / elapsed * 60) if elapsed > 0 else 0
        success_per_minute = (self.successful / elapsed * 60) if elapsed > 0 else 0
        
        # Calculate averages
        avg_build_time = self.total_build_time / self.completed if self.completed > 0 else 0
        avg_success_time = self.successful_build_time / self.successful if self.successful > 0 else 0
        avg_fail_time = self.failed_build_time / self.failed if self.failed > 0 else 0
        
        lines = [
            "Exploration Progress",
            "===================",
            f"Total Configurations: {self.total_configs}",
            f"Completed: {self.completed} ({self.completed / self.total_configs * 100:.1f}%)",
            f"  - Successful: {self.successful} ({self.successful / self.completed * 100:.1f}% of completed)",
            f"  - Failed: {self.failed}",
            f"  - Skipped: {self.skipped}",
            "",
            "Timing Statistics",
            "-----------------",
            f"Total Elapsed: {self._format_duration(elapsed)}",
            f"Total Build Time: {self._format_duration(self.total_build_time)}",
            f"Average Build Time: {avg_build_time:.1f}s",
            f"  - Success: {avg_success_time:.1f}s",
            f"  - Failure: {avg_fail_time:.1f}s",
            "",
            "Performance",
            "-----------",
            f"Builds per minute: {builds_per_minute:.1f}",
            f"Successes per minute: {success_per_minute:.1f}",
        ]
        
        # Add ETA
        eta = self.get_eta()
        if eta:
            remaining = (eta - datetime.now()).total_seconds()
            lines.extend([
                "",
                "Estimated Completion",
                "-------------------",
                f"Time Remaining: {self._format_duration(remaining)}",
                f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}",
            ])
        
        return "\n".join(lines)
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_progress_bar(self, width: int = 50) -> str:
        """
        Get a text-based progress bar.
        
        Args:
            width: Width of the progress bar in characters
            
        Returns:
            Progress bar string
        """
        if self.total_configs == 0:
            return "[" + " " * width + "]"
        
        # Calculate fill
        progress = self.completed / self.total_configs
        filled = int(width * progress)
        
        # Build bar
        bar = "█" * filled + "░" * (width - filled)
        
        return f"[{bar}] {self.completed}/{self.total_configs}"