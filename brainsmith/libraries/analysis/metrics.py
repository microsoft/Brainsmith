"""
Performance and resource metrics for analysis.

Defines metrics collection and calculation methods for FPGA
accelerator design analysis.
"""

from typing import Dict, List, Any, Optional


class PerformanceMetrics:
    """Performance metrics collection and calculation."""
    
    @staticmethod
    def calculate_throughput(pe: int, simd: int, frequency: float) -> float:
        """Calculate theoretical throughput."""
        return pe * simd * frequency / 1000.0  # GOPS
    
    @staticmethod
    def calculate_latency(pipeline_depth: int, pe: int, simd: int) -> float:
        """Calculate pipeline latency."""
        return pipeline_depth + 1.0 / (pe * simd)  # cycles
    
    @staticmethod
    def calculate_efficiency(throughput: float, pe: int, simd: int) -> float:
        """Calculate efficiency per processing element."""
        return throughput / (pe * simd)


class ResourceMetrics:
    """Resource utilization metrics."""
    
    @staticmethod
    def estimate_luts(pe: int, simd: int) -> int:
        """Estimate LUT usage."""
        return pe * simd * 1000
    
    @staticmethod
    def estimate_brams(pe: int) -> int:
        """Estimate BRAM usage."""
        return max(1, pe // 2) * 5
    
    @staticmethod
    def estimate_dsps(pe: int) -> int:
        """Estimate DSP usage."""
        return pe * 2
    
    @staticmethod
    def calculate_utilization(used: int, total: int) -> float:
        """Calculate resource utilization."""
        return used / total if total > 0 else 0.0