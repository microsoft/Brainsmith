"""
Metrics collector for Phase 3.

This module implements standardized metrics extraction from different backend outputs.
"""

import json
import os
from typing import Any, Dict, Optional

from .data_structures import BuildMetrics


class MetricsCollector:
    """Collect and standardize metrics from different backends."""
    
    def collect_from_finn_output(self, output_dir: str) -> BuildMetrics:
        """
        Extract metrics from FINN build outputs.
        
        Args:
            output_dir: Directory containing FINN build outputs
            
        Returns:
            BuildMetrics with extracted values
        """
        metrics = BuildMetrics()
        
        # Resource estimates
        self._extract_resource_estimates(output_dir, metrics)
        
        # Performance data  
        self._extract_performance_data(output_dir, metrics)
        
        # Synthesis results (if available)
        self._extract_synthesis_results(output_dir, metrics)
        
        return metrics
    
    def _extract_resource_estimates(self, output_dir: str, metrics: BuildMetrics):
        """Extract resource utilization estimates."""
        estimate_files = [
            "estimate_layer_resources_hls.json",
            "estimate_layer_resources.json",
            "post_synth_resources.json"
        ]
        
        for filename in estimate_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self._parse_resource_data(data, metrics)
                        # Store raw data
                        metrics.raw_metrics[f"resource_{filename}"] = data
                        break
                except Exception as e:
                    print(f"Failed to parse {filename}: {e}")
    
    def _extract_performance_data(self, output_dir: str, metrics: BuildMetrics):
        """Extract performance metrics."""
        perf_files = [
            "rtlsim_performance.json",
            "performance_summary.json"
        ]
        
        for filename in perf_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self._parse_performance_data(data, metrics)
                        # Store raw data
                        metrics.raw_metrics[f"performance_{filename}"] = data
                        break
                except Exception as e:
                    print(f"Failed to parse {filename}: {e}")
    
    def _extract_synthesis_results(self, output_dir: str, metrics: BuildMetrics):
        """Extract synthesis results if available."""
        synth_files = [
            "synth_report.json",
            "time_per_step.json"
        ]
        
        for filename in synth_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        # Store timing data
                        metrics.raw_metrics[f"synthesis_{filename}"] = data
                except Exception as e:
                    print(f"Failed to parse {filename}: {e}")
    
    def _parse_resource_data(self, data: Dict, metrics: BuildMetrics):
        """Parse resource utilization data."""
        if "total" in data:
            total = data["total"]
            # Extract actual utilization values
            # Note: These might be absolute values, not percentages
            # In real implementation, would need device totals to calculate percentage
            metrics.lut_utilization = self._safe_float(total.get("LUT"))
            metrics.dsp_utilization = self._safe_float(total.get("DSP"))
            metrics.bram_utilization = self._safe_float(total.get("BRAM_18K"))
            metrics.uram_utilization = self._safe_float(total.get("URAM"))
            
            # For now, store as raw counts if no percentage available
            if metrics.lut_utilization and metrics.lut_utilization > 1.0:
                # These are counts, not percentages - store in raw metrics
                metrics.raw_metrics["lut_count"] = metrics.lut_utilization
                metrics.raw_metrics["dsp_count"] = metrics.dsp_utilization
                metrics.raw_metrics["bram_count"] = metrics.bram_utilization
                metrics.raw_metrics["uram_count"] = metrics.uram_utilization
                
                # Set utilization to None since we don't have percentages
                metrics.lut_utilization = None
                metrics.dsp_utilization = None
                metrics.bram_utilization = None
                metrics.uram_utilization = None
            
    def _parse_performance_data(self, data: Dict, metrics: BuildMetrics):
        """Parse performance metrics."""
        # FINN typically reports throughput as FPS (frames per second)
        metrics.throughput = self._safe_float(data.get("throughput_fps"))
        
        # Latency might be in cycles - convert to microseconds if clock frequency known
        latency_cycles = self._safe_float(data.get("latency_cycles"))
        fclk_mhz = self._safe_float(data.get("fclk_mhz"))
        
        if latency_cycles and fclk_mhz:
            # Convert cycles to microseconds
            metrics.latency = latency_cycles / fclk_mhz
        else:
            metrics.latency = self._safe_float(data.get("latency_us"))
            
        metrics.clock_frequency = fclk_mhz
        
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None