"""
Error handling for Phase 3.

This module provides comprehensive error categorization and troubleshooting
for build failures.
"""

from typing import Dict, List

from brainsmith.core_v3.phase2.data_structures import BuildConfig


class BuildErrorHandler:
    """Handle and categorize build errors."""
    
    ERROR_CATEGORIES = {
        "model_load": "Failed to load input model",
        "preprocessing": "Preprocessing step failed", 
        "transform": "Transform application failed",
        "kernel": "Kernel application failed",
        "synthesis": "Hardware synthesis failed",
        "timing": "Timing constraints not met",
        "resource": "Resource constraints exceeded",
        "postprocessing": "Postprocessing failed",
        "metrics": "Metrics extraction failed",
        "unknown": "Unknown error occurred"
    }
    
    def categorize_error(self, error_message: str, logs: Dict[str, str]) -> str:
        """
        Categorize error based on message and logs.
        
        Args:
            error_message: The error message
            logs: Dictionary of log files
            
        Returns:
            Error category string
        """
        error_lower = error_message.lower()
        
        # Check for specific error patterns
        if "timing" in error_lower or "slack" in error_lower or "constraint" in error_lower:
            return "timing"
        elif "resource" in error_lower or "utilization" in error_lower or "exceeded" in error_lower:
            return "resource"
        elif "synthesis" in error_lower or "vivado" in error_lower or "synth" in error_lower:
            return "synthesis"
        elif "transform" in error_lower:
            return "transform"
        elif "kernel" in error_lower:
            return "kernel"
        elif "preprocess" in error_lower:
            return "preprocessing"
        elif "postprocess" in error_lower:
            return "postprocessing"
        elif "model" in error_lower and ("load" in error_lower or "onnx" in error_lower):
            return "model_load"
        elif "metric" in error_lower:
            return "metrics"
        else:
            # Try to infer from logs
            return self._categorize_from_logs(logs)
    
    def _categorize_from_logs(self, logs: Dict[str, str]) -> str:
        """Try to categorize error from log contents."""
        # This is a simplified implementation
        # In real implementation, would parse logs more thoroughly
        for log_name, log_content in logs.items():
            if isinstance(log_content, str) and "error" in log_content.lower():
                # Try to find error patterns in logs
                if "timing" in log_content.lower():
                    return "timing"
                elif "resource" in log_content.lower():
                    return "resource"
                    
        return "unknown"
    
    def generate_error_report(self, config: BuildConfig, error: str, logs: Dict[str, str]) -> str:
        """
        Generate detailed error report.
        
        Args:
            config: Build configuration
            error: Error message
            logs: Dictionary of log files
            
        Returns:
            Formatted error report string
        """
        category = self.categorize_error(error, logs)
        category_description = self.ERROR_CATEGORIES.get(category, "Unknown error")
        
        report_lines = [
            "=" * 70,
            "BUILD ERROR REPORT",
            "=" * 70,
            f"Config ID: {config.id}",
            f"Error Category: {category} - {category_description}",
            f"Error Message: {error}",
            "",
            "CONFIGURATION SUMMARY:",
            "-" * 50,
            f"Model: {config.model_path}",
            f"Kernels: {len(config.kernels)} configured",
            f"Transforms: {self._format_transforms(config.transforms)}",
            f"Output Stage: {config.global_config.output_stage.value}",
            f"Target Device: {config.config_flags.get('target_device', 'Not specified')}",
            f"Clock Period: {config.config_flags.get('target_clock_ns', 'Not specified')}ns",
            "",
            "TROUBLESHOOTING SUGGESTIONS:",
            "-" * 50,
            self._get_troubleshooting_tips(category),
            ""
        ]
        
        # Add log snippet if available
        if logs:
            report_lines.extend([
                "LOG EXCERPTS:",
                "-" * 50,
                self._format_logs(logs),
                ""
            ])
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def _format_transforms(self, transforms: Dict[str, List[str]]) -> str:
        """Format transforms for display."""
        if not transforms:
            return "None"
        
        parts = []
        for stage, transform_list in transforms.items():
            parts.append(f"{stage}[{len(transform_list)}]")
        return ", ".join(parts)
    
    def _get_troubleshooting_tips(self, category: str) -> str:
        """Get troubleshooting tips for error category."""
        tips = {
            "timing": """- Reduce target clock frequency (increase target_clock_ns)
- Increase pipeline depth (adjust folding parameters)
- Check critical path in synthesis reports
- Consider using different optimization directives""",
            
            "resource": """- Reduce parallelization (PE/SIMD parameters)
- Use different memory types (BRAM vs URAM)
- Check device capacity and utilization reports
- Consider using a larger FPGA device""",
            
            "synthesis": """- Check Vivado/HLS tool installation
- Verify tool version compatibility
- Check synthesis directives and constraints
- Review synthesis logs for specific errors""",
            
            "transform": """- Verify transform order and dependencies
- Check model compatibility with transforms
- Review transform parameters
- Ensure all required transforms are available""",
            
            "kernel": """- Verify kernel availability in registry
- Check backend compatibility
- Review kernel parameters and constraints
- Ensure kernel implementations are valid""",
            
            "preprocessing": """- Check preprocessing step configuration
- Verify model format and compatibility
- Review preprocessing parameters
- Ensure input model is valid ONNX""",
            
            "postprocessing": """- Check postprocessing step configuration
- Verify required data is available
- Review postprocessing parameters
- Check for missing dependencies""",
            
            "model_load": """- Verify model file exists and is readable
- Check ONNX model format and version
- Validate model structure
- Ensure model is compatible with target hardware""",
            
            "metrics": """- Check that build outputs exist
- Verify metrics file formats
- Review build completion status
- Ensure all required metrics are generated"""
        }
        
        return tips.get(category, """- Check logs for specific error details
- Verify tool installation and paths
- Review configuration parameters
- Consult documentation for requirements""")
    
    def _format_logs(self, logs: Dict[str, str]) -> str:
        """Format log excerpts for display."""
        formatted_logs = []
        
        for log_name, log_content in logs.items():
            formatted_logs.append(f"\nFrom {log_name}:")
            
            if isinstance(log_content, str):
                # If it's a path, try to read file
                if log_content.endswith('.log') or log_content.endswith('.txt'):
                    formatted_logs.append(f"  Log file: {log_content}")
                else:
                    # It's the actual content - show first few lines
                    lines = log_content.split('\n')
                    for line in lines[:5]:  # Show first 5 lines
                        if line.strip():
                            formatted_logs.append(f"  {line}")
                    if len(lines) > 5:
                        formatted_logs.append("  ... (truncated)")
        
        return "\n".join(formatted_logs) if formatted_logs else "No logs available"