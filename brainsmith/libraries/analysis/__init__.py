"""
Analysis Library - Week 4 Implementation

Provides comprehensive analysis capabilities including performance metrics,
visualization, and reporting for FPGA accelerator designs.
"""

from .library import AnalysisLibrary
from .metrics import PerformanceMetrics, ResourceMetrics

# Convenience functions
def analyze_design(design_config, metrics=['throughput', 'latency', 'efficiency']):
    """Analyze design configuration."""
    library = AnalysisLibrary()
    library.initialize()
    return library.execute("analyze_design", {
        'config': design_config,
        'metrics': metrics
    })

def generate_report(analysis_results, format='html'):
    """Generate analysis report."""
    library = AnalysisLibrary()
    library.initialize()
    return library.execute("generate_report", {
        'results': analysis_results,
        'format': format
    })

__all__ = [
    'AnalysisLibrary',
    'PerformanceMetrics',
    'ResourceMetrics',
    'analyze_design',
    'generate_report'
]

# Version info
__version__ = "1.0.0"  # Week 4 implementation