"""
Analysis Libraries

Auto-discovery and management of analysis tools including profiling,
code generation, and reporting utilities.

Main exports:
- AnalysisRegistry: Registry for analysis tool discovery and management
- discover_all_analysis_tools: Discover all available analysis tools
- get_analysis_tool: Get specific analysis tool by name
- find_tools_by_type: Find tools by analysis type
"""

# Import registry system
from .registry import (
    AnalysisRegistry,
    AnalysisType,
    AnalysisToolInfo,
    get_analysis_registry,
    discover_all_analysis_tools,
    get_analysis_tool,
    find_tools_by_type,
    list_available_analysis_tools,
    refresh_analysis_registry
)

# Import profiling tools
from .profiling import (
    roofline_analysis,
    RooflineProfiler
)

# Import tools (with safe imports)
try:
    from .tools import generate_hw_kernel
except ImportError:
    generate_hw_kernel = None

__all__ = [
    # Registry system
    "AnalysisRegistry",
    "AnalysisType",
    "AnalysisToolInfo", 
    "get_analysis_registry",
    "discover_all_analysis_tools",
    "get_analysis_tool",
    "find_tools_by_type",
    "list_available_analysis_tools",
    "refresh_analysis_registry",
    
    # Profiling tools
    "roofline_analysis",
    "RooflineProfiler",
    
    # Code generation tools
    "generate_hw_kernel"
]