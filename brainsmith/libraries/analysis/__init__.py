"""
BrainSmith Analysis Library - Registry Dictionary Pattern

Simple, explicit analysis tool discovery using registry dictionary.  
No magical filesystem scanning - tools explicitly registered.

Main Functions:
- get_analysis_tool(name): Get analysis tool by name with fail-fast errors
- list_analysis_tools(): List all available analysis tool names

Example Usage:
    from brainsmith.libraries.analysis import get_analysis_tool, list_analysis_tools
    
    # List available tools
    tools = list_analysis_tools()  # ['roofline_analysis', 'roofline_profiler', 'generate_hw_kernel']
    
    # Get specific tool
    roofline_fn = get_analysis_tool('roofline_analysis')
    profiler_cls = get_analysis_tool('roofline_profiler')
"""

from typing import List, Union, Callable, Any

# Import analysis tools - handle missing dependencies gracefully
try:
    from .profiling import roofline_analysis, RooflineProfiler
    from .tools.gen_kernel import generate_hw_kernel
    _ANALYSIS_AVAILABLE = True
    
    # Simple registry maps tool names to their functions/classes
    AVAILABLE_ANALYSIS_TOOLS = {
        "roofline_analysis": roofline_analysis,
        "roofline_profiler": RooflineProfiler,
        "generate_hw_kernel": generate_hw_kernel,
    }

except ImportError as e:
    # Handle missing dependencies gracefully
    _ANALYSIS_AVAILABLE = False
    
    # Define minimal stubs for missing functions
    def roofline_analysis(*args, **kwargs):
        raise ImportError(f"Analysis functionality not available: {e}")
    
    class RooflineProfiler:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Analysis functionality not available: {e}")
    
    def generate_hw_kernel(*args, **kwargs):
        raise ImportError(f"Analysis functionality not available: {e}")
    
    # Create minimal registry with stub functions
    AVAILABLE_ANALYSIS_TOOLS = {
        "roofline_analysis": roofline_analysis,
        "roofline_profiler": RooflineProfiler,
        "generate_hw_kernel": generate_hw_kernel,
    }

def get_analysis_tool(name: str) -> Union[Callable, Any]:
    """
    Get analysis tool by name. Fails fast if not found.
    
    Args:
        name: Analysis tool name to retrieve
        
    Returns:
        Analysis tool function or class
        
    Raises:
        KeyError: If tool not found (with available options)
    """
    if name not in AVAILABLE_ANALYSIS_TOOLS:
        available = ", ".join(sorted(AVAILABLE_ANALYSIS_TOOLS.keys()))
        raise KeyError(f"Analysis tool '{name}' not found. Available: {available}")
    
    return AVAILABLE_ANALYSIS_TOOLS[name]

def list_analysis_tools() -> List[str]:
    """
    List all available analysis tool names.
    
    Returns:
        List of analysis tool names
    """
    return list(AVAILABLE_ANALYSIS_TOOLS.keys())


# Direct imports for backward compatibility
if _ANALYSIS_AVAILABLE:
    # Import everything from profiling and tools for direct access
    from .profiling import *
    from .tools import *

# Export all public functions and types
__all__ = [
    # Registry functions
    'get_analysis_tool',
    'list_analysis_tools',
    'AVAILABLE_ANALYSIS_TOOLS',
    
    # Direct tool access
    'roofline_analysis',
    'RooflineProfiler',
    'generate_hw_kernel',
]

# Module metadata
__version__ = "2.0.0"  # Bumped for registry refactoring
__author__ = "BrainSmith Development Team"