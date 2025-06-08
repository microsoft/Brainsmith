"""
Enhanced Analysis Framework for Hardware Kernel Generator.

This module provides advanced interface analysis and pragma processing
capabilities with comprehensive dataflow integration.
"""

from .enhanced_interface_analyzer import (
    InterfaceClassifier,
    InterfaceAnalyzer, 
    DataflowInterfaceConverter,
    InterfaceValidator,
    create_interface_analyzer,
    analyze_interfaces
)

from .enhanced_pragma_processor import (
    PragmaParser,
    PragmaValidator,
    PragmaProcessor,
    DataflowPragmaConverter,
    create_pragma_processor,
    process_pragmas
)

from .analysis_config import (
    InterfaceAnalysisConfig,
    PragmaAnalysisConfig,
    AnalysisProfile,
    AnalysisMetrics,
    create_analysis_config
)

from .analysis_integration import (
    AnalysisOrchestrator,
    AnalysisResults,
    LegacyAnalysisAdapter,
    AnalysisCache,
    create_analysis_orchestrator,
    run_complete_analysis
)

__all__ = [
    # Interface Analysis
    "InterfaceClassifier",
    "InterfaceAnalyzer", 
    "DataflowInterfaceConverter",
    "InterfaceValidator",
    "create_interface_analyzer",
    "analyze_interfaces",
    
    # Pragma Processing
    "PragmaParser",
    "PragmaValidator", 
    "PragmaProcessor",
    "DataflowPragmaConverter",
    "create_pragma_processor",
    "process_pragmas",
    
    # Configuration
    "InterfaceAnalysisConfig",
    "PragmaAnalysisConfig",
    "AnalysisProfile",
    "AnalysisMetrics",
    "create_analysis_config",
    
    # Integration
    "AnalysisOrchestrator",
    "AnalysisResults",
    "LegacyAnalysisAdapter", 
    "AnalysisCache",
    "create_analysis_orchestrator",
    "run_complete_analysis"
]