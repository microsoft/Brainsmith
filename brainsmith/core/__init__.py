"""
Brainsmith Core Module

This module provides the core orchestration and interface components
for the new extensible Brainsmith architecture using existing components.

Core Components:
- DesignSpaceOrchestrator: Main orchestration engine with hierarchical exit points
- FINNInterface: Legacy FINN support with 4-hook placeholder  
- WorkflowManager: High-level workflow management
- API: Python API functions with backward compatibility
- CLI: Command-line interface

All components are designed to work with existing Brainsmith functionality
while providing extensible structure for future enhancements.
"""

import logging

# Configure module-level logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "0.4.0"
__author__ = "Brainsmith Development Team"
__description__ = "Core orchestration and interface components for extensible Brainsmith architecture"

# Import core components with error handling
try:
    from .design_space_orchestrator import DesignSpaceOrchestrator
    logger.debug("Successfully imported DesignSpaceOrchestrator")
except ImportError as e:
    logger.warning(f"Could not import DesignSpaceOrchestrator: {e}")
    DesignSpaceOrchestrator = None

try:
    from .finn_interface import FINNInterface, FINNHooksPlaceholder
    logger.debug("Successfully imported FINN interface components")
except ImportError as e:
    logger.warning(f"Could not import FINN interface: {e}")
    FINNInterface = None
    FINNHooksPlaceholder = None

try:
    from .workflow import WorkflowManager, WorkflowType, WorkflowStatus
    logger.debug("Successfully imported workflow management components")
except ImportError as e:
    logger.warning(f"Could not import workflow components: {e}")
    WorkflowManager = None
    WorkflowType = None
    WorkflowStatus = None

try:
    from .api import (
        brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
        brainsmith_generate, brainsmith_workflow, explore_design_space, validate_blueprint
    )
    logger.debug("Successfully imported API functions")
except ImportError as e:
    logger.warning(f"Could not import API functions: {e}")
    # Set API functions to None if import fails
    brainsmith_explore = None
    brainsmith_roofline = None
    brainsmith_dataflow_analysis = None
    brainsmith_generate = None
    brainsmith_workflow = None
    explore_design_space = None
    validate_blueprint = None

try:
    from .legacy_support import (
        maintain_existing_api_compatibility, route_to_existing_implementation,
        warn_legacy_usage, install_legacy_compatibility, get_legacy_compatibility_report
    )
    logger.debug("Successfully imported legacy support functions")
except ImportError as e:
    logger.warning(f"Could not import legacy support: {e}")
    maintain_existing_api_compatibility = None
    route_to_existing_implementation = None
    warn_legacy_usage = None
    install_legacy_compatibility = None
    get_legacy_compatibility_report = None

# CLI is imported separately since it's primarily used standalone
try:
    from . import cli
    logger.debug("Successfully imported CLI module")
except ImportError as e:
    logger.warning(f"Could not import CLI module: {e}")
    cli = None

# Define public API
__all__ = [
    # Core orchestration
    'DesignSpaceOrchestrator',
    
    # FINN interface
    'FINNInterface',
    'FINNHooksPlaceholder',
    
    # Workflow management
    'WorkflowManager',
    'WorkflowType', 
    'WorkflowStatus',
    
    # Python API functions
    'brainsmith_explore',
    'brainsmith_roofline',
    'brainsmith_dataflow_analysis', 
    'brainsmith_generate',
    'brainsmith_workflow',
    'validate_blueprint',
    
    # Legacy compatibility
    'explore_design_space',
    'maintain_existing_api_compatibility',
    'route_to_existing_implementation',
    'install_legacy_compatibility',
    'get_legacy_compatibility_report',
    
    # CLI module
    'cli',
    
    # Module metadata
    '__version__',
    '__author__',
    '__description__'
]

# Filter out None values from __all__ if imports failed
__all__ = [name for name in __all__ if globals().get(name) is not None]

def get_core_status() -> dict:
    """
    Get status of all core components.
    
    Returns:
        Dictionary with component availability and status
    """
    status = {
        'version': __version__,
        'components': {
            'DesignSpaceOrchestrator': DesignSpaceOrchestrator is not None,
            'FINNInterface': FINNInterface is not None,
            'WorkflowManager': WorkflowManager is not None,
            'API': brainsmith_explore is not None,
            'CLI': cli is not None,
            'LegacySupport': maintain_existing_api_compatibility is not None
        },
        'api_functions': {
            'brainsmith_explore': brainsmith_explore is not None,
            'brainsmith_roofline': brainsmith_roofline is not None,
            'brainsmith_dataflow_analysis': brainsmith_dataflow_analysis is not None,
            'brainsmith_generate': brainsmith_generate is not None,
            'brainsmith_workflow': brainsmith_workflow is not None,
            'validate_blueprint': validate_blueprint is not None,
            'explore_design_space': explore_design_space is not None
        },
        'workflow_types': {
            'WorkflowType': WorkflowType is not None,
            'WorkflowStatus': WorkflowStatus is not None
        }
    }
    
    # Calculate overall readiness
    component_count = len(status['components'])
    available_components = sum(1 for available in status['components'].values() if available)
    status['readiness'] = available_components / component_count if component_count > 0 else 0.0
    
    # Check legacy compatibility if available
    if maintain_existing_api_compatibility:
        try:
            status['legacy_compatibility'] = maintain_existing_api_compatibility()
        except Exception as e:
            status['legacy_compatibility'] = False
            status['legacy_error'] = str(e)
    else:
        status['legacy_compatibility'] = None
    
    return status

def verify_installation() -> bool:
    """
    Verify that core components are properly installed and functional.
    
    Returns:
        True if installation is complete and functional
    """
    logger.info("Verifying Brainsmith core installation...")
    
    status = get_core_status()
    
    # Check critical components
    critical_components = [
        'DesignSpaceOrchestrator',
        'FINNInterface', 
        'API'
    ]
    
    missing_critical = []
    for component in critical_components:
        if not status['components'].get(component, False):
            missing_critical.append(component)
    
    if missing_critical:
        logger.error(f"Critical components missing: {missing_critical}")
        return False
    
    # Check API functions
    critical_apis = [
        'brainsmith_explore',
        'brainsmith_roofline',
        'brainsmith_dataflow_analysis',
        'brainsmith_generate'
    ]
    
    missing_apis = []
    for api_func in critical_apis:
        if not status['api_functions'].get(api_func, False):
            missing_apis.append(api_func)
    
    if missing_apis:
        logger.error(f"Critical API functions missing: {missing_apis}")
        return False
    
    # Log success
    readiness_percent = status['readiness'] * 100
    logger.info(f"✅ Core installation verified - {readiness_percent:.1f}% components available")
    
    # Report on legacy compatibility
    if status.get('legacy_compatibility') is True:
        logger.info("✅ Legacy compatibility verified")
    elif status.get('legacy_compatibility') is False:
        logger.warning("⚠️ Legacy compatibility issues detected")
    else:
        logger.info("ℹ️ Legacy compatibility check not available")
    
    return True

def get_quick_start_guide() -> str:
    """
    Get quick start guide for using the core components.
    
    Returns:
        Formatted quick start guide string
    """
    guide = f"""
Brainsmith Core {__version__} - Quick Start Guide

🚀 Basic Usage:
   from brainsmith.core import brainsmith_explore
   results, analysis = brainsmith_explore(
       model_path="model.onnx",
       blueprint_path="blueprint.yaml",
       exit_point="roofline"  # or "dataflow_analysis" or "dataflow_generation"
   )

📊 Hierarchical Exit Points:
   • roofline: Quick analytical bounds (~30s)
   • dataflow_analysis: Transform + estimation (~2min)  
   • dataflow_generation: Full RTL/HLS generation (~10min)

🔧 CLI Usage:
   brainsmith explore model.onnx blueprint.yaml --exit-point roofline
   brainsmith roofline model.onnx blueprint.yaml
   brainsmith generate model.onnx blueprint.yaml --output ./results

📚 Components Available:
   • DesignSpaceOrchestrator: {DesignSpaceOrchestrator is not None}
   • FINNInterface: {FINNInterface is not None}
   • WorkflowManager: {WorkflowManager is not None}
   • API Functions: {brainsmith_explore is not None}
   • CLI: {cli is not None}

💡 For detailed documentation, see docs/brainsmith_final_architectural_design.md
"""
    return guide

# Initialize legacy compatibility on import if available
if install_legacy_compatibility:
    try:
        compatibility_installed = install_legacy_compatibility()
        if compatibility_installed:
            logger.debug("Legacy compatibility automatically installed")
        else:
            logger.warning("Could not install legacy compatibility")
    except Exception as e:
        logger.warning(f"Legacy compatibility installation failed: {e}")

# Log module initialization
logger.info(f"Brainsmith core module {__version__} initialized")

# Verify installation on import in development mode
if __name__ != "__main__":
    try:
        installation_ok = verify_installation()
        if not installation_ok:
            logger.warning("Core installation verification failed - some features may not work")
    except Exception as e:
        logger.error(f"Installation verification error: {e}")

# Module-level convenience function
def show_status():
    """Print core module status to console."""
    status = get_core_status()
    print(f"\nBrainsmith Core {__version__} Status:")
    print(f"Overall readiness: {status['readiness']*100:.1f}%")
    print("\nComponents:")
    for component, available in status['components'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component}")
    
    print("\nAPI Functions:")
    for api_func, available in status['api_functions'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {api_func}")
    
    if status.get('legacy_compatibility') is not None:
        compat_icon = "✅" if status['legacy_compatibility'] else "⚠️"
        print(f"\nLegacy Compatibility: {compat_icon}")
    
    print(f"\n💡 Quick start: {__name__}.get_quick_start_guide()")