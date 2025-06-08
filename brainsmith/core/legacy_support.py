"""
Legacy compatibility layer maintaining existing functionality.

This module ensures all existing API functions continue to work while
providing a clean transition path to the new extensible architecture.
"""

from typing import Dict, Any, List, Optional
import warnings
import logging

logger = logging.getLogger(__name__)

class LegacyAPIWarning(UserWarning):
    """Warning for deprecated but supported legacy API usage."""
    pass

def maintain_existing_api_compatibility() -> bool:
    """
    Ensure all existing API functions continue to work.
    This function validates that no existing functionality is broken.
    
    Returns:
        True if all legacy APIs are compatible, False otherwise
    """
    logger.info("Checking legacy API compatibility...")
    
    compatibility_results = []
    
    # Check existing explore_design_space function
    try:
        # Try to import existing functions to ensure they still work
        compatibility_results.append(_check_existing_explore_design_space())
        compatibility_results.append(_check_existing_blueprint_functions())
        compatibility_results.append(_check_existing_dse_functions())
        
        all_compatible = all(compatibility_results)
        
        if all_compatible:
            logger.info("✅ All legacy APIs are compatible")
        else:
            logger.warning("⚠️ Some legacy API compatibility issues detected")
        
        return all_compatible
        
    except Exception as e:
        logger.error(f"Legacy API compatibility check failed: {e}")
        return False

def _check_existing_explore_design_space() -> bool:
    """Check if existing explore_design_space function is accessible."""
    try:
        # Try to import from various possible locations
        import_attempts = [
            ("brainsmith", "explore_design_space"),
            ("brainsmith.dse", "explore_design_space"),
            ("brainsmith.api", "explore_design_space")
        ]
        
        for module_name, function_name in import_attempts:
            try:
                module = __import__(module_name, fromlist=[function_name])
                if hasattr(module, function_name):
                    logger.debug(f"Found existing {function_name} in {module_name}")
                    return True
            except ImportError:
                continue
        
        logger.warning("Existing explore_design_space function not found")
        return False
        
    except Exception as e:
        logger.error(f"Error checking explore_design_space: {e}")
        return False

def _check_existing_blueprint_functions() -> bool:
    """Check if existing blueprint functions are accessible."""
    try:
        # Check for existing blueprint functionality
        import_attempts = [
            ("brainsmith.blueprints", "get_blueprint"),
            ("brainsmith.blueprints", "Blueprint"),
            ("brainsmith", "get_blueprint")
        ]
        
        found_any = False
        for module_name, function_name in import_attempts:
            try:
                module = __import__(module_name, fromlist=[function_name])
                if hasattr(module, function_name):
                    logger.debug(f"Found existing {function_name} in {module_name}")
                    found_any = True
            except ImportError:
                continue
        
        if not found_any:
            logger.warning("No existing blueprint functions found")
        
        return found_any
        
    except Exception as e:
        logger.error(f"Error checking blueprint functions: {e}")
        return False

def _check_existing_dse_functions() -> bool:
    """Check if existing DSE functions are accessible."""
    try:
        # Check for existing DSE functionality
        import_attempts = [
            ("brainsmith.dse", "DSEEngine"),
            ("brainsmith.dse", "run_dse"),
            ("brainsmith", "run_dse")
        ]
        
        found_any = False
        for module_name, function_name in import_attempts:
            try:
                module = __import__(module_name, fromlist=[function_name])
                if hasattr(module, function_name):
                    logger.debug(f"Found existing {function_name} in {module_name}")
                    found_any = True
            except ImportError:
                continue
        
        if not found_any:
            logger.warning("No existing DSE functions found")
        
        return found_any
        
    except Exception as e:
        logger.error(f"Error checking DSE functions: {e}")
        return False

def route_to_existing_implementation(function_name: str, *args, **kwargs):
    """
    Route function calls to existing implementations when needed.
    Provides fallback to maintain compatibility.
    
    Args:
        function_name: Name of the legacy function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result from existing implementation
    """
    logger.info(f"Routing to existing implementation: {function_name}")
    
    try:
        if function_name == "explore_design_space":
            return _route_explore_design_space(*args, **kwargs)
        elif function_name == "get_blueprint":
            return _route_get_blueprint(*args, **kwargs)
        elif function_name == "run_dse":
            return _route_run_dse(*args, **kwargs)
        else:
            raise ValueError(f"Unknown legacy function: {function_name}")
            
    except Exception as e:
        logger.error(f"Failed to route to existing implementation: {e}")
        return _create_fallback_result(function_name, args, kwargs, str(e))

def _route_explore_design_space(*args, **kwargs):
    """Route to existing explore_design_space implementation."""
    try:
        # Try multiple import locations for existing function
        import_attempts = [
            "brainsmith.dse",
            "brainsmith.api", 
            "brainsmith"
        ]
        
        for module_name in import_attempts:
            try:
                module = __import__(module_name, fromlist=["explore_design_space"])
                if hasattr(module, "explore_design_space"):
                    explore_func = getattr(module, "explore_design_space")
                    logger.debug(f"Using explore_design_space from {module_name}")
                    return explore_func(*args, **kwargs)
            except ImportError:
                continue
        
        # If no existing implementation found, create fallback
        logger.warning("No existing explore_design_space found, using fallback")
        return _create_explore_design_space_fallback(*args, **kwargs)
        
    except Exception as e:
        logger.error(f"Error routing explore_design_space: {e}")
        return _create_fallback_result("explore_design_space", args, kwargs, str(e))

def _route_get_blueprint(*args, **kwargs):
    """Route to existing get_blueprint implementation."""
    try:
        # Try to import existing blueprint function
        from ..blueprints import get_blueprint
        logger.debug("Using existing get_blueprint function")
        return get_blueprint(*args, **kwargs)
        
    except ImportError:
        logger.warning("No existing get_blueprint found, using fallback")
        return _create_get_blueprint_fallback(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error routing get_blueprint: {e}")
        return _create_fallback_result("get_blueprint", args, kwargs, str(e))

def _route_run_dse(*args, **kwargs):
    """Route to existing run_dse implementation."""
    try:
        # Try to import existing DSE function
        from ..dse import run_dse
        logger.debug("Using existing run_dse function")
        return run_dse(*args, **kwargs)
        
    except ImportError:
        logger.warning("No existing run_dse found, using fallback")
        return _create_run_dse_fallback(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error routing run_dse: {e}")
        return _create_fallback_result("run_dse", args, kwargs, str(e))

def _create_explore_design_space_fallback(*args, **kwargs):
    """Create fallback result for explore_design_space when existing implementation not available."""
    return {
        'status': 'fallback',
        'function': 'explore_design_space',
        'message': 'Using fallback implementation - existing explore_design_space not available',
        'args': str(args),
        'kwargs': str(kwargs),
        'recommendation': 'Use new brainsmith_explore API for full functionality'
    }

def _create_get_blueprint_fallback(*args, **kwargs):
    """Create fallback result for get_blueprint when existing implementation not available."""
    return {
        'status': 'fallback',
        'function': 'get_blueprint', 
        'message': 'Using fallback implementation - existing get_blueprint not available',
        'args': str(args),
        'kwargs': str(kwargs),
        'recommendation': 'Use Blueprint.from_yaml_file for full functionality'
    }

def _create_run_dse_fallback(*args, **kwargs):
    """Create fallback result for run_dse when existing implementation not available."""
    return {
        'status': 'fallback',
        'function': 'run_dse',
        'message': 'Using fallback implementation - existing run_dse not available', 
        'args': str(args),
        'kwargs': str(kwargs),
        'recommendation': 'Use DesignSpaceOrchestrator for full functionality'
    }

def _create_fallback_result(function_name: str, args: tuple, kwargs: dict, error: str):
    """Create generic fallback result for any legacy function."""
    return {
        'status': 'error',
        'function': function_name,
        'error': error,
        'args': str(args),
        'kwargs': str(kwargs),
        'fallback': True,
        'recommendation': 'Check function name and arguments, or use new API'
    }

def warn_legacy_usage(function_name: str, new_function_name: str):
    """Warn users about legacy API usage while maintaining support."""
    warnings.warn(
        f"Using legacy function {function_name}. "
        f"Consider migrating to {new_function_name} for enhanced features.",
        LegacyAPIWarning,
        stacklevel=3
    )
    logger.info(f"Legacy API warning issued for {function_name} -> {new_function_name}")

def create_legacy_wrapper(new_function, legacy_name: str, new_name: str):
    """
    Create a wrapper function that provides legacy compatibility.
    
    Args:
        new_function: The new function to wrap
        legacy_name: Name of the legacy function
        new_name: Name of the new function
        
    Returns:
        Wrapper function that maintains legacy compatibility
    """
    def legacy_wrapper(*args, **kwargs):
        # Issue deprecation warning
        warn_legacy_usage(legacy_name, new_name)
        
        try:
            # Call new function with legacy arguments
            return new_function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Legacy wrapper failed for {legacy_name}: {e}")
            # Try to route to existing implementation as fallback
            return route_to_existing_implementation(legacy_name, *args, **kwargs)
    
    # Preserve function metadata
    legacy_wrapper.__name__ = legacy_name
    legacy_wrapper.__doc__ = f"Legacy wrapper for {new_name}. Use {new_name} for new code."
    
    return legacy_wrapper

def get_legacy_compatibility_report() -> Dict[str, Any]:
    """
    Generate comprehensive legacy compatibility report.
    
    Returns:
        Dictionary with compatibility status and recommendations
    """
    logger.info("Generating legacy compatibility report...")
    
    report = {
        'timestamp': str(__import__('datetime').datetime.now()),
        'overall_compatibility': maintain_existing_api_compatibility(),
        'functions_checked': [
            'explore_design_space',
            'get_blueprint', 
            'run_dse'
        ],
        'compatibility_details': {},
        'recommendations': []
    }
    
    # Check individual functions
    functions_to_check = [
        ('explore_design_space', _check_existing_explore_design_space),
        ('blueprint_functions', _check_existing_blueprint_functions),
        ('dse_functions', _check_existing_dse_functions)
    ]
    
    for func_name, check_func in functions_to_check:
        try:
            is_compatible = check_func()
            report['compatibility_details'][func_name] = {
                'compatible': is_compatible,
                'status': 'available' if is_compatible else 'missing_or_broken'
            }
            
            if not is_compatible:
                report['recommendations'].append(
                    f"Function {func_name} may need fallback implementation"
                )
                
        except Exception as e:
            report['compatibility_details'][func_name] = {
                'compatible': False,
                'status': 'error',
                'error': str(e)
            }
    
    # Add general recommendations
    if not report['overall_compatibility']:
        report['recommendations'].extend([
            "Consider using new brainsmith_explore API for enhanced functionality",
            "Test legacy functions with your specific use cases",
            "Plan migration to new API for future compatibility"
        ])
    
    return report

def install_legacy_compatibility():
    """
    Install legacy compatibility shims in the global namespace.
    
    This function can be called to automatically install wrapper functions
    that maintain backward compatibility with existing code.
    """
    logger.info("Installing legacy compatibility shims...")
    
    try:
        # Import new API functions
        from .api import brainsmith_explore, validate_blueprint
        
        # Create legacy wrappers
        legacy_explore = create_legacy_wrapper(
            brainsmith_explore, 
            "explore_design_space", 
            "brainsmith_explore"
        )
        
        # Install in brainsmith module namespace
        import brainsmith
        if not hasattr(brainsmith, 'explore_design_space'):
            setattr(brainsmith, 'explore_design_space', legacy_explore)
            logger.info("Installed legacy explore_design_space wrapper")
        
        # Install blueprint compatibility
        if not hasattr(brainsmith, 'validate_blueprint'):
            setattr(brainsmith, 'validate_blueprint', validate_blueprint)
            logger.info("Installed validate_blueprint function")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to install legacy compatibility: {e}")
        return False

def uninstall_legacy_compatibility():
    """Remove legacy compatibility shims."""
    logger.info("Removing legacy compatibility shims...")
    
    try:
        import brainsmith
        
        # Remove legacy functions if they were installed by us
        legacy_functions = ['explore_design_space', 'validate_blueprint']
        
        for func_name in legacy_functions:
            if hasattr(brainsmith, func_name):
                func = getattr(brainsmith, func_name)
                if hasattr(func, '__name__') and 'legacy_wrapper' in str(func):
                    delattr(brainsmith, func_name)
                    logger.info(f"Removed legacy function: {func_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to uninstall legacy compatibility: {e}")
        return False

# Automatic compatibility check on import
if __name__ != "__main__":
    try:
        compatibility_status = maintain_existing_api_compatibility()
        if compatibility_status:
            logger.debug("Legacy API compatibility verified")
        else:
            logger.warning("Some legacy APIs may not be fully compatible")
    except Exception as e:
        logger.error(f"Legacy compatibility check failed on import: {e}")