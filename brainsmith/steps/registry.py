"""
FINN Step Registry

Simple registry for FINN build steps with legacy compatibility.
"""

import logging
from typing import Dict, List, Callable, Any, Optional

logger = logging.getLogger(__name__)

class FinnStepRegistry:
    """
    Simple singleton registry for FINN build steps.
    
    Provides basic registration and lookup functionality with fallback to
    legacy FINN build steps for backward compatibility.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._steps = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._steps: Dict[str, Dict[str, Any]] = {}
            self._initialized = True
    
    def register(self, name: str, func: Callable[[Any, Any], Any], 
                 category: str = "unknown", dependencies: List[str] = None,
                 description: str = ""):
        """Register a FINN build step."""
        if dependencies is None:
            dependencies = []
            
        if name in self._steps:
            logger.warning(f"Overriding existing FINN step: {name}")
        
        self._steps[name] = {
            "function": func,
            "category": category,
            "dependencies": dependencies,
            "description": description
        }
        
        logger.debug(f"Registered FINN step: {name} (category: {category})")
    
    def get_step(self, name: str) -> Callable[[Any, Any], Any]:
        """
        Get FINN build step by name with legacy fallback.
        
        First checks this registry, then falls back to legacy steps in
        brainsmith.libraries.transforms.steps, then to FINN built-in steps.
        """
        # Check our registry first
        if name in self._steps:
            return self._steps[name]["function"]
        
        # Fallback to legacy BrainSmith steps
        try:
            from brainsmith.libraries.transforms.steps import get_step as legacy_get_step
            return legacy_get_step(name)
        except (ImportError, ValueError):
            pass
        
        # Final fallback to FINN built-in steps
        try:
            from finn.builder.build_dataflow_steps import __dict__ as finn_steps
            if name in finn_steps and callable(finn_steps[name]):
                return finn_steps[name]
        except ImportError:
            pass
        
        raise ValueError(f"FINN step '{name}' not found in registry or legacy systems")
    
    def list_steps(self) -> List[str]:
        """List all registered FINN steps."""
        return list(self._steps.keys())
    
    def get_step_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered step."""
        return self._steps.get(name)
    
    def list_by_category(self, category: str) -> List[str]:
        """List steps by category."""
        return [name for name, info in self._steps.items() 
                if info["category"] == category]
    
    def validate_dependencies(self, step_names: List[str]) -> List[str]:
        """
        Validate step dependencies and return any errors.
        
        Args:
            step_names: List of step names to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for step_name in step_names:
            if step_name not in self._steps:
                # Only validate dependencies for steps in our registry
                continue
                
            step_info = self._steps[step_name]
            for dep in step_info["dependencies"]:
                if dep not in step_names:
                    errors.append(f"Step '{step_name}' requires '{dep}'")
                elif step_names.index(dep) > step_names.index(step_name):
                    errors.append(f"Dependency '{dep}' must come before '{step_name}'")
        
        return errors