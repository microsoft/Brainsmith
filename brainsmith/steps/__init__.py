"""
Brainsmith Step Library

A centralized registry for reusable build steps that can be composed into YAML blueprints.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import importlib.util
from pathlib import Path
import inspect

@dataclass
class StepMetadata:
    """Metadata for a build step."""
    name: str
    category: str  # "common", "transformer", etc.
    description: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class StepRegistry:
    """Central registry for all build steps."""
    
    def __init__(self):
        self._steps: Dict[str, Callable] = {}
        self._metadata: Dict[str, StepMetadata] = {}
        self._loaded = False
    
    def register_step(self, metadata: StepMetadata, func: Callable):
        """Register a step function."""
        self._steps[metadata.name] = func
        self._metadata[metadata.name] = metadata
    
    def get_step(self, name: str) -> Callable:
        """Get step function by name."""
        if not self._loaded:
            self._load_all_steps()
        
        # Check Brainsmith step library first
        if name in self._steps:
            return self._steps[name]
        
        # Check FINN steps as fallback
        try:
            from finn.builder.build_dataflow_steps import __dict__ as finn_steps
            if name in finn_steps and callable(finn_steps[name]):
                return finn_steps[name]
        except ImportError:
            pass
        
        raise ValueError(f"Step '{name}' not found in step library or FINN")
    
    def list_steps(self, category: str = None) -> List[str]:
        """List available steps."""
        if not self._loaded:
            self._load_all_steps()
        
        if category:
            return [name for name, meta in self._metadata.items() 
                   if meta.category == category]
        return list(self._steps.keys())
    
    def get_metadata(self, name: str) -> Optional[StepMetadata]:
        """Get metadata for a step."""
        if not self._loaded:
            self._load_all_steps()
        return self._metadata.get(name)
    
    def validate_sequence(self, step_names: List[str]) -> List[str]:
        """Validate step sequence and return any errors."""
        if not self._loaded:
            self._load_all_steps()
            
        errors = []
        for step_name in step_names:
            # Check if step exists (either in Brainsmith or FINN)
            try:
                self.get_step(step_name)
            except ValueError:
                errors.append(f"Step '{step_name}' not found")
                continue
                
            # Check dependencies for Brainsmith steps
            if step_name in self._metadata:
                metadata = self._metadata[step_name]
                for dep in metadata.dependencies:
                    if dep not in step_names:
                        errors.append(f"Step '{step_name}' requires '{dep}'")
                    elif step_names.index(dep) > step_names.index(step_name):
                        errors.append(f"Dependency '{dep}' must come before '{step_name}'")
        
        return errors
    
    def _load_all_steps(self):
        """Auto-discover and load all step modules."""
        steps_dir = Path(__file__).parent
        for category_dir in steps_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('_'):
                self._load_category(category_dir)
        self._loaded = True
    
    def _load_category(self, category_path: Path):
        """Load all steps from a category directory."""
        for py_file in category_path.glob("*.py"):
            if py_file.name != "__init__.py":
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"brainsmith.steps.{category_path.name}.{py_file.stem}",
                        py_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"Warning: Failed to load {py_file}: {e}")

# Global registry instance
STEP_REGISTRY = StepRegistry()

def register_step(name: str, category: str, description: str = "", 
                 dependencies: List[str] = None):
    """Decorator to register a step."""
    def decorator(func: Callable) -> Callable:
        metadata = StepMetadata(
            name=name,
            category=category,
            description=description,
            dependencies=dependencies or []
        )
        STEP_REGISTRY.register_step(metadata, func)
        return func
    return decorator

# Convenience functions
def get_step(name: str) -> Callable:
    """Get a step function by name."""
    return STEP_REGISTRY.get_step(name)

def list_steps(category: str = None) -> List[str]:
    """List available steps."""
    return STEP_REGISTRY.list_steps(category)

def validate_steps(step_names: List[str]) -> List[str]:
    """Validate a sequence of step names."""
    return STEP_REGISTRY.validate_sequence(step_names)