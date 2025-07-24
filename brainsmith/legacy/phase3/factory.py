"""
Build runner factory for Phase 3.

This module provides factory functions for creating build runner instances.
"""

from typing import Callable

from .build_runner import BuildRunner
from .legacy_finn_backend import LegacyFINNBackend
from .future_brainsmith_backend import FutureBrainsmithBackend


def create_build_runner_factory(backend_type: str = "auto") -> Callable[[], BuildRunner]:
    """
    Factory function for creating build runners.
    
    Args:
        backend_type: Type of backend to create. Options:
            - "legacy_finn": Use legacy FINN backend
            - "future_brainsmith": Use future FINN-Brainsmith backend (stub)
            - "auto": Automatically select backend (defaults to legacy)
            
    Returns:
        Factory function that creates BuildRunner instances
        
    Raises:
        ValueError: If backend_type is not recognized
    """
    
    def factory() -> BuildRunner:
        """Create and return a build runner instance."""
        
        if backend_type == "legacy_finn":
            backend = LegacyFINNBackend()
            
        elif backend_type == "future_brainsmith":
            backend = FutureBrainsmithBackend()
            
        elif backend_type == "auto":
            # Auto-select based on configuration
            # Always use LegacyFINNBackend for auto
            backend = LegacyFINNBackend()
            print("Auto-selecting backend: Using Legacy FINN Backend")
            
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Valid options: 'legacy_finn', 'future_brainsmith', 'auto'"
            )
        
        # Wrap the backend in the BuildRunner orchestrator
        return BuildRunner(backend)
    
    return factory