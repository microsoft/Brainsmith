"""
Build runner interfaces for Phase 3.

This module defines the abstract interface that all build backends must implement.
"""

from abc import ABC, abstractmethod
from typing import List

from brainsmith.core.phase1.data_structures import OutputStage
from brainsmith.core.phase2.data_structures import BuildConfig
from .data_structures import BuildResult


class BuildRunnerInterface(ABC):
    """Abstract interface for build execution backends."""
    
    @abstractmethod
    def run(self, config: BuildConfig) -> BuildResult:
        """
        Execute build and return results.
        
        Args:
            config: Build configuration from Phase 2 (includes model_path)
            
        Returns:
            BuildResult with status, metrics, and artifacts
        """
        pass
        
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return human-readable backend name."""
        pass
        
    @abstractmethod
    def get_supported_output_stages(self) -> List[OutputStage]:
        """Return list of supported output stages."""
        pass