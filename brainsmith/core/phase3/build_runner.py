"""
Build runner orchestrator for Phase 3.

This module provides the main BuildRunner class that orchestrates the entire
build process, including preprocessing, backend execution, and postprocessing.
"""

import logging
from typing import Optional

from brainsmith.core.phase2.data_structures import BuildConfig
from .data_structures import BuildResult, BuildStatus
from .interfaces import BuildRunnerInterface
from .preprocessing import PreprocessingPipeline
from .postprocessing import PostprocessingPipeline

logger = logging.getLogger(__name__)


class BuildRunner:
    """
    Orchestrates the complete build process.
    
    This class manages the entire build lifecycle:
    1. Preprocessing pipeline execution
    2. Backend-specific build execution
    3. Postprocessing pipeline execution
    
    The actual build logic is delegated to the backend implementation,
    while this class ensures consistent pipeline application across all backends.
    """
    
    def __init__(self, backend: BuildRunnerInterface):
        """
        Initialize BuildRunner with a specific backend.
        
        Args:
            backend: The backend implementation to use for builds
        """
        self.backend = backend
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.postprocessing_pipeline = PostprocessingPipeline()
        
    def run(self, config: BuildConfig) -> BuildResult:
        """
        Execute the complete build process.
        
        Args:
            config: Build configuration from Phase 2 (includes model_path)
            
        Returns:
            BuildResult with status, metrics, and artifacts
        """
        logger.info(f"Starting build {config.id} using {self.backend.get_backend_name()}")
        
        # Extract model path from config
        model_path = config.model_path
        processed_model_path = None
        
        try:
            # Step 1: Preprocessing
            logger.info("Executing preprocessing pipeline")
            processed_model_path = self.preprocessing_pipeline.execute(config, model_path)
            logger.info(f"Preprocessing complete, processed model at: {processed_model_path}")
            
            # Step 2: Backend execution
            logger.info("Executing backend build")
            result = self.backend.run(config)
            
            # Step 3: Postprocessing (only if build was successful)
            if result.is_successful():
                logger.info("Executing postprocessing pipeline")
                try:
                    self.postprocessing_pipeline.analyze(config, result)
                    logger.info("Postprocessing complete")
                except Exception as e:
                    logger.warning(f"Postprocessing failed but not affecting build result: {str(e)}")
                    # Note: Postprocessing failures don't affect the build result
            else:
                logger.warning(f"Skipping postprocessing due to build failure: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Build process failed with exception: {str(e)}")
            # Create a failed result if backend didn't return one
            result = BuildResult(config_id=config.id)
            result.complete(BuildStatus.FAILED, f"Build process error: {str(e)}")
            return result
    
    def get_backend_name(self) -> str:
        """Get the name of the underlying backend."""
        return self.backend.get_backend_name()
    
    def get_supported_output_stages(self) -> list:
        """Get the output stages supported by the backend."""
        return self.backend.get_supported_output_stages()