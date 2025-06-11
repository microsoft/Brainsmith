"""
Simplified FINN Interface

Clean wrapper around core FINN functionality with 4-hooks preparation.
This module provides a simple, function-based API for FINN integration
while maintaining preparation for future 4-hooks interface evolution.
"""

from typing import Dict, Any, Optional
import logging
from .finn_interface import FINNInterface as CoreFINNInterface
from .types import FINNConfig, FINNResult, FINNHooksConfig

logger = logging.getLogger(__name__)


class FINNInterface:
    """Simplified FINN interface for BrainSmith."""
    
    def __init__(self, config: Optional[FINNConfig] = None):
        """Initialize FINN interface with optional configuration."""
        self.config = config or FINNConfig()
        self.core_interface = CoreFINNInterface(self.config.to_core_dict())
        self.hooks_config = FINNHooksConfig()
        
        logger.info("Simplified FINN interface initialized")
    
    def build_accelerator(self, model_path: str, blueprint_config: Dict[str, Any],
                         output_dir: str = "./output") -> FINNResult:
        """
        Build FPGA accelerator using FINN.
        
        Args:
            model_path: Path to ONNX model file
            blueprint_config: Blueprint configuration dictionary
            output_dir: Output directory for build results
            
        Returns:
            FINNResult with build status and metrics
        """
        logger.info(f"Building accelerator for model: {model_path}")
        
        try:
            # Use core interface for actual build
            core_result = self.core_interface.build_accelerator(
                model_path, blueprint_config, output_dir
            )
            
            # Convert to simplified result format
            result = FINNResult.from_core_result(core_result)
            
            if result.success:
                logger.info("FINN build completed successfully")
            else:
                logger.error(f"FINN build failed: {result.error_message}")
                
            return result
            
        except Exception as e:
            logger.error(f"FINN build exception: {e}")
            return FINNResult(
                success=False,
                model_path=model_path,
                output_dir=output_dir,
                error_message=str(e)
            )
    
    def prepare_4hooks_config(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare configuration for future 4-hooks FINN interface.
        
        Args:
            design_point: Design point parameters
            
        Returns:
            Configuration dictionary prepared for 4-hooks interface
        """
        logger.debug("Preparing 4-hooks configuration")
        return self.hooks_config.prepare_config(design_point)
    
    def validate_config(self, blueprint_config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate blueprint configuration for FINN compatibility.
        
        Args:
            blueprint_config: Blueprint configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return self.core_interface.validate_config(blueprint_config)
    
    def get_supported_devices(self) -> list[str]:
        """Get list of supported FPGA devices."""
        return self.core_interface.get_supported_devices()


# Convenience functions for simple usage
def build_accelerator(model_path: str, blueprint_config: Dict[str, Any],
                     output_dir: str = "./output", config: Optional[FINNConfig] = None) -> FINNResult:
    """
    Simple function interface for FINN builds.
    
    Args:
        model_path: Path to ONNX model file
        blueprint_config: Blueprint configuration dictionary
        output_dir: Output directory for build results
        config: Optional FINN configuration
        
    Returns:
        FINNResult with build status and metrics
    """
    interface = FINNInterface(config)
    return interface.build_accelerator(model_path, blueprint_config, output_dir)


def validate_finn_config(blueprint_config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate FINN configuration.
    
    Args:
        blueprint_config: Blueprint configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    interface = FINNInterface()
    return interface.validate_config(blueprint_config)


def prepare_4hooks_config(design_point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare design point for future 4-hooks interface.
    
    Args:
        design_point: Design point parameters
        
    Returns:
        Configuration dictionary prepared for 4-hooks interface
    """
    interface = FINNInterface()
    return interface.prepare_4hooks_config(design_point)