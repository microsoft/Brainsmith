"""
Legacy FINN backend for Phase 3.

This backend integrates with the existing FINN builder infrastructure using 
explicit build steps.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

# FINN imports
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)

from brainsmith.core.phase1.data_structures import OutputStage
from brainsmith.core.phase2.data_structures import BuildConfig
from .data_structures import BuildMetrics, BuildResult, BuildStatus
from .interfaces import BuildRunnerInterface
from .metrics_collector import MetricsCollector


class LegacyFINNBackend(BuildRunnerInterface):
    """Backend using existing FINN builder with explicit build steps."""
    
    def __init__(self, 
                 finn_build_dir: Optional[str] = None,
                 temp_cleanup: bool = True,
                 preserve_intermediate: bool = False):
        """
        Initialize Legacy FINN backend.
        
        Args:
            finn_build_dir: Directory for FINN build artifacts
            temp_cleanup: Whether to clean up temporary files
            preserve_intermediate: Whether to save intermediate models
        """
        self.finn_build_dir = finn_build_dir or tempfile.mkdtemp(prefix="finn_build_")
        self.temp_cleanup = temp_cleanup
        self.preserve_intermediate = preserve_intermediate
        
    def get_backend_name(self) -> str:
        """Return human-readable backend name."""
        return "FINN Legacy Builder"
        
    def get_supported_output_stages(self) -> List[OutputStage]:
        """Return list of supported output stages."""
        return [OutputStage.RTL, OutputStage.STITCHED_IP]
    
    def run(self, config: BuildConfig) -> BuildResult:
        """Execute build using FINN's build_dataflow infrastructure."""
        result = BuildResult(config_id=config.id)
        
        # Extract model path from config
        model_path = config.model_path
        
        # Set up logging
        logger = logging.getLogger("legacy_finn_backend")
        log_file = os.path.join(config.output_dir, "legacy_finn_backend.log")
        
        try:
            # Create output directory
            os.makedirs(config.output_dir, exist_ok=True)
            
            # Configure logging to file
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
            
            # Set FINN_BUILD_DIR environment variable
            os.environ["FINN_BUILD_DIR"] = self.finn_build_dir
            logger.info(f"FINN_BUILD_DIR set to: {self.finn_build_dir}")
            
            # Create FINN DataflowBuildConfig from BuildConfig
            finn_config = self._create_dataflow_config(config)
            
            # Save FINN config for debugging (only if it contains no functions)
            finn_config_path = os.path.join(config.output_dir, "finn_config.json")
            try:
                # Check if any steps are functions
                has_functions = any(callable(step) for step in finn_config.steps)
                if has_functions:
                    logger.info("FINN config contains function references, skipping JSON save")
                else:
                    with open(finn_config_path, 'w') as f:
                        f.write(finn_config.to_json())
                    logger.info(f"FINN config saved to: {finn_config_path}")
            except Exception as e:
                logger.warning(f"Could not save FINN config to JSON: {e}")
            
            # Execute FINN build with the preprocessed model
            build_exit_code = self._execute_finn_build(model_path, finn_config)
            
            if build_exit_code == 0:
                # Collect metrics and artifacts
                collector = MetricsCollector()
                result.metrics = collector.collect_from_finn_output(config.output_dir)
                result.artifacts = self._collect_artifacts(config.output_dir)
                
                result.complete(BuildStatus.SUCCESS)
                logger.info("Build completed successfully")
            else:
                error_msg = f"FINN build failed with exit code {build_exit_code}"
                result.complete(BuildStatus.FAILED, error_msg)
                logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"Build execution error: {str(e)}"
            result.complete(BuildStatus.FAILED, error_msg)
            logger.exception(error_msg)
        finally:
            # Remove logger handler
            if 'file_handler' in locals():
                logger.removeHandler(file_handler)
                file_handler.close()
                
            # Cleanup temporary files if requested
            if self.temp_cleanup and self.finn_build_dir.startswith("/tmp"):
                shutil.rmtree(self.finn_build_dir, ignore_errors=True)
                
        return result
    
    def _create_dataflow_config(self, config: BuildConfig) -> DataflowBuildConfig:
        """
        Convert BuildConfig to FINN DataflowBuildConfig.
        
        Args:
            config: Build configuration from DSE
            
        Returns:
            DataflowBuildConfig object for FINN builder
        """
        # Map output stage to FINN outputs
        output_mapping = {
            OutputStage.RTL: [
                DataflowOutputType.ESTIMATE_REPORTS,
                DataflowOutputType.RTLSIM_PERFORMANCE,
            ],
            OutputStage.STITCHED_IP: [
                DataflowOutputType.STITCHED_IP,
                DataflowOutputType.ESTIMATE_REPORTS,
                DataflowOutputType.RTLSIM_PERFORMANCE,
            ],
        }
        generate_outputs = output_mapping.get(
            config.global_config.output_stage, 
            [DataflowOutputType.ESTIMATE_REPORTS]
        )
        
        # Extract clock period from config flags
        synth_clk_period_ns = config.config_flags.get("clock_period_ns", 10.0)
        
        # Resolve step configuration
        resolved_steps = self._resolve_build_steps(config)
        
        # Map shell flow type string to enum
        shell_flow_str = config.config_flags.get("shell_flow_type", "vivado_zynq")
        shell_flow_type = (
            ShellFlowType.VIVADO_ZYNQ if shell_flow_str == "vivado_zynq" 
            else ShellFlowType.VITIS_ALVEO
        )
        
        # Create FINN DataflowBuildConfig
        finn_config = DataflowBuildConfig(
            output_dir=config.output_dir,
            synth_clk_period_ns=synth_clk_period_ns,
            generate_outputs=generate_outputs,
            
            # Use resolved build steps
            steps=resolved_steps,
            
            # Basic settings
            save_intermediate_models=self.preserve_intermediate,
            verbose=True,  # Enable verbose to see actual errors
            enable_build_pdb_debug=False,
            
            # Performance settings from config flags
            target_fps=config.config_flags.get("target_fps"),
            board=config.config_flags.get("board", "Pynq-Z1"),
            shell_flow_type=shell_flow_type,
            
            # FIFO and optimization settings
            auto_fifo_depths=config.config_flags.get("auto_fifo_depths", True),
            minimize_bit_width=config.config_flags.get("minimize_bit_width", True),
            
            # Folding config if specified
            folding_config_file=config.config_flags.get("folding_config_file"),
            
            # Start/stop steps for partial builds
            start_step=config.global_config.start_step,
            stop_step=config.global_config.stop_step,
            
            # Verification disabled for DSE (too slow)
            verify_steps=None,
        )
        
        return finn_config
    
    def _resolve_build_steps(self, config: BuildConfig) -> List:
        """
        Get build steps from configuration and resolve to function references.
        
        FINN accepts both string names (for standard FINN steps) and callable
        functions (for custom steps). We need to resolve our custom step names
        to their function references.
        
        Args:
            config: Build configuration with step settings
            
        Returns:
            List of step functions or names to execute
        """
        from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
        from brainsmith.steps import get_step
        
        if not config.build_steps:
            print(f"[LEGACY BACKEND] No build steps specified, using FINN defaults")
            return []
            
        print(f"[LEGACY BACKEND] Resolving {len(config.build_steps)} steps from blueprint")
        resolved_steps = []
        
        for step_name in config.build_steps:
            # First check if it's a standard FINN step - keep as string name
            if step_name in build_dataflow_step_lookup:
                resolved_steps.append(step_name)
                print(f"  - {step_name} (FINN standard step)")
            else:
                # Try to get it from our step registry as a function
                try:
                    step_fn = get_step(step_name)
                    resolved_steps.append(step_fn)
                    print(f"  - {step_name} (Brainsmith custom step)")
                except Exception as e:
                    # If not found in either, raise an error
                    raise ValueError(f"Step '{step_name}' not found in FINN or Brainsmith registries: {e}")
                    
        return resolved_steps
    
    def _execute_finn_build(self, model_path: str, finn_config: DataflowBuildConfig) -> int:
        """
        Execute FINN build_dataflow_cfg and return exit code.
        
        Args:
            model_path: Path to preprocessed ONNX model
            finn_config: FINN DataflowBuildConfig object
            
        Returns:
            Exit code (0 for success, -1 for failure)
        """
        # Set up logger to capture FINN output
        logger = logging.getLogger("legacy_finn_backend")
        
        try:
            # Log build configuration
            logger.info(f"Starting FINN build:")
            logger.info(f"  Model: {model_path}")
            logger.info(f"  Output dir: {finn_config.output_dir}")
            logger.info(f"  Clock period: {finn_config.synth_clk_period_ns}ns")
            logger.info(f"  Steps: {len(finn_config.steps) if finn_config.steps else 'default'}")
            
            # Save current working directory
            old_wd = os.getcwd()
            
            try:
                # Change to output directory for relative path resolution
                # (FINN expects to be run from the output directory)
                os.chdir(finn_config.output_dir)
                
                # Execute FINN build
                logger.info("Executing FINN build_dataflow_cfg...")
                exit_code = build_dataflow_cfg(model_path, finn_config)
                
                if exit_code == 0:
                    logger.info("FINN build completed successfully")
                else:
                    logger.error(f"FINN build failed with exit code {exit_code}")
                
                return exit_code
                
            finally:
                # Always restore working directory
                os.chdir(old_wd)
            
        except FileNotFoundError as e:
            logger.error(f"File not found during FINN build: {e}")
            return -1
        except ImportError as e:
            logger.error(f"Import error during FINN build (check FINN installation): {e}")
            return -1
        except Exception as e:
            logger.error(f"Unexpected error during FINN build: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
            return -1
    
    
    def _collect_artifacts(self, output_dir: str) -> Dict[str, str]:
        """Collect important build artifacts."""
        artifacts = {}
        
        # Common FINN artifacts
        artifact_patterns = {
            "stitched_ip": "stitched_ip/finn_design.xpr",
            "estimate_reports": "estimate_layer_resources_hls.json", 
            "performance_data": "rtlsim_performance.json",
            "build_log": "build_dataflow.log",
            "timing_summary": "time_per_step.json",
            "intermediate_models": "intermediate_models/",
        }
        
        for artifact_name, pattern in artifact_patterns.items():
            artifact_path = os.path.join(output_dir, pattern)
            if os.path.exists(artifact_path):
                artifacts[artifact_name] = artifact_path
                
        return artifacts