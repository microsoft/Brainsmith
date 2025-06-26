"""
Legacy FINN backend for Phase 3.

This backend integrates with the existing FINN builder infrastructure using 
explicit build steps.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from brainsmith.core_v3.phase1.data_structures import OutputStage
from brainsmith.core_v3.phase2.data_structures import BuildConfig
from .data_structures import BuildMetrics, BuildResult, BuildStatus
from .interfaces import BuildRunnerInterface
from .metrics_collector import MetricsCollector
from .step_resolver import StepResolver


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
        self.step_resolver = StepResolver()
        
    def get_backend_name(self) -> str:
        """Return human-readable backend name."""
        return "FINN Legacy Builder"
        
    def get_supported_output_stages(self) -> List[OutputStage]:
        """Return list of supported output stages."""
        return [OutputStage.RTL, OutputStage.STITCHED_IP]
    
    def run(self, config: BuildConfig, model_path: str) -> BuildResult:
        """Execute build using FINN's build_dataflow infrastructure."""
        result = BuildResult(config_id=config.id)
        
        try:
            # Create output directory
            os.makedirs(config.output_dir, exist_ok=True)
            
            # Set FINN_BUILD_DIR environment variable
            os.environ["FINN_BUILD_DIR"] = self.finn_build_dir
            
            # Create FINN DataflowBuildConfig from BuildConfig
            finn_config = self._create_dataflow_config(config)
            
            # Save FINN config for debugging
            finn_config_path = os.path.join(config.output_dir, "finn_config.json")
            with open(finn_config_path, 'w') as f:
                json.dump(finn_config.to_dict() if hasattr(finn_config, 'to_dict') else {}, f, indent=2)
            
            # Execute FINN build with the preprocessed model
            build_exit_code = self._execute_finn_build(model_path, finn_config)
            
            if build_exit_code == 0:
                # Collect metrics and artifacts
                collector = MetricsCollector()
                result.metrics = collector.collect_from_finn_output(config.output_dir)
                result.artifacts = self._collect_artifacts(config.output_dir)
                
                result.complete(BuildStatus.SUCCESS)
            else:
                result.complete(BuildStatus.FAILED, f"FINN build failed with exit code {build_exit_code}")
                
        except Exception as e:
            result.complete(BuildStatus.FAILED, f"Build execution error: {str(e)}")
        finally:
            # Cleanup temporary files if requested
            if self.temp_cleanup and self.finn_build_dir.startswith("/tmp"):
                shutil.rmtree(self.finn_build_dir, ignore_errors=True)
                
        return result
    
    def _create_dataflow_config(self, config: BuildConfig) -> Dict:
        """
        Convert BuildConfig to FINN DataflowBuildConfig.
        
        Note: Returns a dict representation since we can't import actual FINN classes.
        In real implementation, would create actual DataflowBuildConfig object.
        """
        # Map output stage to FINN outputs
        output_mapping = {
            OutputStage.RTL: [
                "inference_cost",
                "rtlsim_perf", 
                "rtlsim_reports"
            ],
            OutputStage.STITCHED_IP: [
                "inference_cost",
                "rtlsim_perf",
                "rtlsim_reports",
                "bitfile",
                "pynq_driver",
                "deployment_package"
            ],
        }
        generate_outputs = output_mapping.get(
            config.global_config.output_stage, 
            ["estimate_reports"]
        )
        
        # Extract clock period from config flags
        synth_clk_period_ns = config.config_flags.get("clock_period_ns", 10.0)
        
        # Resolve step configuration
        resolved_steps = self._resolve_build_steps(config)
        
        # Create FINN config dict
        finn_config = {
            "output_dir": config.output_dir,
            "synth_clk_period_ns": synth_clk_period_ns,
            "generate_outputs": generate_outputs,
            
            # Use resolved build steps
            "steps": resolved_steps,
            
            # Basic settings
            "save_intermediate_models": self.preserve_intermediate,
            "verbose": False,
            "enable_build_pdb_debug": False,
            
            # Performance settings from config flags
            "target_fps": config.config_flags.get("target_fps"),
            "board": config.config_flags.get("board", "Pynq-Z1"),
            "shell_flow_type": config.config_flags.get("shell_flow_type", "vivado_zynq"),
            
            # FIFO and optimization settings
            "auto_fifo_depths": config.config_flags.get("auto_fifo_depths", True),
            "minimize_bit_width": config.config_flags.get("minimize_bit_width", True),
            
            # Verification disabled for DSE (too slow)
            "verify_steps": None,
        }
        
        return finn_config
    
    def _resolve_build_steps(self, config: BuildConfig) -> List[str]:
        """
        Resolve build steps based on step configuration.
        
        Args:
            config: Build configuration with step settings
            
        Returns:
            List of step names to execute
        """
        try:
            # Get base step list (either from config or standard)
            if config.build_steps:
                step_list = config.build_steps
            else:
                step_list = self.step_resolver.get_standard_steps()
            
            # Resolve start/stop steps using global config
            global_config = config.global_config
            start_step, stop_step = self.step_resolver.resolve_step_range(
                start_step=global_config.start_step,
                stop_step=global_config.stop_step,
                input_type=global_config.input_type,
                output_type=global_config.output_type,
                step_list=step_list
            )
            
            # If no step filtering specified, return original steps
            if start_step is None and stop_step is None:
                return step_list
            
            # Get filtered step slice
            filtered_steps = self.step_resolver.get_step_slice(
                step_list=step_list,
                start_step=start_step,
                stop_step=stop_step
            )
            
            print(f"[LEGACY BACKEND] Step filtering:")
            print(f"  Original steps: {len(step_list)} steps")
            if start_step is not None:
                print(f"  Start step: {start_step}")
            if stop_step is not None:
                print(f"  Stop step: {stop_step}")
            print(f"  Filtered steps: {len(filtered_steps)} steps")
            if len(filtered_steps) < 10:  # Only show if reasonable number
                for i, step in enumerate(filtered_steps):
                    print(f"    {i}: {step}")
            
            return filtered_steps
            
        except Exception as e:
            print(f"[LEGACY BACKEND] Warning: Step resolution failed: {e}")
            print(f"[LEGACY BACKEND] Falling back to original build steps")
            return config.build_steps if config.build_steps else self.step_resolver.get_standard_steps()
    
    def _execute_finn_build(self, model_path: str, finn_config: Dict) -> int:
        """
        Execute FINN build_dataflow_cfg and return exit code.
        
        Note: This is a stub implementation. Real implementation would
        import and call actual FINN build functions.
        """
        try:
            print(f"[STUB] Would execute FINN build with:")
            print(f"  Model: {model_path}")
            print(f"  Output dir: {finn_config['output_dir']}")
            print(f"  Clock period: {finn_config['synth_clk_period_ns']}ns")
            print(f"  Steps: {finn_config.get('steps', 'default')}")
            
            # Create mock outputs for testing
            self._create_mock_finn_outputs(finn_config['output_dir'])
            
            # Simulate successful build
            return 0
            
        except Exception as e:
            print(f"FINN build failed: {e}")
            return -1
    
    def _create_mock_finn_outputs(self, output_dir: str):
        """Create mock FINN outputs for testing."""
        # Create mock estimate report
        estimate_data = {
            "total": {
                "LUT": 25000,
                "DSP": 100,
                "BRAM_18K": 200,
                "URAM": 0
            }
        }
        with open(os.path.join(output_dir, "estimate_layer_resources_hls.json"), 'w') as f:
            json.dump(estimate_data, f, indent=2)
        
        # Create mock performance data
        perf_data = {
            "throughput_fps": 1000.0,
            "latency_cycles": 5000,
            "fclk_mhz": 200.0
        }
        with open(os.path.join(output_dir, "rtlsim_performance.json"), 'w') as f:
            json.dump(perf_data, f, indent=2)
        
        # Create mock timing data
        timing_data = {
            "step_tidy_up": 2.5,
            "step_streamline": 5.0,
            "step_convert_to_hw": 10.0
        }
        with open(os.path.join(output_dir, "time_per_step.json"), 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        # Create mock build log
        with open(os.path.join(output_dir, "build_dataflow.log"), 'w') as f:
            f.write("Mock FINN build log\n")
            f.write("Build completed successfully\n")
    
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