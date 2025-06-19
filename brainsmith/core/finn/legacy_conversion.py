"""
Legacy Conversion Layer - Function-based Implementation

Converts Blueprint V2 6-entrypoint configuration to current FINN DataflowBuildConfig.
This bridges the future 6-entrypoint architecture with the current FINN API using
proven step functions from brainsmith.libraries.transforms.steps.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import importlib
from pathlib import Path

# Import proven step functions to replace dynamic generation
from brainsmith.libraries.transforms.steps import (
    qonnx_to_finn_step,       # ✅ Handles FoldConstants correctly
    streamlining_step,        # ✅ Proven transformation sequence
    infer_hardware_step,      # ✅ Complete hardware inference
    cleanup_step,             # ✅ Basic cleanup operations
    cleanup_advanced_step,    # ✅ Advanced cleanup
    remove_head_step,         # ✅ BERT-specific head removal
    remove_tail_step,         # ✅ BERT-specific tail removal
    generate_reference_io_step,  # ✅ IO validation
)

logger = logging.getLogger(__name__)


class LegacyConversionLayer:
    """Converts Blueprint V2 6-entrypoint config to FINN DataflowBuildConfig with blueprint-driven step ordering."""
    
    def __init__(self):
        """Initialize legacy conversion layer with blueprint-driven step mapping."""
        # Import and validate required transformations
        self._validate_brainsmith_transformations()
        
        # Initialize standard FINN steps
        self.standard_finn_steps = self._import_standard_steps()
        
        # Initialize BrainSmith step mapping
        self.brainsmith_steps = self._initialize_brainsmith_steps()
        
        logger.info("LegacyConversionLayer initialized with blueprint-driven step ordering")
    
    def convert_to_dataflow_config(self, entrypoint_config: Dict[str, List[str]], 
                                 blueprint_config: Dict[str, Any]):
        """
        Convert 6-entrypoint configuration to FINN DataflowBuildConfig with blueprint-driven step ordering.
        
        Args:
            entrypoint_config: 6-entrypoint configuration from DSE
            blueprint_config: Blueprint V2 configuration
            
        Returns:
            FINN DataflowBuildConfig object with blueprint-configured step sequence
        """
        try:
            # Import real FINN config classes
            from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
            
            logger.info("Converting to FINN DataflowBuildConfig with blueprint-driven step ordering")
            
            # Build step function list from blueprint configuration
            step_functions = self._build_step_sequence(blueprint_config)
            
            # Extract FINN configuration parameters from blueprint
            finn_params = self._build_finn_config_params(blueprint_config)
            
            # Create DataflowBuildConfig with real FINN
            dataflow_config = DataflowBuildConfig(
                steps=step_functions,
                output_dir=finn_params.get('output_dir', './finn_output'),
                synth_clk_period_ns=finn_params.get('synth_clk_period_ns', 5.0),  # 200 MHz default
                target_fps=finn_params.get('target_fps', None),
                folding_config_file=finn_params.get('folding_config_file', None),
                generate_outputs=[DataflowOutputType.STITCHED_IP],  # Default output
                board=finn_params.get('board', None),
                auto_fifo_depths=finn_params.get('auto_fifo_depths', True),
                verify_steps=finn_params.get('verify_steps', []),
                save_intermediate_models=finn_params.get('save_intermediate_models', True)
            )
            
            logger.info(f"Created DataflowBuildConfig with {len(step_functions)} blueprint-configured steps")
            return dataflow_config
            
        except ImportError as e:
            logger.error(f"FINN import failed: {e}")
            raise RuntimeError("FINN not available - ensure FINN is installed and accessible")
        except Exception as e:
            logger.error(f"DataflowBuildConfig creation failed: {e}")
            raise RuntimeError(f"Failed to create FINN configuration: {str(e)}")
    
    def _validate_brainsmith_transformations(self):
        """Ensure required BrainSmith transformations are available."""
        required_transforms = [
            'brainsmith.libraries.transforms.operations.expand_norms.ExpandNorms',
        ]
        
        missing = []
        for transform_path in required_transforms:
            module_path, class_name = transform_path.rsplit('.', 1)
            try:
                module = importlib.import_module(module_path)
                getattr(module, class_name)
                logger.debug(f"Found BrainSmith transformation: {transform_path}")
            except (ImportError, AttributeError):
                missing.append(transform_path)
                logger.warning(f"Missing BrainSmith transformation: {transform_path}")
        
        if missing:
            logger.warning(f"Some BrainSmith transformations unavailable: {missing}")
            logger.warning("Using proven step functions as alternatives")
    
    def _import_standard_steps(self) -> Dict[str, Callable]:
        """Import standard FINN step functions."""
        try:
            from finn.builder.build_dataflow_steps import (
                step_create_dataflow_partition,
                step_specialize_layers,
                step_target_fps_parallelization,
                step_apply_folding_config,
                step_minimize_bit_width,
                step_generate_estimate_reports,
                step_hw_codegen,
                step_hw_ipgen,
                step_set_fifo_depths,
                step_create_stitched_ip,
                step_measure_rtlsim_performance
            )
            
            steps = {
                'step_create_dataflow_partition': step_create_dataflow_partition,
                'step_specialize_layers': step_specialize_layers,
                'step_target_fps_parallelization': step_target_fps_parallelization,
                'step_apply_folding_config': step_apply_folding_config,
                'step_minimize_bit_width': step_minimize_bit_width,
                'step_generate_estimate_reports': step_generate_estimate_reports,
                'step_hw_codegen': step_hw_codegen,
                'step_hw_ipgen': step_hw_ipgen,
                'step_set_fifo_depths': step_set_fifo_depths,
                'step_create_stitched_ip': step_create_stitched_ip,
                'step_measure_rtlsim_performance': step_measure_rtlsim_performance
            }
            
            logger.info(f"Imported {len(steps)} standard FINN step functions")
            return steps
            
        except ImportError as e:
            logger.warning(f"FINN step functions not available: {e}")
            return {}
    
    def _initialize_brainsmith_steps(self) -> Dict[str, Callable]:
        """Initialize mapping of BrainSmith step names to functions."""
        return {
            'cleanup_step': cleanup_step,
            'cleanup_advanced_step': cleanup_advanced_step,
            'remove_head_step': remove_head_step,
            'remove_tail_step': remove_tail_step,
            'qonnx_to_finn_step': qonnx_to_finn_step,
            'streamlining_step': streamlining_step,
            'infer_hardware_step': infer_hardware_step,
            'generate_reference_io_step': generate_reference_io_step,
        }
    
    def _build_step_sequence(self, blueprint_config: Dict[str, Any]) -> List[Callable]:
        """
        Build step sequence from blueprint configuration using legacy_preproc and legacy_postproc.
        
        Args:
            blueprint_config: Blueprint V2 configuration with step ordering
            
        Returns:
            List of step functions in blueprint-specified order
        """
        steps = []
        
        logger.debug("Building step sequence from blueprint configuration")
        
        # Phase 1: Optional preprocessing steps
        preproc_steps = blueprint_config.get('legacy_preproc', [])
        logger.debug(f"Adding {len(preproc_steps)} preprocessing steps")
        for step_name in preproc_steps:
            step_func = self._resolve_step_function(step_name)
            if step_func:
                steps.append(step_func)
                logger.debug(f"  ✅ Added preproc step: {step_name}")
            else:
                logger.warning(f"  ❌ Preproc step not found: {step_name}")
                self._log_available_steps()
        
        # Phase 2: Standard FINN pipeline (always included)
        logger.debug("Adding standard FINN pipeline steps")
        standard_steps = [
            'step_create_dataflow_partition',
            'step_specialize_layers', 
            'step_target_fps_parallelization',
            'step_apply_folding_config',
            'step_minimize_bit_width',
            'step_generate_estimate_reports',
            'step_hw_codegen',
            'step_hw_ipgen'
        ]
        
        for step_name in standard_steps:
            if step_name in self.standard_finn_steps:
                steps.append(self.standard_finn_steps[step_name])
                logger.debug(f"  ✅ Added FINN step: {step_name}")
            else:
                logger.warning(f"  ❌ FINN step not available: {step_name}")
        
        # Phase 3: Optional postprocessing steps  
        postproc_steps = blueprint_config.get('legacy_postproc', [])
        logger.debug(f"Adding {len(postproc_steps)} postprocessing steps")
        for step_name in postproc_steps:
            step_func = self._resolve_step_function(step_name)
            if step_func:
                steps.append(step_func)
                logger.debug(f"  ✅ Added postproc step: {step_name}")
            else:
                logger.warning(f"  ❌ Postproc step not found: {step_name}")
                self._log_available_steps()
        
        total_steps = len(steps)
        logger.info(f"Built step sequence with {total_steps} steps from blueprint configuration")
        
        # Log complete step sequence for debugging
        step_names = []
        for i, step in enumerate(steps, 1):
            step_name = getattr(step, '__name__', str(step))
            step_names.append(f"[{i}/{total_steps}] {step_name}")
        
        logger.debug("Final step sequence:")
        for step_name in step_names:
            logger.debug(f"  {step_name}")
        
        # Validate we have at least some steps
        if total_steps == 0:
            raise ValueError(
                "No valid steps found in blueprint configuration. "
                "Ensure 'legacy_preproc' and/or 'legacy_postproc' contain valid step names."
            )
        
        return steps
    
    def _resolve_step_function(self, step_name: str) -> Optional[Callable]:
        """
        Resolve step name to step function, checking both BrainSmith and FINN steps.
        
        Args:
            step_name: Name of the step function
            
        Returns:
            Step function or None if not found
        """
        # Check BrainSmith steps first
        if step_name in self.brainsmith_steps:
            logger.debug(f"  Found BrainSmith step: {step_name}")
            return self.brainsmith_steps[step_name]
        
        # Check standard FINN steps
        if step_name in self.standard_finn_steps:
            logger.debug(f"  Found FINN step: {step_name}")
            return self.standard_finn_steps[step_name]
        
        logger.warning(f"  Step function not found: {step_name}")
        return None
    
    def _log_available_steps(self):
        """Log available steps for debugging."""
        logger.debug(f"  Available BrainSmith steps: {list(self.brainsmith_steps.keys())}")
        logger.debug(f"  Available FINN steps: {list(self.standard_finn_steps.keys())}")
    
    def _build_finn_config_params(self, blueprint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract FINN configuration parameters from blueprint.
        
        Args:
            blueprint_config: Blueprint V2 configuration
            
        Returns:
            Dictionary of FINN configuration parameters
        """
        params = {}
        
        # Extract platform settings from blueprint
        if 'platform' in blueprint_config:
            platform = blueprint_config['platform']
            
            # Extract board
            if 'board' in platform:
                params['board'] = platform['board']
            
            # Extract frequency and convert to clock period
            if 'target_frequency' in platform:
                freq_mhz = platform['target_frequency']
                params['synth_clk_period_ns'] = 1000.0 / freq_mhz
        
        # Extract build configuration
        if 'build_configuration' in blueprint_config:
            build_config = blueprint_config['build_configuration']
            
            # Auto FIFO depths
            if 'auto_fifo_depths' in build_config:
                params['auto_fifo_depths'] = build_config['auto_fifo_depths']
            
            # Save intermediate models
            if 'save_intermediate_models' in build_config:
                params['save_intermediate_models'] = build_config['save_intermediate_models']
        
        # Extract output directory
        params['output_dir'] = blueprint_config.get('output_dir', './finn_output')
        
        # Default safe values
        params.setdefault('synth_clk_period_ns', 5.0)  # 200 MHz
        params.setdefault('auto_fifo_depths', True)
        params.setdefault('save_intermediate_models', True)
        
        logger.debug(f"Extracted FINN parameters: {params}")
        return params
    
    def _extract_board_from_platform(self, platform_config_path: str) -> Optional[str]:
        """Extract board name from platform configuration."""
        # Simple heuristic - extract from filename
        platform_name = Path(platform_config_path).stem
        
        # Map common platform configs to board names
        board_mappings = {
            'zynq_ultrascale': 'Pynq-Z1',
            'alveo_u250': 'U250', 
            'alveo_u280': 'U280',
            'kv260': 'KV260'
        }
        
        return board_mappings.get(platform_name, platform_name)


# Convenience function for external usage
def convert_to_dataflow_config(entrypoint_config: Dict[str, List[str]], 
                             blueprint_config: Dict[str, Any]):
    """
    Convenience function to convert Blueprint V2 config to FINN DataflowBuildConfig.
    
    Args:
        entrypoint_config: 6-entrypoint configuration from DSE
        blueprint_config: Blueprint V2 configuration
        
    Returns:
        FINN DataflowBuildConfig object
    """
    converter = LegacyConversionLayer()
    return converter.convert_to_dataflow_config(entrypoint_config, blueprint_config)