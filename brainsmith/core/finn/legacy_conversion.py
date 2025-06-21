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
import json
import sys

# Import proven step functions to replace dynamic generation
from brainsmith.libraries.transforms.steps import (
    onnx_preprocessing_step,  # ✅ ONNX simplify + cleanup for FINN
    qonnx_to_finn_step,       # ✅ Handles FoldConstants correctly
    streamlining_step,        # ✅ Proven transformation sequence
    infer_hardware_step,      # ✅ Complete hardware inference
    cleanup_step,             # ✅ Basic cleanup operations
    cleanup_advanced_step,    # ✅ Advanced cleanup
    fix_dynamic_dimensions_step,  # ✅ Fixes dynamic dimensions
    remove_head_step,         # ✅ BERT-specific head removal
    remove_tail_step,         # ✅ BERT-specific tail removal
    generate_reference_io_step,  # ✅ IO validation
    constrain_folding_and_set_pumped_compute_step,  # ✅ Folding optimization
    shell_metadata_handover_step,  # ✅ Shell metadata extraction
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
            
            # DEBUG: Print complete blueprint config
            print("="*80, file=sys.stderr)
            print("DEBUG: Complete Blueprint Configuration:", file=sys.stderr)
            print(json.dumps(blueprint_config, indent=2), file=sys.stderr)
            print("="*80, file=sys.stderr)
            
            # Build step function list from blueprint configuration
            step_functions = self._build_step_sequence(blueprint_config)
            
            # Extract FINN configuration parameters from blueprint
            finn_params = self._build_finn_config_params(blueprint_config)
            
            # Import enum types for conversion
            from finn.builder.build_dataflow_config import (
                AutoFIFOSizingMethod, ShellFlowType, VerificationStepType,
                VitisOptStrategyCfg, LargeFIFOMemStyle
            )
            
            # Convert enum string values to proper enum types
            enum_conversions = self._convert_finn_enums(finn_params, {
                'generate_outputs': DataflowOutputType,
                'verify_steps': VerificationStepType,
                'shell_flow_type': ShellFlowType,
                'auto_fifo_strategy': AutoFIFOSizingMethod,
                'large_fifo_mem_style': LargeFIFOMemStyle,
                'vitis_opt_strategy': VitisOptStrategyCfg
            })
            
            # DEBUG: Print FINN parameters
            print("="*80, file=sys.stderr)
            print("DEBUG: FINN Parameters from Blueprint:", file=sys.stderr)
            print(json.dumps(finn_params, indent=2), file=sys.stderr)
            print("="*80, file=sys.stderr)
            
            # Create DataflowBuildConfig with all parameters
            # Start with required parameters
            config_params = {
                'steps': step_functions,
                'output_dir': finn_params.get('output_dir', './finn_output'),
                'synth_clk_period_ns': finn_params.get('synth_clk_period_ns', 5.0),
                'generate_outputs': enum_conversions.get('generate_outputs', [DataflowOutputType.STITCHED_IP])
            }
            
            # Add all optional parameters from finn_params
            optional_params = [
                'specialize_layers_config_file', 'folding_config_file', 'target_fps',
                'folding_two_pass_relaxation', 'verify_steps', 'verify_input_npy',
                'verify_expected_output_npy', 'verify_save_full_context',
                'verify_save_rtlsim_waveforms', 'verification_atol', 'stitched_ip_gen_dcp',
                'signature', 'mvau_wwidth_max', 'standalone_thresholds', 'minimize_bit_width',
                'board', 'shell_flow_type', 'fpga_part', 'auto_fifo_depths',
                'split_large_fifos', 'auto_fifo_strategy', 'large_fifo_mem_style',
                'fifosim_input_throttle', 'fifosim_n_inferences', 'fifosim_save_waveform',
                'hls_clk_period_ns', 'default_swg_exception', 'vitis_platform',
                'vitis_floorplan_file', 'vitis_opt_strategy', 'save_intermediate_models',
                'enable_hw_debug', 'enable_build_pdb_debug', 'verbose', 'start_step',
                'stop_step', 'max_multithreshold_bit_width', 'rtlsim_batch_size',
                'rtlsim_use_vivado_comps'
            ]
            
            for param in optional_params:
                if param in finn_params:
                    # Use enum conversion if available, otherwise use raw value
                    if param in enum_conversions:
                        config_params[param] = enum_conversions[param]
                    else:
                        config_params[param] = finn_params[param]
            
            # DEBUG: Print final config params
            print("="*80, file=sys.stderr)
            print("DEBUG: Final DataflowBuildConfig Parameters:", file=sys.stderr)
            debug_params = {k: v for k, v in config_params.items() if k != 'steps'}
            debug_params['steps_count'] = len(config_params['steps'])
            debug_params['step_names'] = [s.__name__ for s in config_params['steps']]
            print(json.dumps(debug_params, indent=2, default=str), file=sys.stderr)
            print("="*80, file=sys.stderr)
            
            dataflow_config = DataflowBuildConfig(**config_params)
            
            logger.info(f"Created DataflowBuildConfig with {len(step_functions)} blueprint-configured steps")
            if dataflow_config.stop_step:
                logger.info(f"DataflowBuildConfig stop_step: {dataflow_config.stop_step}")
            if dataflow_config.start_step:
                logger.info(f"DataflowBuildConfig start_step: {dataflow_config.start_step}")
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
            'onnx_preprocessing_step': onnx_preprocessing_step,
            'cleanup_step': cleanup_step,
            'cleanup_advanced_step': cleanup_advanced_step,
            'fix_dynamic_dimensions_step': fix_dynamic_dimensions_step,
            'remove_head_step': remove_head_step,
            'remove_tail_step': remove_tail_step,
            'qonnx_to_finn_step': qonnx_to_finn_step,
            'streamlining_step': streamlining_step,
            'infer_hardware_step': infer_hardware_step,
            'generate_reference_io_step': generate_reference_io_step,
            'constrain_folding_and_set_pumped_compute_step': constrain_folding_and_set_pumped_compute_step,
            'shell_metadata_handover_step': shell_metadata_handover_step,
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
            
            # Extract stop_step and start_step
            if 'stop_step' in build_config:
                params['stop_step'] = build_config['stop_step']
                logger.info(f"Found stop_step in blueprint: {build_config['stop_step']}")
            
            if 'start_step' in build_config:
                params['start_step'] = build_config['start_step']
                logger.info(f"Found start_step in blueprint: {build_config['start_step']}")
        
        # Extract output directory
        params['output_dir'] = blueprint_config.get('output_dir', './finn_output')
        
        # Also check for stop_step/start_step at top level (for backward compatibility)
        if 'stop_step' in blueprint_config and 'stop_step' not in params:
            params['stop_step'] = blueprint_config['stop_step']
            logger.info(f"Found stop_step at blueprint top level: {blueprint_config['stop_step']}")
        
        if 'start_step' in blueprint_config and 'start_step' not in params:
            params['start_step'] = blueprint_config['start_step']
            logger.info(f"Found start_step at blueprint top level: {blueprint_config['start_step']}")
        
        # Default safe values
        params.setdefault('synth_clk_period_ns', 5.0)  # 200 MHz
        params.setdefault('auto_fifo_depths', True)
        params.setdefault('save_intermediate_models', True)
        
        # Process finn_config section - direct DataflowBuildConfig overrides
        if 'finn_config' in blueprint_config:
            finn_config = blueprint_config['finn_config']
            logger.info(f"Found finn_config section with {len(finn_config)} parameters")
            
            # Process each parameter with type conversion where needed
            for key, value in finn_config.items():
                # Handle enum conversions
                if key == 'generate_outputs' and isinstance(value, list):
                    # Keep as list - will be converted to enums in DataflowBuildConfig
                    params[key] = value
                elif key == 'verify_steps' and isinstance(value, list):
                    # Keep as list - will be converted to enums in DataflowBuildConfig
                    params[key] = value
                elif key == 'shell_flow_type' and isinstance(value, str):
                    # Keep as string - will be converted to enum in DataflowBuildConfig
                    params[key] = value
                elif key == 'auto_fifo_strategy' and isinstance(value, str):
                    # Keep as string - will be converted to enum in DataflowBuildConfig
                    params[key] = value
                elif key == 'large_fifo_mem_style' and isinstance(value, str):
                    # Keep as string - will be converted to enum in DataflowBuildConfig
                    params[key] = value
                elif key == 'vitis_opt_strategy' and isinstance(value, str):
                    # Keep as string - will be converted to enum in DataflowBuildConfig
                    params[key] = value
                else:
                    # Direct pass-through for all other parameters
                    params[key] = value
                
                logger.debug(f"  finn_config override: {key} = {value}")
        
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
    
    def _convert_finn_enums(self, finn_params: Dict[str, Any], enum_mappings: Dict[str, type]) -> Dict[str, Any]:
        """
        Convert string values to FINN enum types.
        
        Args:
            finn_params: Parameters dictionary with potential enum strings
            enum_mappings: Mapping of parameter names to enum types
            
        Returns:
            Dictionary with converted enum values
        """
        converted = {}
        
        for param_name, enum_type in enum_mappings.items():
            if param_name in finn_params:
                value = finn_params[param_name]
                
                if param_name in ['generate_outputs', 'verify_steps'] and isinstance(value, list):
                    # Handle list of enum values
                    converted_list = []
                    for item in value:
                        if isinstance(item, str):
                            try:
                                # Try to convert string to enum
                                converted_list.append(enum_type[item])
                            except KeyError:
                                logger.warning(f"Unknown {param_name} value: {item}, using as-is")
                                converted_list.append(item)
                        else:
                            converted_list.append(item)
                    converted[param_name] = converted_list
                elif isinstance(value, str):
                    # Handle single enum value
                    try:
                        converted[param_name] = enum_type[value]
                    except KeyError:
                        logger.warning(f"Unknown {param_name} value: {value}, using as-is")
                        converted[param_name] = value
                else:
                    # Non-string value, pass through
                    converted[param_name] = value
        
        return converted


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