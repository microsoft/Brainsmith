"""
Legacy Conversion Layer

Converts Blueprint V2 6-entrypoint configuration to current FINN DataflowBuildConfig.
This bridges the future 6-entrypoint architecture with the current FINN API.

References: docs/finn_brainsmith_interfacing_runner.md for entrypoint mappings
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)


class LegacyConversionLayer:
    """Converts Blueprint V2 6-entrypoint config to FINN DataflowBuildConfig."""
    
    def __init__(self):
        """Initialize legacy conversion layer with function-based mappings."""
        # Import and validate required transformations
        self._validate_brainsmith_transformations()
        
        # Initialize function-based mappings
        self.entrypoint_function_mappings = self._initialize_function_mappings()
        self.standard_finn_steps = self._import_standard_steps()
        
        logger.info("LegacyConversionLayer initialized with function-based architecture")
    
    def convert_to_dataflow_config(self, entrypoint_config: Dict[str, List[str]], 
                                 blueprint_config: Dict[str, Any]):
        """
        Convert 6-entrypoint configuration to legacy FINN format.
        
        Uses docs/finn_brainsmith_interfacing_runner.md for mapping:
        - Entrypoint 1: canonical_ops → custom registration steps
        - Entrypoint 2: model_topology → early transformation steps  
        - Entrypoint 3/4: hw_kernels → kernel registration and specialization
        - Entrypoint 5: hw_kernel transforms → optimization steps
        - Entrypoint 6: hw_graph transforms → graph optimization steps
        
        Args:
            entrypoint_config: 6-entrypoint configuration from DSE
            blueprint_config: Blueprint V2 configuration
            
        Returns:
            FINN DataflowBuildConfig object
        """
        try:
            # Import real FINN config classes
            from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
            
            logger.info("Converting 6-entrypoint config to FINN DataflowBuildConfig")
            
            # Build step function list from entrypoint mappings
            step_functions = self._build_step_function_list(entrypoint_config)
            
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
            
            logger.info(f"Created DataflowBuildConfig with {len(step_functions)} step functions")
            return dataflow_config
            
        except ImportError as e:
            logger.error(f"FINN import failed: {e}")
            raise RuntimeError("FINN not available - ensure FINN is installed and accessible")
        except Exception as e:
            logger.error(f"DataflowBuildConfig creation failed: {e}")
            raise RuntimeError(f"Failed to create FINN configuration: {str(e)}")
    
    def _build_step_sequence(self, entrypoint_config: Dict[str, List[str]]) -> List[str]:
        """
        Build FINN step sequence from 6-entrypoint configuration.
        
        Args:
            entrypoint_config: 6-entrypoint configuration
            
        Returns:
            List of FINN step names in execution order
        """
        steps = []
        
        # Entrypoint 1: Register canonical ops (custom steps at start)
        for op in entrypoint_config.get('entrypoint_1', []):
            mapped_steps = self._map_entrypoint_to_steps(1, [op])
            steps.extend(mapped_steps)
        
        # Standard FINN initialization steps
        steps.extend([
            "step_qonnx_to_finn",
            "step_tidy_up"
        ])
        
        # Entrypoint 2: Model topology transforms (early transformation steps)
        for transform in entrypoint_config.get('entrypoint_2', []):
            mapped_steps = self._map_entrypoint_to_steps(2, [transform])
            steps.extend(mapped_steps)
        
        # Standard FINN transformation steps
        steps.extend([
            "step_streamline",
            "step_convert_to_hw",
            "step_create_dataflow_partition"
        ])
        
        # Entrypoint 3 & 4: Hardware kernels and specializations
        kernels = entrypoint_config.get('entrypoint_3', [])
        specializations = entrypoint_config.get('entrypoint_4', [])
        
        if kernels:
            kernel_steps = self._map_entrypoint_to_steps(3, kernels)
            steps.extend(kernel_steps)
        
        steps.append("step_specialize_layers")
        
        if specializations:
            spec_steps = self._map_entrypoint_to_steps(4, specializations)
            steps.extend(spec_steps)
        
        # Entrypoint 5: Hardware kernel transforms (optimization steps)
        hw_kernel_transforms = entrypoint_config.get('entrypoint_5', [])
        if hw_kernel_transforms:
            for transform in hw_kernel_transforms:
                mapped_steps = self._map_entrypoint_to_steps(5, [transform])
                steps.extend(mapped_steps)
        else:
            # Default HW optimization steps
            steps.extend([
                "step_target_fps_parallelization",
                "step_apply_folding_config",
                "step_minimize_bit_width"
            ])
        
        # Standard FINN build steps
        steps.extend([
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen"
        ])
        
        # Entrypoint 6: Hardware graph transforms (graph optimization steps)
        hw_graph_transforms = entrypoint_config.get('entrypoint_6', [])
        if hw_graph_transforms:
            for transform in hw_graph_transforms:
                mapped_steps = self._map_entrypoint_to_steps(6, [transform])
                steps.extend(mapped_steps)
        else:
            # Default graph optimization steps
            steps.extend([
                "step_set_fifo_depths",
                "step_create_stitched_ip"
            ])
        
        # Final performance measurement
        steps.append("step_measure_rtlsim_performance")
        
        logger.debug(f"Built step sequence with {len(steps)} steps: {steps}")
        return steps
    
    def _map_entrypoint_to_steps(self, entrypoint_id: int, components: List[str]) -> List[str]:
        """
        Map entrypoint components to FINN step sequence.
        
        Args:
            entrypoint_id: Entrypoint number (1-6)
            components: List of component names
            
        Returns:
            List of FINN step names
        """
        steps = []
        mappings = self.entrypoint_mappings.get(entrypoint_id, {})
        
        for component in components:
            if component in mappings:
                mapped_steps = mappings[component]
                if isinstance(mapped_steps, str):
                    steps.append(mapped_steps)
                elif isinstance(mapped_steps, list):
                    steps.extend(mapped_steps)
                else:
                    logger.warning(f"Invalid mapping for entrypoint {entrypoint_id}, component {component}")
            else:
                logger.warning(f"No mapping found for entrypoint {entrypoint_id}, component {component}")
        
        return steps
    
    def _build_finn_config_params(self, blueprint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract FINN configuration parameters from blueprint.
        
        Args:
            blueprint_config: Blueprint V2 configuration
            
        Returns:
            Dictionary of FINN configuration parameters
        """
        params = {}
        
        # Extract target device and constraints
        if 'constraints' in blueprint_config:
            constraints = blueprint_config['constraints']
            
            # Map blueprint constraints to FINN parameters
            if 'target_frequency_mhz' in constraints:
                # Convert MHz to nanoseconds
                freq_mhz = constraints['target_frequency_mhz']
                params['synth_clk_period_ns'] = 1000.0 / freq_mhz
            
            if 'target_throughput_fps' in constraints:
                params['target_fps'] = constraints['target_throughput_fps']
        
        # Extract configuration files
        if 'configuration_files' in blueprint_config:
            config_files = blueprint_config['configuration_files']
            
            if 'folding_override' in config_files:
                params['folding_config_file'] = config_files['folding_override']
            
            if 'platform_config' in config_files:
                params['board'] = self._extract_board_from_platform(config_files['platform_config'])
        
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
            logger.warning("Function generation will include placeholder implementations")
    
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
    
    def _build_step_function_list(self, entrypoint_config: Dict[str, List[str]]) -> List[Callable]:
        """Build list of step functions from 6-entrypoint configuration."""
        functions = []
        
        # Entrypoint 1: Custom operations registration
        for op in entrypoint_config.get('entrypoint_1', []):
            if op in self.entrypoint_function_mappings.get(1, {}):
                step_func = self.entrypoint_function_mappings[1][op]({})
                functions.append(step_func)
        
        # Add standard QONNX to FINN conversion if available
        if 'step_qonnx_to_finn' in self.standard_finn_steps:
            functions.append(self.standard_finn_steps['step_qonnx_to_finn'])
        
        # Entrypoint 2: Topology transformations
        for transform in entrypoint_config.get('entrypoint_2', []):
            if transform in self.entrypoint_function_mappings.get(2, {}):
                step_func = self.entrypoint_function_mappings[2][transform]({})
                functions.append(step_func)
        
        # Standard FINN conversion steps
        standard_conversion_steps = [
            'step_create_dataflow_partition'
        ]
        for step_name in standard_conversion_steps:
            if step_name in self.standard_finn_steps:
                functions.append(self.standard_finn_steps[step_name])
        
        # Entrypoint 3: Hardware kernels registration
        hw_kernels = entrypoint_config.get('entrypoint_3', [])
        if hw_kernels:
            # Create combined hardware inference step
            step_func = self._create_hardware_inference_step({'kernels': hw_kernels})
            functions.append(step_func)
        
        # Standard specialization
        if 'step_specialize_layers' in self.standard_finn_steps:
            functions.append(self.standard_finn_steps['step_specialize_layers'])
        
        # Entrypoint 4: Hardware specializations (handled within inference step)
        
        # Entrypoint 5: Kernel optimizations
        hw_transforms = entrypoint_config.get('entrypoint_5', [])
        optimization_steps = ['step_target_fps_parallelization', 'step_apply_folding_config', 'step_minimize_bit_width']
        
        if hw_transforms:
            for transform in hw_transforms:
                if transform in optimization_steps and transform in self.standard_finn_steps:
                    functions.append(self.standard_finn_steps[transform])
        else:
            # Default optimization steps
            for step_name in optimization_steps:
                if step_name in self.standard_finn_steps:
                    functions.append(self.standard_finn_steps[step_name])
        
        # Standard build progression
        build_steps = ['step_generate_estimate_reports', 'step_hw_codegen', 'step_hw_ipgen']
        for step_name in build_steps:
            if step_name in self.standard_finn_steps:
                functions.append(self.standard_finn_steps[step_name])
        
        # Entrypoint 6: Graph optimizations
        graph_transforms = entrypoint_config.get('entrypoint_6', [])
        graph_steps = ['step_set_fifo_depths', 'step_create_stitched_ip']
        
        if graph_transforms:
            for transform in graph_transforms:
                if transform in graph_steps and transform in self.standard_finn_steps:
                    functions.append(self.standard_finn_steps[transform])
        else:
            # Default graph optimization
            for step_name in graph_steps:
                if step_name in self.standard_finn_steps:
                    functions.append(self.standard_finn_steps[step_name])
        
        # Final performance measurement
        if 'step_measure_rtlsim_performance' in self.standard_finn_steps:
            functions.append(self.standard_finn_steps['step_measure_rtlsim_performance'])
        
        logger.debug(f"Built step function list with {len(functions)} functions")
        return functions
    
    def _initialize_function_mappings(self) -> Dict[int, Dict[str, Callable]]:</search>
</search_and_replace>
        """
        Initialize entrypoint to FINN step mappings.
        
        Based on docs/finn_brainsmith_interfacing_runner.md mapping table.
        
        Returns:
            Dictionary mapping entrypoint_id -> {component -> steps}
        """
        return {
            # Entrypoint 1: Register canonical ops
            1: {
                'LayerNorm': 'custom_step_register_layernorm',
                'Softmax': 'custom_step_register_softmax', 
                'GELU': 'custom_step_register_gelu',
                'MultiHeadAttention': 'custom_step_register_mha'
            },
            
            # Entrypoint 2: Topology transformations
            2: {
                'cleanup': 'custom_step_cleanup',
                'streamlining': 'step_streamline',
                'aggressive_streamlining': 'custom_step_aggressive_streamlining',
                'conservative_streamlining': 'custom_step_conservative_streamlining',
                'constant_folding': 'custom_step_constant_folding',
                'remove_head': 'custom_step_remove_head',
                'remove_tail': 'custom_step_remove_tail'
            },
            
            # Entrypoint 3: Register hardware abstraction kernels
            3: {
                'MatMul': 'custom_step_register_matmul_kernel',
                'LayerNorm': 'custom_step_register_layernorm_kernel',
                'Softmax': 'custom_step_register_softmax_kernel'
            },
            
            # Entrypoint 4: Register hardware specializations
            4: {
                'matmul_rtl': 'custom_step_apply_matmul_rtl',
                'matmul_hls': 'custom_step_apply_matmul_hls',
                'layernorm_custom': 'custom_step_apply_layernorm_custom',
                'softmax_hls': 'custom_step_apply_softmax_hls'
            },
            
            # Entrypoint 5: Hardware kernel transformations
            5: {
                'target_fps_parallelization': 'step_target_fps_parallelization',
                'apply_folding_config': 'step_apply_folding_config',
                'minimize_bit_width': 'step_minimize_bit_width',
                'optimize_memory_bandwidth': 'custom_step_optimize_memory_bandwidth'
            },
            
            # Entrypoint 6: Hardware graph transformations  
            6: {
                'set_fifo_depths': 'step_set_fifo_depths',
                'create_stitched_ip': 'step_create_stitched_ip',
                'optimize_pipeline': 'custom_step_optimize_pipeline'
            }
        }