"""
Legacy Conversion Layer - Function-based Implementation

Converts Blueprint V2 6-entrypoint configuration to current FINN DataflowBuildConfig.
This bridges the future 6-entrypoint architecture with the current FINN API using
function-based step mappings as demonstrated in bert.py.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)


class LegacyConversionLayer:
    """Converts Blueprint V2 6-entrypoint config to FINN DataflowBuildConfig with function-based steps."""
    
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
        Convert 6-entrypoint configuration to FINN DataflowBuildConfig with function list.
        
        Args:
            entrypoint_config: 6-entrypoint configuration from DSE
            blueprint_config: Blueprint V2 configuration
            
        Returns:
            FINN DataflowBuildConfig object with function-based steps
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
        
        # Standard FINN initialization (placeholder for step_qonnx_to_finn)
        functions.append(self._create_qonnx_to_finn_step())
        functions.append(self._create_tidy_up_step())
        
        # Entrypoint 2: Topology transformations
        for transform in entrypoint_config.get('entrypoint_2', []):
            if transform in self.entrypoint_function_mappings.get(2, {}):
                step_func = self.entrypoint_function_mappings[2][transform]({})
                functions.append(step_func)
        
        # Standard FINN conversion steps
        if 'step_create_dataflow_partition' in self.standard_finn_steps:
            functions.append(self.standard_finn_steps['step_create_dataflow_partition'])
        
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
    
    def _initialize_function_mappings(self) -> Dict[int, Dict[str, Callable]]:
        """Map entrypoint components to actual step function generators."""
        return {
            # Entrypoint 1: Canonical ops → custom conversion functions
            1: {
                'LayerNorm': self._create_layernorm_registration_step,
                'Softmax': self._create_softmax_registration_step,
                'GELU': self._create_gelu_registration_step,
                'MultiHeadAttention': self._create_mha_registration_step
            },
            
            # Entrypoint 2: Topology transforms → custom streamlining functions
            2: {
                'cleanup': self._create_cleanup_step,
                'streamlining': self._create_streamlining_step,
                'remove_head': self._create_remove_head_step,
                'remove_tail': self._create_remove_tail_step
            }
        }
    
    def _create_layernorm_registration_step(self, config_params):
        """Generate LayerNorm registration step function."""
        def custom_step_register_layernorm(model, cfg):
            """Register LayerNorm operations for FINN processing."""
            try:
                from brainsmith.libraries.transforms.operations.expand_norms import ExpandNorms
                model = model.transform(ExpandNorms())
                logger.debug("Applied ExpandNorms transformation")
            except ImportError:
                logger.warning("ExpandNorms transformation not available, skipping")
            return model
        
        return custom_step_register_layernorm
    
    def _create_softmax_registration_step(self, config_params):
        """Generate Softmax registration step function."""
        def custom_step_register_softmax(model, cfg):
            """Register Softmax operations with quantization handling."""
            try:
                from qonnx.transformation.general import ConvertDivToMul
                from qonnx.transformation.fold_constants import FoldConstants
                
                model = model.transform(FoldConstants())
                model = model.transform(ConvertDivToMul())
                logger.debug("Applied Softmax registration transformations")
            except ImportError:
                logger.warning("Softmax registration transformations not available")
            return model
        
        return custom_step_register_softmax
    
    def _create_gelu_registration_step(self, config_params):
        """Generate GELU registration step function."""
        def custom_step_register_gelu(model, cfg):
            """Register GELU operations for FINN processing."""
            logger.debug("GELU registration step (placeholder)")
            return model
        
        return custom_step_register_gelu
    
    def _create_mha_registration_step(self, config_params):
        """Generate MultiHeadAttention registration step function."""
        def custom_step_register_mha(model, cfg):
            """Register MultiHeadAttention operations for FINN processing."""
            logger.debug("MultiHeadAttention registration step (placeholder)")
            return model
        
        return custom_step_register_mha
    
    def _create_cleanup_step(self, config_params):
        """Generate cleanup step function."""
        def custom_step_cleanup(model, cfg):
            """Custom cleanup steps for model preparation."""
            try:
                from qonnx.transformation.general import (
                    SortCommutativeInputsInitializerLast, 
                    RemoveUnusedTensors
                )
                from qonnx.transformation.remove import RemoveIdentityOps
                
                model = model.transform(SortCommutativeInputsInitializerLast())
                model = model.transform(RemoveIdentityOps())
                model = model.transform(RemoveUnusedTensors())
                logger.debug("Applied cleanup transformations")
            except ImportError:
                logger.warning("Cleanup transformations not available")
            return model
        
        return custom_step_cleanup
    
    def _create_streamlining_step(self, config_params):
        """Generate streamlining step function."""
        def custom_streamlining_step(model, cfg):
            """Custom streamlining with domain-specific optimizations."""
            try:
                import finn.transformation.streamline as absorb
                import finn.transformation.streamline.reorder as reorder
                from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
                from qonnx.transformation.infer_datatypes import InferDataTypes
                from qonnx.transformation.general import GiveUniqueNodeNames
                
                # Apply streamlining sequence
                model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
                model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
                model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
                model = model.transform(RoundAndClipThresholds())
                
                # Apply reordering optimizations
                model = model.transform(reorder.MoveOpPastFork(["Mul"]))
                model = model.transform(reorder.MoveScalarMulPastMatMul())
                model = model.transform(reorder.MoveScalarLinearPastInvariants())
                
                model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
                model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
                model = model.transform(InferDataTypes(allow_scaledint_dtypes=False))
                model = model.transform(GiveUniqueNodeNames())
                logger.debug("Applied custom streamlining transformations")
            except ImportError:
                logger.warning("Streamlining transformations not available")
            return model
        
        return custom_streamlining_step
    
    def _create_remove_head_step(self, config_params):
        """Generate remove head step function.""" 
        def custom_step_remove_head(model, cfg):
            """Remove model head layers."""
            logger.debug("Remove head step (placeholder)")
            return model
        
        return custom_step_remove_head
    
    def _create_remove_tail_step(self, config_params):
        """Generate remove tail step function."""
        def custom_step_remove_tail(model, cfg):
            """Remove model tail layers."""
            logger.debug("Remove tail step (placeholder)")
            return model
        
        return custom_step_remove_tail
    
    def _create_hardware_inference_step(self, config_params):
        """Generate hardware inference step function."""
        def custom_step_infer_hardware(model, cfg):
            """Infer hardware implementations for operations."""
            try:
                import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
                
                # Standard FINN hardware inference
                model = model.transform(to_hw.InferDuplicateStreamsLayer())
                model = model.transform(to_hw.InferElementwiseBinaryOperation())
                model = model.transform(to_hw.InferThresholdingLayer())
                model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
                logger.debug("Applied hardware inference transformations")
            except ImportError:
                logger.warning("Hardware inference transformations not available")
            return model
        
        return custom_step_infer_hardware
    
    def _create_qonnx_to_finn_step(self):
        """Generate QONNX to FINN conversion step."""
        def custom_step_qonnx_to_finn(model, cfg):
            """Convert QONNX model to FINN format."""
            try:
                from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
                model = model.transform(ConvertQONNXtoFINN())
                logger.debug("Applied QONNX to FINN conversion")
            except ImportError:
                logger.warning("QONNX to FINN conversion not available")
            return model
        
        return custom_step_qonnx_to_finn
    
    def _create_tidy_up_step(self):
        """Generate tidy up step."""
        def custom_step_tidy_up(model, cfg):
            """Tidy up model with shape inference and cleanup."""
            try:
                from qonnx.transformation.infer_shapes import InferShapes
                from qonnx.transformation.infer_datatypes import InferDataTypes
                from qonnx.transformation.fold_constants import FoldConstants
                
                model = model.transform(InferShapes())
                model = model.transform(InferDataTypes())
                model = model.transform(FoldConstants())
                logger.debug("Applied tidy up transformations")
            except ImportError:
                logger.warning("Tidy up transformations not available")
            return model
        
        return custom_step_tidy_up
    
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