"""
FINN Transform Discovery for BrainSmith

Discovers FINN transformations and registers them with BrainSmith-specific metadata.
This handles the cross-framework integration while keeping FINN independent.
"""

import inspect
import logging
from typing import Dict, Type, Any

logger = logging.getLogger(__name__)


def discover_finn_transforms() -> int:
    """
    Discover FINN transforms and register with BrainSmith metadata.
    
    Returns:
        Number of transforms discovered and registered
    """
    try:
        # Import to check availability
        from qonnx.transformation.base import Transformation
        from brainsmith.plugin.core import get_registry
    except ImportError as e:
        logger.debug(f"Base transformation not available for FINN discovery: {e}")
        return 0
    
    # Check if FINN is available
    try:
        import finn
    except ImportError:
        logger.debug("FINN not available for transform discovery")
        return 0
    
    registry = get_registry()
    count = 0
    
    # Import FINN transformation modules
    finn_modules = _import_finn_modules()
    
    # Find all Transformation subclasses from FINN modules
    import sys
    for module_name, module in sys.modules.items():
        if module_name.startswith('finn.transformation.'):
            for name, obj in inspect.getmembers(module):
                if (_is_transform_class(obj, Transformation) and 
                    name not in _get_excluded_transforms()):
                    
                    # Get BrainSmith-specific metadata
                    metadata = _get_finn_transform_metadata(name, obj)
                    
                    # Register with BrainSmith
                    registry.register(
                        "transform",
                        f"finn:{name}",
                        obj,
                        framework="finn",
                        **metadata
                    )
                    count += 1
                    logger.debug(f"Registered FINN transform: {name}")
    
    logger.info(f"Discovered {count} FINN transforms")
    return count


def _import_finn_modules() -> list:
    """Import FINN transformation modules."""
    imported = []
    modules_to_import = [
        # Streamline transformations
        'finn.transformation.streamline',
        'finn.transformation.streamline.absorb',
        'finn.transformation.streamline.reorder',
        'finn.transformation.streamline.collapse_repeated',
        'finn.transformation.streamline.round_thresholds',
        'finn.transformation.streamline.sign_to_thres',
        
        # FPGA dataflow transformations
        'finn.transformation.fpgadataflow',
        'finn.transformation.fpgadataflow.annotate_cycles',
        'finn.transformation.fpgadataflow.create_dataflow_partition',
        'finn.transformation.fpgadataflow.create_stitched_ip',
        'finn.transformation.fpgadataflow.hlssynth_ip',
        'finn.transformation.fpgadataflow.insert_dwc',
        'finn.transformation.fpgadataflow.insert_fifo',
        'finn.transformation.fpgadataflow.insert_tlastmarker',
        'finn.transformation.fpgadataflow.make_pynq_driver',
        'finn.transformation.fpgadataflow.make_zynq_proj',
        'finn.transformation.fpgadataflow.minimize_accumulator_width',
        'finn.transformation.fpgadataflow.prepare_cppsim',
        'finn.transformation.fpgadataflow.prepare_ip',
        'finn.transformation.fpgadataflow.prepare_rtlsim',
        'finn.transformation.fpgadataflow.set_exec_mode',
        'finn.transformation.fpgadataflow.set_fifo_depths',
        'finn.transformation.fpgadataflow.specialize_layers',
        'finn.transformation.fpgadataflow.vitis_build',
        
        # QONNX to FINN conversion
        'finn.transformation.qonnx',
        'finn.transformation.qonnx.convert_qonnx_to_finn',
        'finn.transformation.qonnx.qonnx_to_finn_layers',
        
        # Other transformations
        'finn.transformation.move_reshape',
    ]
    
    for module_name in modules_to_import:
        try:
            module = __import__(module_name, fromlist=[''])
            imported.append(module)
        except ImportError as e:
            logger.debug(f"Could not import {module_name}: {e}")
    
    return imported


def _is_transform_class(obj: Any, base_class: Type) -> bool:
    """Check if object is a valid transform class."""
    return (inspect.isclass(obj) and 
            issubclass(obj, base_class) and 
            obj is not base_class)


def _get_excluded_transforms() -> set:
    """Get set of transform names to exclude from registration."""
    return {
        # Abstract base classes
        'Transformation',
        'CodegenTransformation',
        # Test/example classes
        'TestTransformation',
        'ExampleTransformation',
    }


def _get_finn_transform_metadata(name: str, cls: Type) -> Dict[str, Any]:
    """
    Get BrainSmith-specific metadata for a FINN transform.
    
    Maps FINN transforms to BrainSmith's hardware compilation stages.
    """
    
    # BrainSmith-specific metadata for FINN transforms
    FINN_TRANSFORM_METADATA = {
        # === Conversion Stage (QONNX to FINN) ===
        "ConvertQONNXtoFINN": {
            "stage": "conversion", 
            "description": "Convert QONNX model to FINN representation"
        },
        "QONNXToFinnLayers": {
            "stage": "conversion", 
            "description": "Convert QONNX layers to FINN hardware layers"
        },
        
        # === Topology Optimization Stage ===
        "AbsorbSignBiasIntoMultiThreshold": {
            "stage": "topology_opt", 
            "description": "Absorb sign and bias operations into MultiThreshold nodes"
        },
        "AbsorbAddIntoMultiThreshold": {
            "stage": "topology_opt", 
            "description": "Absorb addition operations into MultiThreshold nodes"
        },
        "AbsorbMulIntoMultiThreshold": {
            "stage": "topology_opt", 
            "description": "Absorb multiplication operations into MultiThreshold nodes"
        },
        "AbsorbTransposeIntoMultiThreshold": {
            "stage": "topology_opt", 
            "description": "Absorb transpose operations into MultiThreshold nodes"
        },
        "AbsorbTransposeIntoFlatten": {
            "stage": "topology_opt", 
            "description": "Absorb transpose operations into flatten operations"
        },
        "AbsorbConsecutiveTransposes": {
            "stage": "topology_opt", 
            "description": "Absorb consecutive transpose operations"
        },
        "Absorb1BitMulIntoMatMul": {
            "stage": "topology_opt", 
            "description": "Absorb 1-bit multiplication into matrix multiplication"
        },
        "Absorb1BitMulIntoConv": {
            "stage": "topology_opt", 
            "description": "Absorb 1-bit multiplication into convolution"
        },
        "AbsorbScalarMulAddIntoTopK": {
            "stage": "topology_opt", 
            "description": "Absorb scalar operations into TopK"
        },
        "FactorOutMulSignMagnitude": {
            "stage": "topology_opt", 
            "description": "Factor out multiplication sign and magnitude"
        },
        "ConvertSignToThres": {
            "stage": "topology_opt", 
            "description": "Convert sign operations to threshold operations"
        },
        "RoundAndClipThresholds": {
            "stage": "topology_opt", 
            "description": "Round and clip threshold values"
        },
        
        # === Dataflow Optimization Stage ===
        "MoveLinearPastEltwiseAdd": {
            "stage": "dataflow_opt", 
            "description": "Move linear operations past elementwise addition"
        },
        "MoveLinearPastFork": {
            "stage": "dataflow_opt", 
            "description": "Move linear operations past fork nodes"
        },
        "MoveFlattenPastAffine": {
            "stage": "dataflow_opt", 
            "description": "Move flatten operations past affine layers"
        },
        "MoveFlattenPastTopK": {
            "stage": "dataflow_opt", 
            "description": "Move flatten operations past TopK operations"
        },
        "MoveAddPastMul": {
            "stage": "dataflow_opt", 
            "description": "Move addition past multiplication"
        },
        "MoveAddPastConv": {
            "stage": "dataflow_opt", 
            "description": "Move addition past convolution"
        },
        "MoveAddPastFork": {
            "stage": "dataflow_opt", 
            "description": "Move addition past fork nodes"
        },
        "MoveMulPastMaxPool": {
            "stage": "dataflow_opt", 
            "description": "Move multiplication past max pooling"
        },
        "MoveMulPastDWConv": {
            "stage": "dataflow_opt", 
            "description": "Move multiplication past depthwise convolution"
        },
        "MoveMulPastFork": {
            "stage": "dataflow_opt", 
            "description": "Move multiplication past fork nodes"
        },
        "MoveMaxPoolPastMultiThreshold": {
            "stage": "dataflow_opt", 
            "description": "Move max pooling past multi-threshold operations"
        },
        "MoveScalarAddPastMatMul": {
            "stage": "dataflow_opt", 
            "description": "Move scalar addition past matrix multiplication"
        },
        "MoveScalarMulPastMatMul": {
            "stage": "dataflow_opt", 
            "description": "Move scalar multiplication past matrix multiplication"
        },
        "MoveScalarMulPastConv": {
            "stage": "dataflow_opt", 
            "description": "Move scalar multiplication past convolution"
        },
        "MoveScalarMulPastConvTranspose": {
            "stage": "dataflow_opt", 
            "description": "Move scalar multiplication past transpose convolution"
        },
        "MoveScalarLinearPastInvariants": {
            "stage": "dataflow_opt", 
            "description": "Move scalar linear operations past invariant operations"
        },
        "MoveTransposePastFork": {
            "stage": "dataflow_opt", 
            "description": "Move transpose operations past fork nodes"
        },
        "MoveTransposePastJoinAdd": {
            "stage": "dataflow_opt", 
            "description": "Move transpose past join addition"
        },
        "MoveTransposePastScalarMul": {
            "stage": "dataflow_opt", 
            "description": "Move transpose past scalar multiplication"
        },
        "MoveIdenticalOpPastJoinOp": {
            "stage": "dataflow_opt", 
            "description": "Move identical operations past join operations"
        },
        "CollapseRepeatedAdd": {
            "stage": "dataflow_opt", 
            "description": "Collapse repeated addition operations"
        },
        "CollapseRepeatedMul": {
            "stage": "dataflow_opt", 
            "description": "Collapse repeated multiplication operations"
        },
        "CollapseRepeatedOp": {
            "stage": "dataflow_opt", 
            "description": "Collapse repeated operations"
        },
        
        # === Hardware Stage (FPGA Dataflow) ===
        "SpecializeLayers": {
            "stage": "hardware", 
            "description": "Specialize layers for FPGA implementation"
        },
        "MinimizeAccumulatorWidth": {
            "stage": "hardware", 
            "description": "Minimize accumulator bit width"
        },
        "AnnotateCycles": {
            "stage": "hardware", 
            "description": "Annotate execution cycles for performance analysis"
        },
        "InsertDWC": {
            "stage": "hardware", 
            "description": "Insert data width converters"
        },
        "InsertFIFO": {
            "stage": "hardware", 
            "description": "Insert FIFO buffers for dataflow"
        },
        "InsertTLastMarker": {
            "stage": "hardware", 
            "description": "Insert TLAST markers for AXI streams"
        },
        "SetFIFODepths": {
            "stage": "hardware", 
            "description": "Set FIFO buffer depths"
        },
        "PrepareIP": {
            "stage": "hardware", 
            "description": "Prepare IP blocks for synthesis"
        },
        "HLSSynthIP": {
            "stage": "hardware", 
            "description": "Synthesize HLS IP blocks"
        },
        "PrepareCppSim": {
            "stage": "hardware", 
            "description": "Prepare C++ simulation"
        },
        "PrepareRTLSim": {
            "stage": "hardware", 
            "description": "Prepare RTL simulation"
        },
        "CreateDataflowPartition": {
            "stage": "hardware", 
            "description": "Create dataflow partitions"
        },
        "CreateStitchedIP": {
            "stage": "hardware", 
            "description": "Create stitched IP for complete design"
        },
        "MakeZynqProj": {
            "stage": "hardware", 
            "description": "Create Zynq project"
        },
        "MakePYNQDriver": {
            "stage": "hardware", 
            "description": "Create PYNQ driver"
        },
        "VitisEnv": {
            "stage": "hardware", 
            "description": "Set up Vitis environment"
        },
        "VitisPartition": {
            "stage": "hardware", 
            "description": "Create Vitis partition"
        },
        "VitisLinkStrategy": {
            "stage": "hardware", 
            "description": "Set Vitis linking strategy"
        },
        "VitisBuild": {
            "stage": "hardware", 
            "description": "Execute Vitis build flow"
        },
        "SetExecMode": {
            "stage": "hardware", 
            "description": "Set execution mode for layers"
        },
        
        # === Other/Utility ===
        "MoveReshape": {
            "stage": "layout", 
            "description": "Move reshape operations"
        },
    }
    
    # Get predefined metadata or create default
    metadata = FINN_TRANSFORM_METADATA.get(name, {
        "stage": "general",  # Default stage for unknown transforms
        "description": _extract_description_from_docstring(cls)
    })
    
    # Add common metadata
    metadata.update({
        "author": "finn-team",
        "category": "finn_transform", 
        "tags": [metadata.get("stage", "general"), "finn", "hardware"]
    })
    
    return metadata


def _extract_description_from_docstring(cls: Type) -> str:
    """Extract a brief description from class docstring."""
    if cls.__doc__:
        # Take first line of docstring, strip quotes
        first_line = cls.__doc__.split('\n')[0].strip()
        return first_line.strip('"\'')
    return f"FINN {cls.__name__} transformation"