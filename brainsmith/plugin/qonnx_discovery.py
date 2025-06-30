"""
QONNX Transform Discovery for BrainSmith

Discovers QONNX transformations and registers them with BrainSmith-specific metadata.
QONNX transforms remain simple python classes, BrainSmith adds the hardware compilation metadata.
"""

import inspect
import logging
from typing import Dict, Type, Any

logger = logging.getLogger(__name__)


def discover_qonnx_transforms() -> int:
    """
    Discover QONNX transforms and register with BrainSmith metadata.
    
    Returns:
        Number of transforms discovered and registered
    """
    try:
        # Import to check availability
        from qonnx.transformation.base import Transformation
        from brainsmith.plugin.core import get_registry
    except ImportError as e:
        logger.debug(f"QONNX not available for transform discovery: {e}")
        return 0
    
    registry = get_registry()
    count = 0
    
    # Import QONNX transformation modules to trigger any existing registration
    qonnx_modules = _import_qonnx_modules()
    
    # Find all Transformation subclasses from imported modules
    import sys
    for module_name, module in sys.modules.items():
        if module_name.startswith('qonnx.transformation.'):
            for name, obj in inspect.getmembers(module):
                if (_is_transform_class(obj, Transformation) and 
                    name not in _get_excluded_transforms()):
                    
                    # Get BrainSmith-specific metadata
                    metadata = _get_transform_metadata(name, obj)
                    
                    # Register with BrainSmith
                    registry.register(
                        "transform",
                        f"qonnx:{name}",
                        obj,
                        framework="qonnx",
                        **metadata
                    )
                    count += 1
                    logger.debug(f"Registered QONNX transform: {name}")
    
    logger.info(f"Discovered {count} QONNX transforms")
    return count


def _import_qonnx_modules() -> list:
    """Import QONNX transformation modules."""
    imported = []
    modules_to_import = [
        'qonnx.transformation.general',
        'qonnx.transformation.fold_constants', 
        'qonnx.transformation.infer_shapes',
        'qonnx.transformation.remove',
        'qonnx.transformation.channels_last',
        'qonnx.transformation.batchnorm_to_affine',
        'qonnx.transformation.change_datalayout',
        'qonnx.transformation.extract_conv_bias',
        'qonnx.transformation.gemm_to_matmul',
        'qonnx.transformation.infer_datatypes',
        'qonnx.transformation.insert',
        'qonnx.transformation.lower_convs_to_matmul',
        'qonnx.transformation.quantize_graph',
        'qonnx.transformation.qonnx_to_qcdq',
        'qonnx.transformation.qcdq_to_qonnx',
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
        # Test/example classes that shouldn't be registered
        'TestTransformation',
        'ExampleTransformation',
    }


def _get_transform_metadata(name: str, cls: Type) -> Dict[str, Any]:
    """
    Get BrainSmith-specific metadata for a QONNX transform.
    
    This is where we add hardware compilation context that QONNX doesn't need.
    """
    
    # BrainSmith-specific metadata for QONNX transforms
    # Maps transform names to BrainSmith stages and descriptions
    QONNX_TRANSFORM_METADATA = {
        # === Cleanup Stage ===
        "GiveUniqueNodeNames": {
            "stage": "cleanup", 
            "description": "Assign unique names to all nodes in the graph"
        },
        "GiveReadableTensorNames": {
            "stage": "cleanup", 
            "description": "Assign readable names to tensors"
        },
        "GiveRandomTensorNames": {
            "stage": "cleanup", 
            "description": "Assign random names to tensors"
        },
        "GiveUniqueParameterTensors": {
            "stage": "cleanup", 
            "description": "Ensure parameter tensors have unique names"
        },
        "SortGraph": {
            "stage": "cleanup", 
            "description": "Topologically sort the graph"
        },
        "RemoveUnusedTensors": {
            "stage": "cleanup", 
            "description": "Remove tensors that are not used"
        },
        "RemoveStaticGraphInputs": {
            "stage": "cleanup", 
            "description": "Remove static/constant graph inputs"
        },
        "RemoveIdentityOps": {
            "stage": "cleanup", 
            "description": "Remove identity/no-op operations"
        },
        "InferShapes": {
            "stage": "cleanup", 
            "description": "Infer and set tensor shapes"
        },
        "InferDataTypes": {
            "stage": "cleanup", 
            "description": "Infer and set tensor data types"
        },
        
        # === Optimization Stage ===
        "FoldConstants": {
            "stage": "optimization", 
            "description": "Fold constant expressions into initializers"
        },
        "ConvertDivToMul": {
            "stage": "optimization", 
            "description": "Convert division to multiplication with reciprocal"
        },
        "ConvertSubToAdd": {
            "stage": "optimization", 
            "description": "Convert subtraction to addition with negation"
        },
        "CollapseRepeatedAdd": {
            "stage": "optimization", 
            "description": "Collapse repeated addition operations"
        },
        "CollapseRepeatedMul": {
            "stage": "optimization", 
            "description": "Collapse repeated multiplication operations"
        },
        
        # === Layout/Conversion Stage ===
        "MakeInputChanlast": {
            "stage": "layout", 
            "description": "Convert input to channels-last format"
        },
        "ConvertToChannelsLast": {
            "stage": "layout", 
            "description": "Convert data layout to channels-last"
        },
        "ChangeDataLayoutQuantizedWeights": {
            "stage": "layout", 
            "description": "Change data layout for quantized weights"
        },
        "MovePadAttributeToTensor": {
            "stage": "layout", 
            "description": "Move padding attributes to tensor inputs"
        },
        
        # === Conversion Stage ===
        "GemmToMatMul": {
            "stage": "conversion", 
            "description": "Convert GEMM operations to MatMul"
        },
        "ConvertQONNXtoQCDQ": {
            "stage": "conversion", 
            "description": "Convert QONNX quantization to QCDQ format"
        },
        "ConvertQCDQtoQONNX": {
            "stage": "conversion", 
            "description": "Convert QCDQ quantization to QONNX format"
        },
        "BatchNormToAffine": {
            "stage": "conversion", 
            "description": "Convert BatchNorm to affine transformation"
        },
        
        # === Quantization Stage ===
        "QuantizeGraph": {
            "stage": "quantization", 
            "description": "Apply quantization to graph"
        },
        "ExtractQuantScaleZeropt": {
            "stage": "quantization", 
            "description": "Extract quantization scale and zero point"
        },
        
        # === Hardware Stage ===
        "LowerConvsToMatMul": {
            "stage": "hardware", 
            "description": "Lower convolutions to matrix multiplication"
        },
        "ExtractConvBias": {
            "stage": "hardware", 
            "description": "Extract bias from convolution operations"
        },
    }
    
    # Get predefined metadata or create default
    metadata = QONNX_TRANSFORM_METADATA.get(name, {
        "stage": "general",  # Default stage for unknown transforms
        "description": _extract_description_from_docstring(cls)
    })
    
    # Add common metadata
    metadata.update({
        "author": "qonnx-team",
        "category": "qonnx_transform",
        "tags": [metadata.get("stage", "general"), "qonnx"]
    })
    
    return metadata


def _extract_description_from_docstring(cls: Type) -> str:
    """Extract a brief description from class docstring."""
    if cls.__doc__:
        # Take first line of docstring, strip quotes
        first_line = cls.__doc__.split('\n')[0].strip()
        return first_line.strip('"\'')
    return f"QONNX {cls.__name__} transformation"