"""
Manual QONNX Transform Registration

This module provides manual registration for all QONNX transformations with appropriate
Brainsmith metadata. This is necessary because QONNX transforms lack the rich metadata
required by Brainsmith's stage-based transformation system.

The registration is organized by priority:
1. BERT-required transforms (6) - Essential for BERT pipeline functionality
2. Commonly useful transforms (15) - Generally applicable optimizations
3. Specialized transforms (26) - Domain-specific optimizations

Total: 47 QONNX transforms providing complete coverage of the QONNX ecosystem.
"""

import logging
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass

from .data_models import PluginInfo

logger = logging.getLogger(__name__)

@dataclass
class QONNXTransformInfo:
    """Information about a QONNX transformation for registration."""
    name: str
    class_path: str  # Import path, e.g., "qonnx.transformation.general.RemoveIdentityOps"
    description: str
    stage: str  # Brainsmith stage (cleanup, optimization, etc.)
    priority: str  # "bert_required", "commonly_useful", "specialized"
    dependencies: List[str] = None
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}


# BERT-Required Transforms (8 transforms)
# These are essential for the BERT pipeline to function
BERT_REQUIRED_TRANSFORMS = [
    QONNXTransformInfo(
        name="RemoveIdentityOps",
        class_path="qonnx.transformation.remove.RemoveIdentityOps",
        description="Remove identity operations like Add/Sub with zero or Mul/Div with one",
        stage="cleanup",
        priority="bert_required",
        parameters={"atol": 1e-05}  # Tolerance for zero/one comparison
    ),
    QONNXTransformInfo(
        name="GiveReadableTensorNames", 
        class_path="qonnx.transformation.general.GiveReadableTensorNames",
        description="Give human-readable names to all internal tensors",
        stage="cleanup",
        priority="bert_required",
        dependencies=["GiveUniqueNodeNames"]  # Requires unique node names first
    ),
    QONNXTransformInfo(
        name="GiveUniqueNodeNames",
        class_path="qonnx.transformation.general.GiveUniqueNodeNames", 
        description="Give unique names to each node using enumeration",
        stage="cleanup",
        priority="bert_required",
        parameters={"prefix": ""}  # Optional prefix for node names
    ),
    QONNXTransformInfo(
        name="ConvertDivToMul",
        class_path="qonnx.transformation.general.ConvertDivToMul",
        description="Convert divide by constant nodes to multiply by constant nodes",
        stage="cleanup", 
        priority="bert_required"
    ),
    QONNXTransformInfo(
        name="SortCommutativeInputsInitializerLast",
        class_path="qonnx.transformation.general.SortCommutativeInputsInitializerLast",
        description="Sort inputs of commutative operations to have initializer inputs last",
        stage="cleanup",
        priority="bert_required"
    ),
    QONNXTransformInfo(
        name="InferDataTypes",
        class_path="qonnx.transformation.infer_datatypes.InferDataTypes",
        description="Infer QONNX DataType info for all intermediate/output tensors",
        stage="streamlining",
        priority="bert_required",
        parameters={"allow_scaledint_dtypes": False}  # Default parameter from BERT usage
    ),
]

# Note: FINN transforms (MoveOpPastFork, ConvertToHWLayers) will be handled by FINNAdapter
# We focus only on QONNX transforms here

# Commonly Useful Transforms (15 transforms)
# These are generally applicable and recommended for most workflows
COMMONLY_USEFUL_TRANSFORMS = [
    QONNXTransformInfo(
        name="RemoveStaticGraphInputs",
        class_path="qonnx.transformation.general.RemoveStaticGraphInputs",
        description="Remove any top-level graph inputs that have initializers",
        stage="cleanup",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="RemoveUnusedTensors",
        class_path="qonnx.transformation.general.RemoveUnusedTensors",
        description="Remove unused tensors and their associated initializers/annotations",
        stage="cleanup", 
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="DoubleToSingleFloat",
        class_path="qonnx.transformation.double_to_single_float.DoubleToSingleFloat",
        description="Convert all float64 initializers to float32",
        stage="cleanup",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="RemoveUnusedNodes",
        class_path="qonnx.transformation.remove.RemoveUnusedNodes", 
        description="Remove nodes which do not contribute to any top-level output",
        stage="cleanup",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="InferShapes",
        class_path="qonnx.transformation.infer_shapes.InferShapes",
        description="Ensure every tensor in the graph has a specified shape",
        stage="layout",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="InferDataLayouts",
        class_path="qonnx.transformation.infer_data_layouts.InferDataLayouts",
        description="Infer data layout annotations for tensors",
        stage="layout",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="BatchNormToAffine",
        class_path="qonnx.transformation.batchnorm_to_affine.BatchNormToAffine",
        description="Replace BatchNormalization with Mul-Add affine layers",
        stage="lowering",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="GemmToMatMul", 
        class_path="qonnx.transformation.gemm_to_matmul.GemmToMatMul",
        description="Convert Gemm nodes to MatMul and Add operations",
        stage="lowering",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="SortGraph",
        class_path="qonnx.transformation.general.SortGraph",
        description="Return model with topologically sorted node list",
        stage="utility",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="MovePadAttributeToTensor",
        class_path="qonnx.transformation.general.MovePadAttributeToTensor",
        description="Move padding info from attribute into input tensor for Pad nodes",
        stage="utility",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="ConvertSubToAdd",
        class_path="qonnx.transformation.general.ConvertSubToAdd", 
        description="Convert subtract-a-constant nodes to add-a-constant nodes",
        stage="cleanup",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="GiveUniqueParameterTensors",
        class_path="qonnx.transformation.general.GiveUniqueParameterTensors",
        description="Make every parameter tensor unique to avoid side effects",
        stage="utility",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="FoldConstantsFiltered",
        class_path="qonnx.transformation.fold_constants.FoldConstantsFiltered",
        description="Constant folding with custom filter function",
        stage="optimization",
        priority="commonly_useful",
        parameters={"match_filter_fxn": None}  # Requires filter function
    ),
    QONNXTransformInfo(
        name="QCDQToQuant",
        class_path="qonnx.transformation.qcdq_to_qonnx.QCDQToQuant",
        description="Convert QuantizeLinear+DequantizeLinear chains to QONNX Quant nodes",
        stage="quantization",
        priority="commonly_useful"
    ),
    QONNXTransformInfo(
        name="QuantToQCDQ",
        class_path="qonnx.transformation.qonnx_to_qcdq.QuantToQCDQ",
        description="Convert QONNX Quant nodes to QuantizeLinear+DequantizeLinear",
        stage="quantization", 
        priority="commonly_useful"
    ),
]

# Specialized Transforms (26 transforms)
# These are domain-specific and used for particular optimization scenarios
SPECIALIZED_TRANSFORMS = [
    # ================================
    # Quantization Workflow (4 transforms)
    # ================================
    QONNXTransformInfo(
        name="QuantizeGraph",
        class_path="qonnx.transformation.quantize_graph.QuantizeGraph",
        description="Add Quant nodes at specified locations with given parameters",
        stage="quantization",
        priority="specialized",
        parameters={"quantize_dict": {}}  # Requires quantization specification
    ),
    QONNXTransformInfo(
        name="ExtractQuantScaleZeroPt",
        class_path="qonnx.transformation.extract_quant_scale_zeropt.ExtractQuantScaleZeroPt",
        description="Extract non-identity scale/zero-point from Quant nodes",
        stage="quantization",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="ConvertBipolarMatMulToXnorPopcount",
        class_path="qonnx.transformation.bipolar_to_xnor.ConvertBipolarMatMulToXnorPopcount",
        description="Convert bipolar MatMul to XnorPopcountMatMul for hardware efficiency",
        stage="quantization",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="FoldTransposeIntoQuantInit",
        class_path="qonnx.transformation.quant_constant_folding.FoldTransposeIntoQuantInit",
        description="Fold Transpose operations into quantized initializers",
        stage="quantization",
        priority="specialized"
    ),
    
    # ================================
    # Layout & Dimension Transforms (8 transforms)
    # ================================
    QONNXTransformInfo(
        name="ChangeDataLayoutQuantAvgPool2d",
        class_path="qonnx.transformation.change_datalayout.ChangeDataLayoutQuantAvgPool2d",
        description="Change QuantAvgPool2d data layout from NCHW to NHWC",
        stage="layout",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="ChangeBatchSize",
        class_path="qonnx.transformation.change_batchsize.ChangeBatchSize",
        description="Change batch size for the entire graph",
        stage="layout",
        priority="specialized",
        parameters={"new_batch_size": 1}  # Default batch size
    ),
    QONNXTransformInfo(
        name="Change3DTo4DTensors",
        class_path="qonnx.transformation.change_3d_tensors_to_4d.Change3DTo4DTensors",
        description="Convert 3D tensors to 4D format [N,C,H] -> [N,C,H,1]",
        stage="layout",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="MakeInputChannelsLast",
        class_path="qonnx.transformation.make_input_chanlast.MakeInputChannelsLast",
        description="Convert network input from NCx to NxC layout",
        stage="layout",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="ConvertToChannelsLastAndClean",
        class_path="qonnx.transformation.channels_last.ConvertToChannelsLastAndClean", 
        description="Convert to channels-last layout and perform cleanup",
        stage="layout_optimization",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="InsertChannelsLastDomainsAndTrafos",
        class_path="qonnx.transformation.channels_last.InsertChannelsLastDomainsAndTrafos",
        description="Insert channels-last domains and transpose operations",
        stage="layout_optimization", 
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="RemoveConsecutiveChanFirstAndChanLastTrafos",
        class_path="qonnx.transformation.channels_last.RemoveConsecutiveChanFirstAndChanLastTrafos",
        description="Remove consecutive channel layout transformations",
        stage="layout_optimization",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="MoveChanLastUpstream",
        class_path="qonnx.transformation.channels_last.MoveChanLastUpstream",
        description="Move channels-last transforms upstream in the graph",
        stage="layout_optimization",
        priority="specialized"
    ),
    
    # ================================
    # Advanced Channel Layout Optimization (4 transforms)
    # ================================
    QONNXTransformInfo(
        name="MoveChanFirstDownstream", 
        class_path="qonnx.transformation.channels_last.MoveChanFirstDownstream",
        description="Move channels-first transforms downstream in the graph",
        stage="layout_optimization",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="AbsorbChanFirstIntoMatMul",
        class_path="qonnx.transformation.channels_last.AbsorbChanFirstIntoMatMul",
        description="Absorb channel-first transpose into MatMul weights",
        stage="layout_optimization",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="MoveTransposePastFork",
        class_path="qonnx.transformation.channels_last.MoveTransposePastFork",
        description="Move transpose operations past graph forks",
        stage="layout_optimization",
        priority="specialized"
    ),
    
    # ================================
    # Node Lowering & Conversion (4 transforms)
    # ================================
    QONNXTransformInfo(
        name="LowerConvsToMatMul",
        class_path="qonnx.transformation.lower_convs_to_matmul.LowerConvsToMatMul",
        description="Replace Conv with Im2Col-MatMul pairs",
        stage="lowering",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="ExtractBiasFromConv",
        class_path="qonnx.transformation.extract_conv_bias.ExtractBiasFromConv",
        description="Extract bias from Conv nodes as separate Add operations",
        stage="lowering",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="ResizeConvolutionToDeconvolution",
        class_path="qonnx.transformation.resize_conv_to_deconv.ResizeConvolutionToDeconvolution",
        description="Replace resize+conv with deconvolution using weight convolution",
        stage="lowering",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="SubPixelToDeconvolution",
        class_path="qonnx.transformation.subpixel_to_deconv.SubPixelToDeconvolution",
        description="Replace sub-pixel conv with deconvolution using weight shuffle",
        stage="lowering",
        priority="specialized"
    ),
    
    # ================================
    # Hardware Optimization (1 transform)
    # ================================
    QONNXTransformInfo(
        name="RebalanceIm2Col",
        class_path="qonnx.transformation.rebalance_conv.RebalanceIm2Col",
        description="Reshape Im2Col inputs for optimal channel parallelism",
        stage="optimization",
        priority="specialized"
    ),
    
    # ================================
    # Graph Modification & Utilities (5 transforms)
    # ================================
    QONNXTransformInfo(
        name="GiveRandomTensorNames",
        class_path="qonnx.transformation.general.GiveRandomTensorNames",
        description="Give random tensor names to all tensors",
        stage="utility",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="InsertTopK",
        class_path="qonnx.transformation.insert_topk.InsertTopK",
        description="Add TopK node at network output for classification",
        stage="utility",
        priority="specialized",
        parameters={"k": 5}  # Default top-k value
    ),
    QONNXTransformInfo(
        name="InsertIdentity",
        class_path="qonnx.transformation.insert.InsertIdentity",
        description="Insert Identity node before/after specified tensor",
        stage="utility",
        priority="specialized",
        parameters={"tensor_name": "", "before": True}  # Requires tensor specification
    ),
    QONNXTransformInfo(
        name="InsertIdentityOnAllTopLevelIO",
        class_path="qonnx.transformation.insert.InsertIdentityOnAllTopLevelIO",
        description="Insert Identity nodes on all top-level inputs and outputs",
        stage="utility",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="MergeONNXModels",
        class_path="qonnx.transformation.merge_onnx_models.MergeONNXModels",
        description="Merge two ONNX models end-to-end",
        stage="utility",
        priority="specialized"
    ),
    
    # ================================
    # Intermediate Tensor Management (2 transforms)
    # ================================
    QONNXTransformInfo(
        name="ExposeIntermediateTensorsLambda",
        class_path="qonnx.transformation.expose_intermediate.ExposeIntermediateTensorsLambda",
        description="Expose intermediate tensors as outputs using lambda function",
        stage="utility",
        priority="specialized",
        parameters={"filter_fxn": None}  # Requires filter function
    ),
    QONNXTransformInfo(
        name="ExposeIntermediateTensorsPatternList",
        class_path="qonnx.transformation.expose_intermediate.ExposeIntermediateTensorsPatternList",
        description="Expose tensors matching pattern list as outputs",
        stage="utility",
        priority="specialized",
        parameters={"patterns": []}  # Requires pattern list
    ),
    
    # ================================
    # Partitioning & Deployment (3 transforms)
    # ================================
    QONNXTransformInfo(
        name="PartitionFromLambda",
        class_path="qonnx.transformation.create_generic_partitions.PartitionFromLambda",
        description="Partition graph using lambda function to identify partition boundaries",
        stage="partitioning",
        priority="specialized",
        parameters={"partition_fxn": None}  # Requires partition function
    ),
    QONNXTransformInfo(
        name="PartitionFromDict",
        class_path="qonnx.transformation.create_generic_partitions.PartitionFromDict",
        description="Partition graph using dictionary specification of node assignments",
        stage="partitioning",
        priority="specialized",
        parameters={"partition_dict": {}}  # Requires partition specification
    ),
    QONNXTransformInfo(
        name="ExtendPartition",
        class_path="qonnx.transformation.extend_partition.ExtendPartition",
        description="Extend GenericPartition nodes by inserting sub-graphs",
        stage="partitioning",
        priority="specialized"
    ),
    
    # ================================
    # Model Pruning & Sparsity (4 transforms)
    # ================================
    QONNXTransformInfo(
        name="ApplyMasks",
        class_path="qonnx.transformation.pruning.ApplyMasks",
        description="Apply sparsity masks to tensors in the graph",
        stage="pruning",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="PropagateMasks",
        class_path="qonnx.transformation.pruning.PropagateMasks",
        description="Propagate sparsity masks through the network",
        stage="pruning",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="PruneChannels",
        class_path="qonnx.transformation.pruning.PruneChannels",
        description="High-level channel pruning with dependency handling",
        stage="pruning",
        priority="specialized"
    ),
    QONNXTransformInfo(
        name="RemoveMaskedChannels",
        class_path="qonnx.transformation.pruning.RemoveMaskedChannels",
        description="Remove channels marked by sparsity masks",
        stage="pruning",
        priority="specialized"
    ),
]


def get_all_qonnx_transforms() -> List[QONNXTransformInfo]:
    """Get all QONNX transform information."""
    return BERT_REQUIRED_TRANSFORMS + COMMONLY_USEFUL_TRANSFORMS + SPECIALIZED_TRANSFORMS


def get_transforms_by_priority(priority: str) -> List[QONNXTransformInfo]:
    """Get transforms filtered by priority level."""
    all_transforms = get_all_qonnx_transforms()
    return [t for t in all_transforms if t.priority == priority]


def get_transforms_by_stage(stage: str) -> List[QONNXTransformInfo]:
    """Get transforms filtered by Brainsmith stage."""
    all_transforms = get_all_qonnx_transforms()
    return [t for t in all_transforms if t.stage == stage]


def create_plugin_info_from_qonnx_transform(transform_info: QONNXTransformInfo) -> Optional[PluginInfo]:
    """
    Create a PluginInfo object from QONNXTransformInfo.
    
    Handles safe importing and error cases.
    """
    try:
        # Dynamically import the transform class
        module_path, class_name = transform_info.class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        transform_class = getattr(module, class_name)
        
        # Create metadata dict
        metadata = {
            'discovery_source': 'qonnx_manual_registry',
            'stage': transform_info.stage,
            'priority': transform_info.priority,
            'description': transform_info.description,
            'dependencies': transform_info.dependencies,
            'parameters': transform_info.parameters,
            'class_path': transform_info.class_path
        }
        
        # Create PluginInfo
        plugin_info = PluginInfo(
            name=transform_info.name,
            plugin_class=transform_class,
            plugin_type="transform",
            framework="qonnx",
            metadata=metadata
        )
        
        return plugin_info
        
    except ImportError as e:
        logger.warning(f"Failed to import QONNX transform {transform_info.name}: {e}")
        return None
    except AttributeError as e:
        logger.warning(f"QONNX transform class not found {transform_info.name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to create PluginInfo for {transform_info.name}: {e}")
        return None


def register_qonnx_transforms(manager, priority_filter: Optional[str] = None) -> int:
    """
    Register QONNX transforms with the plugin manager.
    
    Args:
        manager: The plugin manager instance
        priority_filter: Optional priority filter ("bert_required", "commonly_useful", "specialized")
    
    Returns:
        Number of transforms successfully registered
    """
    if priority_filter:
        transforms_to_register = get_transforms_by_priority(priority_filter)
    else:
        transforms_to_register = get_all_qonnx_transforms()
    
    registered_count = 0
    
    for transform_info in transforms_to_register:
        plugin_info = create_plugin_info_from_qonnx_transform(transform_info)
        if plugin_info:
            manager.register_plugin(plugin_info)
            registered_count += 1
            logger.debug(f"Registered QONNX transform: {transform_info.name}")
    
    logger.info(f"Registered {registered_count} QONNX transforms")
    return registered_count


def get_bert_required_transform_names() -> List[str]:
    """Get list of transform names required for BERT pipeline."""
    return [t.name for t in BERT_REQUIRED_TRANSFORMS]


def get_registration_summary() -> Dict[str, Any]:
    """Get summary of available QONNX transforms for registration."""
    all_transforms = get_all_qonnx_transforms()
    
    summary = {
        'total_transforms': len(all_transforms),
        'by_priority': {
            'bert_required': len(get_transforms_by_priority('bert_required')),
            'commonly_useful': len(get_transforms_by_priority('commonly_useful')),
            'specialized': len(get_transforms_by_priority('specialized'))
        },
        'by_stage': {}
    }
    
    # Count by stage
    stages = set(t.stage for t in all_transforms)
    for stage in stages:
        summary['by_stage'][stage] = len(get_transforms_by_stage(stage))
    
    return summary