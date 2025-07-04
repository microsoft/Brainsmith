"""
Framework Adapters - Perfect Code Implementation

Direct registration of external framework transforms.
No wrapper classes needed - register transforms directly with the registry.
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# Transform registration data - (name, module_path, stage)
QONNX_TRANSFORMS = [
    ('BatchNormToAffine', 'qonnx.transformation.batchnorm_to_affine.BatchNormToAffine', 'topology_opt'),
    ('Change3DTo4DTensors', 'qonnx.transformation.change_3d_tensors_to_4d.Change3DTo4DTensors', 'cleanup'),
    ('ChangeBatchSize', 'qonnx.transformation.change_batchsize.ChangeBatchSize', 'cleanup'),
    ('ChangeDataLayoutQuantAvgPool2d', 'qonnx.transformation.change_datalayout.ChangeDataLayoutQuantAvgPool2d', 'cleanup'),
    ('ConvertBipolarMatMulToXnorPopcount', 'qonnx.transformation.bipolar_to_xnor.ConvertBipolarMatMulToXnorPopcount', 'cleanup'),
    ('DoubleToSingleFloat', 'qonnx.transformation.double_to_single_float.DoubleToSingleFloat', 'cleanup'),
    ('ExtractQuantScaleZeroPt', 'qonnx.transformation.extract_quant_scale_zeropt.ExtractQuantScaleZeroPt', 'cleanup'),
    ('QCDQToQuant', 'qonnx.transformation.qcdq_to_qonnx.QCDQToQuant', 'cleanup'),
    ('QuantToQCDQ', 'qonnx.transformation.qonnx_to_qcdq.QuantToQCDQ', 'cleanup'),
    ('FoldTransposeIntoQuantInit', 'qonnx.transformation.quant_constant_folding.FoldTransposeIntoQuantInit', 'cleanup'),
    ('QuantizeGraph', 'qonnx.transformation.quantize_graph.QuantizeGraph', 'cleanup'),
    ('ConvertToChannelsLastAndClean', 'qonnx.transformation.channels_last.ConvertToChannelsLastAndClean', 'cleanup'),
    ('InsertChannelsLastDomainsAndTrafos', 'qonnx.transformation.channels_last.InsertChannelsLastDomainsAndTrafos', 'cleanup'),
    ('RemoveConsecutiveChanFirstAndChanLastTrafos', 'qonnx.transformation.channels_last.RemoveConsecutiveChanFirstAndChanLastTrafos', 'cleanup'),
    ('MoveChanLastUpstream', 'qonnx.transformation.channels_last.MoveChanLastUpstream', 'topology_opt'),
    ('MoveChanFirstDownstream', 'qonnx.transformation.channels_last.MoveChanFirstDownstream', 'topology_opt'),
    ('AbsorbChanFirstIntoMatMul', 'qonnx.transformation.channels_last.AbsorbChanFirstIntoMatMul', 'topology_opt'),
    ('MoveOpPastFork', 'qonnx.transformation.channels_last.MoveOpPastFork', 'topology_opt'),
    ('MakeInputChannelsLast', 'qonnx.transformation.make_input_chanlast.MakeInputChannelsLast', 'cleanup'),
    ('ExtractBiasFromConv', 'qonnx.transformation.extract_conv_bias.ExtractBiasFromConv', 'cleanup'),
    ('GemmToMatMul', 'qonnx.transformation.gemm_to_matmul.GemmToMatMul', 'topology_opt'),
    ('LowerConvsToMatMul', 'qonnx.transformation.lower_convs_to_matmul.LowerConvsToMatMul', 'topology_opt'),
    ('RebalanceIm2Col', 'qonnx.transformation.rebalance_conv.RebalanceIm2Col', 'topology_opt'),
    ('ResizeConvolutionToDeconvolution', 'qonnx.transformation.resize_conv_to_deconv.ResizeConvolutionToDeconvolution', 'topology_opt'),
    ('SubPixelToDeconvolution', 'qonnx.transformation.subpixel_to_deconv.SubPixelToDeconvolution', 'topology_opt'),
    ('PartitionFromLambda', 'qonnx.transformation.create_generic_partitions.PartitionFromLambda', 'dataflow_opt'),
    ('PartitionFromDict', 'qonnx.transformation.create_generic_partitions.PartitionFromDict', 'dataflow_opt'),
    ('ExtendPartition', 'qonnx.transformation.extend_partition.ExtendPartition', 'dataflow_opt'),
    ('ExposeIntermediateTensorsLambda', 'qonnx.transformation.expose_intermediate.ExposeIntermediateTensorsLambda', 'cleanup'),
    ('MergeONNXModels', 'qonnx.transformation.merge_onnx_models.MergeONNXModels', 'cleanup'),
    ('FoldConstantsFiltered', 'qonnx.transformation.fold_constants.FoldConstantsFiltered', 'cleanup'),
    ('FoldConstants', 'qonnx.transformation.fold_constants.FoldConstants', 'cleanup'),
    ('InferDataLayouts', 'qonnx.transformation.infer_data_layouts.InferDataLayouts', 'topology_opt'),
    ('InferDataTypes', 'qonnx.transformation.infer_datatypes.InferDataTypes', 'topology_opt'),
    ('InferShapes', 'qonnx.transformation.infer_shapes.InferShapes', 'topology_opt'),
    ('InsertTopK', 'qonnx.transformation.insert_topk.InsertTopK', 'topology_opt'),
    ('RemoveUnusedTensors', 'qonnx.transformation.remove_unused_tensors.RemoveUnusedTensors', 'cleanup'),
    ('RenameTensors', 'qonnx.transformation.rename.RenameTensors', 'cleanup'),
    ('GiveReadableTensorNames', 'qonnx.transformation.rename.GiveReadableTensorNames', 'cleanup'),
    ('GiveUniqueNodeNames', 'qonnx.transformation.rename.GiveUniqueNodeNames', 'cleanup'),
    ('GiveRandomTensorNames', 'qonnx.transformation.rename.GiveRandomTensorNames', 'cleanup'),
    ('InlineNodeAttributesAsConstants', 'qonnx.transformation.inline_node_attributes_into_graph.InlineNodeAttributesAsConstants', 'cleanup'),
    ('RemoveStaticGraphInputs', 'qonnx.transformation.remove_static_graph_inputs.RemoveStaticGraphInputs', 'cleanup'),
    ('FoldQuantNodesIntoWeights', 'qonnx.transformation.fold_quant_nodes_into_weights.FoldQuantNodesIntoWeights', 'cleanup'),
    ('RemoveIdentityOps', 'qonnx.transformation.remove_identity_ops.RemoveIdentityOps', 'cleanup'),
    ('ConvertSubToAdd', 'qonnx.transformation.convert_sub_to_add.ConvertSubToAdd', 'topology_opt'),
    ('ConvertDivToMul', 'qonnx.transformation.convert_div_to_mul.ConvertDivToMul', 'topology_opt'),
]

FINN_TRANSFORMS = [
    ('RemoveCNVtoFCFlatten', 'finn.transformation.move_reshape.RemoveCNVtoFCFlatten', 'topology_opt'),
    ('ConvertQONNXtoFINN', 'finn.transformation.qonnx.convert_qonnx_to_finn.ConvertQONNXtoFINN', 'cleanup'),
    ('FoldQuantWeights', 'finn.transformation.qonnx.fold_quant_weights.FoldQuantWeights', 'cleanup'),
    ('AvgPoolAndTruncToQuantAvgPool', 'finn.transformation.qonnx.infer_quant_avg_pool_2d.AvgPoolAndTruncToQuantAvgPool', 'cleanup'),
    ('ConvertQuantActToMultiThreshold', 'finn.transformation.qonnx.quant_act_to_multithreshold.ConvertQuantActToMultiThreshold', 'cleanup'),
    ('Streamline', 'finn.transformation.streamline.Streamline', 'topology_opt'),
    ('AbsorbSignBiasIntoMultiThreshold', 'finn.transformation.streamline.absorb.AbsorbSignBiasIntoMultiThreshold', 'topology_opt'),
    ('AbsorbAddIntoMultiThreshold', 'finn.transformation.streamline.absorb.AbsorbAddIntoMultiThreshold', 'topology_opt'),
    ('AbsorbMulIntoMultiThreshold', 'finn.transformation.streamline.absorb.AbsorbMulIntoMultiThreshold', 'topology_opt'),
    ('FactorOutMulSignMagnitude', 'finn.transformation.streamline.absorb.FactorOutMulSignMagnitude', 'topology_opt'),
    ('Absorb1BitMulIntoMatMul', 'finn.transformation.streamline.absorb.Absorb1BitMulIntoMatMul', 'topology_opt'),
    ('Absorb1BitMulIntoConv', 'finn.transformation.streamline.absorb.Absorb1BitMulIntoConv', 'topology_opt'),
    ('AbsorbTransposeIntoMultiThreshold', 'finn.transformation.streamline.absorb.AbsorbTransposeIntoMultiThreshold', 'topology_opt'),
    ('AbsorbTransposeIntoFlatten', 'finn.transformation.streamline.absorb.AbsorbTransposeIntoFlatten', 'topology_opt'),
    ('AbsorbScalarMulAddIntoTopK', 'finn.transformation.streamline.absorb.AbsorbScalarMulAddIntoTopK', 'topology_opt'),
    ('AbsorbConsecutiveTransposes', 'finn.transformation.streamline.absorb.AbsorbConsecutiveTransposes', 'topology_opt'),
    ('AbsorbTransposeIntoResize', 'finn.transformation.streamline.absorb.AbsorbTransposeIntoResize', 'topology_opt'),
    ('CollapseRepeatedOp', 'finn.transformation.streamline.collapse_repeated.CollapseRepeatedOp', 'topology_opt'),
    ('MoveAddPastMul', 'finn.transformation.streamline.reorder.MoveAddPastMul', 'topology_opt'),
    ('MoveScalarMulPastMatMul', 'finn.transformation.streamline.reorder.MoveScalarMulPastMatMul', 'topology_opt'),
    ('MoveScalarAddPastMatMul', 'finn.transformation.streamline.reorder.MoveScalarAddPastMatMul', 'topology_opt'),
    ('MoveAddPastConv', 'finn.transformation.streamline.reorder.MoveAddPastConv', 'topology_opt'),
    ('MoveScalarMulPastConv', 'finn.transformation.streamline.reorder.MoveScalarMulPastConv', 'topology_opt'),
    ('MoveScalarMulPastConvTranspose', 'finn.transformation.streamline.reorder.MoveScalarMulPastConvTranspose', 'topology_opt'),
    ('MoveMulPastDWConv', 'finn.transformation.streamline.reorder.MoveMulPastDWConv', 'topology_opt'),
    ('MoveMulPastMaxPool', 'finn.transformation.streamline.reorder.MoveMulPastMaxPool', 'topology_opt'),
    ('MoveLinearPastEltwiseAdd', 'finn.transformation.streamline.reorder.MoveLinearPastEltwiseAdd', 'topology_opt'),
    ('MoveScalarLinearPastInvariants', 'finn.transformation.streamline.reorder.MoveScalarLinearPastInvariants', 'topology_opt'),
    ('MakeMaxPoolNHWC', 'finn.transformation.streamline.reorder.MakeMaxPoolNHWC', 'topology_opt'),
    ('MakeScaleResizeNHWC', 'finn.transformation.streamline.reorder.MakeScaleResizeNHWC', 'topology_opt'),
    ('MoveOpPastFork', 'finn.transformation.streamline.reorder.MoveOpPastFork', 'topology_opt'),
    ('MoveMaxPoolPastMultiThreshold', 'finn.transformation.streamline.reorder.MoveMaxPoolPastMultiThreshold', 'topology_opt'),
    ('MoveFlattenPastTopK', 'finn.transformation.streamline.reorder.MoveFlattenPastTopK', 'topology_opt'),
    ('MoveFlattenPastAffine', 'finn.transformation.streamline.reorder.MoveFlattenPastAffine', 'topology_opt'),
    ('MoveTransposePastScalarMul', 'finn.transformation.streamline.reorder.MoveTransposePastScalarMul', 'topology_opt'),
    ('MoveTransposePastJoinAdd', 'finn.transformation.streamline.reorder.MoveTransposePastJoinAdd', 'topology_opt'),
    ('MoveTransposePastFork', 'finn.transformation.streamline.reorder.MoveTransposePastFork', 'topology_opt'),
    ('MoveLinearPastFork', 'finn.transformation.streamline.reorder.MoveLinearPastFork', 'topology_opt'),
    ('RoundAndClipThresholds', 'finn.transformation.streamline.round_thresholds.RoundAndClipThresholds', 'topology_opt'),
    ('AdjustBatchNormAxis', 'finn.transformation.streamline.batch_norm.AdjustBatchNormAxis', 'topology_opt'),
    ('ConvertToHLSLayers', 'finn.transformation.fpgadataflow.convert_to_hls_layers.ConvertToHLSLayers', 'kernel_opt'),
    ('MinimizeAccumulatorWidth', 'finn.transformation.fpgadataflow.minimize_accumulator_width.MinimizeAccumulatorWidth', 'dataflow_opt'),
    ('MinimizeWeightBitWidth', 'finn.transformation.fpgadataflow.minimize_weight_bit_width.MinimizeWeightBitWidth', 'dataflow_opt'),
    ('InsertFIFO', 'finn.transformation.fpgadataflow.insert_fifo.InsertFIFO', 'dataflow_opt'),
    ('InsertDWC', 'finn.transformation.fpgadataflow.insert_dwc.InsertDWC', 'dataflow_opt'),
    ('InsertTLastMarker', 'finn.transformation.fpgadataflow.insert_tlastmarker.InsertTLastMarker', 'dataflow_opt'),
    ('RemoveUnusedTensors', 'finn.transformation.fpgadataflow.remove_unused_tensors.RemoveUnusedTensors', 'cleanup'),
    ('GiveUniqueNodeNames', 'finn.transformation.fpgadataflow.floorplan.Floorplan', 'dataflow_opt'),
    ('SpecializeLayers', 'finn.transformation.fpgadataflow.specialize_layers.SpecializeLayers', 'dataflow_opt'),
    ('AddTernaryDirect', 'finn.transformation.fpgadataflow.addtndr.AddTernaryDirect', 'dataflow_opt'),
    ('VitisOptimizer', 'finn.transformation.fpgadataflow.vitis.VitisOptimizer', 'dataflow_opt'),
    ('CollapseBiasIntoConv', 'finn.transformation.streamline.collapse_bias.CollapseBiasIntoConv', 'topology_opt'),
    ('ApplyConfig', 'finn.transformation.fpgadataflow.apply_config.ApplyConfig', 'dataflow_opt'),
    ('MakeBatchSizeOne', 'finn.transformation.fpgadataflow.make_batch_size_one.MakeBatchSizeOne', 'dataflow_opt'),
    ('InferConvActivations', 'finn.transformation.fpgadataflow.infer_conv_activations.InferConvActivations', 'dataflow_opt'),
    ('InferFCActivations', 'finn.transformation.fpgadataflow.infer_fc_activations.InferFCActivations', 'dataflow_opt'),
    ('InferGlobalAccPoolActivations', 'finn.transformation.fpgadataflow.infer_global_accpool_activations.InferGlobalAccPoolActivations', 'dataflow_opt'),
    ('SetExecMode', 'finn.transformation.fpgadataflow.set_exec_mode.SetExecMode', 'dataflow_opt'),
    ('SetFolding', 'finn.transformation.fpgadataflow.set_folding.SetFolding', 'dataflow_opt'),
    ('GiveUniqueDataflowNodeNames', 'finn.transformation.fpgadataflow.set_fifo_depths.GiveUniqueDataflowNodeNames', 'dataflow_opt'),
    ('InsertAndSetFIFODepths', 'finn.transformation.fpgadataflow.set_fifo_depths.InsertAndSetFIFODepths', 'dataflow_opt'),
    ('MoveLinearPastJoinOp', 'finn.transformation.move_reshape.MoveLinearPastJoinOp', 'topology_opt'),
    ('SplitLargeFIFOs', 'finn.transformation.fpgadataflow.split_large_fifos.SplitLargeFIFOs', 'dataflow_opt'),
    ('AddFork', 'finn.transformation.fpgadataflow.add_fork.AddFork', 'dataflow_opt'),
    ('StreamingDataflowMaker', 'finn.transformation.fpgadataflow.make_streamingdataflow.StreamingDataflowMaker', 'dataflow_opt'),
]

# Finn kernels and backends registration data
FINN_KERNELS = [
    ('MultiThreshold', 'finn.custom_op.fpgadataflow.thresholding.MultiThreshold'),
    ('MVAU', 'finn.custom_op.fpgadataflow.mvau.MVAU'),
    ('StreamingFCLayer', 'finn.custom_op.fpgadataflow.streamingfclayer.StreamingFCLayer'),
    ('Conv', 'finn.custom_op.fpgadataflow.conv.Conv'),
    ('Conv_RTL', 'finn.custom_op.fpgadataflow.conv_rtl.Conv_RTL'),
    ('MatrixVectorUnit', 'finn.custom_op.fpgadataflow.matrixvectorunit.MatrixVectorUnit'),
]

FINN_KERNEL_INFERENCES = [
    ('InferQuantizedMatrixVectorActivation', 'finn.transformation.fpgadataflow.infer_quantized_matrixvectoractivation.InferQuantizedMatrixVectorActivation', 'MatrixVectorUnit'),
    ('InferConvolutionInputGenerator', 'finn.transformation.fpgadataflow.infer_convolutioninputgenerator.InferConvolutionInputGenerator', 'ConvolutionInputGenerator'),
]


def _register_transforms(transforms: List[Tuple[str, str, str]], framework: str) -> int:
    """
    Register transforms directly with the registry.
    
    Perfect Code approach: Direct registration without wrapper classes.
    """
    from .registry import get_registry
    
    registry = get_registry()
    registered_count = 0
    
    for name, class_path, stage in transforms:
        try:
            # Dynamic import
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            transform_class = getattr(module, class_name)
            
            # Register directly - no wrapper needed
            registry.register_transform(
                name, 
                transform_class,
                stage=stage, 
                framework=framework,
                original_class=class_path,
                description=f"{framework.upper()} {name} transform"
            )
            
            registered_count += 1
            logger.debug(f"Registered {framework} transform: {name}")
            
        except ImportError as e:
            logger.debug(f"{framework} transform {name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register {framework} transform {name}: {e}")
    
    logger.info(f"Registered {registered_count} {framework} transforms")
    return registered_count


def initialize_framework_integrations() -> Dict[str, int]:
    """
    Initialize all framework integrations.
    
    Returns counts of registered components by type.
    """
    from .registry import get_registry
    
    results = {
        'qonnx_transforms': 0,
        'finn_transforms': 0,
        'finn_kernels': 0,
        'finn_kernel_inferences': 0
    }
    
    # Register QONNX transforms
    try:
        results['qonnx_transforms'] = _register_transforms(QONNX_TRANSFORMS, 'qonnx')
    except Exception as e:
        logger.warning(f"Failed to register QONNX transforms: {e}")
    
    # Register FINN transforms
    try:
        results['finn_transforms'] = _register_transforms(FINN_TRANSFORMS, 'finn')
    except Exception as e:
        logger.warning(f"Failed to register FINN transforms: {e}")
    
    # Register FINN kernels
    registry = get_registry()
    for name, class_path in FINN_KERNELS:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            kernel_class = getattr(module, class_name)
            
            registry.register_kernel(
                name,
                kernel_class,
                framework='finn',
                original_class=class_path
            )
            results['finn_kernels'] += 1
            logger.debug(f"Registered FINN kernel: {name}")
            
        except Exception as e:
            logger.warning(f"Failed to register FINN kernel {name}: {e}")
    
    # Register FINN kernel inferences
    for name, class_path, kernel in FINN_KERNEL_INFERENCES:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            inference_class = getattr(module, class_name)
            
            registry.register_kernel_inference(
                name,
                inference_class,
                kernel=kernel,
                framework='finn',
                original_class=class_path
            )
            results['finn_kernel_inferences'] += 1
            logger.debug(f"Registered FINN kernel inference: {name}")
            
        except Exception as e:
            logger.warning(f"Failed to register FINN kernel inference {name}: {e}")
    
    logger.info(f"Framework initialization complete:")
    logger.info(f"  - QONNX transforms: {results['qonnx_transforms']}")
    logger.info(f"  - FINN transforms: {results['finn_transforms']}")
    logger.info(f"  - FINN kernels: {results['finn_kernels']}")
    logger.info(f"  - FINN kernel inferences: {results['finn_kernel_inferences']}")
    
    return results


# Initialize frameworks on import for immediate availability
# This is a one-time operation that populates the registry
try:
    initialize_framework_integrations()
except Exception as e:
    logger.warning(f"Framework initialization failed: {e}")