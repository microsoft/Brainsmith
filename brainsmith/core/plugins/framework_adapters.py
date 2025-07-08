"""
Framework Adapters - Perfect Code Implementation

Direct registration of external framework transforms.
No wrapper classes needed - register transforms directly with the registry.
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Base paths
QT = 'qonnx.transformation'
FT = 'finn.transformation'
FK = 'finn.custom_op.fpgadataflow'

# Transform registration data - (name, module_path, stage)
QONNX_TRANSFORMS = [
    # Batch/Tensor Operations
    ('BatchNormToAffine', f'{QT}.batchnorm_to_affine.BatchNormToAffine', 'topology_opt'),
    ('Change3DTo4DTensors', f'{QT}.change_3d_tensors_to_4d.Change3DTo4DTensors', 'cleanup'),
    ('ChangeBatchSize', f'{QT}.change_batchsize.ChangeBatchSize', 'cleanup'),
    ('ChangeDataLayoutQuantAvgPool2d', f'{QT}.change_datalayout.ChangeDataLayoutQuantAvgPool2d', 'cleanup'),
    ('DoubleToSingleFloat', f'{QT}.double_to_single_float.DoubleToSingleFloat', 'cleanup'),
    
    # Quantization Operations
    ('ConvertBipolarMatMulToXnorPopcount', f'{QT}.bipolar_to_xnor.ConvertBipolarMatMulToXnorPopcount', 'cleanup'),
    ('ExtractQuantScaleZeroPt', f'{QT}.extract_quant_scale_zeropt.ExtractQuantScaleZeroPt', 'cleanup'),
    ('QCDQToQuant', f'{QT}.qcdq_to_qonnx.QCDQToQuant', 'cleanup'),
    ('QuantToQCDQ', f'{QT}.qonnx_to_qcdq.QuantToQCDQ', 'cleanup'),
    ('FoldTransposeIntoQuantInit', f'{QT}.quant_constant_folding.FoldTransposeIntoQuantInit', 'cleanup'),
    ('QuantizeGraph', f'{QT}.quantize_graph.QuantizeGraph', 'cleanup'),
    
    # Channel Operations
    ('ConvertToChannelsLastAndClean', f'{QT}.channels_last.ConvertToChannelsLastAndClean', 'cleanup'),
    ('InsertChannelsLastDomainsAndTrafos', f'{QT}.channels_last.InsertChannelsLastDomainsAndTrafos', 'cleanup'),
    ('RemoveConsecutiveChanFirstAndChanLastTrafos', f'{QT}.channels_last.RemoveConsecutiveChanFirstAndChanLastTrafos', 'cleanup'),
    ('MoveChanLastUpstream', f'{QT}.channels_last.MoveChanLastUpstream', 'topology_opt'),
    ('MoveChanFirstDownstream', f'{QT}.channels_last.MoveChanFirstDownstream', 'topology_opt'),
    ('AbsorbChanFirstIntoMatMul', f'{QT}.channels_last.AbsorbChanFirstIntoMatMul', 'topology_opt'),
    ('MoveOpPastFork', f'{QT}.channels_last.MoveOpPastFork', 'topology_opt'),
    ('MoveAddPastFork', f'{QT}.channels_last.MoveAddPastFork', 'topology_opt'),
    ('MoveLinearPastFork', f'{QT}.channels_last.MoveLinearPastFork', 'topology_opt'),
    ('MoveMulPastFork', f'{QT}.channels_last.MoveMulPastFork', 'topology_opt'),
    ('MoveTransposePastFork', f'{QT}.channels_last.MoveTransposePastFork', 'topology_opt'),
    ('MakeInputChannelsLast', f'{QT}.make_input_chanlast.MakeInputChannelsLast', 'cleanup'),
    
    # Graph Transformations
    ('ExtractBiasFromConv', f'{QT}.extract_conv_bias.ExtractBiasFromConv', 'cleanup'),
    ('GemmToMatMul', f'{QT}.gemm_to_matmul.GemmToMatMul', 'topology_opt'),
    ('LowerConvsToMatMul', f'{QT}.lower_convs_to_matmul.LowerConvsToMatMul', 'topology_opt'),
    ('RebalanceIm2Col', f'{QT}.rebalance_conv.RebalanceIm2Col', 'topology_opt'),
    ('ResizeConvolutionToDeconvolution', f'{QT}.resize_conv_to_deconv.ResizeConvolutionToDeconvolution', 'topology_opt'),
    ('SubPixelToDeconvolution', f'{QT}.subpixel_to_deconv.SubPixelToDeconvolution', 'topology_opt'),
    
    # Partitioning Operations
    ('PartitionFromLambda', f'{QT}.create_generic_partitions.PartitionFromLambda', 'dataflow_opt'),
    ('PartitionFromDict', f'{QT}.create_generic_partitions.PartitionFromDict', 'dataflow_opt'),
    ('ExtendPartition', f'{QT}.extend_partition.ExtendPartition', 'dataflow_opt'),
    
    # Utility Operations
    ('ExposeIntermediateTensorsLambda', f'{QT}.expose_intermediate.ExposeIntermediateTensorsLambda', 'cleanup'),
    ('MergeONNXModels', f'{QT}.merge_onnx_models.MergeONNXModels', 'cleanup'),
    ('FoldConstantsFiltered', f'{QT}.fold_constants.FoldConstantsFiltered', 'cleanup'),
    ('FoldConstants', f'{QT}.fold_constants.FoldConstants', 'cleanup'),
    
    # Inference Operations
    ('InferDataLayouts', f'{QT}.infer_data_layouts.InferDataLayouts', 'topology_opt'),
    ('InferDataTypes', f'{QT}.infer_datatypes.InferDataTypes', 'topology_opt'),
    ('InferShapes', f'{QT}.infer_shapes.InferShapes', 'topology_opt'),
    
    # Graph Management
    ('InsertTopK', f'{QT}.insert_topk.InsertTopK', 'topology_opt'),
    ('InsertIdentity', f'{QT}.insert.InsertIdentity', 'cleanup'),
    ('RemoveUnusedTensors', f'{QT}.general.RemoveUnusedTensors', 'cleanup'),
    ('RemoveUnusedNodes', f'{QT}.remove.RemoveUnusedNodes', 'cleanup'),
    ('RemoveIdentityOps', f'{QT}.remove.RemoveIdentityOps', 'cleanup'),
    ('RemoveStaticGraphInputs', f'{QT}.general.RemoveStaticGraphInputs', 'cleanup'),
    
    # Naming and Organization
    ('GiveReadableTensorNames', f'{QT}.general.GiveReadableTensorNames', 'cleanup'),
    ('GiveUniqueNodeNames', f'{QT}.general.GiveUniqueNodeNames', 'cleanup'),
    ('GiveRandomTensorNames', f'{QT}.general.GiveRandomTensorNames', 'cleanup'),
    ('GiveUniqueParameterTensors', f'{QT}.general.GiveUniqueParameterTensors', 'cleanup'),
    ('SortCommutativeInputsInitializerLast', f'{QT}.general.SortCommutativeInputsInitializerLast', 'cleanup'),
    ('SortGraph', f'{QT}.general.SortGraph', 'cleanup'),
    
    # Additional Operations
    ('MovePadAttributeToTensor', f'{QT}.general.MovePadAttributeToTensor', 'cleanup'),
    ('ConvertSubToAdd', f'{QT}.general.ConvertSubToAdd', 'topology_opt'),
    ('ConvertDivToMul', f'{QT}.general.ConvertDivToMul', 'topology_opt'),
    
    # Pruning Operations
    ('PropagateMasks', f'{QT}.pruning.PropagateMasks', 'cleanup'),
    ('ApplyMasks', f'{QT}.pruning.ApplyMasks', 'cleanup'),
    ('PruneChannels', f'{QT}.pruning.PruneChannels', 'cleanup'),
    ('RemoveMaskedChannels', f'{QT}.pruning.RemoveMaskedChannels', 'cleanup'),
]

FINN_TRANSFORMS = [
    # Basic/Core transforms
    ('RemoveCNVtoFCFlatten', f'{FT}.move_reshape.RemoveCNVtoFCFlatten', 'topology_opt'),
    
    # QONNX integration transforms  
    ('ConvertQONNXtoFINN', f'{FT}.qonnx.convert_qonnx_to_finn.ConvertQONNXtoFINN', 'cleanup'),
    ('FoldQuantWeights', f'{FT}.qonnx.fold_quant_weights.FoldQuantWeights', 'cleanup'),
    ('AvgPoolAndTruncToQuantAvgPool', f'{FT}.qonnx.infer_quant_avg_pool_2d.AvgPoolAndTruncToQuantAvgPool', 'cleanup'),
    ('ConvertQuantActToMultiThreshold', f'{FT}.qonnx.quant_act_to_multithreshold.ConvertQuantActToMultiThreshold', 'cleanup'),
    
    # Streamline absorb transforms
    ('AbsorbSignBiasIntoMultiThreshold', f'{FT}.streamline.absorb.AbsorbSignBiasIntoMultiThreshold', 'topology_opt'),
    ('AbsorbAddIntoMultiThreshold', f'{FT}.streamline.absorb.AbsorbAddIntoMultiThreshold', 'topology_opt'),
    ('AbsorbMulIntoMultiThreshold', f'{FT}.streamline.absorb.AbsorbMulIntoMultiThreshold', 'topology_opt'),
    ('FactorOutMulSignMagnitude', f'{FT}.streamline.absorb.FactorOutMulSignMagnitude', 'topology_opt'),
    ('Absorb1BitMulIntoMatMul', f'{FT}.streamline.absorb.Absorb1BitMulIntoMatMul', 'topology_opt'),
    ('Absorb1BitMulIntoConv', f'{FT}.streamline.absorb.Absorb1BitMulIntoConv', 'topology_opt'),
    ('AbsorbTransposeIntoMultiThreshold', f'{FT}.streamline.absorb.AbsorbTransposeIntoMultiThreshold', 'topology_opt'),
    ('AbsorbTransposeIntoFlatten', f'{FT}.streamline.absorb.AbsorbTransposeIntoFlatten', 'topology_opt'),
    ('AbsorbScalarMulAddIntoTopK', f'{FT}.streamline.absorb.AbsorbScalarMulAddIntoTopK', 'topology_opt'),
    ('AbsorbConsecutiveTransposes', f'{FT}.streamline.absorb.AbsorbConsecutiveTransposes', 'topology_opt'),
    ('AbsorbTransposeIntoResize', f'{FT}.streamline.absorb.AbsorbTransposeIntoResize', 'topology_opt'),
    
    # Streamline collapse transforms
    ('CollapseRepeatedOp', f'{FT}.streamline.collapse_repeated.CollapseRepeatedOp', 'topology_opt'),
    
    # Streamline reorder transforms
    ('MoveAddPastMul', f'{FT}.streamline.reorder.MoveAddPastMul', 'topology_opt'),
    ('MoveScalarMulPastMatMul', f'{FT}.streamline.reorder.MoveScalarMulPastMatMul', 'topology_opt'),
    ('MoveScalarAddPastMatMul', f'{FT}.streamline.reorder.MoveScalarAddPastMatMul', 'topology_opt'),
    ('MoveAddPastConv', f'{FT}.streamline.reorder.MoveAddPastConv', 'topology_opt'),
    ('MoveScalarMulPastConv', f'{FT}.streamline.reorder.MoveScalarMulPastConv', 'topology_opt'),
    ('MoveScalarMulPastConvTranspose', f'{FT}.streamline.reorder.MoveScalarMulPastConvTranspose', 'topology_opt'),
    ('MoveMulPastDWConv', f'{FT}.streamline.reorder.MoveMulPastDWConv', 'topology_opt'),
    ('MoveMulPastMaxPool', f'{FT}.streamline.reorder.MoveMulPastMaxPool', 'topology_opt'),
    ('MoveLinearPastEltwiseAdd', f'{FT}.streamline.reorder.MoveLinearPastEltwiseAdd', 'topology_opt'),
    ('MoveScalarLinearPastInvariants', f'{FT}.streamline.reorder.MoveScalarLinearPastInvariants', 'topology_opt'),
    ('MakeMaxPoolNHWC', f'{FT}.streamline.reorder.MakeMaxPoolNHWC', 'topology_opt'),
    ('MakeScaleResizeNHWC', f'{FT}.streamline.reorder.MakeScaleResizeNHWC', 'topology_opt'),
    ('MoveOpPastFork', f'{FT}.streamline.reorder.MoveOpPastFork', 'topology_opt'),
    ('MoveMaxPoolPastMultiThreshold', f'{FT}.streamline.reorder.MoveMaxPoolPastMultiThreshold', 'topology_opt'),
    ('MoveFlattenPastTopK', f'{FT}.streamline.reorder.MoveFlattenPastTopK', 'topology_opt'),
    ('MoveFlattenPastAffine', f'{FT}.streamline.reorder.MoveFlattenPastAffine', 'topology_opt'),
    ('MoveTransposePastScalarMul', f'{FT}.streamline.reorder.MoveTransposePastScalarMul', 'topology_opt'),
    ('MoveTransposePastJoinAdd', f'{FT}.streamline.reorder.MoveTransposePastJoinAdd', 'topology_opt'),
    ('MoveTransposePastFork', f'{FT}.streamline.reorder.MoveTransposePastFork', 'topology_opt'),
    ('MoveLinearPastFork', f'{FT}.streamline.reorder.MoveLinearPastFork', 'topology_opt'),
    
    # Streamline other transforms
    ('RoundAndClipThresholds', f'{FT}.streamline.round_thresholds.RoundAndClipThresholds', 'topology_opt'),
    ('ConvertSignToThres', f'{FT}.streamline.sign_to_thres.ConvertSignToThres', 'topology_opt'),
    
    # FPGA dataflow core transforms
    ('MinimizeAccumulatorWidth', f'{FT}.fpgadataflow.minimize_accumulator_width.MinimizeAccumulatorWidth', 'dataflow_opt'),
    ('MinimizeWeightBitWidth', f'{FT}.fpgadataflow.minimize_weight_bit_width.MinimizeWeightBitWidth', 'dataflow_opt'),
    ('InsertFIFO', f'{FT}.fpgadataflow.insert_fifo.InsertFIFO', 'dataflow_opt'),
    ('InsertDWC', f'{FT}.fpgadataflow.insert_dwc.InsertDWC', 'dataflow_opt'),
    ('InsertTLastMarker', f'{FT}.fpgadataflow.insert_tlastmarker.InsertTLastMarker', 'dataflow_opt'),
    ('SpecializeLayers', f'{FT}.fpgadataflow.specialize_layers.SpecializeLayers', 'dataflow_opt'),
    ('SetExecMode', f'{FT}.fpgadataflow.set_exec_mode.SetExecMode', 'dataflow_opt'),
    ('SetFolding', f'{FT}.fpgadataflow.set_folding.SetFolding', 'dataflow_opt'),
    ('InsertAndSetFIFODepths', f'{FT}.fpgadataflow.set_fifo_depths.InsertAndSetFIFODepths', 'dataflow_opt'),
    ('SplitLargeFIFOs', f'{FT}.fpgadataflow.set_fifo_depths.SplitLargeFIFOs', 'dataflow_opt'),
    
    # FPGA dataflow floorplan/config transforms
    ('Floorplan', f'{FT}.fpgadataflow.floorplan.Floorplan', 'dataflow_opt'),
    ('ApplyConfig', f'{FT}.fpgadataflow.floorplan.ApplyConfig', 'dataflow_opt'),
    
    # FPGA dataflow build transforms
    ('PrepareIP', f'{FT}.fpgadataflow.prepare_ip.PrepareIP', 'dataflow_opt'),
    ('HLSSynthIP', f'{FT}.fpgadataflow.hlssynth_ip.HLSSynthIP', 'dataflow_opt'),
    ('CreateStitchedIP', f'{FT}.fpgadataflow.create_stitched_ip.CreateStitchedIP', 'dataflow_opt'),
    ('PrepareRTLSim', f'{FT}.fpgadataflow.prepare_rtlsim.PrepareRTLSim', 'dataflow_opt'),
    ('PrepareCppSim', f'{FT}.fpgadataflow.prepare_cppsim.PrepareCppSim', 'dataflow_opt'),
    
    # FPGA dataflow utility transforms
    ('AnnotateCycles', f'{FT}.fpgadataflow.annotate_cycles.AnnotateCycles', 'dataflow_opt'),
    ('AnnotateResources', f'{FT}.fpgadataflow.annotate_resources.AnnotateResources', 'dataflow_opt'),
    ('CleanUp', f'{FT}.fpgadataflow.cleanup.CleanUp', 'cleanup'),
]

# Finn kernels and backends registration data
FINN_KERNELS = [
    # Verified working kernels with correct class names
    ('Thresholding', f'{FK}.thresholding.Thresholding'),
    ('MVAU', f'{FK}.matrixvectoractivation.MVAU'),
    ('VVAU', f'{FK}.vectorvectoractivation.VVAU'),
    ('ConvolutionInputGenerator', f'{FK}.convolutioninputgenerator.ConvolutionInputGenerator'),
    ('StreamingDataWidthConverter', f'{FK}.streamingdatawidthconverter.StreamingDataWidthConverter'),
    ('GlobalAccPool', f'{FK}.globalaccpool.GlobalAccPool'),
    ('StreamingMaxPool', f'{FK}.streamingmaxpool.StreamingMaxPool'),
    ('StreamingFIFO', f'{FK}.streamingfifo.StreamingFIFO'),
    ('StreamingEltwise', f'{FK}.streamingeltwise.StreamingEltwise'),
    ('ChannelwiseOp', f'{FK}.channelwise_op.ChannelwiseOp'),
    ('Pool', f'{FK}.pool.Pool'),
    ('Lookup', f'{FK}.lookup.Lookup'),
    ('LabelSelect', f'{FK}.labelselect.LabelSelect'),
    ('AddStreams', f'{FK}.addstreams.AddStreams'),
    ('DuplicateStreams', f'{FK}.duplicatestreams.DuplicateStreams'),
    ('FMPadding', f'{FK}.fmpadding.FMPadding'),
    ('FMPadding_Pixel', f'{FK}.fmpadding_pixel.FMPadding_Pixel'),
    ('StreamingDataflowPartition', f'{FK}.streamingdataflowpartition.StreamingDataflowPartition'),
    # ElementwiseBinary variants
    ('ElementwiseBinaryOperation', f'{FK}.elementwise_binary.ElementwiseBinaryOperation'),
    ('ElementwiseAdd', f'{FK}.elementwise_binary.ElementwiseAdd'),
    ('ElementwiseSub', f'{FK}.elementwise_binary.ElementwiseSub'),
    ('ElementwiseMul', f'{FK}.elementwise_binary.ElementwiseMul'),
    ('ElementwiseDiv', f'{FK}.elementwise_binary.ElementwiseDiv'),
    # Other kernels with corrected names
    ('StreamingConcat', f'{FK}.concat.StreamingConcat'),
    ('DownSampler', f'{FK}.downsampler.DownSampler'),
    ('UpsampleNearestNeighbour', f'{FK}.upsampler.UpsampleNearestNeighbour'),
]

FINN_KERNEL_INFERENCES = [
    # These are transforms that infer/convert ONNX ops to FINN HW layers
    # Most are in convert_to_hw_layers.py, some in other files
    
    # From convert_to_hw_layers.py
    ('InferQuantizedMatrixVectorActivation', f'{FT}.fpgadataflow.convert_to_hw_layers.InferQuantizedMatrixVectorActivation', 'MVAU'),
    ('InferConvInpGen', f'{FT}.fpgadataflow.convert_to_hw_layers.InferConvInpGen', 'ConvolutionInputGenerator'),
    ('InferBinaryMatrixVectorActivation', f'{FT}.fpgadataflow.convert_to_hw_layers.InferBinaryMatrixVectorActivation', 'MVAU'),
    ('InferVectorVectorActivation', f'{FT}.fpgadataflow.convert_to_hw_layers.InferVectorVectorActivation', 'VVAU'),
    ('InferThresholdingLayer', f'{FT}.fpgadataflow.convert_to_hw_layers.InferThresholdingLayer', 'Thresholding'),
    ('InferStreamingMaxPool', f'{FT}.fpgadataflow.convert_to_hw_layers.InferStreamingMaxPool', 'StreamingMaxPool'),
    ('InferAddStreamsLayer', f'{FT}.fpgadataflow.convert_to_hw_layers.InferAddStreamsLayer', 'AddStreams'),
    ('InferDuplicateStreamsLayer', f'{FT}.fpgadataflow.convert_to_hw_layers.InferDuplicateStreamsLayer', 'DuplicateStreams'),
    ('InferChannelwiseLinearLayer', f'{FT}.fpgadataflow.convert_to_hw_layers.InferChannelwiseLinearLayer', 'ChannelwiseOp'),
    ('InferLabelSelectLayer', f'{FT}.fpgadataflow.convert_to_hw_layers.InferLabelSelectLayer', 'LabelSelect'),
    ('InferGlobalAccPoolLayer', f'{FT}.fpgadataflow.convert_to_hw_layers.InferGlobalAccPoolLayer', 'GlobalAccPool'),
    ('InferPool', f'{FT}.fpgadataflow.convert_to_hw_layers.InferPool', 'Pool'),
    ('InferConcatLayer', f'{FT}.fpgadataflow.convert_to_hw_layers.InferConcatLayer', 'StreamingConcat'),
    ('InferElementwiseBinaryOperation', f'{FT}.fpgadataflow.convert_to_hw_layers.InferElementwiseBinaryOperation', 'ElementwiseBinaryOperation'),
    ('InferLookupLayer', f'{FT}.fpgadataflow.convert_to_hw_layers.InferLookupLayer', 'Lookup'),
    ('InferStreamingEltwise', f'{FT}.fpgadataflow.convert_to_hw_layers.InferStreamingEltwise', 'StreamingEltwise'),
    ('InferUpsample', f'{FT}.fpgadataflow.convert_to_hw_layers.InferUpsample', 'UpsampleNearestNeighbour'),
    
    # From other files
    ('InferPixelPaddingDeconv', f'{FT}.fpgadataflow.infer_pixel_padding_deconv.InferPixelPaddingDeconv', 'ConvolutionInputGenerator'),
]

# FINN backends - (name, class_path, kernel, language)
FINN_BACKENDS = [
    # HLS Backends
    ('MVAU_hls', f'{FK}.hls.matrixvectoractivation_hls.MVAU_hls', 'MVAU', 'hls'),
    ('Thresholding_hls', f'{FK}.hls.thresholding_hls.Thresholding_hls', 'Thresholding', 'hls'),
    ('VVAU_hls', f'{FK}.hls.vectorvectoractivation_hls.VVAU_hls', 'VVAU', 'hls'),
    ('AddStreams_hls', f'{FK}.hls.addstreams_hls.AddStreams_hls', 'AddStreams', 'hls'),
    ('ChannelwiseOp_hls', f'{FK}.hls.channelwise_op_hls.ChannelwiseOp_hls', 'ChannelwiseOp', 'hls'),
    ('CheckSum_hls', f'{FK}.hls.checksum_hls.CheckSum_hls', 'CheckSum', 'hls'),
    ('StreamingConcat_hls', f'{FK}.hls.concat_hls.StreamingConcat_hls', 'StreamingConcat', 'hls'),
    ('ConvolutionInputGenerator_hls', f'{FK}.hls.convolutioninputgenerator_hls.ConvolutionInputGenerator_hls', 'ConvolutionInputGenerator', 'hls'),
    ('DownSampler_hls', f'{FK}.hls.downsampler_hls.DownSampler_hls', 'DownSampler', 'hls'),
    ('DuplicateStreams_hls', f'{FK}.hls.duplicatestreams_hls.DuplicateStreams_hls', 'DuplicateStreams', 'hls'),
    ('ElementwiseBinaryOperation_hls', f'{FK}.hls.elementwise_binary_hls.ElementwiseBinaryOperation_hls', 'ElementwiseBinaryOperation', 'hls'),
    ('FMPadding_hls', f'{FK}.hls.fmpadding_hls.FMPadding_hls', 'FMPadding', 'hls'),
    ('FMPadding_Pixel_hls', f'{FK}.hls.fmpadding_pixel_hls.FMPadding_Pixel_hls', 'FMPadding_Pixel', 'hls'),
    ('GlobalAccPool_hls', f'{FK}.hls.globalaccpool_hls.GlobalAccPool_hls', 'GlobalAccPool', 'hls'),
    ('IODMA_hls', f'{FK}.hls.iodma_hls.IODMA_hls', 'IODMA', 'hls'),
    ('LabelSelect_hls', f'{FK}.hls.labelselect_hls.LabelSelect_hls', 'LabelSelect', 'hls'),
    ('Lookup_hls', f'{FK}.hls.lookup_hls.Lookup_hls', 'Lookup', 'hls'),
    ('Pool_hls', f'{FK}.hls.pool_hls.Pool_hls', 'Pool', 'hls'),
    ('StreamingDataWidthConverter_hls', f'{FK}.hls.streamingdatawidthconverter_hls.StreamingDataWidthConverter_hls', 'StreamingDataWidthConverter', 'hls'),
    ('StreamingEltwise_hls', f'{FK}.hls.streamingeltwise_hls.StreamingEltwise_hls', 'StreamingEltwise', 'hls'),
    ('StreamingMaxPool_hls', f'{FK}.hls.streamingmaxpool_hls.StreamingMaxPool_hls', 'StreamingMaxPool', 'hls'),
    ('TLastMarker_hls', f'{FK}.hls.tlastmarker_hls.TLastMarker_hls', 'TLastMarker', 'hls'),
    ('UpsampleNearestNeighbour_hls', f'{FK}.hls.upsampler_hls.UpsampleNearestNeighbour_hls', 'UpsampleNearestNeighbour', 'hls'),
    
    # RTL Backends
    ('ConvolutionInputGenerator_rtl', f'{FK}.rtl.convolutioninputgenerator_rtl.ConvolutionInputGenerator_rtl', 'ConvolutionInputGenerator', 'rtl'),
    ('DynMVU_rtl', f'{FK}.rtl.dynmvau_rtl.DynMVU_rtl', 'MVAU', 'rtl'),  # Note: DynMVU_rtl is backend for MVAU kernel
    ('FMPadding_rtl', f'{FK}.rtl.fmpadding_rtl.FMPadding_rtl', 'FMPadding', 'rtl'),
    ('MVAU_rtl', f'{FK}.rtl.matrixvectoractivation_rtl.MVAU_rtl', 'MVAU', 'rtl'),
    ('StreamingDataWidthConverter_rtl', f'{FK}.rtl.streamingdatawidthconverter_rtl.StreamingDataWidthConverter_rtl', 'StreamingDataWidthConverter', 'rtl'),
    ('StreamingFIFO_rtl', f'{FK}.rtl.streamingfifo_rtl.StreamingFIFO_rtl', 'StreamingFIFO', 'rtl'),
    ('Thresholding_rtl', f'{FK}.rtl.thresholding_rtl.Thresholding_rtl', 'Thresholding', 'rtl'),
    ('VVAU_rtl', f'{FK}.rtl.vectorvectoractivation_rtl.VVAU_rtl', 'VVAU', 'rtl'),
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


def _register_backends(backends: List[Tuple[str, str, str, str]], framework: str) -> int:
    """
    Register backends directly with the registry.
    
    Args:
        backends: List of (name, class_path, kernel, language) tuples
        framework: Framework name (e.g. 'finn')
    
    Returns:
        Number of successfully registered backends
    """
    from .registry import get_registry
    
    registry = get_registry()
    registered_count = 0
    
    for name, class_path, kernel, language in backends:
        try:
            # Dynamic import
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            backend_class = getattr(module, class_name)
            
            # Register backend
            registry.register_backend(
                name,
                backend_class,
                kernel=kernel,
                language=language,
                framework=framework,
                original_class=class_path,
                description=f"{framework.upper()} {language.upper()} backend for {kernel}"
            )
            
            registered_count += 1
            logger.debug(f"Registered {framework} backend: {name}")
            
        except ImportError as e:
            logger.debug(f"{framework} backend {name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register {framework} backend {name}: {e}")
    
    logger.info(f"Registered {registered_count} {framework} backends")
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
        'finn_backends': 0,
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
    
    # Register FINN backends
    try:
        results['finn_backends'] = _register_backends(FINN_BACKENDS, 'finn')
    except Exception as e:
        logger.warning(f"Failed to register FINN backends: {e}")
    
    # Register FINN kernel inference transforms (as regular transforms with kernel_opt stage)
    kernel_inference_transforms = [(name, class_path, 'kernel_opt') for name, class_path, kernel in FINN_KERNEL_INFERENCES]
    try:
        results['finn_kernel_inferences'] = _register_transforms(kernel_inference_transforms, 'finn')
    except Exception as e:
        logger.warning(f"Failed to register FINN kernel inference transforms: {e}")
    
    logger.info(f"Framework initialization complete:")
    logger.info(f"  - QONNX transforms: {results['qonnx_transforms']}")
    logger.info(f"  - FINN transforms: {results['finn_transforms']}")
    logger.info(f"  - FINN kernels: {results['finn_kernels']}")
    logger.info(f"  - FINN backends: {results['finn_backends']}")
    logger.info(f"  - FINN kernel inference transforms: {results['finn_kernel_inferences']}")
    
    return results


# Initialize frameworks on import for immediate availability
# This is a one-time operation that populates the registry
initialize_framework_integrations()