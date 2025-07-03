"""
Framework Adapters - Perfect Code Implementation

Simple wrappers for QONNX/FINN plugin integration.
No complex discovery - just registration helpers for external frameworks.
"""

import logging
from typing import Type, Dict, Any, List

logger = logging.getLogger(__name__)


class QONNXTransformWrapper:
    """Simple wrapper for QONNX transforms to integrate with registry."""
    
    def __init__(self, qonnx_transform_class: Type, metadata: Dict[str, Any] = None):
        self.qonnx_class = qonnx_transform_class
        self.metadata = metadata or {}
    
    def apply(self, model):
        """Direct apply method for QONNX compatibility."""
        instance = self.qonnx_class()
        return instance.apply(model)
    
    def __call__(self, *args, **kwargs):
        """Allow both direct calling and apply pattern."""
        if args and hasattr(args[0], 'graph'):  # Likely an ONNX model
            return self.apply(args[0])
        else:
            # Return instance for manual apply calling
            return self.qonnx_class(*args, **kwargs)


class FINNTransformWrapper:
    """Simple wrapper for FINN transforms to integrate with registry."""
    
    def __init__(self, finn_transform_class: Type, metadata: Dict[str, Any] = None):
        self.finn_class = finn_transform_class
        self.metadata = metadata or {}
    
    def apply(self, model):
        """Direct apply method for FINN compatibility."""
        instance = self.finn_class()
        return instance.apply(model)
    
    def __call__(self, *args, **kwargs):
        """Allow both direct calling and apply pattern."""
        if args and hasattr(args[0], 'graph'):  # Likely an ONNX model
            return self.apply(args[0])
        else:
            # Return instance for manual apply calling
            return self.finn_class(*args, **kwargs)


def register_qonnx_transforms() -> int:
    """
    Register commonly used QONNX transforms with the registry.
    
    Perfect Code approach: Explicit registration of known transforms
    instead of complex discovery mechanisms.
    """
    from .registry import get_registry
    
            # QONNX transforms - complete list
    qonnx_transforms = [
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
        ('InsertIdentityOnAllTopLevelIO', 'qonnx.transformation.insert.InsertIdentityOnAllTopLevelIO', 'cleanup'),
        ('InsertIdentity', 'qonnx.transformation.insert.InsertIdentity', 'cleanup'),
        ('InsertTopK', 'qonnx.transformation.insert_topk.InsertTopK', 'topology_opt'),
        ('ApplyMasks', 'qonnx.transformation.pruning.ApplyMasks', 'cleanup'),
        ('PropagateMasks', 'qonnx.transformation.pruning.PropagateMasks', 'cleanup'),
        ('RemoveMaskedChannels', 'qonnx.transformation.pruning.RemoveMaskedChannels', 'cleanup'),
        ('PruneChannels', 'qonnx.transformation.pruning.PruneChannels', 'cleanup'),
        ('RemoveUnusedNodes', 'qonnx.transformation.remove.RemoveUnusedNodes', 'cleanup'),
        ('RemoveIdentityOps', 'qonnx.transformation.remove.RemoveIdentityOps', 'cleanup'),
        ('MovePadAttributeToTensor', 'qonnx.transformation.general.MovePadAttributeToTensor', 'cleanup'),
        ('RemoveUnusedTensors', 'qonnx.transformation.general.RemoveUnusedTensors', 'cleanup'),
        ('RemoveStaticGraphInputs', 'qonnx.transformation.general.RemoveStaticGraphInputs', 'cleanup'),
        ('GiveUniqueNodeNames', 'qonnx.transformation.general.GiveUniqueNodeNames', 'cleanup'),
        ('GiveRandomTensorNames', 'qonnx.transformation.general.GiveRandomTensorNames', 'cleanup'),
        ('GiveReadableTensorNames', 'qonnx.transformation.general.GiveReadableTensorNames', 'cleanup'),
        ('GiveUniqueParameterTensors', 'qonnx.transformation.general.GiveUniqueParameterTensors', 'cleanup'),
        ('SortGraph', 'qonnx.transformation.general.SortGraph', 'cleanup'),
        ('ConvertSubToAdd', 'qonnx.transformation.general.ConvertSubToAdd', 'cleanup'),
        ('ConvertDivToMul', 'qonnx.transformation.general.ConvertDivToMul', 'cleanup'),
        ('ApplyConfig', 'qonnx.transformation.general.ApplyConfig', 'model_specific'),
        ('SortCommutativeInputsInitializerLast', 'qonnx.transformation.general.SortCommutativeInputsInitializerLast', 'cleanup'),
    ]
    
    registry = get_registry()
    registered_count = 0
    
    for name, class_path, stage in qonnx_transforms:
        try:
            # Dynamic import
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            qonnx_class = getattr(module, class_name)
            
            # Create wrapper
            wrapper_class = type(
                f"{name}Wrapper", 
                (QONNXTransformWrapper,), 
                {
                    '__init__': lambda self, qc=qonnx_class: QONNXTransformWrapper.__init__(self, qc),
                    '__doc__': f"QONNX {name} transform wrapper"
                }
            )
            
            # Register with registry
            registry.register_transform(
                name, 
                wrapper_class,
                stage=stage, 
                framework='qonnx',
                original_class=class_path,
                description=f"QONNX {name} transform"
            )
            
            registered_count += 1
            logger.debug(f"Registered QONNX transform: {name}")
            
        except ImportError as e:
            logger.debug(f"QONNX transform {name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register QONNX transform {name}: {e}")
    
    logger.info(f"Registered {registered_count} QONNX transforms")
    return registered_count


def register_finn_transforms() -> int:
    """
    Register commonly used FINN transforms with the registry.
    
    FINN transforms will be added as they become available and needed.
    """
    from .registry import get_registry
    
            # FINN transforms - complete list
    finn_transforms = [
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
        ('CreateDataflowPartition', 'finn.transformation.fpgadataflow.create_dataflow_partition.CreateDataflowPartition', 'dataflow_opt'),
        ('MakePYNQDriver', 'finn.transformation.fpgadataflow.make_pynq_driver.MakePYNQDriver', 'post_proc'),
        ('InsertIODMA', 'finn.transformation.fpgadataflow.insert_iodma.InsertIODMA', 'dataflow_opt'),
        ('AnnotateCycles', 'finn.transformation.fpgadataflow.annotate_cycles.AnnotateCycles', 'post_proc'),
        ('AnnotateResources', 'finn.transformation.fpgadataflow.annotate_resources.AnnotateResources', 'post_proc'),
        ('SetFolding', 'finn.transformation.fpgadataflow.set_folding.SetFolding', 'dataflow_opt'),
        ('CreateStitchedIP', 'finn.transformation.fpgadataflow.create_stitched_ip.CreateStitchedIP', 'dataflow_opt'),
        ('PrepareIP', 'finn.transformation.fpgadataflow.prepare_ip.PrepareIP', 'dataflow_opt'),
        ('SpecializeLayers', 'finn.transformation.fpgadataflow.specialize_layers.SpecializeLayers', 'kernel_opt'),
        ('Floorplan', 'finn.transformation.fpgadataflow.floorplan.Floorplan', 'dataflow_opt'),
        ('ReplaceVerilogRelPaths', 'finn.transformation.fpgadataflow.replace_verilog_relpaths.ReplaceVerilogRelPaths', 'cleanup'),
        ('SynthPYNQProject', 'finn.transformation.fpgadataflow.synth_pynq.SynthPYNQProject', 'dataflow_opt'),
        ('CompileCppSim', 'finn.transformation.fpgadataflow.compile_cppsim.CompileCppSim', 'post_proc'),
        ('HLSSynthIP', 'finn.transformation.fpgadataflow.hlssynth_ip.HLSSynthIP', 'dataflow_opt'),
        ('PrepareRTLSim', 'finn.transformation.fpgadataflow.prepare_rtlsim.PrepareRTLSim', 'post_proc'),
        ('SetExecMode', 'finn.transformation.fpgadataflow.set_exec_mode.SetExecMode', 'post_proc'),
        ('DeriveCharacteristic', 'finn.transformation.fpgadataflow.derive_characteristic.DeriveCharacteristic', 'post_proc'),
    ]
    
    registry = get_registry()
    registered_count = 0
    
    for name, class_path, stage in finn_transforms:
        try:
            # Dynamic import
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            finn_class = getattr(module, class_name)
            
            # Create wrapper
            wrapper_class = type(
                f"{name}Wrapper",
                (FINNTransformWrapper,),
                {
                    '__init__': lambda self, fc=finn_class: FINNTransformWrapper.__init__(self, fc),
                    '__doc__': f"FINN {name} transform wrapper"
                }
            )
            
            # Register with registry
            registry.register_transform(
                name,
                wrapper_class,
                stage=stage,
                framework='finn',
                original_class=class_path,
                description=f"FINN {name} transform"
            )
            
            registered_count += 1
            logger.debug(f"Registered FINN transform: {name}")
            
        except ImportError as e:
            logger.debug(f"FINN transform {name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register FINN transform {name}: {e}")
    
    logger.info(f"Registered {registered_count} FINN transforms")
    return registered_count


def register_finn_kernels() -> int:
    """
    Register FINN HWCustomOp kernels with the registry.
    
    Scans the FINN fpgadataflow directory for HWCustomOp subclasses
    and registers them as kernels.
    """
    from .registry import get_registry
    
    # FINN kernel classes - HWCustomOp subclasses
    finn_kernels = [
        ('MVAU', 'finn.custom_op.fpgadataflow.matrixvectoractivation.MVAU'),
        ('VVAU', 'finn.custom_op.fpgadataflow.vectorvectoractivation.VVAU'), 
        ('Thresholding', 'finn.custom_op.fpgadataflow.thresholding.Thresholding'),
        ('ConvolutionInputGenerator', 'finn.custom_op.fpgadataflow.convolutioninputgenerator.ConvolutionInputGenerator'),
        ('FMPadding', 'finn.custom_op.fpgadataflow.fmpadding.FMPadding'),
        ('FMPadding_Pixel', 'finn.custom_op.fpgadataflow.fmpadding_pixel.FMPadding_Pixel'),
        ('StreamingDataWidthConverter', 'finn.custom_op.fpgadataflow.streamingdatawidthconverter.StreamingDataWidthConverter'),
        ('StreamingFIFO', 'finn.custom_op.fpgadataflow.streamingfifo.StreamingFIFO'),
        ('StreamingMaxPool', 'finn.custom_op.fpgadataflow.streamingmaxpool.StreamingMaxPool'),
        ('Pool', 'finn.custom_op.fpgadataflow.pool.Pool'),
        ('GlobalAccPool', 'finn.custom_op.fpgadataflow.globalaccpool.GlobalAccPool'),
        ('AddStreams', 'finn.custom_op.fpgadataflow.addstreams.AddStreams'),
        ('DuplicateStreams', 'finn.custom_op.fpgadataflow.duplicatestreams.DuplicateStreams'),
        ('StreamingConcat', 'finn.custom_op.fpgadataflow.concat.StreamingConcat'),
        ('LabelSelect', 'finn.custom_op.fpgadataflow.labelselect.LabelSelect'),
        ('Lookup', 'finn.custom_op.fpgadataflow.lookup.Lookup'),
        ('ChannelwiseOp', 'finn.custom_op.fpgadataflow.channelwise_op.ChannelwiseOp'),
        ('ElementwiseBinaryOperation', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseBinaryOperation'),
        ('StreamingEltwise', 'finn.custom_op.fpgadataflow.streamingeltwise.StreamingEltwise'),
        ('DownSampler', 'finn.custom_op.fpgadataflow.downsampler.DownSampler'),
        ('UpsampleNearestNeighbour', 'finn.custom_op.fpgadataflow.upsampler.UpsampleNearestNeighbour'),
        # HLS-only kernels (these are HWCustomOp + HLSBackend classes directly)
        ('TLastMarker', 'finn.custom_op.fpgadataflow.hls.tlastmarker_hls.TLastMarker_hls'),
        ('IODMA', 'finn.custom_op.fpgadataflow.hls.iodma_hls.IODMA_hls'),
        ('CheckSum', 'finn.custom_op.fpgadataflow.hls.checksum_hls.CheckSum_hls'),
    ]
    
    registry = get_registry()
    registered_count = 0
    
    for name, class_path in finn_kernels:
        try:
            # Dynamic import
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            kernel_class = getattr(module, class_name)
            
            # Register kernel
            registry.register_kernel(
                name,
                kernel_class,
                framework='finn',
                description=f"FINN {name} kernel",
                original_class=class_path
            )
            
            registered_count += 1
            logger.debug(f"Registered FINN kernel: {name}")
            
        except ImportError as e:
            logger.debug(f"FINN kernel {name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register FINN kernel {name}: {e}")
    
    logger.info(f"Registered {registered_count} FINN kernels")
    return registered_count


def register_finn_backends() -> int:
    """
    Register FINN HLS and RTL backend implementations.
    
    Scans HLS and RTL directories for backend classes and registers them
    with their corresponding kernels.
    """
    from .registry import get_registry
    
    # FINN HLS backends
    finn_hls_backends = [
        ('MVAU_hls', 'finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls.MVAU_hls', 'MVAU'),
        ('VVAU_hls', 'finn.custom_op.fpgadataflow.hls.vectorvectoractivation_hls.VVAU_hls', 'VVAU'),
        ('Thresholding_hls', 'finn.custom_op.fpgadataflow.hls.thresholding_hls.Thresholding_hls', 'Thresholding'),
        ('ConvolutionInputGenerator_hls', 'finn.custom_op.fpgadataflow.hls.convolutioninputgenerator_hls.ConvolutionInputGenerator_hls', 'ConvolutionInputGenerator'),
        ('FMPadding_hls', 'finn.custom_op.fpgadataflow.hls.fmpadding_hls.FMPadding_hls', 'FMPadding'),
        ('FMPadding_Pixel_hls', 'finn.custom_op.fpgadataflow.hls.fmpadding_pixel_hls.FMPadding_Pixel_hls', 'FMPadding_Pixel'),
        ('StreamingDataWidthConverter_hls', 'finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls.StreamingDataWidthConverter_hls', 'StreamingDataWidthConverter'),
        ('StreamingMaxPool_hls', 'finn.custom_op.fpgadataflow.hls.streamingmaxpool_hls.StreamingMaxPool_hls', 'StreamingMaxPool'),
        ('Pool_hls', 'finn.custom_op.fpgadataflow.hls.pool_hls.Pool_hls', 'Pool'),
        ('GlobalAccPool_hls', 'finn.custom_op.fpgadataflow.hls.globalaccpool_hls.GlobalAccPool_hls', 'GlobalAccPool'),
        ('AddStreams_hls', 'finn.custom_op.fpgadataflow.hls.addstreams_hls.AddStreams_hls', 'AddStreams'),
        ('DuplicateStreams_hls', 'finn.custom_op.fpgadataflow.hls.duplicatestreams_hls.DuplicateStreams_hls', 'DuplicateStreams'),
        ('StreamingConcat_hls', 'finn.custom_op.fpgadataflow.hls.concat_hls.StreamingConcat_hls', 'StreamingConcat'),
        ('LabelSelect_hls', 'finn.custom_op.fpgadataflow.hls.labelselect_hls.LabelSelect_hls', 'LabelSelect'),
        ('Lookup_hls', 'finn.custom_op.fpgadataflow.hls.lookup_hls.Lookup_hls', 'Lookup'),
        ('ChannelwiseOp_hls', 'finn.custom_op.fpgadataflow.hls.channelwise_op_hls.ChannelwiseOp_hls', 'ChannelwiseOp'),
        ('StreamingEltwise_hls', 'finn.custom_op.fpgadataflow.hls.streamingeltwise_hls.StreamingEltwise_hls', 'StreamingEltwise'),
        ('DownSampler_hls', 'finn.custom_op.fpgadataflow.hls.downsampler_hls.DownSampler_hls', 'DownSampler'),
        ('UpsampleNearestNeighbour_hls', 'finn.custom_op.fpgadataflow.hls.upsampler_hls.UpsampleNearestNeighbour_hls', 'UpsampleNearestNeighbour'),
        ('TLastMarker_hls', 'finn.custom_op.fpgadataflow.hls.tlastmarker_hls.TLastMarker_hls', 'TLastMarker'),
        ('IODMA_hls', 'finn.custom_op.fpgadataflow.hls.iodma_hls.IODMA_hls', 'IODMA'),
        ('CheckSum_hls', 'finn.custom_op.fpgadataflow.hls.checksum_hls.CheckSum_hls', 'CheckSum'),
    ]
    
    # FINN RTL backends  
    finn_rtl_backends = [
        ('MVAU_rtl', 'finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl.MVAU_rtl', 'MVAU'),
        ('VVAU_rtl', 'finn.custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl.VVAU_rtl', 'VVAU'),
        ('Thresholding_rtl', 'finn.custom_op.fpgadataflow.rtl.thresholding_rtl.Thresholding_rtl', 'Thresholding'),
        ('ConvolutionInputGenerator_rtl', 'finn.custom_op.fpgadataflow.rtl.convolutioninputgenerator_rtl.ConvolutionInputGenerator_rtl', 'ConvolutionInputGenerator'),
        ('FMPadding_rtl', 'finn.custom_op.fpgadataflow.rtl.fmpadding_rtl.FMPadding_rtl', 'FMPadding'),
        ('StreamingDataWidthConverter_rtl', 'finn.custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl.StreamingDataWidthConverter_rtl', 'StreamingDataWidthConverter'),
        ('StreamingFIFO_rtl', 'finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl.StreamingFIFO_rtl', 'StreamingFIFO'),
        ('DynMVU_rtl', 'finn.custom_op.fpgadataflow.rtl.dynmvau_rtl.DynMVU_rtl', 'MVAU'),
    ]
    
    registry = get_registry()
    registered_count = 0
    
    # Register HLS backends
    for name, class_path, kernel in finn_hls_backends:
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
                language='hls',
                framework='finn',
                description=f"FINN {kernel} HLS backend",
                original_class=class_path
            )
            
            registered_count += 1
            logger.debug(f"Registered FINN HLS backend: {name}")
            
        except ImportError as e:
            logger.debug(f"FINN HLS backend {name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register FINN HLS backend {name}: {e}")
    
    # Register RTL backends
    for name, class_path, kernel in finn_rtl_backends:
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
                language='rtl',
                framework='finn',
                description=f"FINN {kernel} RTL backend",
                original_class=class_path
            )
            
            registered_count += 1
            logger.debug(f"Registered FINN RTL backend: {name}")
            
        except ImportError as e:
            logger.debug(f"FINN RTL backend {name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register FINN RTL backend {name}: {e}")
    
    logger.info(f"Registered {registered_count} FINN backends")
    return registered_count


def register_finn_kernel_inference_transforms() -> int:
    """
    Register FINN kernel inference transforms.
    
    These transforms convert generic operations into specific FINN HW kernels.
    They are registered as kernel_inference type with the target kernel specified.
    """
    from .registry import get_registry
    
    # FINN kernel inference transforms - maps inference transform to target kernel
    finn_kernel_inferences = [
        ('InferThresholdingLayer', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferThresholdingLayer', 'Thresholding'),
        ('InferBinaryMatrixVectorActivation', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferBinaryMatrixVectorActivation', 'MVAU'),
        ('InferQuantizedMatrixVectorActivation', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferQuantizedMatrixVectorActivation', 'MVAU'),
        ('InferVectorVectorActivation', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferVectorVectorActivation', 'VVAU'),
        ('InferStreamingMaxPool', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferStreamingMaxPool', 'StreamingMaxPool'),
        ('InferAddStreamsLayer', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferAddStreamsLayer', 'AddStreams'),
        ('InferDuplicateStreamsLayer', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferDuplicateStreamsLayer', 'DuplicateStreams'),
        ('InferChannelwiseLinearLayer', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferChannelwiseLinearLayer', 'ChannelwiseOp'),
        ('InferStreamingEltwise', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferStreamingEltwise', 'StreamingEltwise'),
        ('InferGlobalAccPoolLayer', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferGlobalAccPoolLayer', 'GlobalAccPool'),
        ('InferLabelSelectLayer', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferLabelSelectLayer', 'LabelSelect'),
        ('InferLookupLayer', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferLookupLayer', 'Lookup'),
        ('InferConvInpGen', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferConvInpGen', 'ConvolutionInputGenerator'),
        ('InferPool', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferPool', 'Pool'),
        ('InferUpsample', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferUpsample', 'UpsampleNearestNeighbour'),
        ('InferConcat', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferConcatLayer', 'StreamingConcat'),
        ('InferElementwiseBinaryOperation', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferElementwiseBinaryOperation', 'ElementwiseBinaryOperation'),
        ('InferPixelPaddingDeconv', 'finn.transformation.fpgadataflow.infer_pixel_padding_deconv.InferPixelPaddingDeconv', 'FMPadding_Pixel'),
    ]
    
    registry = get_registry()
    registered_count = 0
    
    for name, class_path, kernel in finn_kernel_inferences:
        try:
            # Dynamic import
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            inference_class = getattr(module, class_name)
            
            # Create wrapper for kernel inference
            wrapper_class = type(
                f"{name}Wrapper",
                (FINNTransformWrapper,),
                {
                    '__init__': lambda self, fc=inference_class: FINNTransformWrapper.__init__(self, fc),
                    '__doc__': f"FINN {name} kernel inference wrapper"
                }
            )
            
            # Register as kernel_inference transform with plugin_type preserved
            registry.register_transform(
                name,
                wrapper_class,
                stage='kernel_inference',  # Use stage for internal categorization
                framework='finn',
                plugin_type='kernel_inference',  # Preserve the actual type
                kernel=kernel,  # Specify target kernel
                original_class=class_path,
                description=f"FINN kernel inference for {kernel}"
            )
            
            registered_count += 1
            logger.debug(f"Registered FINN kernel inference: {name} -> {kernel}")
            
        except ImportError as e:
            logger.debug(f"FINN kernel inference {name} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to register FINN kernel inference {name}: {e}")
    
    logger.info(f"Registered {registered_count} FINN kernel inference transforms")
    return registered_count


def initialize_framework_integrations() -> Dict[str, int]:
    """
    Initialize all framework integrations.
    
    Perfect Code approach: Explicit initialization instead of complex discovery.
    Called once at import time or explicitly by user.
    """
    results = {}
    
    # Register QONNX transforms
    try:
        qonnx_count = register_qonnx_transforms()
        results['qonnx_transforms'] = qonnx_count
    except Exception as e:
        logger.warning(f"Failed to initialize QONNX integration: {e}")
        results['qonnx_transforms'] = 0
    
    # Register FINN transforms
    try:
        finn_count = register_finn_transforms()
        results['finn_transforms'] = finn_count
    except Exception as e:
        logger.warning(f"Failed to initialize FINN transform integration: {e}")
        results['finn_transforms'] = 0
    
    # Register FINN kernels
    try:
        finn_kernel_count = register_finn_kernels()
        results['finn_kernels'] = finn_kernel_count
    except Exception as e:
        logger.warning(f"Failed to initialize FINN kernel integration: {e}")
        results['finn_kernels'] = 0
    
    # Register FINN backends
    try:
        finn_backend_count = register_finn_backends()
        results['finn_backends'] = finn_backend_count
    except Exception as e:
        logger.warning(f"Failed to initialize FINN backend integration: {e}")
        results['finn_backends'] = 0
    
    # Register FINN kernel inference transforms
    try:
        finn_kernel_inference_count = register_finn_kernel_inference_transforms()
        results['finn_kernel_inferences'] = finn_kernel_inference_count
    except Exception as e:
        logger.warning(f"Failed to initialize FINN kernel inference integration: {e}")
        results['finn_kernel_inferences'] = 0
    
    total_registered = sum(results.values())
    logger.info(f"Framework integration complete: {total_registered} external plugins registered")
    logger.info(f"  - QONNX transforms: {results.get('qonnx_transforms', 0)}")
    logger.info(f"  - FINN transforms: {results.get('finn_transforms', 0)}")  
    logger.info(f"  - FINN kernels: {results.get('finn_kernels', 0)}")
    logger.info(f"  - FINN backends: {results.get('finn_backends', 0)}")
    logger.info(f"  - FINN kernel inferences: {results.get('finn_kernel_inferences', 0)}")
    
    return results


def register_external_plugin(plugin_class: Type, name: str, plugin_type: str, 
                           framework: str, **metadata) -> None:
    """
    Convenience function to register external plugins.
    
    Args:
        plugin_class: The plugin class to register
        name: Plugin name
        plugin_type: Type of plugin ('transform', 'kernel', 'backend')
        framework: Framework name ('qonnx', 'finn', etc.)
        **metadata: Additional metadata
    """
    from .registry import get_registry
    
    registry = get_registry()
    
    if plugin_type == 'transform':
        registry.register_transform(
            name,
            plugin_class,
            framework=framework,
            **metadata
        )
    elif plugin_type == 'kernel':
        registry.register_kernel(name, plugin_class, **metadata)
    elif plugin_type == 'backend':
        kernel = metadata.get('kernel')
        backend_type = metadata.get('backend_type')
        if not kernel or not backend_type:
            raise ValueError("Backend registration requires 'kernel' and 'backend_type' metadata")
        
        registry.register_backend(
            name,
            plugin_class,
            kernel=kernel,
            backend_type=backend_type,
            **{k: v for k, v in metadata.items() if k not in ['kernel', 'backend_type']}
        )
    else:
        raise ValueError(f"Unknown plugin type: {plugin_type}")
    
    logger.info(f"Registered external {plugin_type}: {name} ({framework})")


# Auto-initialize framework integrations on import
# This ensures QONNX/FINN plugins are available immediately
_framework_initialized = False

def ensure_frameworks_initialized():
    """Ensure framework integrations are initialized (called on first access)."""
    global _framework_initialized
    if not _framework_initialized:
        try:
            initialize_framework_integrations()
            _framework_initialized = True
        except Exception as e:
            logger.warning(f"Framework initialization failed: {e}")


# Initialize frameworks on import for immediate availability
try:
    ensure_frameworks_initialized()
except Exception as e:
    logger.debug(f"Framework auto-initialization failed: {e}")  # Non-fatal