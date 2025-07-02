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
        ('MoveTransposePastEltwise', 'finn.transformation.streamline.reorder.MoveTransposePastEltwise', 'topology_opt'),
        ('MoveReshapePastEltwise', 'finn.transformation.streamline.reorder.MoveReshapePastEltwise', 'topology_opt'),
        ('MoveReshapePastJoinOp', 'finn.transformation.streamline.reorder.MoveReshapePastJoinOp', 'topology_opt'),
        ('MoveTransposePastJoinAdd', 'finn.transformation.streamline.reorder.MoveTransposePastJoinAdd', 'topology_opt'),
        ('MoveTransposePastFork', 'finn.transformation.streamline.reorder.MoveTransposePastFork', 'topology_opt'),
        ('MoveLinearPastFork', 'finn.transformation.streamline.reorder.MoveLinearPastFork', 'topology_opt'),
        ('RoundAndClipThresholds', 'finn.transformation.streamline.round_thresholds.RoundAndClipThresholds', 'topology_opt'),
        ('AdjustBatchNormAxis', 'finn.transformation.streamline.batch_norm.AdjustBatchNormAxis', 'topology_opt'),
        ('InferThresholdingLayer', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferThresholdingLayer', 'kernel_opt'),
        ('InferLabelSelectLayer', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferLabelSelectLayer', 'kernel_opt'),
        ('InferGlobalAccPoolLayer', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferGlobalAccPoolLayer', 'kernel_opt'),
        ('InferDuplicateStreamsLayer', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferDuplicateStreamsLayer', 'kernel_opt'),
        ('InferAddStreamsLayer', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferAddStreamsLayer', 'kernel_opt'),
        ('InferLookupLayer', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferLookupLayer', 'kernel_opt'),
        ('InferFlattenLayer', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferFlattenLayer', 'kernel_opt'),
        ('InferStreamingMVU', 'finn.transformation.fpgadataflow.infer_hlslib_mvau.InferStreamingMVU', 'kernel_opt'),
        ('InferChannelwiseLinearLayer', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferChannelwiseLinearLayer', 'kernel_opt'),
        ('InferStreamingEltwise', 'finn.transformation.fpgadataflow.infer_hlslib_layers.InferStreamingEltwise', 'kernel_opt'),
        ('ConvertToHLSLayers', 'finn.transformation.fpgadataflow.convert_to_hls_layers.ConvertToHLSLayers', 'kernel_opt'),
        ('InferBinaryMatrixVectorActivation', 'finn.transformation.fpgadataflow.infer_binarymatrixvectoractivation.InferBinaryMatrixVectorActivation', 'kernel_opt'),
        ('InferQuantizedMatrixVectorActivation', 'finn.transformation.fpgadataflow.infer_quantized_matrixvectoractivation.InferQuantizedMatrixVectorActivation', 'kernel_opt'),
        ('InferVectorVectorActivation', 'finn.transformation.fpgadataflow.infer_vvau.InferVectorVectorActivation', 'kernel_opt'),
        ('InferConvInpGen', 'finn.transformation.fpgadataflow.infer_conv_input_gen.InferConvInpGen', 'kernel_opt'),
        ('InferStreamingMaxPool', 'finn.transformation.fpgadataflow.infer_streamingmaxpool.InferStreamingMaxPool', 'kernel_opt'),
        ('InferPool', 'finn.transformation.fpgadataflow.infer_pool.InferPool', 'kernel_opt'),
        ('InferDownSampler', 'finn.transformation.fpgadataflow.infer_downsampler.InferDownSampler', 'kernel_opt'),
        ('InferUpsample', 'finn.transformation.fpgadataflow.infer_upsampling.InferUpsample', 'kernel_opt'),
        ('InferConcat', 'finn.transformation.fpgadataflow.infer_concat.InferConcat', 'kernel_opt'),
        ('MinimizeAccumulatorWidth', 'finn.transformation.fpgadataflow.minimize_accumulator_width.MinimizeAccumulatorWidth', 'dataflow_opt'),
        ('MinimizeWeightBitWidth', 'finn.transformation.fpgadataflow.minimize_weight_bit_width.MinimizeWeightBitWidth', 'dataflow_opt'),
        ('InsertFIFO', 'finn.transformation.fpgadataflow.insert_fifo.InsertFIFO', 'dataflow_opt'),
        ('InsertDWC', 'finn.transformation.fpgadataflow.insert_dwc.InsertDWC', 'dataflow_opt'),
        ('InsertTLastMarker', 'finn.transformation.fpgadataflow.insert_tlastmarker.InsertTLastMarker', 'dataflow_opt'),
        ('RemoveUnusedTensors', 'finn.transformation.fpgadataflow.remove_unused_tensors.RemoveUnusedTensors', 'cleanup'),
        ('GiveUniqueNodeNames', 'finn.transformation.fpgadataflow.floorplan.Floorplan', 'dataflow_opt'),
        ('CreateDataflowPartition', 'finn.transformation.fpgadataflow.create_dataflow_partition.CreateDataflowPartition', 'dataflow_opt'),
        ('MakePYNQDriver', 'finn.transformation.fpgadataflow.make_pynq_driver.MakePYNQDriver',' post_proc'),
        ('InsertIODMA', 'finn.transformation.fpgadataflow.insert_iodma.InsertIODMA', 'dataflow_opt'),
        ('AnnotateCycles', 'finn.transformation.fpgadataflow.annotate_cycles.AnnotateCycles',' post_proc'),
        ('AnnotateResources', 'finn.transformation.fpgadataflow.annotate_resources.AnnotateResources',' post_proc'),
        ('SetFolding', 'finn.transformation.fpgadataflow.set_folding.SetFolding', 'dataflow_opt'),
        ('CreateStitchedIP', 'finn.transformation.fpgadataflow.create_stitched_ip.CreateStitchedIP', 'dataflow_opt'),
        ('PrepareIP', 'finn.transformation.fpgadataflow.prepare_ip.PrepareIP', 'dataflow_opt'),
        ('SpecializeLayers', 'finn.transformation.fpgadataflow.specialize_layers.SpecializeLayers', 'kernel_opt'),
        ('Floorplan', 'finn.transformation.fpgadataflow.floorplan.Floorplan', 'dataflow_opt'),
        ('ReplaceVerilogRelPaths', 'finn.transformation.fpgadataflow.replace_verilog_relpaths.ReplaceVerilogRelPaths', 'cleanup'),
        ('SynthPYNQProject', 'finn.transformation.fpgadataflow.synth_pynq.SynthPYNQProject', 'dataflow_opt'),
        ('CompileCppSim', 'finn.transformation.fpgadataflow.compile_cppsim.CompileCppSim',' post_proc'),
        ('HLSSynthIP', 'finn.transformation.fpgadataflow.hlssynth_ip.HLSSynthIP', 'dataflow_opt'),
        ('PrepareRTLSim', 'finn.transformation.fpgadataflow.prepare_rtlsim.PrepareRTLSim',' post_proc'),
        ('SetExecMode', 'finn.transformation.fpgadataflow.set_exec_mode.SetExecMode',' post_proc'),
        ('DeriveCharacteristic', 'finn.transformation.fpgadataflow.derive_characteristic.DeriveCharacteristic',' post_proc'),
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
        results['qonnx'] = qonnx_count
    except Exception as e:
        logger.warning(f"Failed to initialize QONNX integration: {e}")
        results['qonnx'] = 0
    
    # Register FINN transforms
    try:
        finn_count = register_finn_transforms()
        results['finn'] = finn_count
    except Exception as e:
        logger.warning(f"Failed to initialize FINN integration: {e}")
        results['finn'] = 0
    
    total_registered = sum(results.values())
    logger.info(f"Framework integration complete: {total_registered} external transforms registered")
    
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