# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Core Brainsmith/FINN/QONNX component registration.

This module registers all core components using the namespace-based registry system.
Components are explicitly imported and registered with source prefixes.

NOTE: This is a partial migration demonstrating the pattern. Full migration requires:
1. Updating all kernel classes to inherit from CustomOp
2. Updating all backend classes to inherit from Backend
3. Converting all FINN/QONNX components
4. Registering all steps, kernels, and backends explicitly

For now, this registers the available components that have been migrated.
"""

from brainsmith.core import registry

# === STEPS ===
# Steps are already importable functions, just need registration

# Brainsmith steps
from brainsmith.steps.core_steps import (
    qonnx_to_finn_step,
    specialize_layers_step,
    constrain_folding_and_set_pumped_compute_step,
)
from brainsmith.steps.kernel_inference import infer_kernels_step
from brainsmith.steps.bert_custom_steps import (
    shell_metadata_handover_step,
    bert_cleanup_step,
    bert_streamlining_step,
)

# Register brainsmith steps
registry.step(qonnx_to_finn_step, source='brainsmith', name='qonnx_to_finn')
registry.step(specialize_layers_step, source='brainsmith', name='specialize_layers')
registry.step(
    constrain_folding_and_set_pumped_compute_step,
    source='brainsmith',
    name='constrain_folding_and_set_pumped_compute'
)
registry.step(infer_kernels_step, source='brainsmith', name='infer_kernels')
registry.step(shell_metadata_handover_step, source='brainsmith', name='shell_metadata_handover')
registry.step(bert_cleanup_step, source='brainsmith', name='bert_cleanup')
registry.step(bert_streamlining_step, source='brainsmith', name='bert_streamlining')

# FINN steps (lazy import to avoid dependency issues)
try:
    from finn.builder.build_dataflow_steps import (
        step_qonnx_to_finn as finn_qonnx_to_finn,
        step_tidy_up,
        step_streamline,
        step_convert_to_hw,
        step_create_dataflow_partition,
        step_specialize_layers as finn_specialize_layers,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_hw_codegen,
        step_hw_ipgen,
        step_set_fifo_depths,
        step_create_stitched_ip,
        step_measure_rtlsim_performance,
        step_out_of_context_synthesis,
        step_synthesize_bitfile,
        step_make_driver,
        step_deployment_package,
    )

    # Register FINN steps
    registry.step(finn_qonnx_to_finn, source='finn', name='qonnx_to_finn')
    registry.step(step_tidy_up, source='finn', name='tidy_up')
    registry.step(step_streamline, source='finn', name='streamline')
    registry.step(step_convert_to_hw, source='finn', name='convert_to_hw')
    registry.step(step_create_dataflow_partition, source='finn', name='create_dataflow_partition')
    registry.step(finn_specialize_layers, source='finn', name='specialize_layers')
    registry.step(step_target_fps_parallelization, source='finn', name='target_fps_parallelization')
    registry.step(step_apply_folding_config, source='finn', name='apply_folding_config')
    registry.step(step_minimize_bit_width, source='finn', name='minimize_bit_width')
    registry.step(step_generate_estimate_reports, source='finn', name='generate_estimate_reports')
    registry.step(step_hw_codegen, source='finn', name='hw_codegen')
    registry.step(step_hw_ipgen, source='finn', name='hw_ipgen')
    registry.step(step_set_fifo_depths, source='finn', name='set_fifo_depths')
    registry.step(step_create_stitched_ip, source='finn', name='create_stitched_ip')
    registry.step(step_measure_rtlsim_performance, source='finn', name='measure_rtlsim_performance')
    registry.step(step_out_of_context_synthesis, source='finn', name='out_of_context_synthesis')
    registry.step(step_synthesize_bitfile, source='finn', name='synthesize_bitfile')
    registry.step(step_make_driver, source='finn', name='make_driver')
    registry.step(step_deployment_package, source='finn', name='deployment_package')

except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import FINN steps: {e}")


# === KERNELS & BACKENDS ===

# Migrated kernels/backends (namespace-based registration)
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.kernels.layernorm.layernorm_hls import LayerNorm_hls

# Register migrated components
registry.kernel(LayerNorm, source='brainsmith')
registry.backend(LayerNorm_hls, source='brainsmith')

# === Remaining Brainsmith Kernels & Backends ===
# Register using metadata parameters (no need to modify class files)

# Softmax kernel and backend
from brainsmith.kernels.softmax.hwsoftmax import HWSoftmax
from brainsmith.kernels.softmax.infer_hwsoftmax import InferHWSoftmax
from brainsmith.kernels.softmax.hwsoftmax_hls import HWSoftmax_hls

registry.kernel(
    HWSoftmax,
    source='brainsmith',
    name='Softmax',
    op_type='Softmax',
    infer_transform=InferHWSoftmax
)
registry.backend(
    HWSoftmax_hls,
    source='brainsmith',
    name='Softmax_HLS',
    target_kernel='brainsmith:Softmax',
    language='hls'
)

# Shuffle kernel and backend
from brainsmith.kernels.shuffle.shuffle import Shuffle
from brainsmith.kernels.shuffle.infer_shuffle import InferShuffle
from brainsmith.kernels.shuffle.shuffle_hls import Shuffle_hls

registry.kernel(
    Shuffle,
    source='brainsmith',
    name='Shuffle',
    op_type='Shuffle',
    infer_transform=InferShuffle
)
registry.backend(
    Shuffle_hls,
    source='brainsmith',
    name='Shuffle_HLS',
    target_kernel='brainsmith:Shuffle',
    language='hls'
)

# Crop kernel and backend
from brainsmith.kernels.crop.crop import Crop
from brainsmith.kernels.crop.infer_crop_from_gather import InferCropFromGather
from brainsmith.kernels.crop.crop_hls import Crop_hls

registry.kernel(
    Crop,
    source='brainsmith',
    name='Crop',
    op_type='Crop',
    infer_transform=InferCropFromGather
)
registry.backend(
    Crop_hls,
    source='brainsmith',
    name='Crop_HLS',
    target_kernel='brainsmith:Crop',
    language='hls'
)


# === FINN Core Kernels & Backends ===
# Bulk register FINN components using metadata parameters
# Data ported from legacy registry_adapters.py

import importlib
import logging

logger = logging.getLogger(__name__)

# FINN kernel data: (name, module_path, infer_transform_path)
_FINN_KERNELS = [
    ('Thresholding', 'finn.custom_op.fpgadataflow.thresholding.Thresholding', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferThresholdingLayer'),
    ('MVAU', 'finn.custom_op.fpgadataflow.matrixvectoractivation.MVAU', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferQuantizedMatrixVectorActivation'),
    ('VVAU', 'finn.custom_op.fpgadataflow.vectorvectoractivation.VVAU', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferVectorVectorActivation'),
    ('ConvolutionInputGenerator', 'finn.custom_op.fpgadataflow.convolutioninputgenerator.ConvolutionInputGenerator', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferConvInpGen'),
    ('StreamingDataWidthConverter', 'finn.custom_op.fpgadataflow.streamingdatawidthconverter.StreamingDataWidthConverter', None),
    ('GlobalAccPool', 'finn.custom_op.fpgadataflow.globalaccpool.GlobalAccPool', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferGlobalAccPoolLayer'),
    ('StreamingFIFO', 'finn.custom_op.fpgadataflow.streamingfifo.StreamingFIFO', None),
    ('StreamingEltwise', 'finn.custom_op.fpgadataflow.streamingeltwise.StreamingEltwise', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferStreamingEltwise'),
    ('ChannelwiseOp', 'finn.custom_op.fpgadataflow.channelwise_op.ChannelwiseOp', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferChannelwiseLinearLayer'),
    ('Pool', 'finn.custom_op.fpgadataflow.pool.Pool', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferPool'),
    ('Lookup', 'finn.custom_op.fpgadataflow.lookup.Lookup', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferLookupLayer'),
    ('LabelSelect', 'finn.custom_op.fpgadataflow.labelselect.LabelSelect', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferLabelSelectLayer'),
    ('AddStreams', 'finn.custom_op.fpgadataflow.addstreams.AddStreams', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferAddStreamsLayer'),
    ('DuplicateStreams', 'finn.custom_op.fpgadataflow.duplicatestreams.DuplicateStreams', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferDuplicateStreamsLayer'),
    ('FMPadding', 'finn.custom_op.fpgadataflow.fmpadding.FMPadding', None),
    ('FMPadding_Pixel', 'finn.custom_op.fpgadataflow.fmpadding_pixel.FMPadding_Pixel', None),
    ('StreamingConcat', 'finn.custom_op.fpgadataflow.concat.StreamingConcat', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferConcatLayer'),
    ('StreamingSplit', 'finn.custom_op.fpgadataflow.split.StreamingSplit', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferSplitLayer'),
    ('UpsampleNearestNeighbour', 'finn.custom_op.fpgadataflow.upsampler.UpsampleNearestNeighbour', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferUpsample'),
    # ElementwiseBinary operations
    ('ElementwiseBinaryOperation', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseBinaryOperation', 'finn.transformation.fpgadataflow.convert_to_hw_layers.InferElementwiseBinaryOperation'),
    ('ElementwiseAdd', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseAdd', None),
    ('ElementwiseSub', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseSub', None),
    ('ElementwiseMul', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseMul', None),
    ('ElementwiseDiv', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseDiv', None),
    ('ElementwiseAnd', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseAnd', None),
    ('ElementwiseOr', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseOr', None),
    ('ElementwiseXor', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseXor', None),
    ('ElementwiseEqual', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseEqual', None),
    ('ElementwiseLess', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseLess', None),
    ('ElementwiseLessOrEqual', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseLessOrEqual', None),
    ('ElementwiseGreater', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseGreater', None),
    ('ElementwiseGreaterOrEqual', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseGreaterOrEqual', None),
    ('ElementwiseBitwiseAnd', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseBitwiseAnd', None),
    ('ElementwiseBitwiseOr', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseBitwiseOr', None),
    ('ElementwiseBitwiseXor', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseBitwiseXor', None),
]

# FINN backend data: (name, module_path, target_kernel, language)
_FINN_BACKENDS = [
    # HLS backends
    ('MVAU_hls', 'finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls.MVAU_hls', 'MVAU', 'hls'),
    ('Thresholding_hls', 'finn.custom_op.fpgadataflow.hls.thresholding_hls.Thresholding_hls', 'Thresholding', 'hls'),
    ('VVAU_hls', 'finn.custom_op.fpgadataflow.hls.vectorvectoractivation_hls.VVAU_hls', 'VVAU', 'hls'),
    ('AddStreams_hls', 'finn.custom_op.fpgadataflow.hls.addstreams_hls.AddStreams_hls', 'AddStreams', 'hls'),
    ('ChannelwiseOp_hls', 'finn.custom_op.fpgadataflow.hls.channelwise_op_hls.ChannelwiseOp_hls', 'ChannelwiseOp', 'hls'),
    ('StreamingConcat_hls', 'finn.custom_op.fpgadataflow.hls.concat_hls.StreamingConcat_hls', 'StreamingConcat', 'hls'),
    ('StreamingSplit_hls', 'finn.custom_op.fpgadataflow.hls.split_hls.StreamingSplit_hls', 'StreamingSplit', 'hls'),
    ('DuplicateStreams_hls', 'finn.custom_op.fpgadataflow.hls.duplicatestreams_hls.DuplicateStreams_hls', 'DuplicateStreams', 'hls'),
    ('ElementwiseBinaryOperation_hls', 'finn.custom_op.fpgadataflow.hls.elementwise_binary_hls.ElementwiseBinaryOperation_hls', 'ElementwiseBinaryOperation', 'hls'),
    ('FMPadding_Pixel_hls', 'finn.custom_op.fpgadataflow.hls.fmpadding_pixel_hls.FMPadding_Pixel_hls', 'FMPadding_Pixel', 'hls'),
    ('GlobalAccPool_hls', 'finn.custom_op.fpgadataflow.hls.globalaccpool_hls.GlobalAccPool_hls', 'GlobalAccPool', 'hls'),
    ('LabelSelect_hls', 'finn.custom_op.fpgadataflow.hls.labelselect_hls.LabelSelect_hls', 'LabelSelect', 'hls'),
    ('Lookup_hls', 'finn.custom_op.fpgadataflow.hls.lookup_hls.Lookup_hls', 'Lookup', 'hls'),
    ('Pool_hls', 'finn.custom_op.fpgadataflow.hls.pool_hls.Pool_hls', 'Pool', 'hls'),
    ('StreamingDataWidthConverter_hls', 'finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls.StreamingDataWidthConverter_hls', 'StreamingDataWidthConverter', 'hls'),
    ('StreamingEltwise_hls', 'finn.custom_op.fpgadataflow.hls.streamingeltwise_hls.StreamingEltwise_hls', 'StreamingEltwise', 'hls'),
    ('UpsampleNearestNeighbour_hls', 'finn.custom_op.fpgadataflow.hls.upsampler_hls.UpsampleNearestNeighbour_hls', 'UpsampleNearestNeighbour', 'hls'),
    # RTL backends
    ('ConvolutionInputGenerator_rtl', 'finn.custom_op.fpgadataflow.rtl.convolutioninputgenerator_rtl.ConvolutionInputGenerator_rtl', 'ConvolutionInputGenerator', 'rtl'),
    ('FMPadding_rtl', 'finn.custom_op.fpgadataflow.rtl.fmpadding_rtl.FMPadding_rtl', 'FMPadding', 'rtl'),
    ('MVAU_rtl', 'finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl.MVAU_rtl', 'MVAU', 'rtl'),
    ('StreamingDataWidthConverter_rtl', 'finn.custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl.StreamingDataWidthConverter_rtl', 'StreamingDataWidthConverter', 'rtl'),
    ('StreamingFIFO_rtl', 'finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl.StreamingFIFO_rtl', 'StreamingFIFO', 'rtl'),
    ('Thresholding_rtl', 'finn.custom_op.fpgadataflow.rtl.thresholding_rtl.Thresholding_rtl', 'Thresholding', 'rtl'),
    ('VVAU_rtl', 'finn.custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl.VVAU_rtl', 'VVAU', 'rtl'),
]


def _import_class(module_path_and_class: str):
    """Dynamically import a class from module.path.ClassName string."""
    module_path, class_name = module_path_and_class.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _lazy_import_class(module_path_and_class: str):
    """Return a callable that lazily imports the class when called."""
    def _importer():
        return _import_class(module_path_and_class)
    return _importer


# Register FINN kernels with LAZY imports (don't import until used)
# This dramatically speeds up startup by deferring ~35 module imports
for name, class_path, infer_path in _FINN_KERNELS:
    try:
        # Store import paths, not actual classes
        # Registry will import on-demand when get_kernel() is called
        from brainsmith.core import registry as reg

        # Create lazy loader that imports when accessed
        kernel_loader = _lazy_import_class(class_path)
        infer_loader = _lazy_import_class(infer_path) if infer_path else None

        # Store metadata with lazy loaders
        full_name = f'finn:{name}'
        reg._kernels[full_name] = {
            'class': kernel_loader,  # Callable that returns class
            'infer': infer_loader,   # Callable that returns class
            'op_type': name,
            'domain': 'finn.custom',
            '_lazy': True  # Mark as lazy so loader knows to call it
        }
    except Exception as e:
        logger.warning(f"Failed to register FINN kernel {name}: {e}")


# Register FINN backends with LAZY imports
for name, class_path, target_kernel, language in _FINN_BACKENDS:
    try:
        from brainsmith.core import registry as reg

        backend_loader = _lazy_import_class(class_path)

        full_name = f'finn:{name}'
        reg._backends[full_name] = {
            'class': backend_loader,  # Callable that returns class
            'target_kernel': f'finn:{target_kernel}',
            'language': language,
            'variant': None,
            '_lazy': True  # Mark as lazy
        }
    except Exception as e:
        logger.warning(f"Failed to register FINN backend {name}: {e}")
