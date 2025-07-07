#!/usr/bin/env python3
"""
Verify FINN kernel imports and generate corrected registration list.
"""

import sys
import os

# Add deps to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../deps/finn/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../deps/qonnx/src'))

# Import current registrations from framework_adapters
from framework_adapters import FINN_KERNELS as CURRENT_FINN_KERNELS

# Discovered kernels to test based on filesystem analysis
DISCOVERED_KERNELS = [
    ('Thresholding', 'finn.custom_op.fpgadataflow.thresholding.Thresholding'),
    ('MatrixVectorActivation', 'finn.custom_op.fpgadataflow.matrixvectoractivation.MatrixVectorActivation'),
    ('VectorVectorActivation', 'finn.custom_op.fpgadataflow.vectorvectoractivation.VectorVectorActivation'),
    ('ConvolutionInputGenerator', 'finn.custom_op.fpgadataflow.convolutioninputgenerator.ConvolutionInputGenerator'),
    ('StreamingDataWidthConverter', 'finn.custom_op.fpgadataflow.streamingdatawidthconverter.StreamingDataWidthConverter'),
    ('GlobalAccPool', 'finn.custom_op.fpgadataflow.globalaccpool.GlobalAccPool'),
    ('StreamingMaxPool', 'finn.custom_op.fpgadataflow.streamingmaxpool.StreamingMaxPool'),
    ('StreamingFIFO', 'finn.custom_op.fpgadataflow.streamingfifo.StreamingFIFO'),
    ('StreamingEltwise', 'finn.custom_op.fpgadataflow.streamingeltwise.StreamingEltwise'),
    ('ChannelwiseOp', 'finn.custom_op.fpgadataflow.channelwise_op.ChannelwiseOp'),
    ('Pool', 'finn.custom_op.fpgadataflow.pool.Pool'),
    ('Lookup', 'finn.custom_op.fpgadataflow.lookup.Lookup'),
    ('LabelSelect', 'finn.custom_op.fpgadataflow.labelselect.LabelSelect'),
    ('ElementwiseBinary', 'finn.custom_op.fpgadataflow.elementwise_binary.ElementwiseBinary'),
    ('AddStreams', 'finn.custom_op.fpgadataflow.addstreams.AddStreams'),
    ('DuplicateStreams', 'finn.custom_op.fpgadataflow.duplicatestreams.DuplicateStreams'),
    ('Concat', 'finn.custom_op.fpgadataflow.concat.Concat'),
    ('Downsampler', 'finn.custom_op.fpgadataflow.downsampler.Downsampler'),
    ('Upsampler', 'finn.custom_op.fpgadataflow.upsampler.Upsampler'),
    ('FMPadding', 'finn.custom_op.fpgadataflow.fmpadding.FMPadding'),
    ('FMPadding_Pixel', 'finn.custom_op.fpgadataflow.fmpadding_pixel.FMPadding_Pixel'),
    ('StreamingDataflowPartition', 'finn.custom_op.fpgadataflow.streamingdataflowpartition.StreamingDataflowPartition'),
]

def test_import(name, class_path):
    """Test if a class can be imported."""
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        return True, cls
    except ImportError as e:
        return False, f"ImportError: {e}"
    except AttributeError as e:
        return False, f"AttributeError: {e}"
    except Exception as e:
        return False, f"Exception: {e}"

def main():
    print("=== FINN Kernel Import Verification ===\n")
    
    print("Testing current registrations:")
    print("-" * 60)
    for name, path in CURRENT_FINN_KERNELS:
        success, result = test_import(name, path)
        status = "✓" if success else "✗"
        print(f"{status} {name:20} {path}")
        if not success:
            print(f"  └─ {result}")
    
    print("\n\nTesting discovered kernels:")
    print("-" * 60)
    working_kernels = []
    for name, path in DISCOVERED_KERNELS:
        success, result = test_import(name, path)
        status = "✓" if success else "✗"
        print(f"{status} {name:25} {path}")
        if not success:
            print(f"  └─ {result}")
        else:
            working_kernels.append((name, path))
    
    print("\n\nWorking kernels for registration:")
    print("-" * 60)
    print("FINN_KERNELS = [")
    for name, path in working_kernels:
        print(f"    ('{name}', '{path}'),")
    print("]")
    
    print(f"\n\nSummary:")
    print(f"- Current registrations: {sum(1 for n, p in CURRENT_FINN_KERNELS if test_import(n, p)[0])}/{len(CURRENT_FINN_KERNELS)} working")
    print(f"- Discovered kernels: {len(working_kernels)}/{len(DISCOVERED_KERNELS)} working")

if __name__ == "__main__":
    main()