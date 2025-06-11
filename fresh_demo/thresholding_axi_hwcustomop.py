############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# AUTO-GENERATED: AutoThresholdingAxiHWCustomOp for thresholding_axi
# Generated: 2025-06-11T05:02:22.625196
# Generator: Unified HWKG with Interface-Wise Dataflow Modeling
#
# DATAFLOW-MODEL-POWERED HARDWARE COMPONENT
# This HWCustomOp uses DataflowModel for mathematical foundation.
# All performance calculations are mathematically derived, not placeholders.
############################################################################

import numpy as np
from typing import Dict, Any, List, Optional

# Import unified dataflow components
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.rtl_integration import create_interface_metadata
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType

# Try to import FINN components (optional for development)
try:
    from qonnx.core.datatype import DataType
    FINN_AVAILABLE = True
except ImportError:
    FINN_AVAILABLE = False


class AutoThresholdingAxiHWCustomOp(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for thresholding_axi kernel.
    
    This class uses the unified HWKG approach with Interface-Wise Dataflow Modeling:
    - DataflowModel provides mathematical foundation for all calculations
    - No placeholders or mocks - all methods have real implementations
    - Runtime dimension extraction from ONNX model when available
    - Automatic performance and resource analysis
    
    Interfaces (4 total):
- ap: config (UINT1)
- s_axis: input (UINT7)
- m_axis: output (UINT7)
- s_axilite: config (UINT31)
    """
    
    def __init__(self, onnx_node, **kwargs):
        """
        Initialize AutoThresholdingAxiHWCustomOp with DataflowModel-based interface metadata.
        
        This uses the simplified 3-tier architecture:
        - Tier 1 (Kernel Data): Interface metadata from RTL parsing
        - Tier 2 (Model Data): Tensor shapes from ONNX when available  
        - Tier 3 (Parallelism): Runtime parallelism configuration
        """
        
        # Create interface metadata using unified RTL integration
        interface_metadata = [
            create_interface_metadata(
                name="ap",
                interface_type="config",
                chunking_strategy={
                    'type': 'interface_based',
                    'tensor_dims': [128],
                    'block_dims': [128]
                },
                dtype_constraints={
                    'finn_type': 'UINT1',
                    'base_type': 'UINT',
                    'bitwidth': 1,
                    'signed': false
                },
                axi_metadata={
                    'protocol': 'axi_stream',
                    'data_width': 1
                }
            ),
            create_interface_metadata(
                name="s_axis",
                interface_type="input",
                chunking_strategy={
                    'type': 'interface_based',
                    'tensor_dims': [7],
                    'block_dims': [7]
                },
                dtype_constraints={
                    'finn_type': 'UINT7',
                    'base_type': 'UINT',
                    'bitwidth': 7,
                    'signed': false
                },
                axi_metadata={
                    'protocol': 'axi_stream',
                    'data_width': 7
                }
            ),
            create_interface_metadata(
                name="m_axis",
                interface_type="output",
                chunking_strategy={
                    'type': 'interface_based',
                    'tensor_dims': [7],
                    'block_dims': [7]
                },
                dtype_constraints={
                    'finn_type': 'UINT7',
                    'base_type': 'UINT',
                    'bitwidth': 7,
                    'signed': false
                },
                axi_metadata={
                    'protocol': 'axi_stream',
                    'data_width': 7
                }
            ),
            create_interface_metadata(
                name="s_axilite",
                interface_type="config",
                chunking_strategy={
                    'type': 'interface_based',
                    'tensor_dims': [31],
                    'block_dims': [31]
                },
                dtype_constraints={
                    'finn_type': 'UINT31',
                    'base_type': 'UINT',
                    'bitwidth': 31,
                    'signed': false
                },
                axi_metadata={
                    'protocol': 'axi_stream',
                    'data_width': 31
                }
            ),
        ]
        
        # Initialize AutoHWCustomOp with interface metadata
        # The base class handles DataflowModel creation and mathematical calculations
        super().__init__(onnx_node, interface_metadata, **kwargs)
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """
        Get node attribute types with dataflow enhancements.
        
        Uses AutoHWCustomOp's enhanced attribute system with mathematical
        foundation rather than placeholder values.
        """
        # Get enhanced attributes from AutoHWCustomOp base class
        attrs = super().get_enhanced_nodeattr_types()
        
        # Add kernel-specific attributes if needed
        kernel_specific_attrs = {
            # Add any thresholding_axi-specific attributes here
            "kernel_name": ("s", False, "thresholding_axi"),
            "generation_method": ("s", False, "unified_hwkg_dataflow_modeling"),
        }
        
        attrs.update(kernel_specific_attrs)
        return attrs
    
    # All other methods (get_exp_cycles, get_instream_width, etc.) are inherited
    # from AutoHWCustomOp and use the DataflowModel for mathematical calculations.
    # No placeholder implementations needed!
    
    def derive_characteristic_fxns(self) -> Dict[str, Any]:
        """
        Derive characteristic functions using DataflowModel.
        
        This overrides the base implementation to add any thresholding_axi-specific
        characteristics while maintaining the mathematical foundation.
        """
        # Get base characteristics from DataflowModel
        base_characteristics = super().derive_characteristic_fxns()
        
        # Add kernel-specific characteristics
        kernel_characteristics = {
            "kernel_type": "thresholding_axi",
            "dataflow_model_interfaces": 4,
            "unified_hwkg_generated": True,
        }
        
        base_characteristics.update(kernel_characteristics)
        return base_characteristics


# Factory function for easy instantiation
def create_thresholding_axi_hwcustomop(onnx_node, **kwargs) -> AutoThresholdingAxiHWCustomOp:
    """
    Factory function for creating AutoThresholdingAxiHWCustomOp instances.
    
    Args:
        onnx_node: ONNX node for this operation
        **kwargs: Additional arguments
        
    Returns:
        AutoThresholdingAxiHWCustomOp: Configured HWCustomOp instance
    """
    return AutoThresholdingAxiHWCustomOp(onnx_node, **kwargs)


# Export the main class for FINN integration
__all__ = ["AutoThresholdingAxiHWCustomOp", "create_thresholding_axi_hwcustomop"]