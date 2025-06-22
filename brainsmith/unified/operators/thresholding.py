############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Unified Thresholding Operator

Reference implementation of a thresholding operator using the unified framework.
"""

import numpy as np
from typing import Dict, Any, List

from qonnx.core.datatype import DataType

from brainsmith.core.dataflow import Shape
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.unified.core import (
    UnifiedHWCustomOp, UnifiedRTLBackend, UnifiedDSEMixin,
    KernelDefinition, InterfaceDefinition, DatatypeConstraint,
    PerformanceModel, ResourceModel
)


class UnifiedThresholding(UnifiedHWCustomOp, UnifiedRTLBackend, UnifiedDSEMixin):
    """
    Thresholding operator using unified framework.
    
    This operator implements multi-threshold comparison with configurable
    activation functions.
    """
    
    @property
    def finn_rtllib_module(self) -> str:
        """Return finn-rtllib module name."""
        return "thresholding"
    
    def get_kernel_definition(self) -> KernelDefinition:
        """Define thresholding kernel."""
        # Define interfaces
        interfaces = [
            InterfaceDefinition(
                name="input",
                type=InterfaceType.INPUT,
                datatype_constraints=[
                    DatatypeConstraint("INT", 4, 16),
                    DatatypeConstraint("UINT", 4, 16)
                ],
                parameter_links={
                    'BDIM': 'N_INPUTS',
                    'SDIM': 'input_parallelism',
                    'datatype': 'input_datatype'
                }
            ),
            InterfaceDefinition(
                name="threshold",
                type=InterfaceType.WEIGHT,
                datatype_constraints=[
                    DatatypeConstraint("INT", 8, 32),
                    DatatypeConstraint("UINT", 8, 32)
                ],
                parameter_links={
                    'BDIM': 'N_THRESHOLDS',
                    'SDIM': 'threshold_parallelism',
                    'datatype': 'threshold_datatype'
                }
            ),
            InterfaceDefinition(
                name="output",
                type=InterfaceType.OUTPUT,
                datatype_constraints=[
                    DatatypeConstraint("UINT", 1, 8),
                    DatatypeConstraint("INT", 2, 8)
                ],
                parameter_links={
                    'BDIM': 'N_OUTPUTS',
                    'SDIM': 'output_parallelism',
                    'datatype': 'output_datatype'
                }
            )
        ]
        
        # Define constraints
        constraints = [
            # Output shape must match input shape
            "output.tensor_shape == input.tensor_shape",
            # Threshold dimension must match last input dimension
            "threshold.tensor_shape[0] == input.tensor_shape[-1]",
            # Number of thresholds determines output bit width
            "threshold.tensor_shape[1] == 2^output_datatype.bitwidth - 1"
        ]
        
        # Define performance model
        performance_model = PerformanceModel(
            base_latency=3,  # Pipeline depth
            throughput_scaling={
                'input': 1.0,  # Linear scaling with input parallelism
                'threshold': 0.5  # Threshold parallelism has less impact
            }
        )
        
        # Define resource model
        resource_model = ResourceModel(
            base_luts=100,
            base_ffs=50,
            base_brams=0,  # No BRAM in base config
            base_dsps=0,   # No DSPs needed
            lut_scaling={
                'input': 20,  # LUTs per input channel
                'threshold': 10  # LUTs per threshold channel
            },
            ff_scaling={
                'input': 10,
                'threshold': 5
            }
        )
        
        return KernelDefinition(
            name="thresholding",
            version="1.0",
            description="Multi-threshold comparison with configurable activation",
            interfaces=interfaces,
            constraints=constraints,
            rtl_parameters={
                "N_INPUTS": "input.block_shape[0]",
                "N_THRESHOLDS": "threshold.block_shape[0]",
                "N_OUTPUTS": "output.block_shape[0]",
                "BIAS": 0,
                "ACTIVATION_TYPE": 0  # 0=binary, 1=relu, 2=custom
            },
            exposed_parameters={"BIAS", "ACTIVATION_TYPE"},
            parameter_defaults={
                "BIAS": 0,
                "ACTIVATION_TYPE": 0
            },
            parameter_ranges={
                "BIAS": (-128, 127),
                "ACTIVATION_TYPE": (0, 2)
            },
            performance_model=performance_model,
            resource_model=resource_model,
            metadata={
                'dse_constraints': {
                    'max_input_parallelism': 64,
                    'max_threshold_parallelism': 16
                }
            }
        )
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """Define node attributes."""
        attrs = super().get_nodeattr_types()
        attrs.update({
            # Algorithm parameters
            "activation": ("s", False, "binary", {"binary", "relu", "custom"}),
            "per_channel_scaling": ("i", False, 0, {0, 1}),
            
            # Memory optimization
            "use_bram": ("i", False, 0, {0, 1}),
            "bram_threshold": ("i", False, 256),  # Use BRAM if thresholds > this
            
            # Precision control
            "signed_input": ("i", False, 0, {0, 1}),
            "num_thresholds": ("i", False, 1)  # Number of threshold levels
        })
        return attrs
    
    def execute_node(self, context, graph):
        """Execute thresholding operation."""
        # Auto-optimize if requested
        if self.get_nodeattr("auto_optimize"):
            self.optimize_for_target({
                "optimization_objective": self.get_nodeattr("optimization_objective")
            })
        
        # Get inputs
        input_tensor = context[self.onnx_node.input[0]]
        
        # Get or generate thresholds
        if len(self.onnx_node.input) > 1:
            threshold_tensor = context[self.onnx_node.input[1]]
        else:
            # Generate default thresholds
            threshold_tensor = self._generate_default_thresholds(input_tensor)
        
        # Apply thresholding
        output_tensor = self._apply_thresholding(
            input_tensor,
            threshold_tensor,
            bias=self.get_nodeattr("BIAS"),
            activation=self.get_nodeattr("activation")
        )
        
        # Store output
        context[self.onnx_node.output[0]] = output_tensor
    
    def _generate_default_thresholds(self, input_tensor: np.ndarray) -> np.ndarray:
        """Generate default threshold values."""
        num_thresholds = self.get_nodeattr("num_thresholds")
        input_shape = input_tensor.shape
        channels = input_shape[-1] if len(input_shape) > 1 else 1
        
        # Generate evenly spaced thresholds
        input_range = input_tensor.max() - input_tensor.min()
        step = input_range / (num_thresholds + 1)
        
        thresholds = np.zeros((channels, num_thresholds))
        for i in range(num_thresholds):
            thresholds[:, i] = input_tensor.min() + step * (i + 1)
        
        return thresholds
    
    def _apply_thresholding(self, input_data: np.ndarray, 
                          thresholds: np.ndarray,
                          bias: int = 0,
                          activation: str = "binary") -> np.ndarray:
        """Apply multi-threshold comparison."""
        # Add bias
        input_biased = input_data + bias
        
        # Reshape for broadcasting
        input_shape = input_biased.shape
        batch_size = np.prod(input_shape[:-1])
        channels = input_shape[-1]
        
        input_flat = input_biased.reshape(batch_size, channels)
        
        # Apply thresholds
        num_thresholds = thresholds.shape[1]
        output_flat = np.zeros((batch_size, channels), dtype=np.uint8)
        
        for b in range(batch_size):
            for c in range(channels):
                value = input_flat[b, c]
                # Count how many thresholds are exceeded
                level = 0
                for t in range(num_thresholds):
                    if value > thresholds[c, t]:
                        level += 1
                
                # Apply activation function
                if activation == "binary":
                    output_flat[b, c] = 1 if level > 0 else 0
                elif activation == "relu":
                    output_flat[b, c] = level
                else:  # custom
                    output_flat[b, c] = level
        
        # Reshape to original shape
        return output_flat.reshape(input_shape)
    
    def bram_estimation(self):
        """Estimate BRAM usage."""
        if not self.get_nodeattr("use_bram"):
            return 0
        
        # Get threshold interface
        threshold_intf = None
        for intf in self.kernel.interfaces:
            if intf.name == "threshold":
                threshold_intf = intf
                break
        
        if not threshold_intf:
            return 0
        
        # Calculate threshold storage requirements
        num_thresholds = np.prod(threshold_intf.tensor_shape)
        threshold_bits = self.get_nodeattr("threshold_datatype")
        if threshold_bits:
            dt = DataType[threshold_bits]
            bits_per_threshold = dt.bitwidth()
        else:
            bits_per_threshold = 16  # Default
        
        total_bits = num_thresholds * bits_per_threshold
        
        # Convert to BRAM18K units (18Kb each)
        brams_needed = (total_bits + 18431) // 18432  # Round up
        
        return brams_needed
    
    def lut_estimation(self):
        """Estimate LUT usage."""
        if not hasattr(self, 'kernel_def') or not self.kernel_def.resource_model:
            return 100  # Default estimate
        
        model = self.kernel_def.resource_model
        luts = model.base_luts
        
        # Add scaling based on parallelism
        if hasattr(self, '_current_config') and self._current_config:
            kernel_config = self._current_config.kernel_configs[self.kernel.name]
            
            for intf_name, scaling in model.lut_scaling.items():
                parallelism = kernel_config.interface_parallelism.get(intf_name, 1)
                luts += parallelism * scaling
        
        return int(luts)
    
    def get_exp_cycles(self):
        """Get expected cycles for operation."""
        # Base latency from performance model
        if hasattr(self, 'kernel_def') and self.kernel_def.performance_model:
            base_latency = self.kernel_def.performance_model.base_latency
        else:
            base_latency = 3
        
        # Add cycles based on data volume and parallelism
        input_intf = self.kernel.get_interfaces_by_type(InterfaceType.INPUT)[0]
        total_elements = np.prod(input_intf.tensor_shape)
        
        if hasattr(self, '_current_config') and self._current_config:
            kernel_config = self._current_config.kernel_configs[self.kernel.name]
            parallelism = kernel_config.interface_parallelism.get('input', 1)
            cycles = base_latency + (total_elements // parallelism)
        else:
            cycles = base_latency + total_elements
        
        return cycles