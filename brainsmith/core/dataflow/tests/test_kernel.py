############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for Kernel class"""

import pytest
from brainsmith.core.dataflow.core.types import InterfaceDirection, INT16, INT32
from brainsmith.core.dataflow.core.interface import Interface
from brainsmith.core.dataflow.core.pragma import TiePragma, ConstrPragma
from brainsmith.core.dataflow.core.kernel import Kernel


class TestKernelCreation:
    """Test kernel creation and basic properties"""
    
    def test_basic_creation(self):
        """Test basic kernel creation"""
        kernel = Kernel(
            name="test_kernel",
            latency_cycles=(100, 80),
            priming_cycles=10,
            flush_cycles=5
        )
        
        assert kernel.name == "test_kernel"
        assert kernel.hw_module == "test_kernel"  # Defaults to name
        assert kernel.latency_cycles == (100, 80)
        assert kernel.priming_cycles == 10
        assert kernel.flush_cycles == 5
        assert kernel.interfaces == []
        assert kernel.pragmas == []
    
    def test_kernel_with_interfaces(self):
        """Test kernel with interfaces"""
        interfaces = [
            Interface(
                name="in",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(32, 64),
                block_dims=(32, 64),
                stream_dims=(4, 8)
            ),
            Interface(
                name="out",
                direction=InterfaceDirection.OUTPUT,
                dtype=INT16,
                tensor_dims=(32, 64),
                block_dims=(32, 64),
                stream_dims=(4, 8)
            )
        ]
        
        kernel = Kernel(
            name="passthrough",
            hw_module="passthrough_rtl",
            interfaces=interfaces,
            latency_cycles=(50, 50)
        )
        
        assert len(kernel.interfaces) == 2
        assert kernel.hw_module == "passthrough_rtl"
        assert len(kernel.input_interfaces) == 1
        assert len(kernel.output_interfaces) == 1
        assert len(kernel.weight_interfaces) == 0
    
    def test_validation_errors(self):
        """Test kernel validation"""
        # Duplicate interface names
        with pytest.raises(ValueError, match="Duplicate interface names"):
            Kernel(
                name="bad",
                interfaces=[
                    Interface("dup", InterfaceDirection.INPUT, INT16, (32,), (32,)),
                    Interface("dup", InterfaceDirection.OUTPUT, INT16, (32,), (32,))
                ]
            )
        
        # Invalid latency
        with pytest.raises(ValueError, match="Worst-case latency"):
            Kernel(
                name="bad",
                latency_cycles=(50, 100)  # worst < average
            )
        
        # Negative pipeline costs
        with pytest.raises(ValueError, match="non-negative"):
            Kernel(
                name="bad",
                priming_cycles=-1
            )


class TestKernelInterfaces:
    """Test kernel interface management"""
    
    def setup_method(self):
        """Create test kernel with various interfaces"""
        self.kernel = Kernel(
            name="complex",
            interfaces=[
                Interface("input1", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("input2", InterfaceDirection.INPUT, INT16, (64,), (64,)),
                Interface("weights", InterfaceDirection.WEIGHT, INT16, (64, 64), (64, 64)),
                Interface("bias", InterfaceDirection.WEIGHT, INT16, (64,), (64,)),
                Interface("output", InterfaceDirection.OUTPUT, INT32, (64,), (64,)),
                Interface("config", InterfaceDirection.CONFIG, INT16, (4,), (4,))
            ]
        )
    
    def test_interface_filtering(self):
        """Test interface filtering by type"""
        assert len(self.kernel.input_interfaces) == 2
        assert len(self.kernel.weight_interfaces) == 2
        assert len(self.kernel.output_interfaces) == 1
        assert len(self.kernel.config_interfaces) == 1
        
        assert self.kernel.input_interfaces[0].name == "input1"
        assert self.kernel.weight_interfaces[0].name == "weights"
        assert self.kernel.has_weights == True
    
    def test_get_interface(self):
        """Test getting interface by name"""
        intf = self.kernel.get_interface("weights")
        assert intf.name == "weights"
        assert intf.direction == InterfaceDirection.WEIGHT
        
        # Non-existent interface
        with pytest.raises(KeyError):
            self.kernel.get_interface("nonexistent")


class TestKernelTiming:
    """Test kernel timing calculations"""
    
    def test_initiation_interval(self):
        """Test II calculation"""
        # Default kernel with no interfaces
        kernel = Kernel(name="empty")
        assert kernel.initiation_interval() == 1
        
        # With calculation_ii specified
        kernel = Kernel(name="test", calculation_ii=10)
        assert kernel.initiation_interval() == 10
        
        # Based on input interfaces
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface(
                    name="in1",
                    direction=InterfaceDirection.INPUT,
                    dtype=INT16,
                    tensor_dims=(100,),
                    block_dims=(32,),
                    stream_dims=(8,)
                ),
                Interface(
                    name="in2", 
                    direction=InterfaceDirection.INPUT,
                    dtype=INT16,
                    tensor_dims=(100,),
                    block_dims=(40,),
                    stream_dims=(8,)
                )
            ]
        )
        # in1: 32/8 = 4 cycles
        # in2: 40/8 = 5 cycles
        assert kernel.initiation_interval() == 5  # max
    
    def test_execution_interval(self):
        """Test execution interval calculation"""
        # No weights
        kernel = Kernel(name="test", calculation_ii=10)
        assert kernel.execution_interval() == 10
        
        # With weights
        kernel = Kernel(
            name="test",
            calculation_ii=10,
            interfaces=[
                Interface(
                    name="weights",
                    direction=InterfaceDirection.WEIGHT,
                    dtype=INT16,
                    tensor_dims=(256, 512),
                    block_dims=(32, 512),
                    stream_dims=(8, 16)
                )
            ]
        )
        # Weight has 256/32 = 8 blocks
        assert kernel.execution_interval() == 10 * 8  # Default behavior
        
        # Override with execution_ii
        kernel = Kernel(name="test", calculation_ii=10, execution_ii=50)
        assert kernel.execution_interval() == 50
    
    def test_inference_latency(self):
        """Test inference latency calculation"""
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface(
                    name="input",
                    direction=InterfaceDirection.INPUT,
                    dtype=INT16,
                    tensor_dims=(128,),
                    block_dims=(32,),
                    stream_dims=(8,)
                )
            ],
            calculation_ii=4,
            execution_ii=4,  # No weights
            priming_cycles=20,
            flush_cycles=10
        )
        
        # Single inference: 4 input blocks * 4 cycles + pipeline
        latency = kernel.inference_latency(batch_size=1)
        assert latency == 20 + (4 * 4) + 10  # 46
        
        # Batch inference
        latency = kernel.inference_latency(batch_size=4)
        assert latency == 20 + (16 * 4) + 10  # 94
    
    def test_throughput(self):
        """Test throughput calculation"""
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface(
                    name="input",
                    direction=InterfaceDirection.INPUT,
                    dtype=INT16,
                    tensor_dims=(100,),
                    block_dims=(25,),
                    stream_dims=(5,)
                )
            ],
            execution_ii=20
        )
        
        # 4 blocks * 20 cycles = 80 cycles per inference
        # At 100 MHz: 100e6 / 80 = 1.25M inferences/sec
        throughput = kernel.throughput(clock_freq_mhz=100.0)
        assert throughput == 1.25e6


class TestKernelPragmas:
    """Test kernel pragma validation"""
    
    def test_pragma_validation(self):
        """Test pragma constraint validation"""
        kernel = Kernel(
            name="matmul",
            interfaces=[
                Interface("vec", InterfaceDirection.INPUT, INT16, (512,), (512,), (16,)),
                Interface("mat", InterfaceDirection.WEIGHT, INT16, (256, 512), (256, 512), (8, 16)),
                Interface("out", InterfaceDirection.OUTPUT, INT32, (256,), (256,), (8,))
            ],
            pragmas=[
                TiePragma("mat[1]", "vec[0]"),  # Matrix cols == vector size
                ConstrPragma("vec[0]", "%", "SIMD"),
                ConstrPragma("mat[0]", "%", "PE")
            ],
            pragma_env={"SIMD": 16, "PE": 8}
        )
        
        # Should validate successfully
        kernel.validate()
        
        # Add failing pragma
        kernel.pragmas.append(ConstrPragma("out[0]", "=", 512))  # out[0] is 256
        
        with pytest.raises(ValueError, match="Pragma violation"):
            kernel.validate()
    
    def test_pragma_error_handling(self):
        """Test pragma error reporting"""
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface("data", InterfaceDirection.INPUT, INT16, (32,), (32,))
            ],
            pragmas=[
                ConstrPragma("data[0]", "=", "UNKNOWN")  # Unknown symbol
            ],
            pragma_env={}
        )
        
        with pytest.raises(ValueError, match="Unknown symbol"):
            kernel.validate()


class TestKernelResources:
    """Test resource estimation"""
    
    def test_bandwidth_requirements(self):
        """Test bandwidth calculation"""
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface("in1", InterfaceDirection.INPUT, INT16, (64,), (64,), (8,)),
                Interface("in2", InterfaceDirection.INPUT, INT16, (64,), (64,), (4,)),
                Interface("cfg", InterfaceDirection.CONFIG, INT16, (4,), (4,), (1,)),
                Interface("out", InterfaceDirection.OUTPUT, INT32, (64,), (64,), (8,))
            ]
        )
        
        bandwidth = kernel.bandwidth_requirements()
        
        assert bandwidth["in1"] == 8 * 16  # 128 bits/cycle
        assert bandwidth["in2"] == 4 * 16  # 64 bits/cycle
        assert "cfg" not in bandwidth  # Config not included
        assert bandwidth["out"] == 8 * 32  # 256 bits/cycle
    
    def test_resource_estimation(self):
        """Test resource estimation"""
        # With explicit resources
        kernel = Kernel(
            name="test",
            resources={"LUT": 1000, "FF": 500, "DSP": 8, "BRAM": 2}
        )
        
        resources = kernel.estimate_resources()
        assert resources["LUT"] == 1000
        assert resources["DSP"] == 8
        
        # Automatic estimation
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (64,), (64,), (8,)),
                Interface("weight", InterfaceDirection.WEIGHT, INT16, (64, 64), (64, 64), (8, 8)),
                Interface("out", InterfaceDirection.OUTPUT, INT32, (64,), (64,), (8,))
            ]
        )
        
        resources = kernel.estimate_resources()
        assert resources["LUT"] > 0
        assert resources["DSP"] > 0  # Should have DSPs for weight multiply


class TestKernelTransformations:
    """Test kernel transformations"""
    
    def test_apply_parallelism(self):
        """Test applying parallelism configuration"""
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (512,), (512,), (8,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (512,), (512,), (8,))
            ]
        )
        
        # Double parallelism
        config = {"in": 16, "out": 16}
        new_kernel = kernel.apply_parallelism(config)
        
        assert new_kernel.name == "test"
        assert new_kernel.get_interface("in").ipar == 16
        assert new_kernel.get_interface("out").ipar == 16
        
        # Original unchanged
        assert kernel.get_interface("in").ipar == 8
    
    def test_adfg_rates(self):
        """Test ADFG rate conversion"""
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface(
                    name="in",
                    direction=InterfaceDirection.INPUT,
                    dtype=INT16,
                    tensor_dims=(100,),
                    block_dims=[(32,), (32,), (36,)],  # CSDF
                    stream_dims=(8,)
                ),
                Interface(
                    name="cfg",
                    direction=InterfaceDirection.CONFIG,
                    dtype=INT16,
                    tensor_dims=(4,),
                    block_dims=(4,)
                ),
                Interface(
                    name="out",
                    direction=InterfaceDirection.OUTPUT,
                    dtype=INT16,
                    tensor_dims=(100,),
                    block_dims=(100,),
                    stream_dims=(10,)
                )
            ]
        )
        
        rates = kernel.to_adfg_rates()
        
        assert "in" in rates
        assert rates["in"] == [4, 4, 4]  # 32/8, 32/8, 36/8 (integer division)
        assert "cfg" not in rates  # Config excluded
        assert "out" in rates
        assert rates["out"] == [10]  # 100/10


class TestKernelIntegration:
    """Integration tests with complete kernel examples"""
    
    def test_matrix_multiply_kernel(self):
        """Test complete matrix multiply kernel"""
        kernel = Kernel(
            name="MatMul",
            hw_module="matmul_rtl",
            interfaces=[
                Interface("vec", InterfaceDirection.INPUT, INT16, (1, 512), (1, 512), (1, 16)),
                Interface("mat", InterfaceDirection.WEIGHT, INT16, (256, 512), (8, 512), (8, 16)),
                Interface("out", InterfaceDirection.OUTPUT, INT32, (1, 256), (1, 256), (1, 8))
            ],
            latency_cycles=(1000, 800),
            calculation_ii=32,  # 512/16 cycles
            execution_ii=1024,  # 32 * 32 weight blocks
            priming_cycles=64,
            flush_cycles=32,
            pragmas=[
                TiePragma("mat[1]", "vec[1]"),
                ConstrPragma("vec[1]", "%", "BURST"),
                ConstrPragma("mat[0]", "%", "PE")  # Changed to modulo check
            ],
            pragma_env={"BURST": 64, "PE": 8}
        )
        
        # Validate
        kernel.validate()
        
        # Check properties
        assert kernel.has_weights
        assert kernel.is_stateful
        assert len(kernel.input_interfaces) == 1
        assert len(kernel.weight_interfaces) == 1
        
        # Timing
        assert kernel.initiation_interval() == 32
        assert kernel.execution_interval() == 1024
        
        # Single inference latency
        latency = kernel.inference_latency()
        assert latency == 64 + 1024 + 32  # priming + exec + flush
        
        # Throughput at 200 MHz
        throughput = kernel.throughput(clock_freq_mhz=200.0)
        assert throughput == pytest.approx(200e6 / 1024)