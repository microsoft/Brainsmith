############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for DSE configuration management"""

import pytest
from brainsmith.core.dataflow.types import INT16, InterfaceDirection
from brainsmith.core.dataflow.interface import Interface
from brainsmith.core.dataflow.kernel import Kernel
from brainsmith.core.dataflow.dse.config import (
    ParallelismConfig, DSEConstraints, ConfigurationSpace
)


class TestParallelismConfig:
    """Test parallelism configuration"""
    
    def test_basic_config(self):
        """Test basic configuration creation"""
        config = ParallelismConfig(
            interface_pars={("matmul", "input"): 16},
            global_par=8
        )
        
        # Specific interface
        assert config.get_parallelism("matmul", "input") == 16
        
        # Global fallback
        assert config.get_parallelism("matmul", "weight") == 8
        assert config.get_parallelism("other", "data") == 8
        
        # No global, no specific -> default
        config2 = ParallelismConfig()
        assert config2.get_parallelism("any", "intf") == 1
    
    def test_apply_to_kernel(self):
        """Test applying configuration to kernel"""
        kernel = Kernel(
            name="conv",
            interfaces=[
                Interface("input", InterfaceDirection.INPUT, INT16, 
                         (224, 224, 3), (224, 8, 3)),  # HWC layout
                Interface("weight", InterfaceDirection.WEIGHT, INT16,
                         (3, 3, 3, 64), (3, 3, 3, 8)),  # KKHWC
                Interface("output", InterfaceDirection.OUTPUT, INT16,
                         (224, 224, 64), (224, 8, 8))
            ]
        )
        
        config = ParallelismConfig(
            interface_pars={
                ("conv", "input"): 4,
                ("conv", "output"): 8
            }
        )
        
        # Apply configuration
        configured = config.apply_to_kernel(kernel, "conv")
        
        # Check stream dimensions updated
        input_intf = next(i for i in configured.interfaces if i.name == "input")
        output_intf = next(i for i in configured.interfaces if i.name == "output")
        
        # Stream dims should reflect parallelism
        assert input_intf.ipar == 4
        assert output_intf.ipar == 8
    
    def test_factorization(self):
        """Test parallelism factorization"""
        config = ParallelismConfig()
        
        # Test various numbers
        assert set(config._factorize(12)) == {1, 2, 3, 4, 6, 12}
        assert set(config._factorize(16)) == {1, 2, 4, 8, 16}
        assert set(config._factorize(7)) == {1, 7}  # Prime
    
    def test_stream_dims_computation(self):
        """Test stream dimension calculation"""
        config = ParallelismConfig()
        
        # Interface with 2D block
        intf = Interface("test", InterfaceDirection.INPUT, INT16,
                        (256, 512), (16, 32))
        
        # Parallelism 8 should factor as (2, 4) or similar
        stream_dims = config._compute_stream_dims(intf, 8)
        assert len(stream_dims) == 2
        assert stream_dims[0] * stream_dims[1] == 8
        
        # Each stream dim should divide block dim
        assert 16 % stream_dims[0] == 0
        assert 32 % stream_dims[1] == 0
    
    def test_validation(self):
        """Test configuration validation"""
        kernel = Kernel(
            name="test",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (256,), (16,))
            ]
        )
        
        # Valid: stream=8 < block=16
        config1 = ParallelismConfig(interface_pars={("k1", "in"): 8})
        valid, msg = config1.validate_kernel(kernel, "k1")
        assert valid
        
        # Invalid: stream=32 > block=16
        config2 = ParallelismConfig(interface_pars={("k1", "in"): 32})
        valid, msg = config2.validate_kernel(kernel, "k1")
        assert not valid
        assert "exceeds" in msg and "block dim" in msg
    
    def test_resource_estimation(self):
        """Test resource usage estimation"""
        kernel = Kernel(
            name="matmul",
            interfaces=[
                Interface("a", InterfaceDirection.INPUT, INT16, (512,), (64,)),
                Interface("b", InterfaceDirection.WEIGHT, INT16, (512, 256), (64, 32)),
                Interface("c", InterfaceDirection.OUTPUT, INT16, (256,), (32,))
            ],
            resources={"DSP": 32, "BRAM": 8}
        )
        
        config = ParallelismConfig(
            interface_pars={
                ("mm", "a"): 8,
                ("mm", "c"): 8
            }
        )
        
        resources = config.estimate_resources(kernel, "mm")
        
        # DSP should scale with parallelism
        assert resources["DSP"] == 32 * 8  # Base DSP * parallelism
        
        # Should include bandwidth
        assert "bandwidth_gbps" in resources


class TestDSEConstraints:
    """Test DSE constraints"""
    
    def test_resource_checking(self):
        """Test resource constraint checking"""
        constraints = DSEConstraints(
            max_dsp=1000,
            max_bram=500,
            max_bandwidth_gbps=50.0
        )
        
        # Within limits
        resources1 = {"DSP": 800, "BRAM": 400, "bandwidth_gbps": 40.0}
        ok, violations = constraints.check_resources(resources1)
        assert ok
        assert len(violations) == 0
        
        # Exceeds DSP
        resources2 = {"DSP": 1200, "BRAM": 400, "bandwidth_gbps": 40.0}
        ok, violations = constraints.check_resources(resources2)
        assert not ok
        assert len(violations) == 1
        assert "DSP" in violations[0]
        
        # Multiple violations
        resources3 = {"DSP": 1200, "BRAM": 600, "bandwidth_gbps": 60.0}
        ok, violations = constraints.check_resources(resources3)
        assert not ok
        assert len(violations) == 3
    
    def test_performance_checking(self):
        """Test performance constraint checking"""
        constraints = DSEConstraints(
            min_throughput=100.0,
            max_latency=1000,
            min_fps=30.0
        )
        
        # Meets requirements
        metrics1 = {"throughput": 150.0, "latency": 800, "fps": 40.0}
        ok, violations = constraints.check_performance(metrics1)
        assert ok
        
        # Low throughput
        metrics2 = {"throughput": 80.0, "latency": 800, "fps": 40.0}
        ok, violations = constraints.check_performance(metrics2)
        assert not ok
        assert "Throughput" in violations[0]
    
    def test_parallelism_range(self):
        """Test parallelism range generation"""
        # Default: powers of 2
        constraints1 = DSEConstraints(min_parallelism=2, max_parallelism=32)
        pars1 = constraints1.get_parallelism_range()
        assert pars1 == [2, 4, 8, 16, 32]
        
        # Custom allowed values
        constraints2 = DSEConstraints(
            allowed_parallelisms=[1, 3, 5, 7, 9, 11],
            min_parallelism=3,
            max_parallelism=9
        )
        pars2 = constraints2.get_parallelism_range()
        assert pars2 == [3, 5, 7, 9]


class TestConfigurationSpace:
    """Test configuration space generation"""
    
    def test_basic_space(self):
        """Test basic configuration space"""
        space = ConfigurationSpace(
            global_options=[1, 2, 4, 8]
        )
        
        # Add interfaces
        space.add_interface("k1", "in", [2, 4])
        space.add_interface("k1", "out", [2, 4])
        
        # Generate configs
        configs = space.generate_configs()
        
        # Should have 2x2 = 4 configurations
        assert len(configs) == 4
        
        # Check all combinations exist
        pars_seen = set()
        for config in configs:
            in_par = config.get_parallelism("k1", "in")
            out_par = config.get_parallelism("k1", "out")
            pars_seen.add((in_par, out_par))
        
        assert pars_seen == {(2,2), (2,4), (4,2), (4,4)}
    
    def test_coupled_interfaces(self):
        """Test coupled interface constraints"""
        space = ConfigurationSpace()
        
        # Add interfaces
        space.add_interface("k1", "a", [1, 2, 4])
        space.add_interface("k1", "b", [1, 2, 4])
        space.add_interface("k2", "c", [1, 2, 4])
        
        # Couple a and c (must have same parallelism)
        space.add_coupling([("k1", "a"), ("k2", "c")])
        
        # Generate configs
        configs = space.generate_configs()
        
        # Check coupling is respected
        for config in configs:
            a_par = config.get_parallelism("k1", "a")
            c_par = config.get_parallelism("k2", "c")
            assert a_par == c_par
        
        # Should have 3 values for coupled pair * 3 values for b = 9 configs
        assert len(configs) == 9
    
    def test_complex_coupling(self):
        """Test multiple coupling groups"""
        space = ConfigurationSpace()
        
        # 4 interfaces
        for i in range(4):
            space.add_interface(f"k{i}", "data", [1, 2, 4])
        
        # Two coupling groups
        space.add_coupling([("k0", "data"), ("k1", "data")])  # Group 1
        space.add_coupling([("k2", "data"), ("k3", "data")])  # Group 2
        
        configs = space.generate_configs()
        
        # Check both couplings respected
        for config in configs:
            # Group 1
            assert config.get_parallelism("k0", "data") == config.get_parallelism("k1", "data")
            # Group 2
            assert config.get_parallelism("k2", "data") == config.get_parallelism("k3", "data")
        
        # 3 choices for group 1 * 3 choices for group 2 = 9 configs
        assert len(configs) == 9
    
    def test_empty_space(self):
        """Test empty configuration space"""
        space = ConfigurationSpace()
        
        # No interfaces added
        configs = space.generate_configs()
        
        # Should have one empty config
        assert len(configs) == 1
        assert len(configs[0].interface_pars) == 0