"""
Unit tests for DataflowModel class

Tests cover unified computational model, initiation interval calculations,
parallelism bounds generation, and FINN optimization integration.
"""

import pytest
import numpy as np

from brainsmith.dataflow.core.dataflow_interface import (
    DataflowInterface,
    DataflowInterfaceType,
    DataflowDataType
)
from brainsmith.dataflow.core.dataflow_model import (
    DataflowModel,
    InitiationIntervals,
    ParallelismBounds,
    ParallelismConfiguration
)

class TestDataflowModel:
    """Test DataflowModel class"""
    
    def create_simple_interfaces(self):
        """Helper to create simple test interfaces"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        input_interface = DataflowInterface(
            name="input0",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[64],
            tDim=[16],
            sDim=[4],
            dtype=dtype
        )
        
        output_interface = DataflowInterface(
            name="output0",
            interface_type=DataflowInterfaceType.OUTPUT,
            qDim=[64],
            tDim=[16],
            sDim=[4],
            dtype=dtype
        )
        
        return [input_interface, output_interface]
    
    def create_multi_interface_setup(self):
        """Helper to create multi-interface test setup"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        input0 = DataflowInterface(
            name="input0",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[64],
            tDim=[16],
            sDim=[4],
            dtype=dtype
        )
        
        input1 = DataflowInterface(
            name="input1",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[32],
            tDim=[8],
            sDim=[2],
            dtype=dtype
        )
        
        weight0 = DataflowInterface(
            name="weights",
            interface_type=DataflowInterfaceType.WEIGHT,
            qDim=[128],
            tDim=[32],
            sDim=[8],
            dtype=dtype
        )
        
        output0 = DataflowInterface(
            name="output0",
            interface_type=DataflowInterfaceType.OUTPUT,
            qDim=[64],
            tDim=[16],
            sDim=[4],
            dtype=dtype
        )
        
        return [input0, input1, weight0, output0]
    
    def test_model_initialization(self):
        """Test DataflowModel initialization"""
        interfaces = self.create_simple_interfaces()
        parameters = {"param1": 16, "param2": 4}
        
        model = DataflowModel(interfaces, parameters)
        
        assert len(model.interfaces) == 2
        assert "input0" in model.interfaces
        assert "output0" in model.interfaces
        assert model.parameters == parameters
        assert len(model.input_interfaces) == 1
        assert len(model.output_interfaces) == 1
        assert len(model.weight_interfaces) == 0
    
    def test_interface_organization(self):
        """Test interface organization by type"""
        interfaces = self.create_multi_interface_setup()
        model = DataflowModel(interfaces, {})
        
        assert len(model.input_interfaces) == 2
        assert len(model.weight_interfaces) == 1
        assert len(model.output_interfaces) == 1
        
        # Check correct classification
        input_names = [iface.name for iface in model.input_interfaces]
        assert "input0" in input_names
        assert "input1" in input_names
        
        weight_names = [iface.name for iface in model.weight_interfaces]
        assert "weights" in weight_names
        
        output_names = [iface.name for iface in model.output_interfaces]
        assert "output0" in output_names
    
    def test_unified_initiation_interval_calculation_simple(self):
        """Test unified calculation with simple interface configuration"""
        interfaces = self.create_simple_interfaces()
        model = DataflowModel(interfaces, {})
        
        iPar = {"input0": 4}
        wPar = {}
        
        intervals = model.calculate_initiation_intervals(iPar, wPar)
        
        # Verify structure
        assert isinstance(intervals, InitiationIntervals)
        assert "input0" in intervals.cII
        assert "input0" in intervals.eII
        assert intervals.L > 0
        assert "bottleneck_input" in intervals.bottleneck_analysis
        
        # With simple setup: cII = tDim / sDim = 16 / 4 = 4
        expected_cII = 16 // 4
        assert intervals.cII["input0"] == expected_cII
        
        # With no weights: eII = cII = 4
        assert intervals.eII["input0"] == expected_cII
        
        # L = eII * num_tensors = 4 * (64/16) = 4 * 4 = 16
        num_tensors = 64 // 16  # qDim / tDim = 64 / 16 = 4
        expected_L = expected_cII * num_tensors
        assert intervals.L == expected_L
    
    def test_unified_initiation_interval_calculation_multi_interface(self):
        """Test unified calculation with complex multi-interface configuration"""
        interfaces = self.create_multi_interface_setup()
        model = DataflowModel(interfaces, {})
        
        iPar = {"input0": 4, "input1": 2}
        wPar = {"weights": 8}
        
        intervals = model.calculate_initiation_intervals(iPar, wPar)
        
        # Verify structure
        assert isinstance(intervals, InitiationIntervals)
        assert len(intervals.cII) == 2  # Two input interfaces
        assert len(intervals.eII) == 2  # Two input interfaces
        assert intervals.L > 0
        
        # Verify all inputs have calculations
        assert "input0" in intervals.cII
        assert "input1" in intervals.cII
        assert "input0" in intervals.eII
        assert "input1" in intervals.eII
        
        # Verify bottleneck analysis with new structure
        bottleneck_analysis = intervals.bottleneck_analysis
        assert "bottleneck_input" in bottleneck_analysis
        assert "bottleneck_eII" in bottleneck_analysis
        assert "bottleneck_cII" in bottleneck_analysis
        assert "interface_counts" in bottleneck_analysis
        assert "total_cycles_breakdown" in bottleneck_analysis
        
        interface_counts = bottleneck_analysis["interface_counts"]
        assert interface_counts["total_inputs"] == 2
        assert interface_counts["total_weights"] == 1
        assert interface_counts["total_outputs"] == 1
    
    def test_parallelism_bounds_generation(self):
        """Test parallelism bounds generation for FINN optimization"""
        interfaces = self.create_multi_interface_setup()
        model = DataflowModel(interfaces, {})
        
        bounds = model.get_parallelism_bounds()
        
        # Should have bounds for inputs and weights
        assert "input0_iPar" in bounds
        assert "input1_iPar" in bounds
        assert "weights_wPar" in bounds
        
        # Check input0 bounds
        input0_bounds = bounds["input0_iPar"]
        assert isinstance(input0_bounds, ParallelismBounds)
        assert input0_bounds.interface_name == "input0"
        assert input0_bounds.min_value == 1
        assert input0_bounds.max_value == 16  # tDim = 16
        assert len(input0_bounds.divisibility_constraints) > 0
        
        # Check weights bounds (now based on num_tensors instead of qDim)
        weights_bounds = bounds["weights_wPar"]
        assert weights_bounds.interface_name == "weights"
        assert weights_bounds.min_value == 1
        # num_tensors = qDim/tDim = 128/32 = 4, so max_value = np.prod([4]) = 4
        assert weights_bounds.max_value == 4
    
    def test_bottleneck_analysis(self):
        """Test bottleneck analysis functionality"""
        interfaces = self.create_multi_interface_setup()
        model = DataflowModel(interfaces, {})
        
        # Set up different parallelism to create clear bottleneck
        iPar = {"input0": 1, "input1": 1}  # Low parallelism
        wPar = {"weights": 1}
        
        intervals = model.calculate_initiation_intervals(iPar, wPar)
        
        bottleneck = intervals.bottleneck_analysis
        assert bottleneck["bottleneck_input"] in ["input0", "input1"]
        assert bottleneck["bottleneck_eII"] > 0
        assert len(bottleneck["bottleneck_qDim"]) > 0
    
    def test_mathematical_constraint_validation(self):
        """Test mathematical constraint validation"""
        # Create interface with valid constraints
        interfaces = self.create_simple_interfaces()
        model = DataflowModel(interfaces, {})
        
        result = model.validate_mathematical_constraints()
        assert result.success == True
        
        # Create interface with invalid streaming constraint (tDim % sDim != 0)
        dtype = DataflowDataType("INT", 8, True, "")
        # First create valid interface to pass construction
        invalid_interface = DataflowInterface(
            name="invalid",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[15],  # qDim can be any value
            tDim=[15],  # Valid for construction: 15 % 5 == 0
            sDim=[5],
            dtype=dtype
        )
        
        # Now modify to create invalid streaming constraint for testing
        invalid_interface.tDim = [16]  # 16 % 5 != 0, invalid streaming
        
        invalid_model = DataflowModel([invalid_interface], {})
        result = invalid_model.validate_mathematical_constraints()
        assert result.success == False
        assert len(result.errors) > 0
    
    def test_resource_requirements_calculation(self):
        """Test resource requirements estimation"""
        interfaces = self.create_multi_interface_setup()
        model = DataflowModel(interfaces, {})
        
        config = ParallelismConfiguration(
            iPar={"input0": 4, "input1": 2},
            wPar={"weights": 8},
            derived_sDim={}
        )
        
        requirements = model.get_resource_requirements(config)
        
        assert "memory_bits" in requirements
        assert "transfer_bandwidth" in requirements
        assert "computation_cycles" in requirements
        
        assert requirements["memory_bits"] > 0
        assert requirements["transfer_bandwidth"] > 0
        assert requirements["computation_cycles"] > 0
    
    def test_parallelism_optimization_placeholder(self):
        """Test parallelism optimization placeholder functionality"""
        interfaces = self.create_simple_interfaces()
        model = DataflowModel(interfaces, {})
        
        constraints = {"max_memory": 1000, "max_bandwidth": 500}
        
        config = model.optimize_parallelism(constraints)
        
        # Currently returns default parallelism of 1
        assert isinstance(config, ParallelismConfiguration)
        assert len(config.iPar) == 1
        assert config.iPar["input0"] == 1
        assert len(config.wPar) == 0  # No weight interfaces
        assert "input0" in config.derived_sDim
    
    def test_empty_interfaces_handling(self):
        """Test model behavior with empty interface list"""
        model = DataflowModel([], {})
        
        intervals = model.calculate_initiation_intervals({}, {})
        
        assert len(intervals.cII) == 0
        assert len(intervals.eII) == 0
        assert intervals.L == 1  # Default latency
        assert isinstance(intervals.bottleneck_analysis, dict)
    
    def test_single_input_calculation(self):
        """Test calculation with single input interface"""
        dtype = DataflowDataType("INT", 8, True, "")
        
        single_input = DataflowInterface(
            name="only_input",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[32],
            tDim=[8],
            sDim=[2],
            dtype=dtype
        )
        
        model = DataflowModel([single_input], {})
        
        iPar = {"only_input": 2}
        wPar = {}
        
        intervals = model.calculate_initiation_intervals(iPar, wPar)
        
        # cII = tDim / sDim = 8 / 2 = 4
        assert intervals.cII["only_input"] == 4
        
        # eII = cII (no weights) = 4
        assert intervals.eII["only_input"] == 4
        
        # L = eII * num_tensors = 4 * (32/8) = 4 * 4 = 16
        num_tensors = 32 // 8  # qDim / tDim = 32 / 8 = 4
        expected_L = 4 * num_tensors
        assert intervals.L == expected_L
        
        # Bottleneck should be the only input
        assert intervals.bottleneck_analysis["bottleneck_input"] == "only_input"

class TestInitiationIntervals:
    """Test InitiationIntervals data structure"""
    
    def test_initiation_intervals_creation(self):
        """Test creation of InitiationIntervals"""
        cII = {"input0": 4, "input1": 2}
        eII = {"input0": 8, "input1": 4}
        L = 128
        bottleneck_analysis = {
            "bottleneck_input": "input0",
            "bottleneck_eII": 8,
            "bottleneck_qDim": [64],
            "total_inputs": 2,
            "total_weights": 1
        }
        
        intervals = InitiationIntervals(
            cII=cII,
            eII=eII,
            L=L,
            bottleneck_analysis=bottleneck_analysis
        )
        
        assert intervals.cII == cII
        assert intervals.eII == eII
        assert intervals.L == L
        assert intervals.bottleneck_analysis == bottleneck_analysis

class TestParallelismBounds:
    """Test ParallelismBounds data structure"""
    
    def test_parallelism_bounds_creation(self):
        """Test creation of ParallelismBounds"""
        bounds = ParallelismBounds(
            interface_name="input0",
            min_value=1,
            max_value=16,
            divisibility_constraints=[1, 2, 4, 8, 16]
        )
        
        assert bounds.interface_name == "input0"
        assert bounds.min_value == 1
        assert bounds.max_value == 16
        assert bounds.divisibility_constraints == [1, 2, 4, 8, 16]

class TestParallelismConfiguration:
    """Test ParallelismConfiguration data structure"""
    
    def test_parallelism_configuration_creation(self):
        """Test creation of ParallelismConfiguration"""
        config = ParallelismConfiguration(
            iPar={"input0": 4, "input1": 2},
            wPar={"weights": 8},
            derived_sDim={"input0": [4], "input1": [2], "weights": [8]}
        )
        
        assert config.iPar == {"input0": 4, "input1": 2}
        assert config.wPar == {"weights": 8}
        assert config.derived_sDim["input0"] == [4]
        assert config.derived_sDim["input1"] == [2]
        assert config.derived_sDim["weights"] == [8]

if __name__ == "__main__":
    pytest.main([__file__])
