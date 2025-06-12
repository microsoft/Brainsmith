############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Test file for InterfaceBuilder.build_interface_metadata() method
############################################################################

import pytest
from unittest.mock import Mock

from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_builder import InterfaceBuilder
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction, Pragma, PragmaType
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy


class TestInterfaceBuilderMetadata:
    """Test cases for InterfaceBuilder.build_interface_metadata() method."""
    
    def test_basic_axi_stream_interface_creation(self):
        """Test creation of basic AXI-Stream interfaces without pragmas."""
        builder = InterfaceBuilder(debug=True)
        
        # Create basic AXI-Stream input ports
        ports = [
            Port("s_axis_tdata", Direction.INPUT, "32"),
            Port("s_axis_tvalid", Direction.INPUT, "1"),
            Port("s_axis_tready", Direction.OUTPUT, "1"),
            Port("m_axis_tdata", Direction.OUTPUT, "32"),
            Port("m_axis_tvalid", Direction.OUTPUT, "1"),
            Port("m_axis_tready", Direction.INPUT, "1"),
        ]
        
        # No pragmas for basic test
        pragmas = []
        
        # Call new method
        metadata_list, unassigned_ports = builder.build_interface_metadata(ports, pragmas)
        
        # Verify results
        assert len(metadata_list) == 2, f"Expected 2 interfaces, got {len(metadata_list)}"
        assert len(unassigned_ports) == 0, f"Expected 0 unassigned ports, got {len(unassigned_ports)}"
        
        # Check interface names and types
        interface_names = {meta.name for meta in metadata_list}
        assert "s_axis" in interface_names, "Missing s_axis interface"
        assert "m_axis" in interface_names, "Missing m_axis interface"
        
        # Verify interface types are correct
        for metadata in metadata_list:
            assert isinstance(metadata, InterfaceMetadata)
            assert metadata.interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT]
            assert isinstance(metadata.allowed_datatypes, list)
            assert len(metadata.allowed_datatypes) == 1  # Should have default UINT8 datatype
            assert isinstance(metadata.chunking_strategy, DefaultChunkingStrategy)
    
    def test_global_control_interface_creation(self):
        """Test creation of global control interface."""
        builder = InterfaceBuilder(debug=True)
        
        # Create global control ports
        ports = [
            Port("clk", Direction.INPUT, "1"),
            Port("rst_n", Direction.INPUT, "1"),
        ]
        
        # No pragmas
        pragmas = []
        
        # Call new method
        metadata_list, unassigned_ports = builder.build_interface_metadata(ports, pragmas)
        
        # Verify results
        assert len(metadata_list) == 1, f"Expected 1 interface, got {len(metadata_list)}"
        assert len(unassigned_ports) == 0, f"Expected 0 unassigned ports, got {len(unassigned_ports)}"
        
        # Check interface
        metadata = metadata_list[0]
        assert metadata.interface_type == InterfaceType.CONTROL
        assert "<NO_PREFIX>" in metadata.name  # Global signals have no prefix
    
    def test_with_datatype_pragma(self):
        """Test InterfaceMetadata creation with DATATYPE pragma."""
        builder = InterfaceBuilder(debug=True)
        
        # Create AXI-Stream input ports
        ports = [
            Port("s_axis_tdata", Direction.INPUT, "8"),
            Port("s_axis_tvalid", Direction.INPUT, "1"),
            Port("s_axis_tready", Direction.OUTPUT, "1"),
        ]
        
        # Create DATATYPE pragma
        pragma = Mock()
        pragma.type = PragmaType.DATATYPE
        pragma.applies_to_interface = Mock(return_value=True)
        pragma.apply_to_interface_metadata = Mock(side_effect=self._mock_datatype_pragma)
        
        pragmas = [pragma]
        
        # Call new method
        metadata_list, unassigned_ports = builder.build_interface_metadata(ports, pragmas)
        
        # Verify pragma was applied
        assert len(metadata_list) == 1
        pragma.applies_to_interface.assert_called()
        pragma.apply_to_interface_metadata.assert_called()
        
        # Verify metadata has datatype constraints (from mock)
        metadata = metadata_list[0]
        assert len(metadata.allowed_datatypes) == 1
        assert metadata.allowed_datatypes[0].finn_type == "UINT8"
    
    def test_with_weight_pragma(self):
        """Test InterfaceMetadata creation with WEIGHT pragma."""
        builder = InterfaceBuilder(debug=True)
        
        # Create AXI-Stream input ports (will be changed to WEIGHT by pragma)
        ports = [
            Port("weights_tdata", Direction.INPUT, "8"),
            Port("weights_tvalid", Direction.INPUT, "1"),
            Port("weights_tready", Direction.OUTPUT, "1"),
        ]
        
        # Create WEIGHT pragma
        pragma = Mock()
        pragma.type = PragmaType.WEIGHT
        pragma.applies_to_interface = Mock(return_value=True)
        pragma.apply_to_interface_metadata = Mock(side_effect=self._mock_weight_pragma)
        
        pragmas = [pragma]
        
        # Call new method
        metadata_list, unassigned_ports = builder.build_interface_metadata(ports, pragmas)
        
        # Verify pragma was applied
        assert len(metadata_list) == 1
        metadata = metadata_list[0]
        assert metadata.interface_type == InterfaceType.WEIGHT  # Should be overridden
    
    def test_mixed_interface_types(self):
        """Test that mixed interface types are correctly detected."""
        builder = InterfaceBuilder(debug=True)
        
        # Create test ports with multiple interface types
        ports = [
            # AXI-Stream input
            Port("s_axis_tdata", Direction.INPUT, "32"),
            Port("s_axis_tvalid", Direction.INPUT, "1"),
            Port("s_axis_tready", Direction.OUTPUT, "1"),
            # Global control
            Port("clk", Direction.INPUT, "1"),
            Port("rst_n", Direction.INPUT, "1"),
            # AXI-Lite config
            Port("s_axi_control_awaddr", Direction.INPUT, "12"),
            Port("s_axi_control_awvalid", Direction.INPUT, "1"),
            Port("s_axi_control_awready", Direction.OUTPUT, "1"),
        ]
        
        # No pragmas for basic test
        pragmas = []
        
        # Call new method
        metadata_list, unassigned_metadata = builder.build_interface_metadata(ports, pragmas)
        
        # Verify results
        assert len(metadata_list) >= 2, f"Expected at least 2 interfaces, got {len(metadata_list)}"
        
        # Check that different interface types are detected
        detected_types = {meta.interface_type for meta in metadata_list}
        assert InterfaceType.INPUT in detected_types, "Should detect AXI-Stream input"
        assert InterfaceType.CONTROL in detected_types, "Should detect global control"
        
        # Verify metadata properties
        for metadata in metadata_list:
            assert isinstance(metadata, InterfaceMetadata)
            assert metadata.name  # Should have a name
            assert metadata.allowed_datatypes  # Should have default datatypes
            assert metadata.chunking_strategy  # Should have a chunking strategy
    
    def test_unassigned_ports_handling(self):
        """Test handling of unassigned ports."""
        builder = InterfaceBuilder(debug=True)
        
        # Create ports that don't match any interface pattern
        ports = [
            Port("random_signal", Direction.INPUT, "1"),
            Port("another_random", Direction.OUTPUT, "8"),
            Port("s_axis_tdata", Direction.INPUT, "32"),  # This should match
            Port("s_axis_tvalid", Direction.INPUT, "1"),
            Port("s_axis_tready", Direction.OUTPUT, "1"),
        ]
        
        pragmas = []
        
        # Call new method
        metadata_list, unassigned_ports = builder.build_interface_metadata(ports, pragmas)
        
        # Verify that some ports are unassigned
        assert len(metadata_list) == 1, "Should detect 1 interface (s_axis)"
        assert len(unassigned_ports) == 2, "Should have 2 unassigned ports"
        
        # Check unassigned port names
        unassigned_names = {port.name for port in unassigned_ports}
        assert "random_signal" in unassigned_names
        assert "another_random" in unassigned_names
    
    def _mock_datatype_pragma(self, interface, metadata):
        """Mock DataType pragma application."""
        from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
        constraint = DataTypeConstraint(
            finn_type="UINT8",
            bit_width=8,
            signed=False
        )
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=metadata.interface_type,
            allowed_datatypes=[constraint],
            chunking_strategy=metadata.chunking_strategy,
            description=metadata.description
        )
    
    def _mock_weight_pragma(self, interface, metadata):
        """Mock Weight pragma application."""
        from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
        return InterfaceMetadata(
            name=metadata.name,
            interface_type=InterfaceType.WEIGHT,  # Override to WEIGHT
            allowed_datatypes=metadata.allowed_datatypes,
            chunking_strategy=metadata.chunking_strategy,
            description=metadata.description
        )


if __name__ == "__main__":
    pytest.main([__file__])