############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Comprehensive integration tests for RTL-to-InterfaceMetadata pipeline
############################################################################

import pytest
from pathlib import Path
import tempfile

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy, DefaultChunkingStrategy


class TestComprehensiveRTLToMetadata:
    """Comprehensive integration tests for the entire RTL-to-InterfaceMetadata pipeline."""
    
    def test_end_to_end_thresholding_with_pragmas(self):
        """Test complete RTL parsing with pragmas through to InterfaceMetadata creation."""
        parser = RTLParser(debug=True)
        
        # Comprehensive thresholding module with all pragma types
        systemverilog_code = """
// @brainsmith datatype in0_V_data_V INT,UINT 4 16
// @brainsmith datatype out0_V_data_V UINT 8 8
// @brainsmith weight weights_V_data_V
// @brainsmith bdim in0_V_data_V [PE]
// @brainsmith bdim out0_V_data_V [CHANNELS,PE] RINDEX=1
module comprehensive_thresholding #(
    parameter N = 16,
    parameter BIAS = 128,
    parameter PE = 4,
    parameter CHANNELS = 64
)(
    input ap_clk,
    input ap_rst_n,
    
    // Input AXI-Stream
    input [31:0] in0_V_data_V_TDATA,
    input in0_V_data_V_TVALID,
    output in0_V_data_V_TREADY,
    
    // Weight AXI-Stream
    input [7:0] weights_V_data_V_TDATA,
    input weights_V_data_V_TVALID,
    output weights_V_data_V_TREADY,
    
    // Output AXI-Stream
    output [31:0] out0_V_data_V_TDATA,
    output out0_V_data_V_TVALID,
    input out0_V_data_V_TREADY,
    
    // AXI-Lite Configuration
    input [11:0] s_axi_control_AWADDR,
    input s_axi_control_AWVALID,
    output s_axi_control_AWREADY,
    input [31:0] s_axi_control_WDATA,
    input [3:0] s_axi_control_WSTRB,
    input s_axi_control_WVALID,
    output s_axi_control_WREADY,
    output [1:0] s_axi_control_BRESP,
    output s_axi_control_BVALID,
    input s_axi_control_BREADY,
    input [11:0] s_axi_control_ARADDR,
    input s_axi_control_ARVALID,
    output s_axi_control_ARREADY,
    output [31:0] s_axi_control_RDATA,
    output [1:0] s_axi_control_RRESP,
    output s_axi_control_RVALID,
    input s_axi_control_RREADY
);
    // Module implementation...
endmodule
"""
        
        # Parse RTL and get KernelMetadata
        kernel_metadata = parser.parse(systemverilog_code, source_name="comprehensive_test")
        
        # Verify basic parsing
        assert kernel_metadata.name == "comprehensive_thresholding"
        assert len(kernel_metadata.parameters) == 4  # N, BIAS, PE, CHANNELS
        assert len(kernel_metadata.pragmas) == 5  # 2 DATATYPE, 1 WEIGHT, 2 BDIM
        assert len(kernel_metadata.interfaces) >= 4  # Control, Input, Weight, Output, Config
        
        # Verify interface metadata creation and pragma application
        interface_types = {iface.interface_type for iface in kernel_metadata.interfaces}
        assert InterfaceType.CONTROL in interface_types
        assert InterfaceType.INPUT in interface_types
        assert InterfaceType.WEIGHT in interface_types  # Should be overridden by WEIGHT pragma
        assert InterfaceType.OUTPUT in interface_types
        assert InterfaceType.CONFIG in interface_types
        
        # Test DATATYPE pragma application
        input_interface = next((iface for iface in kernel_metadata.interfaces 
                              if "in0_V_data_V" in iface.name), None)
        assert input_interface is not None
        
        # Should have INT and UINT datatypes with 4-16 bit range
        datatype_constraints = input_interface.allowed_datatypes
        finn_types = {dt.finn_type for dt in datatype_constraints}
        assert any("INT" in ft for ft in finn_types), f"Expected INT datatypes, got: {finn_types}"
        assert any("UINT" in ft for ft in finn_types), f"Expected UINT datatypes, got: {finn_types}"
        
        # Test WEIGHT pragma application
        weight_interface = next((iface for iface in kernel_metadata.interfaces 
                               if iface.interface_type == InterfaceType.WEIGHT), None)
        assert weight_interface is not None
        assert "weights" in weight_interface.name
        
        # Test BDIM pragma application (block chunking)
        chunking_interfaces = [iface for iface in kernel_metadata.interfaces 
                             if not isinstance(iface.chunking_strategy, DefaultChunkingStrategy)]
        assert len(chunking_interfaces) >= 1, "BDIM pragmas should create custom chunking strategies"
        
        # Verify BlockChunkingStrategy for BDIM pragmas
        block_chunking_interface = next((iface for iface in chunking_interfaces
                                       if hasattr(iface.chunking_strategy, 'block_shape')), None)
        if block_chunking_interface:
            strategy = block_chunking_interface.chunking_strategy
            assert hasattr(strategy, 'block_shape'), "BDIM pragma should create BlockChunkingStrategy"
            assert hasattr(strategy, 'rindex'), "BlockChunkingStrategy should have rindex attribute"
        
        print(f"✅ End-to-end RTL parsing with pragmas successful:")
        print(f"  - Module: {kernel_metadata.name}")
        print(f"  - Parameters: {len(kernel_metadata.parameters)}")
        print(f"  - Interfaces: {len(kernel_metadata.interfaces)}")
        print(f"  - Pragmas: {len(kernel_metadata.pragmas)}")
        for iface in kernel_metadata.interfaces:
            print(f"    * {iface.name}: {iface.interface_type.value}")
    
    def test_multi_interface_datatype_conflicts(self):
        """Test handling of multiple interfaces with different datatype requirements."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith datatype input_stream INT 8 8
// @brainsmith datatype output_stream UINT 16 16
// @brainsmith datatype weight_stream FIXED 8 8
module multi_datatype_test #(
    parameter WIDTH = 32
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] input_stream_TDATA,
    input input_stream_TVALID,
    output input_stream_TREADY,
    
    input [7:0] weight_stream_TDATA,
    input weight_stream_TVALID,
    output weight_stream_TREADY,
    
    output [31:0] output_stream_TDATA,
    output output_stream_TVALID,
    input output_stream_TREADY
);
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="multi_datatype_test")
        
        # Find specific interfaces
        input_iface = next((iface for iface in kernel_metadata.interfaces 
                          if "input_stream" in iface.name), None)
        output_iface = next((iface for iface in kernel_metadata.interfaces 
                           if "output_stream" in iface.name), None)
        weight_iface = next((iface for iface in kernel_metadata.interfaces 
                           if "weight_stream" in iface.name), None)
        
        assert input_iface is not None
        assert output_iface is not None
        assert weight_iface is not None
        
        # Verify different datatype constraints
        input_types = {dt.finn_type for dt in input_iface.allowed_datatypes}
        output_types = {dt.finn_type for dt in output_iface.allowed_datatypes}
        weight_types = {dt.finn_type for dt in weight_iface.allowed_datatypes}
        
        assert any("INT8" in ft for ft in input_types), f"Expected INT8 for input, got: {input_types}"
        assert any("UINT16" in ft for ft in output_types), f"Expected UINT16 for output, got: {output_types}"
        assert any("FIXED8" in ft for ft in weight_types), f"Expected FIXED8 for weight, got: {weight_types}"
        
        print(f"✅ Multi-interface datatype handling successful")
    
    def test_pragma_error_resilience(self):
        """Test that invalid pragmas don't break the entire parsing process."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith datatype invalid_interface INVALID_TYPE 8 8
// @brainsmith bdim nonexistent_interface [PE]  // This should fail gracefully
// @brainsmith datatype valid_interface UINT 8 8
module pragma_error_test #(
    parameter PE = 4
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] valid_interface_TDATA,
    input valid_interface_TVALID,
    output valid_interface_TREADY,
    
    output [31:0] out_TDATA,
    output out_TVALID,
    input out_TREADY
);
endmodule
"""
        
        # Should not raise exception despite invalid pragmas
        kernel_metadata = parser.parse(systemverilog_code, source_name="pragma_error_test")
        
        # Should still parse the module successfully
        assert kernel_metadata.name == "pragma_error_test"
        assert len(kernel_metadata.interfaces) >= 2  # Should detect at least some interfaces
        
        # Valid pragma should still apply
        valid_iface = next((iface for iface in kernel_metadata.interfaces 
                          if "valid_interface" in iface.name), None)
        if valid_iface:
            # Should have UINT8 from valid pragma
            types = {dt.finn_type for dt in valid_iface.allowed_datatypes}
            assert any("UINT8" in ft for ft in types), f"Valid pragma should apply, got: {types}"
        
        print(f"✅ Pragma error resilience test successful")
    
    def test_interface_metadata_serialization_compatibility(self):
        """Test that InterfaceMetadata objects can be used in downstream systems."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// @brainsmith datatype in0 UINT 8 8
// @brainsmith bdim in0 [CHANNELS]
module serialization_test #(
    parameter CHANNELS = 32
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] in0_TDATA,
    input in0_TVALID,
    output in0_TREADY,
    
    output [31:0] out0_TDATA,
    output out0_TVALID,
    input out0_TREADY
);
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="serialization_test")
        
        # Test that InterfaceMetadata objects have all required fields for downstream use
        for interface in kernel_metadata.interfaces:
            assert hasattr(interface, 'name')
            assert hasattr(interface, 'interface_type')
            assert hasattr(interface, 'allowed_datatypes')
            assert hasattr(interface, 'chunking_strategy')
            
            # Test that they can be converted to dict (for JSON serialization)
            interface_dict = {
                'name': interface.name,
                'interface_type': interface.interface_type.value,
                'allowed_datatypes': [
                    {
                        'finn_type': dt.finn_type,
                        'bit_width': dt.bit_width,
                        'signed': dt.signed
                    } for dt in interface.allowed_datatypes
                ],
                'chunking_strategy_type': interface.chunking_strategy.chunking_type.value
            }
            
            # Verify serialization works
            assert isinstance(interface_dict['name'], str)
            assert isinstance(interface_dict['interface_type'], str)
            assert isinstance(interface_dict['allowed_datatypes'], list)
            assert isinstance(interface_dict['chunking_strategy_type'], str)
        
        print(f"✅ InterfaceMetadata serialization compatibility verified")
    
    def test_performance_with_complex_module(self):
        """Test performance and memory usage with a complex module."""
        parser = RTLParser(debug=False)
        
        # Generate a complex module with many interfaces and pragmas
        pragmas = []
        ports = ["    input ap_clk,", "    input ap_rst_n,"]
        
        # Add pragmas and ports for multiple streams
        for i in range(10):
            pragmas.extend([
                f"// @brainsmith datatype stream_{i}_in UINT 8 8",
                f"// @brainsmith bdim stream_{i}_in [DATA_WIDTH]"  # Use DATA_WIDTH parameter for all streams
            ])
            ports.extend([
                f"    input [31:0] stream_{i}_in_TDATA,",
                f"    input stream_{i}_in_TVALID,",
                f"    output stream_{i}_in_TREADY,",
                f"    output [31:0] stream_{i}_out_TDATA,",
                f"    output stream_{i}_out_TVALID,",
                f"    input stream_{i}_out_TREADY,"
            ])
        
        # Remove trailing comma
        if ports:
            ports[-1] = ports[-1].rstrip(',')
        
        systemverilog_code = f"""
{chr(10).join(pragmas)}
module complex_performance_test #(
    parameter NUM_STREAMS = 10,
    parameter DATA_WIDTH = 32,
    parameter BUFFER_SIZE = 1024
)(
{chr(10).join(ports)}
);
    // Complex module implementation...
endmodule
"""
        
        import time
        import sys
        
        # Measure parsing time and memory
        start_time = time.time()
        initial_objects = len(sys.modules)
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="complex_performance_test")
        
        end_time = time.time()
        final_objects = len(sys.modules)
        parse_time = end_time - start_time
        
        # Verify parsing completed successfully
        assert kernel_metadata.name == "complex_performance_test"
        assert len(kernel_metadata.interfaces) >= 20  # 10 inputs + 10 outputs + control
        assert len(kernel_metadata.pragmas) == 20  # 10 DATATYPE + 10 BDIM
        
        # Performance requirements
        assert parse_time < 3.0, f"Parsing took too long: {parse_time:.2f}s"
        
        # Memory usage should be reasonable
        memory_growth = final_objects - initial_objects
        assert memory_growth < 50, f"Too many new modules loaded: {memory_growth}"
        
        # Test InterfaceMetadata access performance
        start_time = time.time()
        for interface in kernel_metadata.interfaces:
            _ = interface.name
            _ = interface.interface_type
            _ = len(interface.allowed_datatypes)
            _ = interface.chunking_strategy.chunking_type
        access_time = time.time() - start_time
        
        assert access_time < 0.1, f"InterfaceMetadata access too slow: {access_time:.3f}s"
        
        print(f"✅ Complex module performance test successful:")
        print(f"  - Interfaces: {len(kernel_metadata.interfaces)}")
        print(f"  - Parse time: {parse_time:.3f}s")
        print(f"  - Access time: {access_time:.3f}s")
        print(f"  - Memory growth: {memory_growth} modules")
    
    def test_edge_case_pragma_combinations(self):
        """Test edge cases with complex pragma combinations."""
        parser = RTLParser(debug=False)
        
        systemverilog_code = """
// Multiple pragmas on same interface
// @brainsmith datatype shared_interface UINT 8 8
// @brainsmith weight shared_interface
// @brainsmith bdim shared_interface [SIZE]

// Conflicting pragmas (should handle gracefully)
// @brainsmith datatype conflict_interface INT 8 8
// @brainsmith datatype conflict_interface UINT 16 16

module edge_case_pragmas #(
    parameter SIZE = 64
)(
    input ap_clk,
    input ap_rst_n,
    
    input [31:0] shared_interface_TDATA,
    input shared_interface_TVALID,
    output shared_interface_TREADY,
    
    input [31:0] conflict_interface_TDATA,
    input conflict_interface_TVALID,
    output conflict_interface_TREADY,
    
    output [31:0] output_TDATA,
    output output_TVALID,
    input output_TREADY
);
endmodule
"""
        
        kernel_metadata = parser.parse(systemverilog_code, source_name="edge_case_test")
        
        # Find shared interface (should have all pragmas applied)
        shared_iface = next((iface for iface in kernel_metadata.interfaces 
                           if "shared_interface" in iface.name), None)
        assert shared_iface is not None
        
        # Should be marked as WEIGHT (overriding INPUT)
        assert shared_iface.interface_type == InterfaceType.WEIGHT
        
        # Should have UINT8 datatype
        types = {dt.finn_type for dt in shared_iface.allowed_datatypes}
        assert any("UINT8" in ft for ft in types), f"Expected UINT8, got: {types}"
        
        # Should have custom chunking strategy from BDIM pragma
        assert not isinstance(shared_iface.chunking_strategy, DefaultChunkingStrategy)
        assert hasattr(shared_iface.chunking_strategy, 'block_shape')
        assert shared_iface.chunking_strategy.block_shape == ["SIZE"]
        
        # Find conflict interface (should handle gracefully)
        conflict_iface = next((iface for iface in kernel_metadata.interfaces 
                             if "conflict_interface" in iface.name), None)
        assert conflict_iface is not None
        
        # Should have some datatype (last pragma wins or error handling)
        assert len(conflict_iface.allowed_datatypes) > 0
        
        print(f"✅ Edge case pragma combinations handled successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])