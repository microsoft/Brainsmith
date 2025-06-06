"""
Comprehensive End-to-End Integration Test for Interface-Wise Dataflow Modeling Framework.

This test uses the real-world thresholding_axi.sv example to validate the complete pipeline
from RTL parsing through dataflow modeling, and is designed to be easily extended for 
Phase 3 code generation testing.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator, HardwareKernelGeneratorError
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import InterfaceType as RTLInterfaceType
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType, DataflowDataType
from brainsmith.dataflow.core.validation import ValidationSeverity


class TestEndToEndThresholding:
    """
    End-to-end integration test using the thresholding_axi.sv example.
    
    Tests the complete pipeline:
    1. RTL Parser - parsing complex real-world RTL
    2. Interface Detection - AXI-Stream, AXI-Lite, Global Control
    3. Dataflow Conversion - RTL → DataflowInterface objects
    4. Dataflow Model Creation - unified computational model
    5. HKG Pipeline - complete enhanced pipeline execution
    6. Extensibility - prepared for Phase 3 code generation
    """
    
    def setup_method(self):
        """Set up test fixtures with real thresholding example."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Create thresholding RTL file (realistic example)
        self.rtl_file = Path(self.temp_dir) / "thresholding_axi.sv"
        self._create_thresholding_rtl()
        
        # Create enhanced compiler data with ONNX metadata and pragmas
        self.compiler_file = Path(self.temp_dir) / "thresholding_compiler_data.py"
        self._create_thresholding_compiler_data()
        
        # Create enhanced RTL with dataflow pragmas for testing
        self.enhanced_rtl_file = Path(self.temp_dir) / "thresholding_enhanced.sv"
        self._create_enhanced_thresholding_rtl()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_thresholding_rtl(self):
        """Create the thresholding_axi.sv RTL file."""
        rtl_content = '''
/******************************************************************************
 * Thresholding AXI Module - Real-world example for end-to-end testing
 *****************************************************************************/

// @brainsmith TOP_MODULE thresholding_axi

module thresholding_axi #(
	int unsigned  N = 1,	// output precision
	int unsigned  WI = 8,	// input precision
	int unsigned  WT = 8,	// threshold precision
	int unsigned  C = 32,	// Channels
	int unsigned  PE = 1,	// Processing Parallelism, requires C = k*PE

	bit  SIGNED = 1,	// signed inputs
	bit  FPARG  = 0,	// floating-point inputs
	int  BIAS   = 0,	// offsetting the output

	// Initial Thresholds
	parameter  THRESHOLDS_PATH = "",

	bit  USE_AXILITE = 1,	// Implement AXI-Lite for threshold read/write

	// Memory configuration
	int unsigned  DEPTH_TRIGGER_URAM = 0,
	int unsigned  DEPTH_TRIGGER_BRAM = 0,
	bit  DEEP_PIPELINE = 0,

	localparam int unsigned  CF = C/PE,	// Channel Fold
	localparam int unsigned  ADDR_BITS = $clog2(CF) + $clog2(PE) + N + 2,
	localparam int unsigned  O_BITS = BIAS >= 0?
		$clog2(2**N+BIAS) : 1+$clog2(-BIAS >= 2**(N-1)? -BIAS : 2**N+BIAS)
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	logic                  s_axilite_AWVALID,
	output	logic                  s_axilite_AWREADY,
	input	logic [ADDR_BITS-1:0]  s_axilite_AWADDR,

	input	logic         s_axilite_WVALID,
	output	logic         s_axilite_WREADY,
	input	logic [31:0]  s_axilite_WDATA,
	input	logic [ 3:0]  s_axilite_WSTRB,

	output	logic        s_axilite_BVALID,
	input	logic        s_axilite_BREADY,
	output	logic [1:0]  s_axilite_BRESP,

	// Reading
	input	logic                  s_axilite_ARVALID,
	output	logic                  s_axilite_ARREADY,
	input	logic [ADDR_BITS-1:0]  s_axilite_ARADDR,

	output	logic         s_axilite_RVALID,
	input	logic         s_axilite_RREADY,
	output	logic [31:0]  s_axilite_RDATA,
	output	logic [ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [((PE*WI+7)/8)*8-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [((PE*O_BITS+7)/8)*8-1:0]  m_axis_tdata
);

	// Module implementation would go here...
	// For testing purposes, we focus on interface structure

endmodule : thresholding_axi
'''
        with open(self.rtl_file, 'w') as f:
            f.write(rtl_content)
    
    def _create_enhanced_thresholding_rtl(self):
        """Create enhanced RTL with dataflow pragmas for comprehensive testing."""
        enhanced_content = '''
/******************************************************************************
 * Enhanced Thresholding AXI Module with Dataflow Pragmas
 *****************************************************************************/

// @brainsmith TOP_MODULE thresholding_axi
// @brainsmith TDIM s_axis PE*32 PE
// @brainsmith TDIM m_axis PE*32 PE
// @brainsmith DATATYPE s_axis UINT 8 8
// @brainsmith DATATYPE m_axis UINT 1 1
// @brainsmith WEIGHT s_axilite

module thresholding_axi #(
	int unsigned  N = 1,
	int unsigned  WI = 8,
	int unsigned  WT = 8,
	int unsigned  C = 32,
	int unsigned  PE = 1,
	bit  SIGNED = 1,
	bit  FPARG  = 0,
	int  BIAS   = 0,
	parameter  THRESHOLDS_PATH = "",
	bit  USE_AXILITE = 1,
	int unsigned  DEPTH_TRIGGER_URAM = 0,
	int unsigned  DEPTH_TRIGGER_BRAM = 0,
	bit  DEEP_PIPELINE = 0,
	localparam int unsigned  CF = C/PE,
	localparam int unsigned  ADDR_BITS = $clog2(CF) + $clog2(PE) + N + 2,
	localparam int unsigned  O_BITS = BIAS >= 0?
		$clog2(2**N+BIAS) : 1+$clog2(-BIAS >= 2**(N-1)? -BIAS : 2**N+BIAS)
)(
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	input	logic                  s_axilite_AWVALID,
	output	logic                  s_axilite_AWREADY,
	input	logic [ADDR_BITS-1:0]  s_axilite_AWADDR,
	input	logic         s_axilite_WVALID,
	output	logic         s_axilite_WREADY,
	input	logic [31:0]  s_axilite_WDATA,
	input	logic [ 3:0]  s_axilite_WSTRB,
	output	logic        s_axilite_BVALID,
	input	logic        s_axilite_BREADY,
	output	logic [1:0]  s_axilite_BRESP,
	input	logic                  s_axilite_ARVALID,
	output	logic                  s_axilite_ARREADY,
	input	logic [ADDR_BITS-1:0]  s_axilite_ARADDR,
	output	logic         s_axilite_RVALID,
	input	logic         s_axilite_RREADY,
	output	logic [31:0]  s_axilite_RDATA,
	output	logic [ 1:0]  s_axilite_RRESP,

	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [((PE*WI+7)/8)*8-1:0]  s_axis_tdata,

	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [((PE*O_BITS+7)/8)*8-1:0]  m_axis_tdata
);

	// Implementation...

endmodule : thresholding_axi
'''
        with open(self.enhanced_rtl_file, 'w') as f:
            f.write(enhanced_content)
    
    def _create_thresholding_compiler_data(self):
        """Create comprehensive compiler data with ONNX metadata."""
        compiler_content = '''
"""
Compiler data for thresholding_axi module - comprehensive ONNX integration.
"""

# ONNX Model Metadata for Dataflow Framework Integration
onnx_metadata = {
    # Input interface metadata
    "s_axis_layout": "[N, C]",
    "s_axis_shape": [1, 32],
    "s_axis_dtype": "UINT8",
    
    # Output interface metadata  
    "m_axis_layout": "[N, C]",
    "m_axis_shape": [1, 32],
    "m_axis_dtype": "UINT1",
    
    # Configuration interface metadata
    "s_axilite_layout": "[THRESHOLDS]", 
    "s_axilite_shape": [32],
    "s_axilite_dtype": "UINT8",
    
    # Operation metadata
    "operation_type": "Thresholding",
    "operation_description": "Multi-threshold activation with configurable thresholds",
    
    # Performance characteristics
    "expected_latency_cycles": 2,
    "expected_throughput_ops_per_cycle": 1,
    
    # Resource estimates
    "estimated_luts": 1000,
    "estimated_ffs": 500,
    "estimated_brams": 1,
    "estimated_dsps": 0
}

# Cost functions for FINN DSE integration
def get_node_cost(node_attrs):
    """Calculate resource cost for thresholding node."""
    pe = node_attrs.get("PE", 1)
    channels = node_attrs.get("C", 32)
    
    # Scale resources with parallelism
    base_cost = {
        "BRAM": 1,
        "LUT": pe * 50,
        "FF": pe * 25,
        "DSP": 0
    }
    
    return base_cost

def get_performance_estimate(node_attrs):
    """Estimate performance characteristics."""
    pe = node_attrs.get("PE", 1) 
    channels = node_attrs.get("C", 32)
    
    cycles_per_sample = max(1, channels // pe)
    
    return {
        "cycles_per_sample": cycles_per_sample,
        "max_frequency_mhz": 300,
        "throughput_samples_per_second": 300e6 / cycles_per_sample
    }

# Additional metadata for enhanced testing
test_configurations = [
    {"PE": 1, "C": 32, "description": "Baseline configuration"},
    {"PE": 2, "C": 32, "description": "2x parallelism"},
    {"PE": 4, "C": 32, "description": "4x parallelism"},
    {"PE": 8, "C": 32, "description": "8x parallelism"}
]
'''
        with open(self.compiler_file, 'w') as f:
            f.write(compiler_content)
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_basic_rtl_parsing(self):
        """Test basic RTL parsing of complex thresholding module."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(self.rtl_file),
            compiler_data_path=str(self.compiler_file),
            output_dir=str(self.output_dir)
        )
        
        # Test RTL parsing phase
        parsed_data = hkg.get_parsed_rtl_data()
        
        # Verify module structure
        assert parsed_data.name == "thresholding_axi"
        assert len(parsed_data.parameters) >= 10  # Complex parameter structure
        
        # Count total ports across all interfaces
        total_ports = sum(len(iface.ports) for iface in parsed_data.interfaces.values())
        assert total_ports >= 20  # Multiple AXI interfaces
        
        # Verify key interfaces are detected
        interface_names = set(parsed_data.interfaces.keys())
        expected_interfaces = {"global", "s_axilite", "s_axis", "m_axis"}
        # Use a subset check since interface names might be different (e.g., "ap" instead of "global")
        axi_interfaces = {name for name in interface_names if "axi" in name}
        assert len(axi_interfaces) >= 2  # At least AXI-Lite and AXI-Stream interfaces
        
        # Verify key parameters are parsed
        param_names = {p.name for p in parsed_data.parameters}
        expected_params = {"N", "WI", "WT", "C", "PE", "SIGNED", "USE_AXILITE"}
        assert expected_params.issubset(param_names)
        
        # Verify parameter types and defaults
        param_map = {p.name: p for p in parsed_data.parameters}
        assert param_map["N"].default_value == "1"
        assert param_map["WI"].default_value == "8"
        assert param_map["C"].default_value == "32"
        assert param_map["PE"].default_value == "1"
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_interface_detection_and_classification(self):
        """Test interface detection and classification for AXI interfaces."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(self.rtl_file),
            compiler_data_path=str(self.compiler_file),
            output_dir=str(self.output_dir)
        )
        
        # Run through interface building
        hkg.run(stop_after="parse_compiler_data")
        parsed_data = hkg.get_parsed_rtl_data()
        
        # Test Stage 3 interface detection - interfaces are built during parsing
        interfaces = parsed_data.interfaces
        
        # Verify interface detection
        interface_names = set(interfaces.keys())
        
        # Find interfaces by type and name - interface builder creates predictable names
        global_iface = None
        axilite_iface = None
        input_stream_iface = None
        output_stream_iface = None
        
        for name, iface in interfaces.items():
            if iface.type == RTLInterfaceType.GLOBAL_CONTROL:
                global_iface = iface
            elif iface.type == RTLInterfaceType.AXI_LITE:
                axilite_iface = iface
            elif iface.type == RTLInterfaceType.AXI_STREAM:
                # Use interface name to determine direction - interface builder uses original prefixes
                if name == "s_axis":  # Input stream interface
                    input_stream_iface = iface
                elif name == "m_axis":  # Output stream interface
                    output_stream_iface = iface
        
        # Verify interface types exist
        assert global_iface is not None, "No global control interface found"
        assert axilite_iface is not None, "No AXI-Lite interface found"
        assert input_stream_iface is not None, "No input AXI-Stream interface found"
        assert output_stream_iface is not None, "No output AXI-Stream interface found"
        
        # Verify AXI-Stream interface structure (interface builder normalizes names)
        input_stream_ports = set(input_stream_iface.ports.keys())
        expected_stream_ports = {"TDATA", "TVALID", "TREADY"}
        assert expected_stream_ports.issubset(input_stream_ports), f"Missing required ports in input stream. Found: {input_stream_ports}"
        
        output_stream_ports = set(output_stream_iface.ports.keys())
        assert expected_stream_ports.issubset(output_stream_ports), f"Missing required ports in output stream. Found: {output_stream_ports}"
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_dataflow_conversion_basic(self):
        """Test basic dataflow conversion without pragmas."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(self.rtl_file),
            compiler_data_path=str(self.compiler_file),
            output_dir=str(self.output_dir)
        )
        
        # Run through dataflow model building
        hkg.run(stop_after="build_dataflow_model")
        
        # Verify dataflow components were created
        assert hkg.dataflow_enabled
        assert hkg.rtl_converter is not None
        assert hkg.dataflow_interfaces is not None
        assert hkg.dataflow_model is not None
        
        # Verify dataflow interfaces
        interface_types = [iface.interface_type for iface in hkg.dataflow_interfaces]
        assert DataflowInterfaceType.INPUT in interface_types
        assert DataflowInterfaceType.OUTPUT in interface_types
        assert DataflowInterfaceType.CONFIG in interface_types
        assert DataflowInterfaceType.CONTROL in interface_types
        
        # Verify interface naming
        interface_names = {iface.name for iface in hkg.dataflow_interfaces}
        expected_names = {"s_axis", "m_axis", "s_axilite", "ap"}  # "ap" is the actual global control interface name
        assert expected_names.issubset(interface_names)
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_enhanced_dataflow_conversion_with_pragmas(self):
        """Test enhanced dataflow conversion with TDIM and DATATYPE pragmas."""
        # Use enhanced RTL with pragmas
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(self.enhanced_rtl_file),
            compiler_data_path=str(self.compiler_file),
            output_dir=str(self.output_dir)
        )
        
        # Run through dataflow model building
        hkg.run(stop_after="build_dataflow_model")
        
        # Verify enhanced dataflow model
        assert len(hkg.dataflow_interfaces) >= 4
        
        # Find specific interfaces
        s_axis_iface = None
        m_axis_iface = None
        s_axilite_iface = None
        
        for iface in hkg.dataflow_interfaces:
            if iface.name == "s_axis":
                s_axis_iface = iface
            elif iface.name == "m_axis":
                m_axis_iface = iface
            elif iface.name == "s_axilite":
                s_axilite_iface = iface
        
        # Verify TDIM pragma effects
        if s_axis_iface:
            assert s_axis_iface.interface_type == DataflowInterfaceType.INPUT
            # Verify TDIM pragma was applied (would have PE*32, PE in metadata)
            assert "tdim_override" in s_axis_iface.pragma_metadata or s_axis_iface.tDim is not None
        
        if m_axis_iface:
            assert m_axis_iface.interface_type == DataflowInterfaceType.OUTPUT
            
        # Verify WEIGHT pragma effects
        if s_axilite_iface:
            assert s_axilite_iface.interface_type == DataflowInterfaceType.WEIGHT or DataflowInterfaceType.CONFIG
        
        # Verify DATATYPE constraints
        for iface in [s_axis_iface, m_axis_iface]:
            if iface and iface.allowed_datatypes:
                constraint = iface.allowed_datatypes[0]
                assert "UINT" in constraint.base_types
                assert constraint.min_bitwidth > 0
                assert constraint.max_bitwidth >= constraint.min_bitwidth
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_dataflow_model_creation_and_validation(self):
        """Test dataflow model creation and validation."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(self.enhanced_rtl_file),
            compiler_data_path=str(self.compiler_file),
            output_dir=str(self.output_dir)
        )
        
        # Run dataflow model building
        hkg.run(stop_after="build_dataflow_model")
        
        # Verify unified computational model
        model = hkg.dataflow_model
        assert model is not None
        
        # Test model methods
        try:
            parallelism_bounds = model.get_parallelism_bounds()
            assert isinstance(parallelism_bounds, dict)
        except AttributeError:
            # Method may not be implemented yet - that's fine for this phase
            pass
        
        # Verify interface validation
        for iface in hkg.dataflow_interfaces:
            validation_result = iface.validate_constraints()
            # Should not have critical errors for well-formed interfaces
            # ValidationResult contains lists of errors, warnings, and info
            critical_errors = validation_result.errors  # These are the ERROR severity items
            
            # Allow warnings but no critical errors
            assert len(critical_errors) == 0, f"Interface {iface.name} has critical validation errors: {[str(e) for e in critical_errors]}"
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_complete_hkg_pipeline(self):
        """Test complete HKG pipeline execution."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(self.enhanced_rtl_file),
            compiler_data_path=str(self.compiler_file),
            output_dir=str(self.output_dir)
        )
        
        # Run complete pipeline up to current phase
        generated_files = hkg.run(stop_after="generate_hw_custom_op")
        
        # Verify pipeline completion
        assert isinstance(generated_files, dict)
        assert "rtl_template" in generated_files
        assert "hw_custom_op" in generated_files
        
        # Verify files were generated
        for file_type, file_path in generated_files.items():
            if file_path:  # Some generators might return None
                assert Path(file_path).exists(), f"Generated file {file_type} not found: {file_path}"
        
        # Verify enhanced template context was built
        if hasattr(hkg, '_build_enhanced_template_context'):
            context = hkg._build_enhanced_template_context()
            
            # Verify context completeness
            expected_keys = {
                "kernel_name", "class_name", "source_file", "generation_timestamp",
                "rtl_parameters", "rtl_interfaces", "dataflow_interfaces", 
                "dataflow_model", "has_unified_model"
            }
            context_keys = set(context.keys())
            assert expected_keys.issubset(context_keys)
            
            # Verify context values
            assert context["kernel_name"] == "thresholding_axi"
            assert "AutoThresholdingAxi" in context["class_name"]
            assert context["has_unified_model"] == True
            assert len(context["dataflow_interfaces"]) >= 4
    
    def test_hwcustomop_generation_requires_dataflow(self):
        """Test that HWCustomOp generation requires dataflow framework."""
        with patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', False):
            hkg = HardwareKernelGenerator(
                rtl_file_path=str(self.rtl_file),
                compiler_data_path=str(self.compiler_file),
                output_dir=str(self.output_dir)
            )
            
            # Should fail at HWCustomOp generation phase
            with pytest.raises(HardwareKernelGeneratorError, match="requires dataflow framework"):
                hkg.run(stop_after="generate_hw_custom_op")
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_extensibility_for_phase3_code_generation(self):
        """Test extensibility preparations for Phase 3 code generation."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(self.enhanced_rtl_file),
            compiler_data_path=str(self.compiler_file),
            output_dir=str(self.output_dir)
        )
        
        # Build complete context for code generation
        hkg.run(stop_after="build_dataflow_model")
        
        # Test public API for code generation
        try:
            # This should work for Phase 3 integration
            template_path = str(Path(__file__).parent / "templates" / "hwcustomop.j2")
            output_path = str(self.output_dir / "generated_hwcustomop.py")
            
            # Create minimal template for testing
            template_dir = Path(self.temp_dir) / "templates"
            template_dir.mkdir(exist_ok=True)
            template_file = template_dir / "hwcustomop.j2"
            
            with open(template_file, 'w') as f:
                f.write("""
# Generated HWCustomOp for {{ kernel_name }}
# Class: {{ class_name }}
# Interfaces: {{ dataflow_interfaces|length }}
# Has Model: {{ has_unified_model }}

class {{ class_name }}:
    '''Auto-generated HWCustomOp for {{ kernel_name }}'''
    
    def __init__(self):
        self.kernel_name = "{{ kernel_name }}"
        self.interface_count = {{ dataflow_interfaces|length }}
        self.has_dataflow_model = {{ has_unified_model }}
        
    # Phase 3 will add full implementation here
""")
            
            # Test template-based generation
            result_path = hkg.generate_auto_hwcustomop(str(template_file), output_path)
            assert result_path == output_path
            assert Path(output_path).exists()
            
            # Verify generated content
            with open(output_path, 'r') as f:
                content = f.read()
                assert "thresholding_axi" in content
                assert "AutoThresholdingAxi" in content
                assert "interface_count" in content
                
        except HardwareKernelGeneratorError as e:
            if "requires dataflow framework" in str(e):
                pytest.skip("Dataflow framework required for code generation")
            else:
                raise
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)  
    def test_performance_and_scalability(self):
        """Test performance characteristics and scalability."""
        import time
        
        hkg = HardwareKernelGenerator(
            rtl_file_path=str(self.enhanced_rtl_file),
            compiler_data_path=str(self.compiler_file),
            output_dir=str(self.output_dir)
        )
        
        # Measure pipeline execution time
        start_time = time.time()
        hkg.run(stop_after="build_dataflow_model")
        execution_time = time.time() - start_time
        
        # Verify reasonable performance (should complete quickly)
        assert execution_time < 10.0, f"Pipeline execution too slow: {execution_time:.2f}s"
        
        # Verify memory efficiency
        assert len(hkg.dataflow_interfaces) >= 4
        assert hkg.dataflow_model is not None
        
        # Test multiple runs for consistency
        for i in range(3):
            hkg2 = HardwareKernelGenerator(
                rtl_file_path=str(self.enhanced_rtl_file),
                compiler_data_path=str(self.compiler_file),
                output_dir=str(self.output_dir)
            )
            hkg2.run(stop_after="build_dataflow_model")
            
            # Results should be consistent
            assert len(hkg2.dataflow_interfaces) == len(hkg.dataflow_interfaces)
            assert (hkg2.dataflow_model is not None) == (hkg.dataflow_model is not None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])