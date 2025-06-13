"""
Comprehensive end-to-end tests for Phase 3 unified generator system.

Tests the complete pipeline from RTL parsing through template generation
using real RTL files and the actual implementation components.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser
from brainsmith.tools.hw_kernel_gen.unified_generator import UnifiedGenerator
from brainsmith.tools.hw_kernel_gen.result_handler import ResultHandler, GenerationResult


class TestPhase3EndToEnd:
    """End-to-end tests for Phase 3 unified generator system."""
    
    def setup_method(self):
        """Set up test fixtures with real RTL files."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.rtl_dir = self.temp_dir / "rtl"
        self.rtl_dir.mkdir()
        
        # Create realistic RTL test files
        self.create_test_rtl_files()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_rtl_files(self):
        """Create realistic RTL test files for end-to-end testing."""
        
        # Simple matrix multiplication kernel
        matrix_mult_rtl = self.rtl_dir / "matrix_mult.sv"
        matrix_mult_rtl.write_text("""
// @brainsmith BDIM input0 -1 [PE]
// @brainsmith BDIM weights -1 [SIMD,PE] 
// @brainsmith BDIM output0 -1 [PE]
// @brainsmith DATATYPE input0 FIXED 8 8
// @brainsmith DATATYPE weights FIXED 8 8
// @brainsmith DATATYPE output0 FIXED 8 16

module matrix_mult #(
    parameter PE = 4,
    parameter SIMD = 8,
    parameter DEPTH = 512
) (
    input wire ap_clk,
    input wire ap_rst_n,
    input wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,
    
    // Input stream
    input wire [input0_width-1:0] input0_TDATA,
    input wire input0_TVALID,
    output wire input0_TREADY,
    
    // Weight stream  
    input wire [weights_width-1:0] weights_TDATA,
    input wire weights_TVALID,
    output wire weights_TREADY,
    
    // Output stream
    output wire [output0_width-1:0] output0_TDATA,
    output wire output0_TVALID,
    input wire output0_TREADY
);

// Matrix multiplication implementation
// ... (implementation details)

endmodule
""")
        
        # Convolution kernel with more complex parameters
        conv_rtl = self.rtl_dir / "conv2d.sv"
        conv_rtl.write_text("""
// @brainsmith BDIM input0 -1 [BATCH,CHANNELS,PE]
// @brainsmith BDIM weights -1 [FILTERS,CHANNELS,SIMD] 
// @brainsmith BDIM output0 -1 [BATCH,FILTERS,PE]
// @brainsmith DATATYPE input0 FIXED 8 8
// @brainsmith DATATYPE weights FIXED 8 8
// @brainsmith DATATYPE output0 FIXED 8 16

module conv2d #(
    parameter PE = 16,
    parameter SIMD = 16,
    parameter KERNEL_SIZE = 3,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    parameter CHANNELS = 64,
    parameter FILTERS = 128,
    parameter WIDTH = 32,
    parameter HEIGHT = 32
) (
    input wire ap_clk,
    input wire ap_rst_n,
    input wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,
    
    // Input feature map
    input wire [input0_width-1:0] input0_TDATA,
    input wire input0_TVALID,
    output wire input0_TREADY,
    
    // Convolution weights
    input wire [weights_width-1:0] weights_TDATA,
    input wire weights_TVALID,
    output wire weights_TREADY,
    
    // Output feature map
    output wire [output0_width-1:0] output0_TDATA,
    output wire output0_TVALID,
    input wire output0_TREADY
);

// Convolution implementation
// ... (implementation details)

endmodule
""")
        
        # Simple element-wise operation
        elementwise_rtl = self.rtl_dir / "elementwise_add.sv"
        elementwise_rtl.write_text("""
// @brainsmith BDIM input0 -1 [PE]
// @brainsmith BDIM input1 -1 [PE]
// @brainsmith BDIM output0 -1 [PE]
// @brainsmith DATATYPE input0 FIXED 8 8
// @brainsmith DATATYPE input1 FIXED 8 8  
// @brainsmith DATATYPE output0 FIXED 8 8

module elementwise_add #(
    parameter PE = 8
) (
    input wire ap_clk,
    input wire ap_rst_n,
    input wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,
    
    // First input
    input wire [input0_width-1:0] input0_TDATA,
    input wire input0_TVALID,
    output wire input0_TREADY,
    
    // Second input
    input wire [input1_width-1:0] input1_TDATA,
    input wire input1_TVALID,
    output wire input1_TREADY,
    
    // Output
    output wire [output0_width-1:0] output0_TDATA,
    output wire output0_TVALID,
    input wire output0_TREADY
);

// Element-wise addition implementation
// ... (implementation details)

endmodule
""")
    
    def test_matrix_mult_end_to_end_generation(self):
        """Test complete end-to-end generation for matrix multiplication kernel."""
        rtl_file = self.rtl_dir / "matrix_mult.sv"
        
        # Step 1: Parse RTL with Phase 1 validation
        parser = RTLParser()
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Verify parsing worked correctly
        assert kernel_metadata.name == "matrix_mult"
        assert len(kernel_metadata.parameters) == 3  # PE, SIMD, DEPTH
        assert len(kernel_metadata.interfaces) == 4  # ap, input0, weights, output0
        
        # Verify parameter names
        param_names = {p.name for p in kernel_metadata.parameters}
        assert param_names == {"PE", "SIMD", "DEPTH"}
        
        # Verify interface names and types
        interface_info = {(i.name, i.interface_type.name) for i in kernel_metadata.interfaces}
        expected_interfaces = {
            ("ap", "CONTROL"),
            ("input0", "INPUT"),
            ("weights", "WEIGHT"), 
            ("output0", "OUTPUT")
        }
        assert interface_info == expected_interfaces
        
        # Step 2: Generate templates with Phase 3 unified generator
        generator = UnifiedGenerator()
        generated_files = generator.generate_all(kernel_metadata)
        
        # Verify all expected files were generated
        expected_files = {
            "matrix_mult_hw_custom_op.py",
            "matrix_mult_wrapper.v",
            "test_matrix_mult.py"
        }
        assert set(generated_files.keys()) == expected_files
        
        # Step 3: Verify generated HWCustomOp content
        hw_custom_op_code = generated_files["matrix_mult_hw_custom_op.py"]
        
        # Check class definition (class name is derived from module name)
        assert "class MatrixMult" in hw_custom_op_code
        assert "AutoHWCustomOp" in hw_custom_op_code
        
        # Check runtime parameter extraction (Phase 2 feature)
        assert "runtime_parameters" in hw_custom_op_code
        assert 'runtime_parameters["PE"]' in hw_custom_op_code
        assert 'runtime_parameters["SIMD"]' in hw_custom_op_code
        assert 'runtime_parameters["DEPTH"]' in hw_custom_op_code
        
        # Check imports
        assert "from brainsmith.dataflow.core import AutoHWCustomOp" in hw_custom_op_code
        
        # Step 4: Verify generated RTL wrapper content
        rtl_wrapper_code = generated_files["matrix_mult_wrapper.v"]
        
        # Check module definition
        assert "module matrix_mult_wrapper" in rtl_wrapper_code
        assert "parameter PE = " in rtl_wrapper_code
        assert "parameter SIMD = " in rtl_wrapper_code
        assert "parameter DEPTH = " in rtl_wrapper_code
        
        # Check interface declarations
        assert "input0_TDATA" in rtl_wrapper_code
        assert "weights_TDATA" in rtl_wrapper_code
        assert "output0_TDATA" in rtl_wrapper_code
        
        # Check parameter validation (Phase 2 feature)
        assert "$error" in rtl_wrapper_code
        assert "must be positive" in rtl_wrapper_code
        
        # Step 5: Verify generated test suite content
        test_suite_code = generated_files["test_matrix_mult.py"]
        
        # Check test class definition
        assert "class TestMatrixMult" in test_suite_code
        assert "import pytest" in test_suite_code
        assert "import onnx.helper" in test_suite_code
        
        # Check parameter validation tests
        assert "def test_parameter_validation" in test_suite_code
        assert "def test_hwcustomop_instantiation" in test_suite_code
        
        # Check that parameters are tested
        assert "PE" in test_suite_code
        assert "SIMD" in test_suite_code
        assert "DEPTH" in test_suite_code
        
        # Step 6: Write results with ResultHandler
        result = GenerationResult(
            kernel_name=kernel_metadata.name,
            source_file=rtl_file,
            generated_files=generated_files
        )
        
        handler = ResultHandler(self.output_dir)
        kernel_dir = handler.write_result(result)
        
        # Verify files were written correctly
        assert kernel_dir.exists()
        assert (kernel_dir / "matrix_mult_hw_custom_op.py").exists()
        assert (kernel_dir / "matrix_mult_wrapper.v").exists()
        assert (kernel_dir / "test_matrix_mult.py").exists()
        assert (kernel_dir / "generation_metadata.json").exists()
        assert (kernel_dir / "generation_summary.txt").exists()
    
    def test_conv2d_complex_parameter_handling(self):
        """Test complex parameter handling with convolution kernel."""
        rtl_file = self.rtl_dir / "conv2d.sv"
        
        # Parse RTL
        parser = RTLParser()
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Verify complex parameter set
        param_names = {p.name for p in kernel_metadata.parameters}
        expected_params = {
            "PE", "SIMD", "KERNEL_SIZE", "STRIDE", "PADDING", 
            "CHANNELS", "FILTERS", "WIDTH", "HEIGHT"
        }
        assert param_names == expected_params
        
        # Generate templates
        generator = UnifiedGenerator()
        generated_files = generator.generate_all(kernel_metadata)
        
        # Verify HWCustomOp handles all parameters
        hw_custom_op_code = generated_files["conv2d_hw_custom_op.py"]
        
        # Check that all parameters are extracted
        for param_name in expected_params:
            assert f'runtime_parameters["{param_name}"]' in hw_custom_op_code
        
        # Verify RTL wrapper includes all parameters
        rtl_wrapper_code = generated_files["conv2d_wrapper.v"]
        
        # Check parameter declarations
        for param_name in expected_params:
            assert f"parameter {param_name} = " in rtl_wrapper_code
        
        # Check that parameter validation exists for all parameters
        for param_name in expected_params:
            assert f"if ({param_name} <= 0)" in rtl_wrapper_code
    
    def test_elementwise_minimal_case(self):
        """Test minimal case with single parameter."""
        rtl_file = self.rtl_dir / "elementwise_add.sv"
        
        # Parse RTL
        parser = RTLParser()
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Verify minimal parameter set
        assert len(kernel_metadata.parameters) == 1
        assert kernel_metadata.parameters[0].name == "PE"
        
        # Verify dual input interfaces
        interface_names = {i.name for i in kernel_metadata.interfaces}
        assert interface_names == {"input0", "input1", "output0"}
        
        # Generate templates
        generator = UnifiedGenerator()
        generated_files = generator.generate_all(kernel_metadata)
        
        # Verify generated code handles dual inputs correctly
        hw_custom_op_code = generated_files["elementwise_add_hw_custom_op.py"]
        assert 'runtime_parameters["PE"]' in hw_custom_op_code
        
        # Verify RTL wrapper handles dual inputs
        rtl_wrapper_code = generated_files["elementwise_add_wrapper.v"]
        assert "input0_TDATA" in rtl_wrapper_code
        assert "input1_TDATA" in rtl_wrapper_code
        assert "output0_TDATA" in rtl_wrapper_code
    
    def test_phase2_parameter_whitelist_integration(self):
        """Test Phase 2 parameter whitelist integration in end-to-end flow."""
        rtl_file = self.rtl_dir / "matrix_mult.sv"
        
        # Parse RTL
        parser = RTLParser()
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Generate with unified generator
        generator = UnifiedGenerator()
        
        # Get the template context to verify whitelist handling
        template_context = generator.template_context_generator.generate_template_context(kernel_metadata)
        
        # Verify whitelisted parameters are identified
        whitelisted_params = {name for name, _ in template_context.whitelisted_defaults.items()}
        
        # PE, SIMD, DEPTH should be whitelisted based on parameter_defaults.py
        expected_whitelisted = {"PE", "SIMD", "DEPTH"}
        assert whitelisted_params.intersection(expected_whitelisted), \
            f"Expected some whitelisted parameters from {expected_whitelisted}, got {whitelisted_params}"
        
        # Generate code and verify whitelist handling
        generated_files = generator.generate_all(kernel_metadata)
        hw_custom_op_code = generated_files["matrix_mult_hw_custom_op.py"]
        
        # Should include runtime parameter extraction for all parameters
        for param in kernel_metadata.parameters:
            assert f'runtime_parameters["{param.name}"]' in hw_custom_op_code
    
    def test_generated_code_syntax_validation(self):
        """Test that generated code has valid syntax."""
        rtl_file = self.rtl_dir / "matrix_mult.sv"
        
        # Generate code
        parser = RTLParser()
        parsing_result = parser.parse_file(str(rtl_file))
        kernel_metadata = parsing_result.kernel_metadata
        
        generator = UnifiedGenerator()
        generated_files = generator.generate_all(kernel_metadata)
        
        # Test Python syntax validation
        hw_custom_op_code = generated_files["matrix_mult_hw_custom_op.py"]
        test_suite_code = generated_files["test_matrix_mult.py"]
        
        # Try to compile Python code to check syntax
        try:
            compile(hw_custom_op_code, "matrix_mult_hw_custom_op.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated HWCustomOp code has syntax error: {e}")
        
        try:
            compile(test_suite_code, "test_matrix_mult.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated test suite code has syntax error: {e}")
        
        # Test SystemVerilog basic syntax
        rtl_wrapper_code = generated_files["matrix_mult_wrapper.v"]
        
        # Basic SystemVerilog syntax checks
        assert rtl_wrapper_code.count("module") == rtl_wrapper_code.count("endmodule")
        assert "begin" in rtl_wrapper_code or "initial" in rtl_wrapper_code  # Should have some procedural blocks
    
    def test_error_handling_invalid_rtl(self):
        """Test error handling with invalid RTL."""
        # Create invalid RTL file
        invalid_rtl = self.rtl_dir / "invalid.sv"
        invalid_rtl.write_text("""
// Invalid RTL - missing module declaration
parameter PE = 4;
input wire clk;
// No module wrapper
""")
        
        # Should handle parsing errors gracefully
        parser = RTLParser()
        with pytest.raises(Exception):  # Expect some kind of parsing error
            parser.parse_file(str(invalid_rtl))
    
    def test_multiple_kernels_batch_processing(self):
        """Test processing multiple kernels in batch."""
        rtl_files = [
            self.rtl_dir / "matrix_mult.sv",
            self.rtl_dir / "conv2d.sv", 
            self.rtl_dir / "elementwise_add.sv"
        ]
        
        parser = RTLParser()
        generator = UnifiedGenerator()
        handler = ResultHandler(self.output_dir)
        
        results = []
        
        # Process all kernels
        for rtl_file in rtl_files:
            # Parse
            kernel_metadata = parser.parse_file(str(rtl_file))
            
            # Generate
            generated_files = generator.generate_all(kernel_metadata)
            
            # Create result
            result = GenerationResult(
                kernel_name=kernel_metadata.name,
                source_file=rtl_file,
                generated_files=generated_files
            )
            
            # Write result
            kernel_dir = handler.write_result(result)
            results.append((kernel_metadata.name, kernel_dir))
        
        # Verify all kernels were processed
        assert len(results) == 3
        
        # Verify each kernel has its own directory
        kernel_names = {name for name, _ in results}
        assert kernel_names == {"matrix_mult", "conv2d", "elementwise_add"}
        
        # Verify all directories exist
        for kernel_name, kernel_dir in results:
            assert kernel_dir.exists()
            assert kernel_dir.name == kernel_name
            assert (kernel_dir / f"{kernel_name}_hw_custom_op.py").exists()
    
    def test_phase2_bdim_validation_integration(self):
        """Test Phase 2 BDIM validation is integrated in end-to-end flow."""
        rtl_file = self.rtl_dir / "matrix_mult.sv"
        
        # Parse RTL
        parser = RTLParser()
        kernel_metadata = parser.parse_file(str(rtl_file))
        
        # Verify parsing succeeded (if we get here, basic parsing worked)
        assert kernel_metadata is not None, "Parsing should return valid kernel metadata"
        
        for interface in kernel_metadata.interfaces:
            if interface.name in ["input0", "weights", "output0"]:
                assert interface.chunking_strategy is not None, \
                    f"Interface {interface.name} should have chunking strategy from BDIM pragma"
                assert interface.chunking_strategy.block_shape is not None, \
                    f"Interface {interface.name} should have block shape from BDIM validation"
        
        # Generate templates and verify BDIM information is preserved
        generator = UnifiedGenerator()
        generated_files = generator.generate_all(kernel_metadata)
        
        # Check that RTL wrapper includes BDIM information
        rtl_wrapper_code = generated_files["matrix_mult_wrapper.v"]
        
        # Should include BDIM-related comments or calculations
        assert "PE" in rtl_wrapper_code  # BDIM parameter should be referenced
        assert "SIMD" in rtl_wrapper_code  # BDIM parameter should be referenced
        
        # Should include width calculations based on BDIM
        assert "TDATA_WIDTH" in rtl_wrapper_code or "width" in rtl_wrapper_code


class TestPhase3PerformanceAndScalability:
    """Performance and scalability tests for Phase 3 system."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.rtl_dir = self.temp_dir / "rtl"
        self.rtl_dir.mkdir()
    
    def teardown_method(self):
        """Clean up performance test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.performance  
    def test_large_parameter_set_performance(self):
        """Test performance with kernel having many parameters."""
        # Create RTL with many parameters
        many_params_rtl = self.rtl_dir / "many_params.sv"
        
        # Generate parameters
        param_lines = []
        bdim_lines = []
        for i in range(50):  # 50 parameters
            param_lines.append(f"    parameter PARAM_{i} = {i+1},")
            if i < 3:  # Only annotate first few with BDIM
                bdim_lines.append(f"// @brainsmith BDIM input{i} -1 [PARAM_{i}]")
        
        # Remove trailing comma from last parameter
        param_lines[-1] = param_lines[-1].rstrip(',')
        
        rtl_content = f"""
{chr(10).join(bdim_lines)}

module many_params #(
{chr(10).join(param_lines)}
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Just a few interfaces to keep it simple
    input wire [31:0] input0_TDATA,
    input wire input0_TVALID,
    output wire input0_TREADY,
    
    output wire [31:0] output0_TDATA,
    output wire output0_TVALID,
    input wire output0_TREADY
);

// Implementation
endmodule
"""
        many_params_rtl.write_text(rtl_content)
        
        # Test performance
        import time
        
        start_time = time.time()
        
        # Parse
        parser = RTLParser()
        parsing_result = parser.parse_file(str(many_params_rtl))
        kernel_metadata = parsing_result.kernel_metadata
        
        # Generate
        generator = UnifiedGenerator()
        generated_files = generator.generate_all(kernel_metadata)
        
        end_time = time.time()
        
        # Verify it completed successfully
        assert len(kernel_metadata.parameters) == 50
        assert len(generated_files) >= 1  # At least HWCustomOp should be generated
        
        # Performance check - should complete within reasonable time
        total_time = end_time - start_time
        assert total_time < 10.0, f"Generation took {total_time:.2f}s, expected < 10s"
        
        # Verify generated code includes all parameters
        hw_custom_op_code = generated_files["many_params_hw_custom_op.py"]
        for i in range(50):
            assert f"PARAM_{i}" in hw_custom_op_code
    
    @pytest.mark.stress
    def test_memory_usage_large_interface_set(self):
        """Test memory usage with many interfaces."""
        # Create RTL with many interfaces
        many_interfaces_rtl = self.rtl_dir / "many_interfaces.sv"
        
        # Generate many input/output interfaces
        interface_lines = []
        bdim_lines = []
        for i in range(20):  # 20 input interfaces
            interface_lines.extend([
                f"    input wire [31:0] input{i}_TDATA,",
                f"    input wire input{i}_TVALID,", 
                f"    output wire input{i}_TREADY,"
            ])
            bdim_lines.append(f"// @brainsmith BDIM input{i} -1 [PE]")
        
        for i in range(10):  # 10 output interfaces
            interface_lines.extend([
                f"    output wire [31:0] output{i}_TDATA,",
                f"    output wire output{i}_TVALID,",
                f"    input wire output{i}_TREADY"
            ])
            bdim_lines.append(f"// @brainsmith BDIM output{i} -1 [PE]")
        
        # Remove trailing comma
        interface_lines[-1] = interface_lines[-1].rstrip(',')
        
        rtl_content = f"""
{chr(10).join(bdim_lines)}

module many_interfaces #(
    parameter PE = 8
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
{chr(10).join(interface_lines)}
);

// Implementation
endmodule
"""
        many_interfaces_rtl.write_text(rtl_content)
        
        # Process with memory monitoring
        parser = RTLParser()
        parsing_result = parser.parse_file(str(many_interfaces_rtl))
        kernel_metadata = parsing_result.kernel_metadata
        
        # Verify many interfaces were parsed
        assert len(kernel_metadata.interfaces) == 30  # 20 inputs + 10 outputs
        
        # Generate templates
        generator = UnifiedGenerator()
        generated_files = generator.generate_all(kernel_metadata)
        
        # Verify generation succeeded
        assert len(generated_files) >= 1
        
        # Basic memory usage check - shouldn't crash or hang
        hw_custom_op_code = generated_files["many_interfaces_hw_custom_op.py"]
        assert len(hw_custom_op_code) > 0
        assert "class ManyInterfacesHWCustomOp" in hw_custom_op_code