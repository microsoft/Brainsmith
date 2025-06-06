"""
Tests for enhanced Hardware Kernel Generator with dataflow modeling support.

This module tests the HKG enhancements for Phase 2 of the Interface-Wise 
Dataflow Modeling Framework.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator, HardwareKernelGeneratorError


class TestEnhancedHKG:
    """Test enhanced HKG with dataflow modeling capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.rtl_file = os.path.join(self.temp_dir, "test.sv")
        self.compiler_file = os.path.join(self.temp_dir, "compiler_data.py")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create minimal test files
        with open(self.rtl_file, 'w') as f:
            f.write("module test(); endmodule")
            
        with open(self.compiler_file, 'w') as f:
            f.write("# Test compiler data\nonnx_metadata = {}")
            
        os.makedirs(self.output_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_hkg_initialization_with_dataflow(self):
        """Test HKG initialization when dataflow framework is available."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        assert hkg.dataflow_enabled == True
        assert hkg.dataflow_interfaces is None  # Not built yet
        assert hkg.dataflow_model is None  # Not built yet
        assert hkg.rtl_converter is None  # Not initialized yet
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', False)
    def test_hkg_initialization_without_dataflow(self):
        """Test HKG initialization when dataflow framework is not available."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        assert hkg.dataflow_enabled == False
        assert hkg.dataflow_interfaces is None
        assert hkg.dataflow_model is None
        assert hkg.rtl_converter is None
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_build_dataflow_model_success(self):
        """Test successful dataflow model building."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        # Mock RTL data
        mock_hw_kernel = Mock()
        mock_hw_kernel.interfaces = {"in0": Mock(), "out0": Mock()}
        mock_hw_kernel.parameters = [Mock(name="PE", default_value="4")]
        hkg.hw_kernel_data = mock_hw_kernel
        
        # Mock converter and its methods
        with patch('brainsmith.tools.hw_kernel_gen.hkg.RTLInterfaceConverter') as mock_converter_class:
            mock_converter = Mock()
            mock_converter.convert_interfaces.return_value = [Mock(), Mock()]
            mock_converter_class.return_value = mock_converter
            
            with patch('brainsmith.tools.hw_kernel_gen.hkg.validate_conversion_result') as mock_validate:
                mock_validate.return_value = []  # No errors
                
                with patch('brainsmith.tools.hw_kernel_gen.hkg.DataflowModel') as mock_model_class:
                    mock_model = Mock()
                    mock_model_class.return_value = mock_model
                    
                    # Test the method
                    hkg._build_dataflow_model()
                    
                    # Verify results
                    assert hkg.rtl_converter is not None
                    assert hkg.dataflow_interfaces is not None
                    assert len(hkg.dataflow_interfaces) == 2
                    assert hkg.dataflow_model == mock_model
                    
                    # Verify method calls
                    mock_converter.convert_interfaces.assert_called_once()
                    mock_validate.assert_called_once()
                    mock_model_class.assert_called_once()
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', False)
    def test_build_dataflow_model_disabled(self):
        """Test dataflow model building when framework is disabled."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        # Mock RTL data
        hkg.hw_kernel_data = Mock()
        
        # Test the method
        hkg._build_dataflow_model()
        
        # Verify no dataflow components were created
        assert hkg.rtl_converter is None
        assert hkg.dataflow_interfaces is None
        assert hkg.dataflow_model is None
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_enhanced_hwcustomop_generation(self):
        """Test enhanced HWCustomOp generation with dataflow modeling."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        # Mock required data
        mock_hw_kernel = Mock()
        mock_hw_kernel.name = "test_kernel"
        hkg.hw_kernel_data = mock_hw_kernel
        
        hkg.dataflow_interfaces = [Mock(), Mock()]
        hkg.dataflow_model = Mock()
        
        # Mock template context generation
        with patch.object(hkg, '_build_enhanced_template_context') as mock_context:
            mock_context.return_value = {"test": "context"}
            
            # Test enhanced generation
            output_file = hkg._generate_auto_hwcustomop_with_dataflow()
            
            # Verify output file was created
            assert output_file.exists()
            assert "autotestkernel" in str(output_file).lower()
            
            # Verify template context was built
            mock_context.assert_called_once()
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_hwcustomop_generation_requires_dataflow_model(self):
        """Test that HWCustomOp generation requires successful dataflow modeling."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        # Mock required data
        mock_hw_kernel = Mock()
        mock_hw_kernel.name = "test_kernel"
        mock_hw_kernel.interfaces = {"in0": Mock()}
        hkg.hw_kernel_data = mock_hw_kernel
        
        # No dataflow components (simulating failure)
        hkg.dataflow_interfaces = None
        hkg.dataflow_model = None
        
        # Test should raise error when dataflow model is not available
        with pytest.raises(HardwareKernelGeneratorError, match="requires successful dataflow model initialization"):
            hkg._generate_hw_custom_op()
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_template_context_building(self):
        """Test enhanced template context building."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        # Mock required data
        mock_hw_kernel = Mock()
        mock_hw_kernel.name = "test_kernel"
        mock_hw_kernel.parameters = []
        mock_hw_kernel.interfaces = {}
        mock_hw_kernel.pragmas = []
        hkg.hw_kernel_data = mock_hw_kernel
        
        # Mock dataflow components
        mock_interface = Mock()
        mock_interface.interface_type = Mock()
        mock_interface.interface_type.__str__ = Mock(return_value="DataflowInterfaceType.INPUT")
        hkg.dataflow_interfaces = [mock_interface]
        
        mock_model = Mock()
        mock_model.get_parallelism_bounds.return_value = {"test": "bounds"}
        hkg.dataflow_model = mock_model
        
        hkg.compiler_data_module = Mock()
        
        # Test context building
        context = hkg._build_enhanced_template_context()
        
        # Verify context structure
        assert "kernel_name" in context
        assert "class_name" in context
        assert "source_file" in context
        assert "generation_timestamp" in context
        assert "dataflow_interfaces" in context
        assert "dataflow_model" in context
        assert "has_unified_model" in context
        assert "parallelism_bounds" in context
        assert "compiler_data_available" in context
        
        # Verify values
        assert context["kernel_name"] == "test_kernel"
        assert context["class_name"] == "AutoTestKernel"
        assert context["has_unified_model"] == True
        assert context["compiler_data_available"] == True
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_public_generate_auto_hwcustomop_method(self):
        """Test public AutoHWCustomOp generation method."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        # Mock RTL parsing to avoid file format issues
        mock_hw_kernel = Mock()
        mock_hw_kernel.name = "test_kernel"
        mock_hw_kernel.parameters = []
        mock_hw_kernel.interfaces = {}
        mock_hw_kernel.pragmas = []
        hkg.hw_kernel_data = mock_hw_kernel
        
        # Mock dataflow model and interfaces
        hkg.dataflow_model = Mock()
        hkg.dataflow_interfaces = []  # Empty list to avoid None iteration
        
        # Mock template rendering
        with patch('jinja2.Environment') as mock_env_class:
            mock_env = Mock()
            mock_template = Mock()
            mock_template.render.return_value = "# Generated code"
            mock_env.get_template.return_value = mock_template
            mock_env_class.return_value = mock_env
            
            # Test template path and output path
            template_path = os.path.join(self.temp_dir, "template.j2")
            output_path = os.path.join(self.output_dir, "generated.py")
            
            # Create template file
            with open(template_path, 'w') as f:
                f.write("# Template")
            
            # Test the method
            result_path = hkg.generate_auto_hwcustomop(template_path, output_path)
            
            # Verify results
            assert result_path == output_path
            assert os.path.exists(output_path)
            
            # Verify template was rendered
            mock_template.render.assert_called_once()
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', False)
    def test_public_generate_auto_hwcustomop_method_disabled(self):
        """Test public AutoHWCustomOp generation method when dataflow is disabled."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        template_path = os.path.join(self.temp_dir, "template.j2")
        output_path = os.path.join(self.output_dir, "generated.py")
        
        # Test should raise error when dataflow is disabled
        with pytest.raises(HardwareKernelGeneratorError, match="requires dataflow framework"):
            hkg.generate_auto_hwcustomop(template_path, output_path)
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_enhanced_run_pipeline(self):
        """Test enhanced run pipeline with dataflow model building phase."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        # Mock all phase methods
        with patch.object(hkg, '_parse_rtl') as mock_parse_rtl, \
             patch.object(hkg, '_parse_compiler_data') as mock_parse_compiler, \
             patch.object(hkg, '_load_custom_documentation') as mock_load_doc, \
             patch.object(hkg, '_build_dataflow_model') as mock_build_dataflow, \
             patch.object(hkg, '_generate_rtl_template') as mock_gen_rtl, \
             patch.object(hkg, '_generate_hw_custom_op') as mock_gen_hwop, \
             patch.object(hkg, '_generate_rtl_backend') as mock_gen_backend, \
             patch.object(hkg, '_generate_test_suite') as mock_gen_test, \
             patch.object(hkg, '_generate_documentation') as mock_gen_doc:
            
            # Test full pipeline
            result = hkg.run()
            
            # Verify all phases were called in correct order
            mock_parse_rtl.assert_called_once()
            mock_parse_compiler.assert_called_once()
            mock_load_doc.assert_called_once()
            mock_build_dataflow.assert_called_once()  # New phase
            mock_gen_rtl.assert_called_once()
            mock_gen_hwop.assert_called_once()
            mock_gen_backend.assert_called_once()
            mock_gen_test.assert_called_once()  # New phase
            mock_gen_doc.assert_called_once()
            
            # Verify result is generated files dict
            assert isinstance(result, dict)
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_enhanced_run_pipeline_stop_after_dataflow(self):
        """Test enhanced run pipeline stopping after dataflow model building."""
        hkg = HardwareKernelGenerator(
            rtl_file_path=self.rtl_file,
            compiler_data_path=self.compiler_file,
            output_dir=self.output_dir
        )
        
        # Mock phase methods
        with patch.object(hkg, '_parse_rtl') as mock_parse_rtl, \
             patch.object(hkg, '_parse_compiler_data') as mock_parse_compiler, \
             patch.object(hkg, '_load_custom_documentation') as mock_load_doc, \
             patch.object(hkg, '_build_dataflow_model') as mock_build_dataflow, \
             patch.object(hkg, '_generate_rtl_template') as mock_gen_rtl:
            
            # Test pipeline stopping after dataflow phase
            result = hkg.run(stop_after="build_dataflow_model")
            
            # Verify phases up to dataflow were called
            mock_parse_rtl.assert_called_once()
            mock_parse_compiler.assert_called_once()
            mock_load_doc.assert_called_once()
            mock_build_dataflow.assert_called_once()
            
            # Verify later phases were not called
            mock_gen_rtl.assert_not_called()


class TestEnhancedHKGIntegration:
    """Integration tests for enhanced HKG with actual components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('brainsmith.tools.hw_kernel_gen.hkg.DATAFLOW_AVAILABLE', True)
    def test_dataflow_model_building_integration(self):
        """Test dataflow model building with actual converter components."""
        # Create test RTL file
        rtl_file = os.path.join(self.temp_dir, "test.sv")
        with open(rtl_file, 'w') as f:
            f.write("""
            module test_kernel(
                input wire clk,
                input wire rst_n,
                input wire [31:0] in0_TDATA,
                input wire in0_TVALID,
                output wire in0_TREADY,
                output wire [31:0] out0_TDATA,
                output wire out0_TVALID,
                input wire out0_TREADY
            );
            endmodule
            """)
        
        # Create compiler data file
        compiler_file = os.path.join(self.temp_dir, "compiler_data.py")
        with open(compiler_file, 'w') as f:
            f.write("""
# Test compiler data
onnx_metadata = {
    'in0_layout': '[N, C]',
    'in0_shape': [1, 32],
    'out0_layout': '[N, C]',
    'out0_shape': [1, 32]
}
            """)
        
        # Mock actual RTL parsing since we don't have full RTL parser setup
        with patch('brainsmith.tools.hw_kernel_gen.hkg.RTLParser') as mock_parser_class:
            mock_parser = Mock()
            mock_hw_kernel = Mock()
            mock_hw_kernel.name = "test_kernel"
            mock_hw_kernel.parameters = []
            mock_hw_kernel.interfaces = {
                "in0": Mock(name="in0", type=Mock(), metadata={}),
                "out0": Mock(name="out0", type=Mock(), metadata={})
            }
            # Make parameters iterable
            mock_param = Mock()
            mock_param.name = "test_param"
            mock_param.default_value = "test_value"
            mock_hw_kernel.parameters = [mock_param]
            mock_hw_kernel.pragmas = []
            mock_parser.parse_file.return_value = mock_hw_kernel
            mock_parser_class.return_value = mock_parser
            
            hkg = HardwareKernelGenerator(
                rtl_file_path=rtl_file,
                compiler_data_path=compiler_file,
                output_dir=self.output_dir
            )
            
            # Test dataflow model building phase
            with patch('brainsmith.tools.hw_kernel_gen.hkg.RTLInterfaceConverter') as mock_converter_class, \
                 patch('brainsmith.tools.hw_kernel_gen.hkg.DataflowModel') as mock_model_class, \
                 patch('brainsmith.tools.hw_kernel_gen.hkg.validate_conversion_result') as mock_validate:
                
                mock_converter = Mock()
                # Return mock interfaces instead of empty list to trigger model creation
                mock_interface = Mock()
                mock_converter.convert_interfaces.return_value = [mock_interface]
                mock_converter_class.return_value = mock_converter
                
                # Mock validation to return no errors
                mock_validate.return_value = []
                
                mock_model = Mock()
                mock_model_class.return_value = mock_model
                
                # Run pipeline up to dataflow model building
                hkg.run(stop_after="build_dataflow_model")
                
                # Verify dataflow components were initialized
                assert hkg.dataflow_enabled == True
                assert hkg.rtl_converter is not None
                assert hkg.dataflow_interfaces is not None
                assert len(hkg.dataflow_interfaces) == 1
                assert hkg.dataflow_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])