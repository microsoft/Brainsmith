"""
Test suite for Phase 3A enhanced template system.

Tests the enhanced template system with complete method implementations,
resource estimation, and multi-file generation capabilities.
"""

import pytest
import tempfile
import os
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator, HardwareKernelGeneratorError


class TestPhase3TemplateSystem:
    """Test Phase 3A enhanced template system implementation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def hkg_instance(self, temp_output_dir):
        """Create HKG instance with thresholding example."""
        rtl_file = "examples/thresholding/thresholding_axi.sv"
        compiler_data = "examples/thresholding/dummy_compiler_data.py"
        
        hkg = HardwareKernelGenerator(
            rtl_file_path=rtl_file,
            compiler_data_path=compiler_data,
            output_dir=str(temp_output_dir)
        )
        
        # Parse RTL and build dataflow model
        hkg._parse_rtl()
        hkg._parse_compiler_data()
        hkg._build_dataflow_model()
        
        return hkg

    def test_enhanced_hwcustomop_template_generation(self, hkg_instance, temp_output_dir):
        """Test enhanced HWCustomOp template generation with complete methods."""
        # Generate HWCustomOp using enhanced template
        output_path = hkg_instance._generate_auto_hwcustomop_with_dataflow()
        
        assert output_path.exists()
        assert output_path.suffix == ".py"
        
        # Read generated content
        with open(output_path, 'r') as f:
            generated_content = f.read()
        
        # Verify essential components are present
        assert "class AutoThresholdingAxi(HWCustomOp):" in generated_content
        assert "def get_input_datatype(self, ind: int = 0) -> DataType:" in generated_content
        assert "def get_output_datatype(self, ind: int = 0) -> DataType:" in generated_content
        assert "def bram_estimation(self) -> int:" in generated_content
        assert "def lut_estimation(self) -> int:" in generated_content
        assert "def dsp_estimation(self, fpgapart: str) -> int:" in generated_content
        assert "def get_exp_cycles(self) -> int:" in generated_content
        
        # Verify dataflow interface integration
        assert "self.dataflow_interfaces = {" in generated_content
        assert "ConstraintValidator" in generated_content
        
        # Verify import statements
        assert "from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp" in generated_content
        assert "from qonnx.core.datatype import DataType" in generated_content

    def test_enhanced_rtlbackend_template_generation(self, hkg_instance, temp_output_dir):
        """Test enhanced RTLBackend template generation."""
        # Generate RTLBackend using enhanced template
        output_path = hkg_instance._generate_auto_rtlbackend_with_dataflow()
        
        assert output_path.exists()
        assert output_path.suffix == ".py"
        
        # Read generated content
        with open(output_path, 'r') as f:
            generated_content = f.read()
        
        # Verify essential components are present
        assert "class AutoThresholdingAxiRTLBackend(RTLBackend):" in generated_content
        assert "def generate_params(self, model, path):" in generated_content
        assert "def code_generation_dict(self):" in generated_content
        assert "_generate_interface_definitions" in generated_content
        assert "_generate_signal_assignments" in generated_content
        
        # Verify dataflow interface integration
        assert "self.dataflow_interfaces = {" in generated_content

    def test_enhanced_test_suite_generation(self, hkg_instance, temp_output_dir):
        """Test enhanced test suite generation."""
        # Generate test suite using enhanced template
        output_path = hkg_instance._generate_auto_test_suite_with_dataflow()
        
        assert output_path.exists()
        assert output_path.suffix == ".py"
        
        # Read generated content
        with open(output_path, 'r') as f:
            generated_content = f.read()
        
        # Verify essential components are present
        assert "class TestAutoThresholdingAxi:" in generated_content
        assert "def test_node_creation(self, test_node):" in generated_content
        assert "def test_input_datatypes(self, test_node, input_index):" in generated_content
        assert "def test_resource_estimation(self, test_node):" in generated_content
        assert "def test_constraint_validation(self, test_node):" in generated_content
        assert "def test_rtl_backend_integration(self, test_node):" in generated_content
        
        # Verify pytest integration
        assert "import pytest" in generated_content
        assert "@pytest.fixture" in generated_content

    def test_multi_file_package_generation(self, hkg_instance, temp_output_dir):
        """Test complete multi-file package generation."""
        # Generate complete package
        package_files = hkg_instance.generate_complete_package()
        
        # Verify all expected files are generated
        expected_files = ["hwcustomop", "rtlbackend", "test_suite", "documentation", "rtl_template"]
        assert all(file_type in package_files for file_type in expected_files)
        
        # Verify files exist and have content
        for file_type, file_path in package_files.items():
            assert file_path.exists()
            assert file_path.stat().st_size > 0
            print(f"Generated {file_type}: {file_path} ({file_path.stat().st_size} bytes)")

    def test_template_context_building(self, hkg_instance):
        """Test enhanced template context building."""
        context = hkg_instance._build_enhanced_template_context()
        
        # Verify essential context variables
        assert "kernel_name" in context
        assert "class_name" in context
        assert "source_file" in context
        assert "generation_timestamp" in context
        assert "dataflow_interfaces" in context
        assert "input_interfaces" in context
        assert "output_interfaces" in context
        
        # Verify dataflow-specific context
        assert isinstance(context["dataflow_interfaces"], list)
        assert len(context["dataflow_interfaces"]) > 0
        
        # Verify interface type organization
        assert isinstance(context["input_interfaces"], list)
        assert isinstance(context["output_interfaces"], list)
        assert isinstance(context["weight_interfaces"], list)
        assert isinstance(context["config_interfaces"], list)

    def test_template_validation_and_syntax(self, hkg_instance, temp_output_dir):
        """Test that generated code has valid Python syntax."""
        import ast
        
        # Generate all components
        hwcustomop_path = hkg_instance._generate_auto_hwcustomop_with_dataflow()
        rtlbackend_path = hkg_instance._generate_auto_rtlbackend_with_dataflow()
        test_path = hkg_instance._generate_auto_test_suite_with_dataflow()
        
        # Validate syntax of generated files
        for file_path in [hwcustomop_path, rtlbackend_path, test_path]:
            with open(file_path, 'r') as f:
                generated_code = f.read()
            
            # Parse to validate syntax
            try:
                ast.parse(generated_code)
                print(f"✅ Valid Python syntax: {file_path.name}")
            except SyntaxError as e:
                pytest.fail(f"Invalid Python syntax in {file_path.name}: {e}")

    def test_resource_estimation_integration(self, hkg_instance, temp_output_dir):
        """Test resource estimation integration in generated code."""
        # Generate HWCustomOp
        output_path = hkg_instance._generate_auto_hwcustomop_with_dataflow()
        
        with open(output_path, 'r') as f:
            generated_content = f.read()
        
        # Verify resource estimation methods are comprehensive
        assert "def bram_estimation(self) -> int:" in generated_content
        assert "def lut_estimation(self) -> int:" in generated_content
        assert "def dsp_estimation(self, fpgapart: str) -> int:" in generated_content
        
        # Verify estimation modes are supported
        assert "resource_estimation_mode" in generated_content
        assert "conservative" in generated_content
        assert "optimistic" in generated_content
        assert "automatic" in generated_content

    def test_constraint_validation_integration(self, hkg_instance, temp_output_dir):
        """Test constraint validation integration in generated code."""
        # Generate HWCustomOp
        output_path = hkg_instance._generate_auto_hwcustomop_with_dataflow()
        
        with open(output_path, 'r') as f:
            generated_content = f.read()
        
        # Verify constraint validation is integrated
        assert "ConstraintValidator" in generated_content
        assert "_validate_datatype_constraints" in generated_content
        assert "enable_constraint_validation" in generated_content
        assert "def verify_node(self):" in generated_content

    def test_dataflow_interface_metadata_integration(self, hkg_instance, temp_output_dir):
        """Test dataflow interface metadata integration."""
        # Generate HWCustomOp
        output_path = hkg_instance._generate_auto_hwcustomop_with_dataflow()
        
        with open(output_path, 'r') as f:
            generated_content = f.read()
        
        # Verify interface metadata is properly integrated
        assert '"interface_type":' in generated_content
        assert '"qDim":' in generated_content
        assert '"tDim":' in generated_content
        assert '"sDim":' in generated_content
        assert '"dtype":' in generated_content
        assert '"finn_type":' in generated_content

    def test_complete_pipeline_with_enhanced_templates(self, temp_output_dir):
        """Test complete HKG pipeline with enhanced template system."""
        rtl_file = "examples/thresholding/thresholding_axi.sv"
        compiler_data = "examples/thresholding/dummy_compiler_data.py"
        
        # Create HKG instance
        hkg = HardwareKernelGenerator(
            rtl_file_path=rtl_file,
            compiler_data_path=compiler_data,
            output_dir=str(temp_output_dir)
        )
        
        # Run complete pipeline
        generated_files = hkg.run()
        
        # Verify all phases completed successfully
        expected_phases = [
            "rtl_template", "hw_custom_op", "rtl_backend", 
            "test_suite", "documentation"
        ]
        
        for phase in expected_phases:
            assert phase in generated_files
            assert generated_files[phase].exists()
            
        print("✅ Complete enhanced pipeline executed successfully")
        print(f"Generated {len(generated_files)} files:")
        for phase, path in generated_files.items():
            print(f"  - {phase}: {path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])