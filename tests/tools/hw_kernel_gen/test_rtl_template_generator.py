############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import pytest
import os
from pathlib import Path

# Import the HardwareKernelGenerator class and related components
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator, HardwareKernelGeneratorError
from brainsmith.tools.hw_kernel_gen.generators.rtl_template_generator import generate_rtl_template

# Define the path to the example RTL file relative to the test file
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples"))
THRESHOLDING_RTL_PATH = os.path.join(EXAMPLES_DIR, "thresholding", "thresholding_axi.sv")
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "generated"))

# For testing purposes, we need a dummy compiler data path
# This file should exist but doesn't need to have meaningful content for RTL template tests
DUMMY_COMPILER_DATA_PATH = os.path.join(EXAMPLES_DIR, "thresholding", "dummy_compiler_data.py")

# Check if the example file exists
if not os.path.exists(THRESHOLDING_RTL_PATH):
    pytest.skip(f"Example RTL file not found: {THRESHOLDING_RTL_PATH}", allow_module_level=True)

# Create dummy compiler data file if it doesn't exist
if not os.path.exists(DUMMY_COMPILER_DATA_PATH):
    os.makedirs(os.path.dirname(DUMMY_COMPILER_DATA_PATH), exist_ok=True)
    with open(DUMMY_COMPILER_DATA_PATH, 'w') as f:
        f.write("# Dummy compiler data file for testing\n")
        f.write("onnx_patterns = []\n")
        f.write("def cost_function(*args, **kwargs):\n")
        f.write("    return 1.0\n")

@pytest.fixture(scope="module")
def hkg_instance():
    """Creates a HardwareKernelGenerator instance configured for testing."""
    try:
        # Create HKG instance that will parse thresholding_axi.sv
        hkg = HardwareKernelGenerator(
            rtl_file_path=THRESHOLDING_RTL_PATH,
            compiler_data_path=DUMMY_COMPILER_DATA_PATH,
            output_dir=OUTPUT_DIR
        )
        return hkg
    except (FileNotFoundError, HardwareKernelGeneratorError) as e:
        pytest.fail(f"Failed to create HKG instance: {e}")

def test_generate_rtl_template_from_hkg(hkg_instance):
    """Test RTL template generation using the HKG."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Get the parsed RTL data
        hw_kernel_data = hkg_instance.get_parsed_rtl_data()
        assert hw_kernel_data is not None, "Failed to parse RTL data from thresholding_axi.sv"
        
        # Generate RTL template
        output_path = generate_rtl_template(hw_kernel_data, Path(OUTPUT_DIR))
        assert output_path.exists(), f"Output file not created: {output_path}"
        
        # Read the generated file to verify its content
        with open(output_path, 'r') as f:
            content = f.read()
            
        # Verify some basic content expectations
        assert "$THRESHOLDING_AXI_WRAPPER_NAME$" in content, "Wrapper module name placeholder missing"
        assert "module $THRESHOLDING_AXI_WRAPPER_NAME$" in content, "Module declaration missing"
        assert "thresholding_axi" in content, "Original module name missing in instantiation"
        
        # Check for parameter passing in instantiation
        assert "#(" in content, "Parameter section missing in instantiation"
        
        # Check for interface connections
        for if_name in hw_kernel_data.interfaces.keys():
            assert if_name in content, f"Interface {if_name} not found in generated wrapper"
        
        print(f"Successfully generated and verified RTL template at {output_path}")
        
    except Exception as e:
        pytest.fail(f"Failed to generate RTL template: {e}")

def test_rtl_template_generation_via_hkg_pipeline(hkg_instance):
    """Test RTL template generation through the HKG pipeline."""
    try:
        # Run the HKG pipeline, stopping after RTL template generation
        generated_files = hkg_instance.run(stop_after="generate_rtl_template")
        
        # Verify that the RTL template file was generated
        assert "rtl_template" in generated_files, "RTL template file not in generated_files dict"
        template_path = generated_files["rtl_template"]
        assert template_path.exists(), f"RTL template file not found at {template_path}"
        
        # Read the generated file to verify its content
        with open(template_path, 'r') as f:
            content = f.read()
            
        # Verify some basic content expectations
        assert "$THRESHOLDING_AXI_WRAPPER_NAME$" in content, "Wrapper module name placeholder missing"
        assert "module $THRESHOLDING_AXI_WRAPPER_NAME$" in content, "Module declaration missing"
        assert "thresholding_axi" in content, "Original module name missing in instantiation"
        
        print(f"Successfully generated and verified RTL template via HKG pipeline at {template_path}")
        
    except Exception as e:
        pytest.fail(f"Failed to generate RTL template via HKG pipeline: {e}")
