"""Tests for the enhanced base generator."""
import pytest
from pathlib import Path
from brainsmith.tools.kernel_integrator.generators.base_v2 import GeneratorBase
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata


class MockGenerator(GeneratorBase):
    """Mock generator for testing."""
    
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def template_file(self) -> str:
        return "mock_template.j2"
    
    @property
    def output_pattern(self) -> str:
        return "mock_{kernel_name}.txt"
    
    def _get_specific_vars(self, metadata: KernelMetadata) -> dict:
        return {
            "mock_var": "test_value",
            "computed": f"{metadata.name}_computed"
        }


def test_base_generator_initialization():
    """Test generator initialization."""
    gen = MockGenerator()
    assert gen.name == "mock"
    assert gen.template_file == "mock_template.j2"
    assert gen.output_pattern == "mock_{kernel_name}.txt"
    assert gen.template_dir.name == "templates"


def test_get_output_filename():
    """Test output filename generation."""
    gen = MockGenerator()
    assert gen.get_output_filename("test_kernel") == "mock_test_kernel.txt"


def test_common_vars():
    """Test common variable extraction."""
    gen = MockGenerator()
    metadata = KernelMetadata(
        name="test_kernel",
        interfaces=[],
        parameters=[],
        source_file="test.cpp"
    )
    
    common_vars = gen._get_common_vars(metadata)
    
    # Now only passes kernel_metadata directly
    assert "kernel_metadata" in common_vars
    assert common_vars["kernel_metadata"] == metadata
    assert len(common_vars) == 1  # Only one key


def test_snake_to_camel():
    """Test snake_case to CamelCase conversion."""
    gen = MockGenerator()
    assert gen._snake_to_camel("test_kernel_name") == "TestKernelName"
    assert gen._snake_to_camel("simple") == "Simple"
    assert gen._snake_to_camel("a_b_c") == "ABC"


def test_camel_to_snake():
    """Test CamelCase to snake_case conversion."""
    gen = MockGenerator()
    assert gen._camel_to_snake("TestKernelName") == "test_kernel_name"
    assert gen._camel_to_snake("Simple") == "simple"
    assert gen._camel_to_snake("ABC") == "abc"  # All caps converts to lowercase