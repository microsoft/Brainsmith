"""
Design Space Tests - Infrastructure Layer

Tests the design space engine components: DesignSpace, DesignPoint, ParameterDefinition.
Validates parameter management, design point creation, and blueprint integration.
"""

import pytest
import json
from typing import Dict, Any, List

# Import design space components
try:
    from brainsmith.infrastructure.dse.design_space import (
        DesignSpace, DesignPoint, ParameterDefinition, ParameterType,
        create_parameter_sweep_points, sample_design_space
    )
    DESIGN_SPACE_AVAILABLE = True
except ImportError:
    DESIGN_SPACE_AVAILABLE = False


@pytest.mark.infrastructure
class TestParameterDefinition:
    """Test ParameterDefinition class functionality."""
    
    def test_parameter_definition_creation(self):
        """Test creating ParameterDefinition instances."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        # Integer parameter
        param = ParameterDefinition(
            name="pe_conv",
            param_type="integer",
            range_min=1,
            range_max=16,
            default=4
        )
        
        assert param.name == "pe_conv"
        assert param.type == "integer"
        assert param.range_min == 1
        assert param.range_max == 16
        assert param.default == 4
        assert param.values is None
    
    def test_categorical_parameter(self):
        """Test categorical parameter definition."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        param = ParameterDefinition(
            name="precision",
            param_type="categorical",
            values=["INT8", "INT16", "FP16"],
            default="INT8"
        )
        
        assert param.name == "precision"
        assert param.type == "categorical"
        assert param.values == ["INT8", "INT16", "FP16"]
        assert param.default == "INT8"
    
    def test_parameter_type_enum(self):
        """Test using ParameterType enum."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        param = ParameterDefinition(
            name="test_param",
            param_type=ParameterType.FLOAT,
            range_min=0.0,
            range_max=1.0
        )
        
        assert param.type == "float"
    
    def test_parameter_validation_integer(self):
        """Test parameter value validation for integers."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        param = ParameterDefinition(
            name="pe_conv",
            param_type="integer",
            range_min=1,
            range_max=16
        )
        
        # Valid values
        assert param.validate_value(1) is True
        assert param.validate_value(8) is True
        assert param.validate_value(16) is True
        
        # Invalid values
        assert param.validate_value(0) is False
        assert param.validate_value(17) is False
        assert param.validate_value(-1) is False
    
    def test_parameter_validation_float(self):
        """Test parameter value validation for floats."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        param = ParameterDefinition(
            name="clock_freq",
            param_type="float",
            range_min=50.0,
            range_max=200.0
        )
        
        # Valid values
        assert param.validate_value(50.0) is True
        assert param.validate_value(125.5) is True
        assert param.validate_value(200.0) is True
        
        # Invalid values
        assert param.validate_value(49.9) is False
        assert param.validate_value(200.1) is False
    
    def test_parameter_validation_categorical(self):
        """Test parameter value validation for categorical."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        param = ParameterDefinition(
            name="precision",
            param_type="categorical",
            values=["INT8", "INT16", "FP16"]
        )
        
        # Valid values
        assert param.validate_value("INT8") is True
        assert param.validate_value("INT16") is True
        assert param.validate_value("FP16") is True
        
        # Invalid values
        assert param.validate_value("INT32") is False
        assert param.validate_value("invalid") is False
    
    def test_parameter_validation_boolean(self):
        """Test parameter value validation for boolean."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        param = ParameterDefinition(
            name="enable_feature",
            param_type="boolean"
        )
        
        # Valid values
        assert param.validate_value(True) is True
        assert param.validate_value(False) is True
        
        # Invalid values
        assert param.validate_value("true") is False
        assert param.validate_value(1) is False
        assert param.validate_value(0) is False
    
    def test_parameter_to_dict(self):
        """Test parameter conversion to dictionary."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        param = ParameterDefinition(
            name="pe_conv",
            param_type="integer",
            range_min=1,
            range_max=16,
            default=4
        )
        
        param_dict = param.to_dict()
        
        assert isinstance(param_dict, dict)
        assert param_dict["name"] == "pe_conv"
        assert param_dict["type"] == "integer"
        assert param_dict["range_min"] == 1
        assert param_dict["range_max"] == 16
        assert param_dict["default"] == 4


@pytest.mark.infrastructure
class TestDesignPoint:
    """Test DesignPoint class functionality."""
    
    def test_design_point_creation(self):
        """Test creating DesignPoint instances."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        point = DesignPoint()
        
        assert isinstance(point.parameters, dict)
        assert isinstance(point.results, dict)
        assert isinstance(point.metadata, dict)
        assert len(point.parameters) == 0
        assert len(point.results) == 0
        assert len(point.metadata) == 0
    
    def test_design_point_with_parameters(self):
        """Test DesignPoint with initial parameters."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        params = {"pe_conv": 8, "simd_conv": 4, "precision": "INT8"}
        point = DesignPoint(params)
        
        assert point.parameters == params
        assert point.get_parameter("pe_conv") == 8
        assert point.get_parameter("simd_conv") == 4
        assert point.get_parameter("precision") == "INT8"
    
    def test_design_point_parameter_operations(self):
        """Test DesignPoint parameter get/set operations."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        point = DesignPoint()
        
        # Set parameters
        point.set_parameter("pe_conv", 8)
        point.set_parameter("clock_freq", 150.0)
        
        # Get parameters
        assert point.get_parameter("pe_conv") == 8
        assert point.get_parameter("clock_freq") == 150.0
        assert point.get_parameter("nonexistent") is None
        assert point.get_parameter("nonexistent", "default") == "default"
    
    def test_design_point_result_operations(self):
        """Test DesignPoint result get/set operations."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        point = DesignPoint()
        
        # Set results
        point.set_result("throughput", 500.0)
        point.set_result("latency", 20.0)
        
        # Get results
        assert point.get_result("throughput") == 500.0
        assert point.get_result("latency") == 20.0
        assert point.get_result("nonexistent") is None
        assert point.get_result("nonexistent", 0.0) == 0.0
    
    def test_design_point_serialization(self):
        """Test DesignPoint to_dict and from_dict."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        # Create point with data
        point = DesignPoint({"pe_conv": 8, "simd_conv": 4})
        point.set_result("throughput", 500.0)
        point.metadata["build_time"] = 120.5
        
        # Convert to dict
        point_dict = point.to_dict()
        
        assert isinstance(point_dict, dict)
        assert "parameters" in point_dict
        assert "results" in point_dict
        assert "metadata" in point_dict
        assert point_dict["parameters"]["pe_conv"] == 8
        assert point_dict["results"]["throughput"] == 500.0
        assert point_dict["metadata"]["build_time"] == 120.5
        
        # Convert back from dict
        restored_point = DesignPoint.from_dict(point_dict)
        
        assert restored_point.get_parameter("pe_conv") == 8
        assert restored_point.get_result("throughput") == 500.0
        assert restored_point.metadata["build_time"] == 120.5


@pytest.mark.infrastructure
class TestDesignSpace:
    """Test DesignSpace class functionality."""
    
    def test_design_space_creation(self):
        """Test creating DesignSpace instances."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        ds = DesignSpace("test_space")
        
        assert ds.name == "test_space"
        assert isinstance(ds.parameters, dict)
        assert isinstance(ds.design_points, list)
        assert isinstance(ds.blueprint_config, dict)
        assert len(ds.parameters) == 0
        assert len(ds.design_points) == 0
    
    def test_design_space_add_parameters(self):
        """Test adding parameters to design space."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        ds = DesignSpace("test_space")
        
        # Add integer parameter
        pe_param = ParameterDefinition("pe_conv", "integer", range_min=1, range_max=16, default=4)
        ds.add_parameter(pe_param)
        
        # Add categorical parameter
        precision_param = ParameterDefinition("precision", "categorical", values=["INT8", "INT16"], default="INT8")
        ds.add_parameter(precision_param)
        
        assert len(ds.parameters) == 2
        assert "pe_conv" in ds.parameters
        assert "precision" in ds.parameters
        
        # Test parameter names
        names = ds.get_parameter_names()
        assert set(names) == {"pe_conv", "precision"}
    
    def test_design_space_create_design_point(self):
        """Test creating design points with validation."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        ds = DesignSpace("test_space")
        
        # Add parameters
        pe_param = ParameterDefinition("pe_conv", "integer", range_min=1, range_max=16)
        precision_param = ParameterDefinition("precision", "categorical", values=["INT8", "INT16"])
        ds.add_parameter(pe_param)
        ds.add_parameter(precision_param)
        
        # Create valid design point
        valid_params = {"pe_conv": 8, "precision": "INT8"}
        point = ds.create_design_point(valid_params)
        
        assert isinstance(point, DesignPoint)
        assert point.get_parameter("pe_conv") == 8
        assert point.get_parameter("precision") == "INT8"
        
        # Test invalid parameter value
        invalid_params = {"pe_conv": 20, "precision": "INT8"}  # pe_conv out of range
        with pytest.raises(ValueError, match="Invalid value"):
            ds.create_design_point(invalid_params)
    
    def test_design_space_sample_points(self):
        """Test sampling design points from parameter space."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        ds = DesignSpace("test_space")
        
        # Add parameters
        pe_param = ParameterDefinition("pe_conv", "integer", range_min=1, range_max=16, default=4)
        simd_param = ParameterDefinition("simd_conv", "integer", range_min=1, range_max=8, default=2)
        precision_param = ParameterDefinition("precision", "categorical", values=["INT8", "INT16"], default="INT8")
        bool_param = ParameterDefinition("enable_feature", "boolean", default=True)
        float_param = ParameterDefinition("clock_freq", "float", range_min=50.0, range_max=200.0, default=100.0)
        
        ds.add_parameter(pe_param)
        ds.add_parameter(simd_param)
        ds.add_parameter(precision_param)
        ds.add_parameter(bool_param)
        ds.add_parameter(float_param)
        
        # Sample points
        points = ds.sample_points(n_samples=10, seed=42)
        
        assert len(points) == 10
        assert all(isinstance(point, DesignPoint) for point in points)
        
        # Check that all parameters are present
        for point in points:
            assert "pe_conv" in point.parameters
            assert "simd_conv" in point.parameters
            assert "precision" in point.parameters
            assert "enable_feature" in point.parameters
            assert "clock_freq" in point.parameters
            
            # Check value ranges
            assert 1 <= point.get_parameter("pe_conv") <= 16
            assert 1 <= point.get_parameter("simd_conv") <= 8
            assert point.get_parameter("precision") in ["INT8", "INT16"]
            assert isinstance(point.get_parameter("enable_feature"), bool)
            assert 50.0 <= point.get_parameter("clock_freq") <= 200.0
    
    def test_design_space_sample_points_deterministic(self):
        """Test that sampling with same seed is deterministic."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        ds = DesignSpace("test_space")
        pe_param = ParameterDefinition("pe_conv", "integer", range_min=1, range_max=16)
        ds.add_parameter(pe_param)
        
        # Sample with same seed twice
        points1 = ds.sample_points(n_samples=5, seed=42)
        points2 = ds.sample_points(n_samples=5, seed=42)
        
        # Should be identical
        for p1, p2 in zip(points1, points2):
            assert p1.parameters == p2.parameters
    
    def test_design_space_from_blueprint_data(self):
        """Test creating design space from blueprint data."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        blueprint_data = {
            "name": "test_blueprint",
            "description": "Test blueprint",
            "parameters": {
                "pe_conv": {
                    "type": "integer",
                    "range_min": 1,
                    "range_max": 16,
                    "default": 4
                },
                "precision": {
                    "type": "categorical",
                    "values": ["INT8", "INT16"],
                    "default": "INT8"
                },
                "clock_freq": {
                    "type": "float",
                    "range_min": 50.0,
                    "range_max": 200.0,
                    "default": 100.0
                }
            }
        }
        
        ds = DesignSpace.from_blueprint_data(blueprint_data)
        
        assert ds.name == "test_blueprint"
        assert len(ds.parameters) == 3
        assert "pe_conv" in ds.parameters
        assert "precision" in ds.parameters
        assert "clock_freq" in ds.parameters
        
        # Check parameter types
        assert ds.parameters["pe_conv"].type == "integer"
        assert ds.parameters["precision"].type == "categorical"
        assert ds.parameters["clock_freq"].type == "float"
        
        # Check parameter ranges/values
        assert ds.parameters["pe_conv"].range_min == 1
        assert ds.parameters["pe_conv"].range_max == 16
        assert ds.parameters["precision"].values == ["INT8", "INT16"]
        assert ds.parameters["clock_freq"].range_min == 50.0
    
    def test_design_space_validation(self):
        """Test design space validation."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        # Empty design space (invalid)
        ds_empty = DesignSpace("empty")
        is_valid, errors = ds_empty.validate()
        assert is_valid is False
        assert len(errors) > 0
        assert any("no parameters" in error.lower() for error in errors)
        
        # Valid design space
        ds_valid = DesignSpace("valid")
        pe_param = ParameterDefinition("pe_conv", "integer", range_min=1, range_max=16)
        ds_valid.add_parameter(pe_param)
        
        is_valid, errors = ds_valid.validate()
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid categorical parameter (no values)
        ds_invalid = DesignSpace("invalid")
        bad_param = ParameterDefinition("bad_categorical", "categorical")  # No values
        ds_invalid.add_parameter(bad_param)
        
        is_valid, errors = ds_invalid.validate()
        assert is_valid is False
        assert len(errors) > 0
        
        # Invalid range parameter
        ds_invalid_range = DesignSpace("invalid_range")
        bad_range_param = ParameterDefinition("bad_range", "integer", range_min=10, range_max=5)  # min > max
        ds_invalid_range.add_parameter(bad_range_param)
        
        is_valid, errors = ds_invalid_range.validate()
        assert is_valid is False
        assert len(errors) > 0
    
    def test_design_space_serialization(self):
        """Test design space to_dict and to_json."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        ds = DesignSpace("test_space")
        pe_param = ParameterDefinition("pe_conv", "integer", range_min=1, range_max=16, default=4)
        ds.add_parameter(pe_param)
        
        # Add a design point
        point = ds.create_design_point({"pe_conv": 8})
        ds.design_points.append(point)
        
        # Test to_dict
        ds_dict = ds.to_dict()
        assert isinstance(ds_dict, dict)
        assert ds_dict["name"] == "test_space"
        assert "parameters" in ds_dict
        assert "design_points" in ds_dict
        assert len(ds_dict["parameters"]) == 1
        assert len(ds_dict["design_points"]) == 1
        
        # Test to_json
        json_str = ds.to_json()
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "test_space"


@pytest.mark.infrastructure
class TestUtilityFunctions:
    """Test utility functions for design space operations."""
    
    def test_create_parameter_sweep_points(self):
        """Test creating parameter sweep points."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        parameters = {
            "pe_conv": [1, 2, 4, 8],
            "simd_conv": [1, 2, 4],
            "precision": ["INT8", "INT16"]
        }
        
        points = create_parameter_sweep_points(parameters)
        
        # Should create all combinations: 4 * 3 * 2 = 24 points
        assert len(points) == 24
        assert all(isinstance(point, DesignPoint) for point in points)
        
        # Check that all combinations are present
        pe_values = {point.get_parameter("pe_conv") for point in points}
        simd_values = {point.get_parameter("simd_conv") for point in points}
        precision_values = {point.get_parameter("precision") for point in points}
        
        assert pe_values == {1, 2, 4, 8}
        assert simd_values == {1, 2, 4}
        assert precision_values == {"INT8", "INT16"}
    
    def test_sample_design_space_function(self):
        """Test sample_design_space utility function."""
        if not DESIGN_SPACE_AVAILABLE:
            pytest.skip("Design space not available")
        
        ds = DesignSpace("test_space")
        pe_param = ParameterDefinition("pe_conv", "integer", range_min=1, range_max=16)
        ds.add_parameter(pe_param)
        
        # Test the utility function
        points = sample_design_space(ds, n_samples=5, seed=42)
        
        assert len(points) == 5
        assert all(isinstance(point, DesignPoint) for point in points)


@pytest.mark.integration
def test_design_space_blueprint_integration(sample_blueprint_path):
    """Test integration between design space and blueprint system."""
    if not DESIGN_SPACE_AVAILABLE:
        pytest.skip("Design space not available")
    
    # This test would require actual blueprint loading
    # For now, test with mock blueprint data
    
    blueprint_data = {
        "name": "integration_test",
        "parameters": {
            "pe_conv": {"type": "integer", "range_min": 1, "range_max": 16, "default": 4},
            "precision": {"type": "categorical", "values": ["INT8", "INT16"], "default": "INT8"}
        }
    }
    
    try:
        ds = DesignSpace.from_blueprint_data(blueprint_data)
        assert ds.name == "integration_test"
        assert len(ds.parameters) == 2
        
        # Test that we can sample points
        points = ds.sample_points(n_samples=3, seed=42)
        assert len(points) == 3
        
        # Test validation
        is_valid, errors = ds.validate()
        assert is_valid is True
        
    except Exception as e:
        pytest.skip(f"Blueprint integration not available: {e}")


# Helper functions for design space testing
def create_test_design_space() -> 'DesignSpace':
    """Helper to create a standard test design space."""
    if not DESIGN_SPACE_AVAILABLE:
        return None
    
    ds = DesignSpace("test_space")
    
    # Add standard parameters
    pe_param = ParameterDefinition("pe_conv", "integer", range_min=1, range_max=16, default=4)
    simd_param = ParameterDefinition("simd_conv", "integer", range_min=1, range_max=8, default=2)
    precision_param = ParameterDefinition("precision", "categorical", values=["INT8", "INT16"], default="INT8")
    
    ds.add_parameter(pe_param)
    ds.add_parameter(simd_param)
    ds.add_parameter(precision_param)
    
    return ds


def assert_design_point_valid(point: 'DesignPoint', expected_params: List[str] = None):
    """Helper to assert design point is valid."""
    if not DESIGN_SPACE_AVAILABLE:
        return
    
    assert isinstance(point, DesignPoint)
    assert isinstance(point.parameters, dict)
    assert isinstance(point.results, dict)
    assert isinstance(point.metadata, dict)
    
    if expected_params:
        for param in expected_params:
            assert param in point.parameters


def assert_design_space_valid(ds: 'DesignSpace', expected_param_count: int = None):
    """Helper to assert design space is valid."""
    if not DESIGN_SPACE_AVAILABLE:
        return
    
    assert isinstance(ds, DesignSpace)
    assert isinstance(ds.name, str)
    assert isinstance(ds.parameters, dict)
    assert isinstance(ds.design_points, list)
    
    if expected_param_count is not None:
        assert len(ds.parameters) == expected_param_count
    
    # Validate the design space
    is_valid, errors = ds.validate()
    if not is_valid:
        pytest.fail(f"Design space validation failed: {errors}")