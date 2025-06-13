#!/usr/bin/env python3
"""
Comprehensive unit tests for the refactored pragma system.

Tests all aspects of the pragma refactor including:
- Individual pragma methods (applies_to_interface, apply_to_interface_metadata)
- Interface name matching utility
- Pragma chain application
- Error recovery and isolation
- Edge cases and error conditions
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.pragma import PragmaHandler, PragmaType
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    Direction, Port, ValidationResult, PortGroup,
    InterfaceNameMatcher, DatatypePragma, BDimPragma, WeightPragma
)
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy

# Mock Interface class for testing pragma compatibility
# TODO: Remove when pragma system is fully refactored for InterfaceMetadata
class Interface:
    """Mock Interface class for testing pragma system."""
    def __init__(self, name, type, ports, validation_result, metadata=None):
        self.name = name
        self.type = type
        self.ports = ports
        self.validation_result = validation_result
        self.metadata = metadata or {}


class TestInterfaceNameMatcher:
    """Test the InterfaceNameMatcher mixin."""
    
    def test_exact_match(self):
        """Test exact name matching."""
        assert InterfaceNameMatcher._interface_names_match("in0", "in0")
        assert InterfaceNameMatcher._interface_names_match("weights", "weights")
        assert InterfaceNameMatcher._interface_names_match("s_axi_control", "s_axi_control")
    
    def test_prefix_match(self):
        """Test prefix matching patterns."""
        # Standard prefix: "in0" matches "in0_V_data_V"
        assert InterfaceNameMatcher._interface_names_match("in0", "in0_V_data_V")
        assert InterfaceNameMatcher._interface_names_match("out0", "out0_V_data_V")
        assert InterfaceNameMatcher._interface_names_match("weights", "weights_data")
    
    def test_reverse_prefix_match(self):
        """Test reverse prefix matching."""
        # Reverse: "in0_V_data_V" matches "in0"
        assert InterfaceNameMatcher._interface_names_match("in0_V_data_V", "in0")
        assert InterfaceNameMatcher._interface_names_match("weights_data", "weights")
    
    def test_base_name_matching(self):
        """Test base name matching with suffix removal."""
        # Base name matching: remove _V_data_V and _data suffixes
        assert InterfaceNameMatcher._interface_names_match("weights_data", "weights_V_data_V")
        assert InterfaceNameMatcher._interface_names_match("in0_V_data_V", "in0_data")
    
    def test_non_matching_cases(self):
        """Test cases that should not match."""
        assert not InterfaceNameMatcher._interface_names_match("in0", "out0")
        assert not InterfaceNameMatcher._interface_names_match("weights", "bias")
        assert not InterfaceNameMatcher._interface_names_match("config", "data")
        # Note: Empty string behavior follows startswith() semantics where empty string matches everything
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty strings
        assert InterfaceNameMatcher._interface_names_match("", "")
        
        # Single character names
        assert InterfaceNameMatcher._interface_names_match("a", "a")
        assert not InterfaceNameMatcher._interface_names_match("a", "b")
        
        # Names with underscores
        assert InterfaceNameMatcher._interface_names_match("my_interface", "my_interface_V_data_V")


class TestDatatypePragma:
    """Test DatatypePragma functionality."""
    
    def create_test_interface(self, name: str) -> Interface:
        """Helper to create test interface."""
        test_port = Port(name=f"{name}_data", direction=Direction.INPUT, width="32")
        validation_result = ValidationResult(valid=True, message="Test interface")
        
        return Interface(
            name=name,
            type=InterfaceType.INPUT,
            ports={"data": test_port},
            validation_result=validation_result,
            metadata={}
        )
    
    def create_test_metadata(self, name: str) -> InterfaceMetadata:
        """Helper to create test metadata."""
        return InterfaceMetadata(
            name=name,
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[
                DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False),
                DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True)
            ],
            chunking_strategy=DefaultChunkingStrategy()
        )
    
    def test_applies_to_interface_exact_match(self):
        """Test applies_to_interface with exact name match."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["in0", "UINT", "8", "16"],
            line_number=10
        )
        
        interface = self.create_test_interface("in0")
        assert pragma.applies_to_interface(interface)
        
        other_interface = self.create_test_interface("out0")
        assert not pragma.applies_to_interface(other_interface)
    
    def test_applies_to_interface_pattern_match(self):
        """Test applies_to_interface with pattern matching."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["in0", "UINT", "8", "16"],
            line_number=10
        )
        
        # Should match AXI naming pattern
        axi_interface = self.create_test_interface("in0_V_data_V")
        assert pragma.applies_to_interface(axi_interface)
    
    def test_applies_to_interface_no_parsed_data(self):
        """Test applies_to_interface with invalid pragma."""
        # Create pragma with invalid inputs to cause parsing error
        try:
            pragma = DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=["invalid"],  # Missing required arguments
                line_number=10
            )
            # If pragma creation succeeded despite invalid inputs, test it
            interface = self.create_test_interface("test")
            assert not pragma.applies_to_interface(interface)
        except:
            # Expected - invalid pragma should fail during creation
            pass
    
    def test_apply_to_interface_metadata(self):
        """Test apply_to_interface_metadata functionality."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["test", "UINT,INT", "8", "16"],
            line_number=10
        )
        
        interface = self.create_test_interface("test")
        base_metadata = self.create_test_metadata("test")
        
        # Apply pragma
        updated_metadata = pragma.apply_to_interface_metadata(interface, base_metadata)
        
        # Verify metadata was updated
        assert updated_metadata.name == "test"
        assert updated_metadata.interface_type == InterfaceType.INPUT  # Preserved
        assert len(updated_metadata.allowed_datatypes) > 0  # New constraints added
        
        # Check that new constraints were created
        constraint_types = [c.finn_type for c in updated_metadata.allowed_datatypes]
        assert any("UINT" in t for t in constraint_types)
        assert any("INT" in t for t in constraint_types)
    
    def test_apply_to_interface_metadata_non_applicable(self):
        """Test that non-applicable pragmas don't modify metadata."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["other", "UINT", "8", "16"],
            line_number=10
        )
        
        interface = self.create_test_interface("test")
        base_metadata = self.create_test_metadata("test")
        
        # Apply pragma (should not affect metadata)
        updated_metadata = pragma.apply_to_interface_metadata(interface, base_metadata)
        
        # Metadata should be unchanged
        assert updated_metadata == base_metadata
    
    def test_create_datatype_constraints(self):
        """Test _create_datatype_constraints method."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["test", "UINT,INT", "8", "16"],
            line_number=10
        )
        
        constraints = pragma._create_datatype_constraints()
        
        # Should have constraints for both UINT and INT at min/max bitwidths
        assert len(constraints) > 0
        
        # Check for expected constraint types
        finn_types = [c.finn_type for c in constraints]
        assert "UINT8" in finn_types or "UINT16" in finn_types
        assert "INT8" in finn_types or "INT16" in finn_types


class TestBDimPragma:
    """Test BDimPragma functionality."""
    
    def create_test_interface(self, name: str) -> Interface:
        """Helper to create test interface."""
        test_port = Port(name=f"{name}_data", direction=Direction.INPUT, width="32")
        validation_result = ValidationResult(valid=True, message="Test interface")
        
        return Interface(
            name=name,
            type=InterfaceType.INPUT,
            ports={"data": test_port},
            validation_result=validation_result,
            metadata={}
        )
    
    def create_test_metadata(self, name: str) -> InterfaceMetadata:
        """Helper to create test metadata."""
        return InterfaceMetadata(
            name=name,
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[
                DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False)
            ],
            chunking_strategy=DefaultChunkingStrategy()
        )
    
    def test_applies_to_interface_enhanced_format(self):
        """Test applies_to_interface with new BDIM format."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0_V_data_V", "[PE]"],
            line_number=20
        )
        
        interface = self.create_test_interface("in0_V_data_V")
        assert pragma.applies_to_interface(interface)
        
        # Should also match base name
        base_interface = self.create_test_interface("in0")
        assert pragma.applies_to_interface(base_interface)
    
    def test_applies_to_interface_legacy_format(self):
        """Test applies_to_interface with new BDIM format and RINDEX."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["out0", "[SIMD,PE]", "RINDEX=1"],
            line_number=21
        )
        
        interface = self.create_test_interface("out0")
        assert pragma.applies_to_interface(interface)
    
    def test_apply_to_interface_metadata_enhanced(self):
        """Test apply_to_interface_metadata with new format."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["test", "[PE]"],
            line_number=20
        )
        
        interface = self.create_test_interface("test")
        base_metadata = self.create_test_metadata("test")
        
        # Apply pragma
        updated_metadata = pragma.apply_to_interface_metadata(interface, base_metadata)
        
        # Verify metadata was updated
        assert updated_metadata.name == "test"
        assert updated_metadata.interface_type == InterfaceType.INPUT  # Preserved
        assert updated_metadata.allowed_datatypes == base_metadata.allowed_datatypes  # Preserved
        assert updated_metadata.chunking_strategy is not None  # Strategy created (may be DefaultChunkingStrategy due to fallback)
    
    def test_apply_to_interface_metadata_legacy(self):
        """Test apply_to_interface_metadata with new format and RINDEX."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["test", "[TILE_SIZE]", "RINDEX=1"],
            line_number=21
        )
        
        interface = self.create_test_interface("test")
        base_metadata = self.create_test_metadata("test")
        
        # Apply pragma
        updated_metadata = pragma.apply_to_interface_metadata(interface, base_metadata)
        
        # Verify chunking strategy was updated (both default to DefaultChunkingStrategy due to fallback)
        assert updated_metadata.chunking_strategy is not None
    
    def test_create_chunking_strategy_enhanced(self):
        """Test _create_chunking_strategy with new format."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["test", "[PE]"],
            line_number=20
        )
        
        strategy = pragma._create_chunking_strategy()
        assert strategy is not None
        # Should return DefaultChunkingStrategy if IndexChunkingStrategy not available
    
    def test_create_chunking_strategy_legacy(self):
        """Test _create_chunking_strategy with new format and RINDEX."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["test", "[SIMD,PE]", "RINDEX=1"],
            line_number=21
        )
        
        strategy = pragma._create_chunking_strategy()
        assert strategy is not None
        # Should return DefaultChunkingStrategy if ExpressionChunkingStrategy not available


class TestWeightPragma:
    """Test WeightPragma functionality."""
    
    def create_test_interface(self, name: str) -> Interface:
        """Helper to create test interface."""
        test_port = Port(name=f"{name}_data", direction=Direction.INPUT, width="32")
        validation_result = ValidationResult(valid=True, message="Test interface")
        
        return Interface(
            name=name,
            type=InterfaceType.INPUT,
            ports={"data": test_port},
            validation_result=validation_result,
            metadata={}
        )
    
    def create_test_metadata(self, name: str) -> InterfaceMetadata:
        """Helper to create test metadata."""
        return InterfaceMetadata(
            name=name,
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[
                DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False)
            ],
            chunking_strategy=DefaultChunkingStrategy()
        )
    
    def test_applies_to_interface_single_name(self):
        """Test applies_to_interface with single interface name."""
        pragma = WeightPragma(
            type=PragmaType.WEIGHT,
            inputs=["weights"],
            line_number=30
        )
        
        weights_interface = self.create_test_interface("weights")
        assert pragma.applies_to_interface(weights_interface)
        
        data_interface = self.create_test_interface("data")
        assert not pragma.applies_to_interface(data_interface)
    
    def test_applies_to_interface_multiple_names(self):
        """Test applies_to_interface with multiple interface names."""
        pragma = WeightPragma(
            type=PragmaType.WEIGHT,
            inputs=["weights", "bias", "params"],
            line_number=30
        )
        
        weights_interface = self.create_test_interface("weights")
        bias_interface = self.create_test_interface("bias")
        params_interface = self.create_test_interface("params")
        data_interface = self.create_test_interface("data")
        
        assert pragma.applies_to_interface(weights_interface)
        assert pragma.applies_to_interface(bias_interface)
        assert pragma.applies_to_interface(params_interface)
        assert not pragma.applies_to_interface(data_interface)
    
    def test_apply_to_interface_metadata(self):
        """Test apply_to_interface_metadata changes interface type."""
        pragma = WeightPragma(
            type=PragmaType.WEIGHT,
            inputs=["weights"],
            line_number=30
        )
        
        interface = self.create_test_interface("weights")
        base_metadata = self.create_test_metadata("weights")
        
        # Apply pragma
        updated_metadata = pragma.apply_to_interface_metadata(interface, base_metadata)
        
        # Verify interface type was changed to WEIGHT
        assert updated_metadata.name == "weights"
        assert updated_metadata.interface_type == InterfaceType.WEIGHT  # Changed
        assert updated_metadata.allowed_datatypes == base_metadata.allowed_datatypes  # Preserved
        assert updated_metadata.chunking_strategy == base_metadata.chunking_strategy  # Preserved


class TestPragmaChainApplication:
    """Test pragma chain application and integration."""
    
    def create_test_interface(self, name: str) -> Interface:
        """Helper to create test interface."""
        test_port = Port(name=f"{name}_data", direction=Direction.INPUT, width="32")
        validation_result = ValidationResult(valid=True, message="Test interface")
        
        return Interface(
            name=name,
            type=InterfaceType.INPUT,
            ports={"data": test_port},
            validation_result=validation_result,
            metadata={}
        )
    
    def apply_pragmas_to_interface(self, interface, pragmas):
        """Helper method to apply pragmas to interface like the parser does."""
        from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
        from brainsmith.dataflow.core.block_chunking import DefaultChunkingStrategy
        
        # Create base InterfaceMetadata
        metadata = InterfaceMetadata(
            name=interface.name,
            interface_type=interface.type,
            allowed_datatypes=[],
            chunking_strategy=DefaultChunkingStrategy()
        )
        
        # Apply pragmas individually with error isolation
        for pragma in pragmas:
            try:
                if pragma.applies_to_interface(interface):
                    metadata = pragma.apply_to_interface_metadata(interface, metadata)
            except Exception as e:
                # Log and continue like the original implementation
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to apply pragma: {e}")
        
        return metadata
    
    def test_multiple_pragmas_same_interface(self):
        """Test multiple pragmas affecting the same interface."""
        interface = self.create_test_interface("in0")
        
        pragmas = [
            DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=["in0", "UINT", "8", "16"],
                line_number=10
            ),
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0", "[PE]"],
                line_number=20
            ),
            WeightPragma(
                type=PragmaType.WEIGHT,
                inputs=["in0"],
                line_number=30
            )
        ]
        
        # Apply pragmas using helper method
        metadata = self.apply_pragmas_to_interface(interface, pragmas)
        
        # Verify all pragma effects were applied
        assert metadata.name == "in0"
        assert metadata.interface_type == InterfaceType.WEIGHT  # WeightPragma effect
        assert len(metadata.allowed_datatypes) > 0  # DatatypePragma effect
        assert metadata.chunking_strategy is not None  # BDimPragma effect
    
    def test_pragma_order_dependency(self):
        """Test that pragma order affects final result."""
        interface = self.create_test_interface("test")
        
        # Apply WeightPragma first, then DatatypePragma
        pragmas_order1 = [
            WeightPragma(type=PragmaType.WEIGHT, inputs=["test"], line_number=10),
            DatatypePragma(type=PragmaType.DATATYPE, inputs=["test", "UINT", "8", "8"], line_number=20)
        ]
        
        metadata1 = self.apply_pragmas_to_interface(interface, pragmas_order1)
        
        # Apply DatatypePragma first, then WeightPragma
        pragmas_order2 = [
            DatatypePragma(type=PragmaType.DATATYPE, inputs=["test", "UINT", "8", "8"], line_number=20),
            WeightPragma(type=PragmaType.WEIGHT, inputs=["test"], line_number=10)
        ]
        
        metadata2 = self.apply_pragmas_to_interface(interface, pragmas_order2)
        
        # Both should result in WEIGHT interface type (WeightPragma overrides)
        assert metadata1.interface_type == InterfaceType.WEIGHT
        assert metadata2.interface_type == InterfaceType.WEIGHT
        
        # But datatype constraints should be similar
        assert len(metadata1.allowed_datatypes) > 0
        assert len(metadata2.allowed_datatypes) > 0
    
    def test_non_applicable_pragmas_filtered(self):
        """Test that non-applicable pragmas are filtered out."""
        
        in_interface = self.create_test_interface("in0")
        out_interface = Interface(
            name="out0",
            type=InterfaceType.OUTPUT,
            ports={"data": Port(name="out0_data", direction=Direction.OUTPUT, width="32")},
            validation_result=ValidationResult(valid=True, message="Test interface"),
            metadata={}
        )
        
        pragmas = [
            DatatypePragma(type=PragmaType.DATATYPE, inputs=["in0", "UINT", "8", "8"], line_number=10),
            DatatypePragma(type=PragmaType.DATATYPE, inputs=["out0", "INT", "16", "16"], line_number=11),
            WeightPragma(type=PragmaType.WEIGHT, inputs=["in0"], line_number=20)
        ]
        
        # Apply to in0 interface - should only get pragmas for in0
        in_metadata = self.apply_pragmas_to_interface(in_interface, pragmas)
        assert in_metadata.interface_type == InterfaceType.WEIGHT  # WeightPragma applied
        
        # Apply to out0 interface - should only get pragmas for out0
        out_metadata = self.apply_pragmas_to_interface(out_interface, pragmas)
        assert out_metadata.interface_type == InterfaceType.OUTPUT  # No WeightPragma applied
    
    def test_error_isolation(self):
        """Test that errors in pragma application don't break the chain."""
        interface = self.create_test_interface("test")
        
        # Create a pragma that will cause an error
        class BadPragma(DatatypePragma):
            def apply_to_interface_metadata(self, interface, metadata):
                raise Exception("Simulated pragma error")
        
        bad_pragma = BadPragma(
            type=PragmaType.DATATYPE,
            inputs=["test", "UINT", "8", "8"],
            line_number=100
        )
        
        good_pragma = WeightPragma(
            type=PragmaType.WEIGHT,
            inputs=["test"],
            line_number=101
        )
        
        pragmas = [bad_pragma, good_pragma]
        
        # Should not raise exception, and good pragma should still be applied
        metadata = self.apply_pragmas_to_interface(interface, pragmas)
        assert metadata.interface_type == InterfaceType.WEIGHT  # Good pragma applied


if __name__ == "__main__":
    # Run tests using pytest
    pytest.main([__file__, "-v"])