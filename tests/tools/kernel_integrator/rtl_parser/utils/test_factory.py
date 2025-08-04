############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Test data factory for creating consistent test data and expected results.

This factory provides methods to create RTL code and corresponding expected
KernelMetadata objects, making it easier to write comprehensive tests.
"""

from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path

from brainsmith.tools.kernel_integrator.metadata import (
    KernelMetadata, InterfaceMetadata, DatatypeMetadata, DimensionRelationship
)
from brainsmith.tools.kernel_integrator.rtl_parser.rtl_data import Parameter
from brainsmith.tools.kernel_integrator.rtl_parser.pragmas.base import Pragma
from brainsmith.tools.kernel_integrator.rtl_parser.rtl_data import PragmaType
from brainsmith.tools.kernel_integrator.data import InterfaceType, DatatypeConstraintGroup
from brainsmith.tools.kernel_integrator.rtl_parser.rtl_data import PortDirection
from .rtl_builder import RTLBuilder, StrictRTLBuilder


class TestDataFactory:
    """Factory for creating test data and expected results."""
    
    @staticmethod
    def minimal_strict_kernel() -> Tuple[str, KernelMetadata]:
        """Returns (rtl_code, expected_metadata) for minimal strict module."""
        # Build RTL
        rtl = (StrictRTLBuilder()
               .module("minimal_strict")
               .add_stream_input("s_axis_input", bdim_value="16", sdim_value="256")
               .add_stream_output("m_axis_output", bdim_value="16")
               .parameter("DATA_WIDTH", "32")
               .assign("m_axis_output_tdata", "s_axis_input_tdata")
               .assign("m_axis_output_tvalid", "s_axis_input_tvalid")
               .assign("s_axis_input_tready", "m_axis_output_tready")
               .build())
        
        # Build expected metadata
        metadata = KernelMetadata(
            name="minimal_strict",
            source_file=Path("minimal_strict.sv"),
            parameters=[
                Parameter("s_axis_input_BDIM", "16", "integer"),
                Parameter("s_axis_input_SDIM", "256", "integer"),
                Parameter("m_axis_output_BDIM", "16", "integer"),
                Parameter("DATA_WIDTH", "32", "integer")
            ],
            interfaces=[
                InterfaceMetadata(
                    name="global_control",
                    interface_type=InterfaceType.CONTROL,
                ),
                InterfaceMetadata(
                    name="s_axis_input",
                    interface_type=InterfaceType.INPUT,
                    bdim_params=["s_axis_input_BDIM"],
                    sdim_params=["s_axis_input_SDIM"]
                ),
                InterfaceMetadata(
                    name="m_axis_output",
                    interface_type=InterfaceType.OUTPUT,
                    bdim_params=["m_axis_output_BDIM"]
                )
            ],
            exposed_parameters=["DATA_WIDTH"],
            pragmas=[],
            internal_datatypes=[]
        )
        
        return rtl, metadata
    
    @staticmethod
    def interface_with_pragmas(interface_type: InterfaceType, 
                             pragmas: List[str]) -> Tuple[str, InterfaceMetadata]:
        """Create interface with specific pragmas applied."""
        # Build RTL with pragmas
        builder = StrictRTLBuilder().module("pragma_test")
        
        interface_name = "s_axis_test" if interface_type != InterfaceType.OUTPUT else "m_axis_test"
        
        # Add pragmas
        for pragma in pragmas:
            if pragma == "datatype_uint":
                builder.pragma("DATATYPE", interface_name, "UINT", "8", "32")
            elif pragma == "weight":
                builder.pragma("WEIGHT", interface_name)
            elif pragma.startswith("bdim:"):
                params = pragma.split(":", 1)[1].split(",")
                if len(params) == 1:
                    builder.pragma("BDIM", interface_name, params[0])
                else:
                    builder.pragma("BDIM", interface_name, f"[{', '.join(params)}]")
        
        # Add interface
        if interface_type == InterfaceType.OUTPUT:
            builder.add_stream_output(interface_name)
        else:
            builder.add_stream_input(interface_name)
        
        # Add dummy output if needed
        if interface_type != InterfaceType.OUTPUT:
            builder.add_stream_output("m_axis_dummy")
        
        rtl = builder.build()
        
        # Build expected interface metadata
        interface = InterfaceMetadata(
            name=interface_name,
            interface_type=interface_type
        )
        
        # Apply pragma effects
        if "datatype_uint" in pragmas:
            interface.datatype_constraints = [
                DatatypeConstraintGroup("UINT", 8, 32)
            ]
        
        if "weight" in pragmas:
            interface.interface_type = InterfaceType.WEIGHT
            compiler_name = "weight0"
        
        for pragma in pragmas:
            if pragma.startswith("bdim:"):
                params = pragma.split(":", 1)[1].split(",")
                interface.bdim_params = params
        
        return rtl, interface
    
    @staticmethod
    def multi_interface_kernel(num_inputs: int = 2, num_outputs: int = 1,
                             num_weights: int = 1) -> Tuple[str, KernelMetadata]:
        """Create kernel with multiple interfaces of each type."""
        builder = StrictRTLBuilder().module("multi_interface")
        
        interfaces = []
        
        # Add control interface (auto-added by StrictRTLBuilder)
        interfaces.append(InterfaceMetadata(
            name="global_control",
            interface_type=InterfaceType.CONTROL,
        ))
        
        # Add inputs
        for i in range(num_inputs):
            name = f"s_axis_in{i}"
            builder.add_stream_input(name, bdim_value=str(16 * (i + 1)))
            interfaces.append(InterfaceMetadata(
                name=name,
                interface_type=InterfaceType.INPUT,
                    bdim_params=[f"{name}_BDIM"],
                sdim_params=[f"{name}_SDIM"]
            ))
        
        # Add weights
        for i in range(num_weights):
            name = f"s_axis_weight{i}"
            builder.add_stream_weight(name, bdim_value=str(32 * (i + 1)))
            interfaces.append(InterfaceMetadata(
                name=name,
                interface_type=InterfaceType.WEIGHT,
                    bdim_params=[f"{name}_BDIM"],
                sdim_params=[f"{name}_SDIM"]
            ))
        
        # Add outputs
        for i in range(num_outputs):
            name = f"m_axis_out{i}"
            builder.add_stream_output(name, bdim_value=str(64 * (i + 1)))
            interfaces.append(InterfaceMetadata(
                name=name,
                interface_type=InterfaceType.OUTPUT,
                bdim_params=[f"{name}_BDIM"]
            ))
        
        rtl = builder.build()
        
        # Build metadata
        metadata = KernelMetadata(
            name="multi_interface",
            source_file=Path("multi_interface.sv"),
            interfaces=interfaces,
            parameters=[],  # Will be populated by parser
            exposed_parameters=[],  # BDIM/SDIM params removed by auto-linking
            pragmas=[],
            internal_datatypes=[]
        )
        
        return rtl, metadata
    
    @staticmethod
    def create_pragma_test_set() -> List[Pragma]:
        """Create common pragma combinations for testing."""
        
        pragmas = []
        
        # BDIM pragmas
        pragmas.append(Pragma(
            type=PragmaType.BDIM,
            inputs={
                'raw': ['s_axis_input', 'INPUT_BDIM'],
                'positional': ['s_axis_input', 'INPUT_BDIM'],
                'named': {}
            },
            line_number=1
        ))
        
        # DATATYPE pragmas
        pragmas.append(Pragma(
            type=PragmaType.DATATYPE,
            inputs={
                'raw': ['s_axis_input', 'UINT', '8', '32'],
                'positional': ['s_axis_input', 'UINT', '8', '32'],
                'named': {}
            },
            line_number=2
        ))
        
        # ALIAS pragma
        pragmas.append(Pragma(
            type=PragmaType.ALIAS,
            inputs={
                'raw': ['PE', 'ProcessingElements'],
                'positional': ['PE', 'ProcessingElements'],
                'named': {}
            },
            line_number=3
        ))
        
        return pragmas
    
    @staticmethod
    def create_invalid_pragma_set() -> List[Pragma]:
        """Create pragmas that should fail validation."""
        
        invalid_pragmas = []
        
        # BDIM with magic numbers
        invalid_pragmas.append(Pragma(
            type=PragmaType.BDIM,
            inputs={
                'raw': ['s_axis_input', '[32, 64]'],
                'positional': ['s_axis_input', ['32', '64']],
                'named': {}
            },
            line_number=1
        ))
        
        # DATATYPE with invalid range
        invalid_pragmas.append(Pragma(
            type=PragmaType.DATATYPE,
            inputs={
                'raw': ['s_axis_input', 'UINT', '32', '8'],  # min > max
                'positional': ['s_axis_input', 'UINT', '32', '8'],
                'named': {}
            },
            line_number=2
        ))
        
        # RELATIONSHIP missing args
        invalid_pragmas.append(Pragma(
            type=PragmaType.RELATIONSHIP,
            inputs={
                'raw': ['s_axis_a'],  # Missing target and type
                'positional': ['s_axis_a'],
                'named': {}
            },
            line_number=3
        ))
        
        return invalid_pragmas
    
    @staticmethod
    def kernel_with_relationships(relationship_type: str = "EQUAL") -> Tuple[str, KernelMetadata]:
        """Create kernel with interface relationships."""
        builder = (StrictRTLBuilder()
                  .module("relationship_test")
                  .add_stream_input("s_axis_a", bdim_value="32", sdim_value="512")
                  .add_stream_input("s_axis_b", bdim_value="32", sdim_value="512")
                  .add_stream_output("m_axis_out", bdim_value="32"))
        
        # Add relationship pragma
        if relationship_type == "EQUAL":
            builder.pragma("RELATIONSHIP", "s_axis_a", "s_axis_b", "EQUAL")
        elif relationship_type == "DEPENDENT":
            builder.pragma("RELATIONSHIP", "s_axis_a", "m_axis_out", "DEPENDENT", "0", "0", "scaled", "2")
        
        rtl = builder.build()
        
        # Build expected metadata with relationship
        relationships = []
        if relationship_type == "EQUAL":
            relationships.append(RelationshipMetadata(
                source_interface="s_axis_a",
                target_interface="s_axis_b",
                relationship_type="EQUAL"
            ))
        elif relationship_type == "DEPENDENT":
            relationships.append(RelationshipMetadata(
                source_interface="s_axis_a",
                target_interface="m_axis_out",
                relationship_type="DEPENDENT",
                source_dim=0,
                target_dim=0,
                dependency_type="scaled",
                scale_factor=2
            ))
        
        metadata = KernelMetadata(
            name="relationship_test",
            source_file=Path("relationship_test.sv"),
            interfaces=[],  # Will be populated by parser
            parameters=[],
            exposed_parameters=[],
            pragmas=[],
            relationships=relationships,
            internal_datatypes=[]
        )
        
        return rtl, metadata
    
    @staticmethod
    def kernel_with_internal_datatypes() -> Tuple[str, KernelMetadata]:
        """Create kernel with internal datatype definitions."""
        rtl = (StrictRTLBuilder()
               .module("internal_dt_test")
               .add_stream_input("s_axis_input")
               .add_stream_output("m_axis_output")
               .parameter("ACC_WIDTH", "48")
               .parameter("ACC_SIGNED", "1")
               .parameter("THRESH_WIDTH", "32")
               .pragma("DATATYPE_PARAM", "accumulator", "width", "ACC_WIDTH")
               .pragma("DATATYPE_PARAM", "accumulator", "signed", "ACC_SIGNED")
               .pragma("DATATYPE_PARAM", "threshold", "width", "THRESH_WIDTH")
               .build())
        
        # Expected internal datatypes
        internal_dts = [
            DatatypeMetadata(
                name="accumulator",
                width="ACC_WIDTH",
                signed="ACC_SIGNED"
            ),
            DatatypeMetadata(
                name="threshold",
                width="THRESH_WIDTH"
            )
        ]
        
        metadata = KernelMetadata(
            name="internal_dt_test",
            source_file=Path("internal_dt_test.sv"),
            interfaces=[],  # Will be populated by parser
            parameters=[],
            exposed_parameters=[],  # ACC_WIDTH etc. removed by DATATYPE_PARAM
            pragmas=[],
            internal_datatypes=internal_dts
        )
        
        return rtl, metadata
    
    @staticmethod
    def non_strict_minimal() -> Tuple[str, KernelMetadata]:
        """Create minimal non-strict module for basic tests."""
        rtl = (RTLBuilder()
               .module("non_strict")
               .parameter("WIDTH", "32")
               .port("clk", "input")
               .port("rst", "input") 
               .port("data_in", "input", "WIDTH-1:0")
               .port("data_out", "output", "WIDTH-1:0")
               .assign("data_out", "data_in")
               .build())
        
        metadata = KernelMetadata(
            name="non_strict",
            source_file=Path("non_strict.sv"),
            parameters=[Parameter("WIDTH", "32", "integer")],
            interfaces=[],  # No AXI interfaces detected
            exposed_parameters=["WIDTH"],
            pragmas=[],
            internal_datatypes=[]
        )
        
        return rtl, metadata