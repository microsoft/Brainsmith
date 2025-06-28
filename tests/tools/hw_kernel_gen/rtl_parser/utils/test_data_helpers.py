############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Test data helpers for creating consistent test objects.

This module provides factory methods for creating Parameter, Port, PortGroup,
and other objects used in RTL parser tests, reducing duplication and ensuring
consistency.
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from brainsmith.tools.hw_kernel_gen.rtl_parser.rtl_data import (
    Parameter, Port, PortGroup, PragmaType
)
from brainsmith.tools.hw_kernel_gen.rtl_parser.pragmas.base import Pragma
from brainsmith.tools.hw_kernel_gen.data import InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser.rtl_data import PortDirection
from brainsmith.tools.hw_kernel_gen.metadata import InterfaceMetadata, KernelMetadata


class TestDataHelpers:
    """Factory methods for creating test data objects."""
    
    @staticmethod
    def create_port(name: str, direction: str = "input", 
                   width: str = "1", protocol: str = None) -> Port:
        """Create a Port object with common defaults.
        
        Args:
            name: Port name
            direction: "input", "output", or "inout"
            width: Port width expression (e.g., "32", "WIDTH-1:0")
            protocol: Optional protocol hint (e.g., "axi_stream", "axi_lite")
        
        Returns:
            Port object
        """
        # Convert simple width to Verilog format
        if width.isdigit() and int(width) > 1:
            width = f"{width}-1:0"
        
        return Port(name=name, direction=direction, width=width)
    
    @staticmethod
    def create_axi_stream_ports(prefix: str, direction: str = "slave",
                               data_width: str = "32",
                               has_last: bool = False,
                               has_keep: bool = False) -> List[Port]:
        """Create a set of AXI-Stream ports.
        
        Args:
            prefix: Interface prefix (e.g., "s_axis_input")
            direction: "slave" or "master"
            data_width: Data width
            has_last: Include TLAST signal
            has_keep: Include TKEEP signal
        
        Returns:
            List of Port objects
        """
        ports = []
        
        # Determine signal directions based on interface direction
        if direction == "slave":
            data_dir = "input"
            ready_dir = "output"
        else:
            data_dir = "output"
            ready_dir = "input"
        
        # Core AXI-Stream signals
        ports.append(TestDataHelpers.create_port(
            f"{prefix}_tdata", data_dir, data_width, "axi_stream"
        ))
        ports.append(TestDataHelpers.create_port(
            f"{prefix}_tvalid", data_dir, "1", "axi_stream"
        ))
        ports.append(TestDataHelpers.create_port(
            f"{prefix}_tready", ready_dir, "1", "axi_stream"
        ))
        
        # Optional signals
        if has_last:
            ports.append(TestDataHelpers.create_port(
                f"{prefix}_tlast", data_dir, "1", "axi_stream"
            ))
        
        if has_keep:
            keep_width = f"({data_width})/8" if not data_width.isdigit() else str(int(data_width) // 8)
            ports.append(TestDataHelpers.create_port(
                f"{prefix}_tkeep", data_dir, keep_width, "axi_stream"
            ))
        
        return ports
    
    @staticmethod
    def create_parameter(name: str, value: str = "32", 
                        param_type: str = "integer") -> Parameter:
        """Create a Parameter object with defaults.
        
        Args:
            name: Parameter name
            value: Default value
            param_type: Parameter type (e.g., "integer", None)
        
        Returns:
            Parameter object
        """
        return Parameter(name=name, default_value=value, param_type=param_type)
    
    @staticmethod
    def create_parameter_set(pattern: str = "standard") -> List[Parameter]:
        """Create common parameter sets.
        
        Args:
            pattern: Pattern name:
                - "standard": Basic WIDTH, DEPTH parameters
                - "axi_stream": Parameters for AXI-Stream interface
                - "indexed": Multi-dimensional indexed parameters
                - "conv2d": Convolution-specific parameters
        
        Returns:
            List of Parameter objects
        """
        if pattern == "standard":
            return [
                TestDataHelpers.create_parameter("WIDTH", "32"),
                TestDataHelpers.create_parameter("DEPTH", "16"),
                TestDataHelpers.create_parameter("SIGNED", "0", None)
            ]
        
        elif pattern == "axi_stream":
            return [
                TestDataHelpers.create_parameter("DATA_WIDTH", "32"),
                TestDataHelpers.create_parameter("s_axis_input_BDIM", "64"),
                TestDataHelpers.create_parameter("s_axis_input_SDIM", "512"),
                TestDataHelpers.create_parameter("m_axis_output_BDIM", "64")
            ]
        
        elif pattern == "indexed":
            return [
                TestDataHelpers.create_parameter("in0_BDIM0", "16"),
                TestDataHelpers.create_parameter("in0_BDIM1", "16"),
                TestDataHelpers.create_parameter("in0_BDIM2", "3"),
                TestDataHelpers.create_parameter("in0_SDIM0", "224"),
                TestDataHelpers.create_parameter("in0_SDIM1", "224")
            ]
        
        elif pattern == "conv2d":
            return [
                TestDataHelpers.create_parameter("INPUT_HEIGHT", "28"),
                TestDataHelpers.create_parameter("INPUT_WIDTH", "28"),
                TestDataHelpers.create_parameter("KERNEL_HEIGHT", "3"),
                TestDataHelpers.create_parameter("KERNEL_WIDTH", "3"),
                TestDataHelpers.create_parameter("OUTPUT_HEIGHT", "26"),
                TestDataHelpers.create_parameter("OUTPUT_WIDTH", "26"),
                TestDataHelpers.create_parameter("IN_CHANNELS", "3"),
                TestDataHelpers.create_parameter("OUT_CHANNELS", "32")
            ]
        
        else:
            raise ValueError(f"Unknown parameter pattern: {pattern}")
    
    @staticmethod
    def create_port_group(interface_type: InterfaceType,
                         name: str,
                         data_width: str = "32") -> PortGroup:
        """Create a PortGroup for interface scanning.
        
        Args:
            interface_type: Type of interface
            name: Interface name (e.g., "input0")
            data_width: Data width for the interface
        
        Returns:
            PortGroup object with appropriate ports
        """
        port_group = PortGroup(interface_type=interface_type, name=name)
        
        if interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT, InterfaceType.WEIGHT]:
            # AXI-Stream interface
            prefix = f"s_axis_{name}" if interface_type != InterfaceType.OUTPUT else f"m_axis_{name}"
            direction = "slave" if interface_type != InterfaceType.OUTPUT else "master"
            
            ports = TestDataHelpers.create_axi_stream_ports(prefix, direction, data_width)
            for port in ports:
                # Extract signal type (e.g., "tdata", "tvalid")
                signal_type = port.name.split('_')[-1]
                port_group.add_port(port, signal_type.upper())
        
        elif interface_type == InterfaceType.CONTROL:
            # Global control interface
            port_group.add_port(
                TestDataHelpers.create_port("ap_clk", "input"), "CLK"
            )
            port_group.add_port(
                TestDataHelpers.create_port("ap_rst_n", "input"), "RST"
            )
        
        return port_group
    
    @staticmethod
    def create_pragma(pragma_type: str, interface: str, 
                     *args: str, line_number: int = 1) -> Pragma:
        """Create a Pragma object consistently.
        
        Args:
            pragma_type: Type of pragma (e.g., "BDIM", "DATATYPE")
            interface: Interface name
            args: Additional pragma arguments
            line_number: Line number for error reporting
        
        Returns:
            Pragma object
        """
        # Convert string to PragmaType enum
        pragma_enum = PragmaType(pragma_type.lower())
        
        # Build inputs dictionary
        inputs = {
            'raw': [interface] + list(args),
            'positional': [interface] + list(args),
            'named': {}
        }
        
        # Handle special parsing for lists
        processed_positional = []
        for arg in inputs['positional']:
            if arg.startswith('[') and arg.endswith(']'):
                # Parse as list
                list_content = arg[1:-1].strip()
                if list_content:
                    parsed_list = [item.strip() for item in list_content.split(',')]
                else:
                    parsed_list = []
                processed_positional.append(parsed_list)
            else:
                processed_positional.append(arg)
        
        inputs['positional'] = processed_positional
        
        # Import the appropriate pragma class
        if pragma_enum == PragmaType.BDIM:
            from brainsmith.tools.hw_kernel_gen.rtl_parser.pragmas.dimension import BdimPragma
            return BdimPragma(type=pragma_enum, inputs=inputs, line_number=line_number)
        elif pragma_enum == PragmaType.SDIM:
            from brainsmith.tools.hw_kernel_gen.rtl_parser.pragmas.dimension import SdimPragma
            return SdimPragma(type=pragma_enum, inputs=inputs, line_number=line_number)
        elif pragma_enum == PragmaType.DATATYPE:
            from brainsmith.tools.hw_kernel_gen.rtl_parser.pragmas.interface import DatatypePragma
            return DatatypePragma(type=pragma_enum, inputs=inputs, line_number=line_number)
        elif pragma_enum == PragmaType.WEIGHT:
            from brainsmith.tools.hw_kernel_gen.rtl_parser.pragmas.interface import WeightPragma
            return WeightPragma(type=pragma_enum, inputs=inputs, line_number=line_number)
        elif pragma_enum == PragmaType.ALIAS:
            from brainsmith.tools.hw_kernel_gen.rtl_parser.pragmas.parameter import AliasPragma
            return AliasPragma(type=pragma_enum, inputs=inputs, line_number=line_number)
        else:
            # Generic pragma
            return Pragma(type=pragma_enum, inputs=inputs, line_number=line_number)
    
    @staticmethod
    def create_interface_metadata(name: str,
                                 interface_type: InterfaceType,
                                 has_params: bool = True) -> InterfaceMetadata:
        """Create InterfaceMetadata with common patterns.
        
        Args:
            name: Interface name
            interface_type: Type of interface
            has_params: Whether to include BDIM/SDIM parameters
        
        Returns:
            InterfaceMetadata object
        """
        # Determine compiler name based on type
        if interface_type == InterfaceType.INPUT:
            compiler_name = f"input{name[-1]}" if name[-1].isdigit() else "input0"
        elif interface_type == InterfaceType.OUTPUT:
            compiler_name = f"output{name[-1]}" if name[-1].isdigit() else "output0"
        elif interface_type == InterfaceType.WEIGHT:
            compiler_name = f"weight{name[-1]}" if name[-1].isdigit() else "weight0"
        else:
            compiler_name = name
        
        metadata = InterfaceMetadata(
            name=name,
            interface_type=interface_type
        )
        
        # Add parameters if requested
        if has_params and interface_type != InterfaceType.CONTROL:
            metadata.bdim_params = [f"{name}_BDIM"]
            if interface_type != InterfaceType.OUTPUT:
                metadata.sdim_params = [f"{name}_SDIM"]
        
        return metadata