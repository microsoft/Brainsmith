############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""RTL Builder utility for programmatic SystemVerilog generation.

This utility allows test cases to programmatically generate SystemVerilog RTL
code with precise control over module structure, parameters, ports, and pragmas.
Follows PD-3 by generating real SystemVerilog that the parser must handle.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class RTLParameter:
    """Represents a module parameter."""
    name: str
    value: str
    param_type: str = "integer"
    
    def to_verilog(self) -> str:
        """Generate Verilog parameter declaration."""
        if self.param_type:
            return f"parameter {self.param_type} {self.name} = {self.value}"
        else:
            return f"parameter {self.name} = {self.value}"


@dataclass
class RTLPort:
    """Represents a module port."""
    name: str
    direction: str  # input, output, inout
    width: str = "1"
    port_type: str = "wire"
    
    def to_verilog(self) -> str:
        """Generate Verilog port declaration."""
        if self.width != "1":
            # Handle pre-formatted width expressions like "WIDTH-1:0"
            if ":" in self.width:
                width_str = f"[{self.width}] "
            else:
                width_str = f"[{self.width}-1:0] "
        else:
            width_str = ""
        return f"{self.direction} {self.port_type} {width_str}{self.name}"


@dataclass
class RTLPragma:
    """Represents a Brainsmith pragma."""
    pragma_type: str
    args: List[str]
    location: str = "before_module"  # before_module, after_ports, inline
    target_line: Optional[int] = None
    
    def to_comment(self) -> str:
        """Generate pragma as Verilog comment."""
        args_str = " ".join(self.args)
        return f"// @brainsmith {self.pragma_type} {args_str}"


class RTLBuilder:
    """Programmatic RTL generation for test cases."""
    
    def __init__(self):
        """Initialize empty RTL builder."""
        self._module_name: Optional[str] = None
        self._parameters: List[RTLParameter] = []
        self._ports: List[RTLPort] = []
        self._pragmas: List[RTLPragma] = []
        self._body_lines: List[str] = []
        self._current_indent = 0
        
    def module(self, name: str) -> 'RTLBuilder':
        """Set the module name."""
        self._module_name = name
        return self
        
    def parameter(self, name: str, value: str, param_type: str = "integer") -> 'RTLBuilder':
        """Add a parameter to the module."""
        self._parameters.append(RTLParameter(name, value, param_type))
        return self
        
    def port(self, name: str, direction: str, width: str = "1", port_type: str = "wire") -> 'RTLBuilder':
        """Add a port to the module."""
        self._ports.append(RTLPort(name, direction, width, port_type))
        return self
        
    def pragma(self, pragma_type: str, *args, location: str = "before_module") -> 'RTLBuilder':
        """Add a pragma comment."""
        self._pragmas.append(RTLPragma(pragma_type, list(args), location))
        return self
        
    def body(self, *lines: str) -> 'RTLBuilder':
        """Add lines to the module body."""
        self._body_lines.extend(lines)
        return self
        
    def assign(self, signal: str, value: str) -> 'RTLBuilder':
        """Add an assign statement."""
        self._body_lines.append(f"assign {signal} = {value};")
        return self
        
    def always_comb(self, *lines: str) -> 'RTLBuilder':
        """Add an always_comb block."""
        self._body_lines.append("always_comb begin")
        for line in lines:
            self._body_lines.append(f"    {line}")
        self._body_lines.append("end")
        return self
        
    def always_ff(self, sensitivity: str, *lines: str) -> 'RTLBuilder':
        """Add an always_ff block."""
        self._body_lines.append(f"always_ff @({sensitivity}) begin")
        for line in lines:
            self._body_lines.append(f"    {line}")
        self._body_lines.append("end")
        return self
        
    def comment(self, text: str) -> 'RTLBuilder':
        """Add a regular comment."""
        self._body_lines.append(f"// {text}")
        return self
        
    def axi_stream_slave(self, prefix: str, data_width: str = "32", 
                        has_last: bool = False, has_keep: bool = False) -> 'RTLBuilder':
        """Add a complete AXI-Stream slave interface."""
        self.port(f"{prefix}_tdata", "input", data_width)
        self.port(f"{prefix}_tvalid", "input")
        self.port(f"{prefix}_tready", "output")
        if has_last:
            self.port(f"{prefix}_tlast", "input")
        if has_keep:
            keep_width = f"({data_width})/8" if data_width != "8" else "1"
            self.port(f"{prefix}_tkeep", "input", keep_width)
        return self
        
    def axi_stream_master(self, prefix: str, data_width: str = "32",
                         has_last: bool = False, has_keep: bool = False) -> 'RTLBuilder':
        """Add a complete AXI-Stream master interface."""
        self.port(f"{prefix}_tdata", "output", data_width)
        self.port(f"{prefix}_tvalid", "output")
        self.port(f"{prefix}_tready", "input")
        if has_last:
            self.port(f"{prefix}_tlast", "output")
        if has_keep:
            keep_width = f"({data_width})/8" if data_width != "8" else "1"
            self.port(f"{prefix}_tkeep", "output", keep_width)
        return self
        
    def axi_lite_slave(self, prefix: str, addr_width: str = "32", data_width: str = "32") -> 'RTLBuilder':
        """Add a complete AXI-Lite slave interface."""
        # Write address channel
        self.port(f"{prefix}_awaddr", "input", addr_width)
        self.port(f"{prefix}_awvalid", "input")
        self.port(f"{prefix}_awready", "output")
        
        # Write data channel
        self.port(f"{prefix}_wdata", "input", data_width)
        self.port(f"{prefix}_wstrb", "input", f"({data_width})/8")
        self.port(f"{prefix}_wvalid", "input")
        self.port(f"{prefix}_wready", "output")
        
        # Write response channel
        self.port(f"{prefix}_bresp", "output", "2")
        self.port(f"{prefix}_bvalid", "output")
        self.port(f"{prefix}_bready", "input")
        
        # Read address channel
        self.port(f"{prefix}_araddr", "input", addr_width)
        self.port(f"{prefix}_arvalid", "input")
        self.port(f"{prefix}_arready", "output")
        
        # Read data channel
        self.port(f"{prefix}_rdata", "output", data_width)
        self.port(f"{prefix}_rresp", "output", "2")
        self.port(f"{prefix}_rvalid", "output")
        self.port(f"{prefix}_rready", "input")
        
        return self
        
    def add_global_control(self, clk_name: str = "ap_clk", rst_name: str = "ap_rst_n") -> 'RTLBuilder':
        """Add standard global control interface."""
        self.port(clk_name, "input")
        self.port(rst_name, "input")
        return self
    
    def add_complete_axi_stream(self, prefix: str, direction: str, 
                               data_width: str = "32",
                               has_bdim: bool = True, has_sdim: bool = True,
                               bdim_value: str = "32", sdim_value: str = "512",
                               has_last: bool = False, has_keep: bool = False) -> 'RTLBuilder':
        """Add AXI-Stream with automatic BDIM/SDIM parameters."""
        # Add parameters if requested
        if has_bdim:
            bdim_param = f"{prefix}_BDIM"
            self.parameter(bdim_param, bdim_value)
        
        if has_sdim and direction == "input":
            sdim_param = f"{prefix}_SDIM"
            self.parameter(sdim_param, sdim_value)
        
        # Add interface
        if direction == "input":
            self.axi_stream_slave(prefix, data_width, has_last, has_keep)
        else:
            self.axi_stream_master(prefix, data_width, has_last, has_keep)
        
        return self
    
    def strict_module(self, name: str) -> 'RTLBuilder':
        """Initialize a module that will pass strict validation."""
        self.module(name)
        self.add_global_control()
        return self
    
    def add_pragma_set(self, interface: str, pragmas: List[str]) -> 'RTLBuilder':
        """Add a common set of pragmas for an interface."""
        for pragma in pragmas:
            if pragma == "datatype_uint":
                self.pragma("DATATYPE", interface, "UINT", "8", "32")
            elif pragma == "datatype_int":
                self.pragma("DATATYPE", interface, "INT", "8", "16")
            elif pragma == "weight":
                self.pragma("WEIGHT", interface)
            elif pragma.startswith("bdim:"):
                _, param = pragma.split(":", 1)
                self.pragma("BDIM", interface, param)
            elif pragma.startswith("sdim:"):
                _, param = pragma.split(":", 1)
                self.pragma("SDIM", interface, param)
        return self
    
    def add_axi_stream_pair(self, name: str, data_width: str = "32", 
                           with_params: bool = True,
                           bdim_value: str = "32", sdim_value: str = "512") -> 'RTLBuilder':
        """Add matching input/output AXI-Stream interfaces with optional BDIM/SDIM."""
        if with_params:
            self.parameter(f"s_axis_{name}_BDIM", bdim_value)
            self.parameter(f"s_axis_{name}_SDIM", sdim_value)
            self.parameter(f"m_axis_{name}_BDIM", bdim_value)
        
        self.axi_stream_slave(f"s_axis_{name}", data_width)
        self.axi_stream_master(f"m_axis_{name}", data_width)
        return self
    
    def add_pragma_block(self, pragmas: List[Tuple[str, List[str]]]) -> 'RTLBuilder':
        """Add multiple pragmas efficiently."""
        for pragma_type, args in pragmas:
            self.pragma(pragma_type, *args)
        return self
    
    def add_standard_params(self, include_dims: bool = True, 
                           include_widths: bool = True) -> 'RTLBuilder':
        """Add common parameter sets."""
        if include_widths:
            self.parameter("DATA_WIDTH", "32")
            self.parameter("ADDR_WIDTH", "16")
        
        if include_dims:
            self.parameter("INPUT_DIM", "32")
            self.parameter("OUTPUT_DIM", "32")
        
        return self
    
    def add_malformed_port(self, name: str, error_type: str) -> 'RTLBuilder':
        """Add intentionally malformed port for error testing.
        
        Args:
            name: Port name
            error_type: Type of error to introduce
                - "missing_width": Vector without width
                - "invalid_width": Invalid width syntax  
                - "duplicate_name": Port already exists
                - "invalid_direction": Bad direction
        """
        if error_type == "missing_width":
            self._body_lines.append(f"input wire [] {name};  // Missing width")
        elif error_type == "invalid_width":
            self._body_lines.append(f"input wire [32] {name};  // Invalid width syntax")
        elif error_type == "duplicate_name":
            # Add same port twice
            self.port(name, "input")
            self.port(name, "output")  # Duplicate with different direction
        elif error_type == "invalid_direction":
            self._body_lines.append(f"inputoutput wire {name};  // Invalid direction")
        else:
            raise ValueError(f"Unknown error type: {error_type}")
        return self
    
    def add_partial_interface(self, protocol: str, completeness: float) -> 'RTLBuilder':
        """Add partially complete interface (0.0 to 1.0).
        
        Args:
            protocol: Interface protocol ("axi_stream", "axi_lite", etc.)
            completeness: How complete the interface should be (0.0-1.0)
        """
        import random
        random.seed(42)  # Deterministic for testing
        
        if protocol == "axi_stream":
            signals = [
                ("tdata", "input", "31:0"),
                ("tvalid", "input", ""),
                ("tready", "output", ""),
                ("tlast", "input", ""),
                ("tkeep", "input", "3:0")
            ]
            prefix = f"s_axis_partial_{int(completeness*100)}"
            
            # Add subset of signals based on completeness
            num_signals = max(1, int(len(signals) * completeness))
            selected = random.sample(signals, num_signals)
            
            for signal, direction, width in selected:
                port_name = f"{prefix}_{signal}"
                self.port(port_name, direction, width if width else "1")
        
        return self
    
    def add_syntax_error(self, location: str, error_type: str) -> 'RTLBuilder':
        """Insert syntax error at specific location.
        
        Args:
            location: Where to insert error ("module", "port", "body") 
            error_type: Type of syntax error
        """
        if location == "module" and error_type == "missing_paren":
            self._module_name = self._module_name + " #(WIDTH=32"  # Missing )
        elif location == "port" and error_type == "missing_comma":
            # This will be handled during build by skipping commas
            self._syntax_error_location = "port_list"
        elif location == "body" and error_type == "missing_semi":
            self._body_lines.append("assign bad_signal = 1'b1  // Missing semicolon")
        elif location == "body" and error_type == "unclosed_block":
            self._body_lines.append("always @(posedge clk) begin")
            self._body_lines.append("    // Missing end")
        return self
    
    def build(self) -> str:
        """Generate the complete SystemVerilog RTL."""
        if not self._module_name:
            raise ValueError("Module name not set")
            
        lines = []
        
        # Add pragmas that should appear before module
        for pragma in self._pragmas:
            if pragma.location == "before_module":
                lines.append(pragma.to_comment())
        
        # Module declaration
        if self._parameters:
            lines.append(f"module {self._module_name} #(")
            param_lines = []
            for i, param in enumerate(self._parameters):
                comma = "," if i < len(self._parameters) - 1 else ""
                param_lines.append(f"    {param.to_verilog()}{comma}")
            lines.extend(param_lines)
            lines.append(") (")
        else:
            lines.append(f"module {self._module_name} (")
        
        # Ports
        if self._ports:
            for i, port in enumerate(self._ports):
                comma = "," if i < len(self._ports) - 1 else ""
                # Check for inline pragmas for this port
                port_pragmas = [p for p in self._pragmas 
                              if p.location == "inline" and port.name in p.args]
                if port_pragmas:
                    for pp in port_pragmas:
                        lines.append(f"    {pp.to_comment()}")
                lines.append(f"    {port.to_verilog()}{comma}")
        
        lines.append(");")
        
        # Add pragmas that should appear after ports
        for pragma in self._pragmas:
            if pragma.location == "after_ports":
                lines.append(f"    {pragma.to_comment()}")
        
        # Module body
        if self._body_lines:
            lines.append("")
            for line in self._body_lines:
                lines.append(f"    {line}")
        else:
            lines.append("    // Empty module")
        
        lines.append("")
        lines.append("endmodule")
        
        return "\n".join(lines)


# Convenience functions for common patterns
def create_minimal_module(name: str = "minimal") -> str:
    """Create a minimal valid SystemVerilog module."""
    return (RTLBuilder()
            .module(name)
            .port("clk", "input")
            .port("rst", "input")
            .build())


def create_axi_stream_module(name: str = "axi_module", 
                           data_width: int = 32,
                           num_inputs: int = 1,
                           num_outputs: int = 1) -> str:
    """Create a module with AXI-Stream interfaces."""
    builder = RTLBuilder().module(name)
    builder.parameter("DATA_WIDTH", str(data_width))
    builder.port("clk", "input")
    builder.port("rst", "input")
    
    for i in range(num_inputs):
        prefix = f"s_axis_input{i}" if num_inputs > 1 else "s_axis_input"
        builder.axi_stream_slave(prefix, "DATA_WIDTH")
    
    for i in range(num_outputs):
        prefix = f"m_axis_output{i}" if num_outputs > 1 else "m_axis_output"
        builder.axi_stream_master(prefix, "DATA_WIDTH")
    
    return builder.build()


def create_pragma_test_module(name: str = "pragma_test") -> str:
    """Create a module with various pragmas for testing."""
    return (RTLBuilder()
            .module(name)
            .pragma("TOP_MODULE", name)
            .parameter("WIDTH", "16")
            .parameter("DEPTH", "8")
            .port("clk", "input")
            .port("rst", "input")
            .pragma("DATATYPE", "s_axis_data", "UINT", "8", "32", location="inline")
            .axi_stream_slave("s_axis_data", "WIDTH")
            .pragma("ALIAS", "WIDTH", "data_width", location="after_ports")
            .pragma("DERIVED_PARAMETER", "TOTAL_BITS", "WIDTH * DEPTH", location="after_ports")
            .axi_stream_master("m_axis_result", "WIDTH")
            .build())


class StrictRTLBuilder(RTLBuilder):
    """RTL builder that ensures strict validation compliance.
    
    This builder automatically includes required components for strict mode:
    - Global control interface (ap_clk, ap_rst_n)
    - Ensures interfaces have proper BDIM/SDIM parameters
    - Validates builder state before build
    """
    
    def __init__(self):
        """Initialize with automatic global control."""
        super().__init__()
        self._has_input = False
        self._has_output = False
        self._interface_params = {}  # Track BDIM/SDIM for interfaces
    
    def module(self, name: str) -> 'StrictRTLBuilder':
        """Set module name and add global control."""
        super().module(name)
        self.add_global_control()
        return self
    
    def add_stream_input(self, name: str, data_width: str = "32", 
                        bdim_param: Optional[str] = None, bdim_value: str = "32",
                        sdim_param: Optional[str] = None, sdim_value: str = "512",
                        **kwargs) -> 'StrictRTLBuilder':
        """Add compliant AXI-Stream input with proper parameters."""
        # Auto-generate parameter names if not provided
        if bdim_param is None:
            bdim_param = f"{name}_BDIM"
        if sdim_param is None:
            sdim_param = f"{name}_SDIM"
        
        # Add parameters
        self.parameter(bdim_param, bdim_value)
        self.parameter(sdim_param, sdim_value)
        
        # Add interface
        self.axi_stream_slave(name, data_width, **kwargs)
        
        # Track for validation
        self._has_input = True
        self._interface_params[name] = {
            'type': 'input',
            'bdim': bdim_param,
            'sdim': sdim_param
        }
        
        return self
    
    def add_stream_output(self, name: str, data_width: str = "32",
                         bdim_param: Optional[str] = None, bdim_value: str = "32",
                         **kwargs) -> 'StrictRTLBuilder':
        """Add compliant AXI-Stream output with BDIM parameter."""
        # Auto-generate parameter name if not provided
        if bdim_param is None:
            bdim_param = f"{name}_BDIM"
        
        # Add parameter
        self.parameter(bdim_param, bdim_value)
        
        # Add interface
        self.axi_stream_master(name, data_width, **kwargs)
        
        # Track for validation
        self._has_output = True
        self._interface_params[name] = {
            'type': 'output',
            'bdim': bdim_param
        }
        
        return self
    
    def add_stream_weight(self, name: str, data_width: str = "8",
                         bdim_param: Optional[str] = None, bdim_value: str = "64",
                         sdim_param: Optional[str] = None, sdim_value: str = "512",
                         **kwargs) -> 'StrictRTLBuilder':
        """Add compliant weight interface with pragmas."""
        # Add as input with parameters
        self.add_stream_input(name, data_width, bdim_param, bdim_value, 
                            sdim_param, sdim_value, **kwargs)
        
        # Add WEIGHT pragma
        self.pragma("WEIGHT", name)
        
        # Update tracking
        self._interface_params[name]['type'] = 'weight'
        
        return self
    
    def validate(self) -> List[str]:
        """Validate that builder state will pass strict validation."""
        errors = []
        
        if not self._module_name:
            errors.append("Module name not set")
        
        if not self._has_input:
            errors.append("No input interface added")
        
        if not self._has_output:
            errors.append("No output interface added")
        
        # Check that we have global control (should always be true)
        has_ap_clk = any(p.name == "ap_clk" for p in self._ports)
        has_ap_rst_n = any(p.name == "ap_rst_n" for p in self._ports)
        
        if not has_ap_clk or not has_ap_rst_n:
            errors.append("Missing global control interface")
        
        return errors
    
    def build(self) -> str:
        """Build with validation check."""
        errors = self.validate()
        if errors:
            raise ValueError(f"StrictRTLBuilder validation failed: {'; '.join(errors)}")
        
        return super().build()