"""
Analysis Patterns for Interface and Pragma Detection.

This module contains pattern definitions and utilities for detecting
different types of interfaces and pragmas in RTL code.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Optional, Set, Any
from enum import Enum


class InterfaceType(Enum):
    """Types of interfaces that can be detected."""
    AXI_STREAM = "axi_stream"
    AXI_LITE = "axi_lite"
    AXI_FULL = "axi_full"
    CONTROL = "control"
    CLOCK_RESET = "clock_reset"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class SignalRole(Enum):
    """Roles that signals can play in interfaces."""
    DATA = "data"
    VALID = "valid"
    READY = "ready"
    LAST = "last"
    KEEP = "keep"
    STRB = "strb"
    USER = "user"
    ID = "id"
    DEST = "dest"
    ADDR = "addr"
    PROT = "prot"
    RESP = "resp"
    CLOCK = "clock"
    RESET = "reset"
    ENABLE = "enable"
    INTERRUPT = "interrupt"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass
class SignalPattern:
    """Pattern for detecting signal roles."""
    role: SignalRole
    patterns: List[str] = field(default_factory=list)
    compiled_patterns: List[Pattern] = field(default_factory=list, init=False)
    required: bool = True
    direction: Optional[str] = None  # input, output, inout
    width_range: Optional[tuple] = None  # (min, max) width
    
    def __post_init__(self):
        """Compile regex patterns."""
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns]
    
    def matches(self, signal_name: str) -> bool:
        """Check if signal name matches any pattern."""
        return any(pattern.search(signal_name) for pattern in self.compiled_patterns)


@dataclass
class InterfacePattern:
    """Pattern for detecting interface types."""
    interface_type: InterfaceType
    signal_patterns: List[SignalPattern] = field(default_factory=list)
    prefix_patterns: List[str] = field(default_factory=list)
    suffix_patterns: List[str] = field(default_factory=list)
    required_signals: Set[SignalRole] = field(default_factory=set)
    optional_signals: Set[SignalRole] = field(default_factory=set)
    min_signals: int = 1
    
    def matches_prefix(self, interface_name: str) -> bool:
        """Check if interface name matches prefix patterns."""
        if not self.prefix_patterns:
            return True
        return any(interface_name.startswith(prefix) for prefix in self.prefix_patterns)
    
    def matches_suffix(self, interface_name: str) -> bool:
        """Check if interface name matches suffix patterns.""" 
        if not self.suffix_patterns:
            return True
        return any(interface_name.endswith(suffix) for suffix in self.suffix_patterns)


class PragmaType(Enum):
    """Types of pragmas that can be processed."""
    BRAINSMITH = "brainsmith"
    HLS = "hls"
    INTERFACE = "interface"
    PARALLELISM = "parallelism"
    DATAFLOW = "dataflow"
    CUSTOM = "custom"


@dataclass
class PragmaPattern:
    """Pattern for detecting and parsing pragmas."""
    pragma_type: PragmaType
    patterns: List[str] = field(default_factory=list)
    compiled_patterns: List[Pattern] = field(default_factory=list, init=False)
    parameter_patterns: Dict[str, str] = field(default_factory=dict)
    required_parameters: Set[str] = field(default_factory=set)
    optional_parameters: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Compile regex patterns."""
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns]
    
    def matches(self, pragma_text: str) -> bool:
        """Check if pragma text matches any pattern."""
        return any(pattern.search(pragma_text) for pattern in self.compiled_patterns)


# Default AXI-Stream patterns
AXI_STREAM_PATTERNS = [
    SignalPattern(SignalRole.DATA, [r".*_tdata$", r".*_TDATA$"], required=True, direction="input", width_range=(8, 1024)),
    SignalPattern(SignalRole.VALID, [r".*_tvalid$", r".*_TVALID$"], required=True, direction="input", width_range=(1, 1)),
    SignalPattern(SignalRole.READY, [r".*_tready$", r".*_TREADY$"], required=True, direction="output", width_range=(1, 1)),
    SignalPattern(SignalRole.LAST, [r".*_tlast$", r".*_TLAST$"], required=False, direction="input", width_range=(1, 1)),
    SignalPattern(SignalRole.KEEP, [r".*_tkeep$", r".*_TKEEP$"], required=False, direction="input"),
    SignalPattern(SignalRole.STRB, [r".*_tstrb$", r".*_TSTRB$"], required=False, direction="input"),
    SignalPattern(SignalRole.USER, [r".*_tuser$", r".*_TUSER$"], required=False, direction="input"),
    SignalPattern(SignalRole.ID, [r".*_tid$", r".*_TID$"], required=False, direction="input"),
    SignalPattern(SignalRole.DEST, [r".*_tdest$", r".*_TDEST$"], required=False, direction="input"),
]

# Default AXI-Lite patterns
AXI_LITE_PATTERNS = [
    # Write Address Channel
    SignalPattern(SignalRole.ADDR, [r".*_awaddr$", r".*_AWADDR$"], required=True, direction="input"),
    SignalPattern(SignalRole.VALID, [r".*_awvalid$", r".*_AWVALID$"], required=True, direction="input", width_range=(1, 1)),
    SignalPattern(SignalRole.READY, [r".*_awready$", r".*_AWREADY$"], required=True, direction="output", width_range=(1, 1)),
    SignalPattern(SignalRole.PROT, [r".*_awprot$", r".*_AWPROT$"], required=False, direction="input", width_range=(3, 3)),
    
    # Write Data Channel
    SignalPattern(SignalRole.DATA, [r".*_wdata$", r".*_WDATA$"], required=True, direction="input"),
    SignalPattern(SignalRole.STRB, [r".*_wstrb$", r".*_WSTRB$"], required=True, direction="input"),
    SignalPattern(SignalRole.VALID, [r".*_wvalid$", r".*_WVALID$"], required=True, direction="input", width_range=(1, 1)),
    SignalPattern(SignalRole.READY, [r".*_wready$", r".*_WREADY$"], required=True, direction="output", width_range=(1, 1)),
    
    # Write Response Channel
    SignalPattern(SignalRole.RESP, [r".*_bresp$", r".*_BRESP$"], required=True, direction="output", width_range=(2, 2)),
    SignalPattern(SignalRole.VALID, [r".*_bvalid$", r".*_BVALID$"], required=True, direction="output", width_range=(1, 1)),
    SignalPattern(SignalRole.READY, [r".*_bready$", r".*_BREADY$"], required=True, direction="input", width_range=(1, 1)),
    
    # Read Address Channel
    SignalPattern(SignalRole.ADDR, [r".*_araddr$", r".*_ARADDR$"], required=True, direction="input"),
    SignalPattern(SignalRole.VALID, [r".*_arvalid$", r".*_ARVALID$"], required=True, direction="input", width_range=(1, 1)),
    SignalPattern(SignalRole.READY, [r".*_arready$", r".*_ARREADY$"], required=True, direction="output", width_range=(1, 1)),
    SignalPattern(SignalRole.PROT, [r".*_arprot$", r".*_ARPROT$"], required=False, direction="input", width_range=(3, 3)),
    
    # Read Data Channel
    SignalPattern(SignalRole.DATA, [r".*_rdata$", r".*_RDATA$"], required=True, direction="output"),
    SignalPattern(SignalRole.RESP, [r".*_rresp$", r".*_RRESP$"], required=True, direction="output", width_range=(2, 2)),
    SignalPattern(SignalRole.VALID, [r".*_rvalid$", r".*_RVALID$"], required=True, direction="output", width_range=(1, 1)),
    SignalPattern(SignalRole.READY, [r".*_rready$", r".*_RREADY$"], required=True, direction="input", width_range=(1, 1)),
]

# Control signal patterns
CONTROL_PATTERNS = [
    SignalPattern(SignalRole.CLOCK, [r".*clk$", r".*_clk$", r"ap_clk$"], required=True, direction="input", width_range=(1, 1)),
    SignalPattern(SignalRole.RESET, [r".*rst$", r".*_rst$", r".*rst_n$", r".*_rst_n$", r"ap_rst_n$"], required=True, direction="input", width_range=(1, 1)),
    SignalPattern(SignalRole.ENABLE, [r".*_en$", r".*_enable$", r"ap_start$"], required=False, direction="input", width_range=(1, 1)),
    SignalPattern(SignalRole.INTERRUPT, [r".*_done$", r".*_ready$", r".*_idle$", r"ap_done$", r"ap_ready$", r"ap_idle$"], required=False, direction="output", width_range=(1, 1)),
]

# Interface detection patterns
INTERFACE_PATTERNS = [
    InterfacePattern(
        InterfaceType.AXI_STREAM,
        signal_patterns=AXI_STREAM_PATTERNS,
        prefix_patterns=["s_axis_", "m_axis_", "axis_"],
        required_signals={SignalRole.DATA, SignalRole.VALID, SignalRole.READY},
        optional_signals={SignalRole.LAST, SignalRole.KEEP, SignalRole.STRB, SignalRole.USER, SignalRole.ID, SignalRole.DEST},
        min_signals=3
    ),
    InterfacePattern(
        InterfaceType.AXI_LITE,
        signal_patterns=AXI_LITE_PATTERNS,
        prefix_patterns=["s_axi_", "s_axilite_", "axi_"],
        required_signals={SignalRole.ADDR, SignalRole.DATA, SignalRole.VALID, SignalRole.READY},
        optional_signals={SignalRole.PROT, SignalRole.STRB, SignalRole.RESP},
        min_signals=4
    ),
    InterfacePattern(
        InterfaceType.CONTROL,
        signal_patterns=CONTROL_PATTERNS,
        required_signals={SignalRole.CLOCK, SignalRole.RESET},
        optional_signals={SignalRole.ENABLE, SignalRole.INTERRUPT},
        min_signals=2
    ),
]

# Pragma patterns
PRAGMA_PATTERNS = [
    PragmaPattern(
        PragmaType.BRAINSMITH,
        patterns=[r"//\s*@brainsmith\s+(\w+)(.*)"],
        parameter_patterns={
            "interface": r"interface\s+(\w+)\s+(\w+)",
            "parallelism": r"parallelism\s+(\w+)=(\d+)",
            "dataflow": r"dataflow\s+(\w+)=(.+)"
        }
    ),
    PragmaPattern(
        PragmaType.HLS,
        patterns=[r"#pragma\s+HLS\s+(\w+)(.*)"],
        parameter_patterns={
            "interface": r"INTERFACE\s+(\w+)=(\w+)",
            "pipeline": r"PIPELINE\s+II=(\d+)",
            "array_partition": r"ARRAY_PARTITION\s+variable=(\w+)"
        }
    ),
    PragmaPattern(
        PragmaType.INTERFACE,
        patterns=[r"//\s*@interface\s+(\w+)(.*)"],
        parameter_patterns={
            "type": r"type=(\w+)",
            "direction": r"direction=(\w+)",
            "width": r"width=(\d+)"
        },
        required_parameters={"type"}
    ),
]


def get_interface_patterns() -> List[InterfacePattern]:
    """Get default interface detection patterns."""
    return INTERFACE_PATTERNS.copy()


def get_pragma_patterns() -> List[PragmaPattern]:
    """Get default pragma detection patterns."""
    return PRAGMA_PATTERNS.copy()


def create_custom_interface_pattern(
    interface_type: InterfaceType,
    signal_patterns: List[SignalPattern],
    prefix_patterns: List[str] = None,
    suffix_patterns: List[str] = None,
    required_signals: Set[SignalRole] = None,
    optional_signals: Set[SignalRole] = None,
    min_signals: int = 1
) -> InterfacePattern:
    """Create a custom interface pattern."""
    return InterfacePattern(
        interface_type=interface_type,
        signal_patterns=signal_patterns,
        prefix_patterns=prefix_patterns or [],
        suffix_patterns=suffix_patterns or [],
        required_signals=required_signals or set(),
        optional_signals=optional_signals or set(),
        min_signals=min_signals
    )


def create_custom_signal_pattern(
    role: SignalRole,
    patterns: List[str],
    required: bool = True,
    direction: str = None,
    width_range: tuple = None
) -> SignalPattern:
    """Create a custom signal pattern."""
    return SignalPattern(
        role=role,
        patterns=patterns,
        required=required,
        direction=direction,
        width_range=width_range
    )


def create_custom_pragma_pattern(
    pragma_type: PragmaType,
    patterns: List[str],
    parameter_patterns: Dict[str, str] = None,
    required_parameters: Set[str] = None,
    optional_parameters: Set[str] = None
) -> PragmaPattern:
    """Create a custom pragma pattern."""
    return PragmaPattern(
        pragma_type=pragma_type,
        patterns=patterns,
        parameter_patterns=parameter_patterns or {},
        required_parameters=required_parameters or set(),
        optional_parameters=optional_parameters or set()
    )