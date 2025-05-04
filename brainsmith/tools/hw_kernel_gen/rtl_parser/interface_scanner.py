# interface_scanner.py
"""Identifies potential interfaces from port lists."""

import re
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import logging # Added

# Assuming data structures are in these paths based on context
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Port, Direction
from brainsmith.tools.hw_kernel_gen.rtl_parser.interface_types import InterfaceType, PortGroup

logger = logging.getLogger(__name__) # Added

# Define known signal patterns based on RTL_Parser-Data-Analysis.md
GLOBAL_SIGNALS = {
    "ap_clk": {"direction": Direction.INPUT, "required": True},
    "ap_rst_n": {"direction": Direction.INPUT, "required": True},
    "ap_clk2x": {"direction": Direction.INPUT, "required": False},
}

# Suffixes for AXI-Stream. Direction depends on prefix (in/out) and will be validated later.
# The key used in PortGroup will be the suffix (e.g., _TDATA)
AXI_STREAM_SUFFIXES = {
    "_TDATA": {"required": True},
    "_TVALID": {"required": True},
    "_TREADY": {"required": True},
    "_TLAST": {"required": False},
}

# AXI-Lite signals. The key used in PortGroup will be the full signal name.
AXI_LITE_WRITE_SIGNALS = {
    "config_AWADDR": {"direction": Direction.INPUT, "required": True},
    "config_AWPROT": {"direction": Direction.INPUT, "required": True},
    "config_AWVALID": {"direction": Direction.INPUT, "required": True},
    "config_AWREADY": {"direction": Direction.OUTPUT, "required": True},
    "config_WDATA": {"direction": Direction.INPUT, "required": True},
    "config_WSTRB": {"direction": Direction.INPUT, "required": True},
    "config_WVALID": {"direction": Direction.INPUT, "required": True},
    "config_WREADY": {"direction": Direction.OUTPUT, "required": True},
    "config_BRESP": {"direction": Direction.OUTPUT, "required": True},
    "config_BVALID": {"direction": Direction.OUTPUT, "required": True},
    "config_BREADY": {"direction": Direction.INPUT, "required": True},
}

AXI_LITE_READ_SIGNALS = {
    "config_ARADDR": {"direction": Direction.INPUT, "required": True},
    "config_ARPROT": {"direction": Direction.INPUT, "required": True},
    "config_ARVALID": {"direction": Direction.INPUT, "required": True},
    "config_ARREADY": {"direction": Direction.OUTPUT, "required": True},
    "config_RDATA": {"direction": Direction.OUTPUT, "required": True},
    "config_RRESP": {"direction": Direction.OUTPUT, "required": True},
    "config_RVALID": {"direction": Direction.OUTPUT, "required": True},
    "config_RREADY": {"direction": Direction.INPUT, "required": True},
}

# Combined AXI-Lite for easier checking
AXI_LITE_SIGNALS = {**AXI_LITE_WRITE_SIGNALS, **AXI_LITE_READ_SIGNALS}

# Regex to capture AXI-Stream prefix and suffix, handling optional "_V_"
# Example: in0_V_TDATA -> prefix=in0, suffix=_TDATA
# Example: out1_TDATA -> prefix=out1, suffix=_TDATA
# Corrected Regex: Make the _V_ part truly optional and non-capturing
AXI_STREAM_SPLIT_REGEX = re.compile(r"^(?P<prefix>.*?)(?:_V)?(?P<suffix>_(?:TDATA|TVALID|TREADY|TLAST))$")


class InterfaceScanner:
    """Identifies potential interfaces from port lists."""

    def __init__(self, debug: bool = False):
        self.debug = debug # Placeholder for potential future debugging logs

    def _is_global_signal(self, port_name: str) -> bool:
        """Check if a port name matches a known global signal."""
        return port_name in GLOBAL_SIGNALS

    def _get_axi_stream_parts(self, port_name: str) -> Optional[Tuple[str, str]]:
        """Extract prefix and suffix from potential AXI-Stream port name."""
        match = AXI_STREAM_SPLIT_REGEX.match(port_name)
        if match:
            prefix = match.group("prefix")
            suffix = match.group("suffix")
            # Ensure the extracted suffix is one we care about (redundant check, but safe)
            if suffix in AXI_STREAM_SUFFIXES:
                # Added check: Ensure prefix is not empty
                if prefix:
                    return prefix, suffix
                else:
                    logger.debug(f"Port '{port_name}' matched AXI stream suffix but has empty prefix. Ignoring.")
        return None

    def _is_axi_lite_signal(self, port_name: str) -> bool:
        """Check if a port name matches a known AXI-Lite signal."""
        # Primarily check prefix, but also ensure it's a known full signal name
        return port_name.startswith("config_") and port_name in AXI_LITE_SIGNALS

    def scan(self, ports: List[Port]) -> Tuple[List[PortGroup], List[Port]]:
        """
        Scans a list of ports and groups them into potential interfaces.

        Args:
            ports: List of Port objects from the parsed module.

        Returns:
            A tuple containing:
            - List of identified PortGroup objects (potential interfaces).
            - List of remaining ports that didn't fit into any known interface type.
        """
        identified_groups: List[PortGroup] = []
        remaining_ports: List[Port] = []
        processed_ports = set() # Keep track of ports already assigned to a group

        # 1. Scan for Global Signals
        global_group = PortGroup(InterfaceType.GLOBAL_CONTROL, name="global")
        for port in ports:
            if self._is_global_signal(port.name):
                # Use the full signal name as the key within the group
                global_group.add_port(port, key=port.name)
                processed_ports.add(port.name)
        # Add group only if it contains any ports
        if global_group.ports:
            identified_groups.append(global_group)

        # 2. Scan for AXI-Lite Signals
        lite_group = PortGroup(InterfaceType.AXI_LITE, name="config")
        for port in ports:
            # Check if not already processed and is an AXI-Lite signal
            if port.name not in processed_ports and self._is_axi_lite_signal(port.name):
                 # Use the full signal name as the key within the group
                lite_group.add_port(port, key=port.name)
                processed_ports.add(port.name)
        # Add group only if it contains any ports
        if lite_group.ports:
            identified_groups.append(lite_group)

        # 3. Scan for AXI-Stream Signals
        # Group ports by potential AXI-Stream prefix
        stream_candidates = defaultdict(list)
        temp_processed_stream_ports = set() # Track ports matched by regex initially

        for port in ports:
            if port.name not in processed_ports:
                parts = self._get_axi_stream_parts(port.name)
                if parts:
                    prefix, _ = parts # We only need the prefix for grouping here
                    stream_candidates[prefix].append(port)
                    # Mark as potentially processed now
                    temp_processed_stream_ports.add(port.name)

        # Create PortGroup for each identified prefix
        for prefix, stream_ports in stream_candidates.items():
            stream_group = PortGroup(InterfaceType.AXI_STREAM, name=prefix)
            valid_group = False # Flag to check if any port was actually added
            for port in stream_ports:
                # Extract suffix again to use as the key within the group
                # Use the same reliable method _get_axi_stream_parts
                port_parts = self._get_axi_stream_parts(port.name)
                if port_parts:
                    _, suffix = port_parts
                    stream_group.add_port(port, key=suffix)
                    processed_ports.add(port.name) # Mark as fully processed ONLY if added to a group
                    valid_group = True
                # else: # This should ideally not happen if it was added to stream_candidates
                    # logger.warning(f"Internal inconsistency: Port '{port.name}' in candidate group '{prefix}' failed suffix extraction.")

            # Add group only if it contains any ports
            if valid_group:
                identified_groups.append(stream_group)
            # else: # Log if a candidate group ended up empty (shouldn't happen with current logic)
                # logger.warning(f"AXI-Stream candidate group '{prefix}' resulted in an empty final group.")


        # 4. Collect Remaining/Unknown Ports
        # Any port not processed belongs to an unknown interface or is standalone
        for port in ports:
            if port.name not in processed_ports:
                remaining_ports.append(port)
                # Optionally, create an 'UNKNOWN' PortGroup for these
                # unknown_group = PortGroup(InterfaceType.UNKNOWN, name=f"unknown_{port.name}")
                # unknown_group.add_port(port, key=port.name)
                # identified_groups.append(unknown_group)


        return identified_groups, remaining_ports
