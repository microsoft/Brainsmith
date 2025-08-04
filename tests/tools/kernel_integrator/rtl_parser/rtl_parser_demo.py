#!/usr/bin/env python3
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
RTL Parser End-to-End Demo Application

This demo showcases the RTL parser's capabilities by taking any SystemVerilog
RTL file as input and generating rich, human-readable output of the parsed
KernelMetadata for inspection and validation.

Features:
- Parse any SystemVerilog RTL file
- Rich console output with visual hierarchy
- JSON and Markdown export options
- Performance metrics and diagnostics
- Comprehensive metadata inspection
- Error handling and user feedback

Usage:
    python rtl_parser_demo.py <rtl_file> [options]
    python rtl_parser_demo.py thresholding_axi.sv --format markdown --output demo_output.md
    python rtl_parser_demo.py mvu_vvu_axi.sv --format json --output metadata.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

# Add parent directories to path for brainsmith imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.tools.kernel_integrator.rtl_parser.parser import RTLParser, ParserError
from brainsmith.tools.kernel_integrator.rtl_parser.ast_parser import SyntaxError as RTLSyntaxError
from brainsmith.tools.kernel_integrator.metadata import KernelMetadata, InterfaceMetadata, DatatypeMetadata


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def colored(cls, text: str, color: str) -> str:
        """Wrap text with color codes."""
        return f"{color}{text}{cls.END}"


class RTLParserDemo:
    """Main demo application for RTL parser showcase."""
    
    def __init__(self, debug: bool = False, strict: bool = True):
        """Initialize demo with optional debug mode and strict validation."""
        self.debug = debug
        self.strict = strict
        self.parser = RTLParser(debug=debug, strict=strict)
        
    def parse_rtl_file(self, rtl_path: Path) -> Dict[str, Any]:
        """Parse RTL file and return comprehensive result data."""
        result = {
            'success': False,
            'parse_time_ms': 0,
            'error': None,
            'kernel_metadata': None,
            'file_info': {
                'path': str(rtl_path.absolute()),
                'name': rtl_path.name,
                'size_bytes': 0,
                'exists': rtl_path.exists()
            }
        }
        
        if not rtl_path.exists():
            result['error'] = f"File not found: {rtl_path}"
            return result
            
        try:
            # Get file info
            result['file_info']['size_bytes'] = rtl_path.stat().st_size
            
            # Parse with timing
            start_time = time.perf_counter()
            kernel_metadata = self.parser.parse_file(str(rtl_path))
            end_time = time.perf_counter()
            
            result['parse_time_ms'] = (end_time - start_time) * 1000
            result['kernel_metadata'] = kernel_metadata
            result['success'] = True
            
        except (ParserError, RTLSyntaxError, FileNotFoundError) as e:
            result['error'] = str(e)
        except Exception as e:
            result['error'] = f"Unexpected error: {e}"
            
        return result
    
    def format_console_output(self, result: Dict[str, Any]) -> str:
        """Generate rich console output with colors and hierarchy."""
        output = []
        c = Colors
        
        # Header
        output.append(c.colored("=" * 80, c.HEADER))
        output.append(c.colored("RTL PARSER DEMO - COMPREHENSIVE METADATA INSPECTION", c.HEADER + c.BOLD))
        output.append(c.colored("=" * 80, c.HEADER))
        output.append("")
        
        # File Information
        output.append(c.colored("📁 FILE INFORMATION", c.BLUE + c.BOLD))
        output.append(c.colored("-" * 40, c.BLUE))
        file_info = result['file_info']
        output.append(f"📄 File Path: {file_info['path']}")
        output.append(f"📏 File Size: {file_info['size_bytes']:,} bytes")
        output.append(f"⏱️  Parse Time: {result['parse_time_ms']:.2f}ms")
        output.append(f"✅ Parse Success: {result['success']}")
        output.append("")
        
        if not result['success']:
            output.append(c.colored("❌ PARSING FAILED", c.RED + c.BOLD))
            output.append(c.colored("-" * 40, c.RED))
            output.append(f"Error: {result['error']}")
            return "\n".join(output)
            
        metadata = result['kernel_metadata']
        
        # Module Overview
        output.append(c.colored("🏗️  MODULE OVERVIEW", c.GREEN + c.BOLD))
        output.append(c.colored("-" * 40, c.GREEN))
        output.append(f"Module Name: {metadata.name}")
        output.append(f"Class Name: {metadata.get_class_name()}")
        output.append(f"Source File: {metadata.source_file}")
        output.append(f"Total Parameters: {len(metadata.parameters)}")
        output.append(f"Exposed Parameters: {len(metadata.exposed_parameters)}")
        output.append(f"Total Interfaces: {len(metadata.interfaces)}")
        output.append(f"Total Pragmas: {len(metadata.pragmas)}")
        output.append("")
        
        # Parameters Section
        output.append(c.colored("⚙️  PARAMETERS", c.CYAN + c.BOLD))
        output.append(c.colored("-" * 40, c.CYAN))
        
        if metadata.parameters:
            output.append(f"All Parameters ({len(metadata.parameters)}):")
            for param in metadata.parameters:
                param_info = f"  • {param.name}"
                if hasattr(param, 'default_value') and param.default_value:
                    param_info += f" = {param.default_value}"
                if hasattr(param, 'param_type') and param.param_type:
                    param_info += f" [{param.param_type}]"
                output.append(param_info)
            output.append("")
            
            output.append(f"Exposed Parameters ({len(metadata.exposed_parameters)}):")
            for exp_param in metadata.exposed_parameters:
                output.append(f"  • {exp_param}")
            output.append("")
            
            # Linked parameters
            if metadata.linked_parameters:
                output.append("Linked Parameters:")
                if metadata.linked_parameters.get('aliases'):
                    output.append("  Aliases:")
                    for rtl_name, nodeattr_name in metadata.linked_parameters['aliases'].items():
                        output.append(f"    {rtl_name} → {nodeattr_name}")
                if metadata.linked_parameters.get('derived'):
                    output.append("  Derived:")
                    for param_name, expression in metadata.linked_parameters['derived'].items():
                        output.append(f"    {param_name} = {expression}")
                output.append("")
        else:
            output.append("  No parameters found")
            output.append("")
        
        # Interfaces Section
        output.append(c.colored("🔌 INTERFACES", c.YELLOW + c.BOLD))
        output.append(c.colored("-" * 40, c.YELLOW))
        
        if metadata.interfaces:
            for i, interface in enumerate(metadata.interfaces, 1):
                output.append(f"{i}. {interface.name} ({interface.interface_type.value})")
                
                if interface.compiler_name:
                    output.append(f"   Compiler Name: {interface.compiler_name}")
                    
                if interface.datatype_constraints:
                    constraint_desc = interface.get_constraint_description()
                    output.append(f"   Datatype Constraints: {constraint_desc}")
                
                if interface.datatype_metadata:
                    dt = interface.datatype_metadata
                    output.append(f"   Datatype Metadata: {dt.name}")
                    if dt.width:
                        output.append(f"     Width Parameter: {dt.width}")
                    if dt.signed:
                        output.append(f"     Signed Parameter: {dt.signed}")
                    if dt.format:
                        output.append(f"     Format Parameter: {dt.format}")
                
                # Shape parameters - handle both single and indexed
                if interface.bdim_params:
                    output.append(f"   BDIM Parameters (indexed): {interface.bdim_params}")
                elif interface.bdim_param:
                    output.append(f"   BDIM Parameter: {interface.bdim_param}")
                else:
                    output.append(f"   BDIM Parameter (default): {interface.get_bdim_parameter_name()}")
                
                if interface.sdim_params:
                    output.append(f"   SDIM Parameters (indexed): {interface.sdim_params}")
                elif interface.sdim_param:
                    output.append(f"   SDIM Parameter: {interface.sdim_param}")
                else:
                    output.append(f"   SDIM Parameter (default): {interface.get_sdim_parameter_name()}")
                
                if interface.shape_params:
                    shape_info = interface.shape_params
                    if 'shape' in shape_info:
                        output.append(f"   Block Shape: {shape_info['shape']}")
                    if 'rindex' in shape_info:
                        output.append(f"   R-Index: {shape_info['rindex']}")
                
                if interface.description:
                    output.append(f"   Description: {interface.description}")
                    
                output.append("")
        else:
            output.append("  No interfaces found")
            output.append("")
        
        # Pragmas Section
        output.append(c.colored("📋 PRAGMAS", c.HEADER + c.BOLD))
        output.append(c.colored("-" * 40, c.HEADER))
        
        if metadata.pragmas:
            pragma_types = {}
            for pragma in metadata.pragmas:
                pragma_type = pragma.type.value
                if pragma_type not in pragma_types:
                    pragma_types[pragma_type] = []
                pragma_types[pragma_type].append(pragma)
            
            for pragma_type, pragmas in pragma_types.items():
                output.append(f"{pragma_type} Pragmas ({len(pragmas)}):")
                for pragma in pragmas:
                    output.append(f"  Line {pragma.line_number}: {str(pragma)}")
                output.append("")
        else:
            output.append("  No pragmas found")
            output.append("")
        
        # Internal Datatypes
        if metadata.internal_datatypes:
            output.append(c.colored("🔧 INTERNAL DATATYPES", c.CYAN + c.BOLD))
            output.append(c.colored("-" * 40, c.CYAN))
            for dt in metadata.internal_datatypes:
                output.append(f"• {dt.name}")
                params = dt.get_all_parameters()
                if params:
                    output.append(f"  Parameters: {', '.join(params)}")
            output.append("")
        
        # Relationships
        if metadata.relationships:
            output.append(c.colored("🔗 RELATIONSHIPS", c.GREEN + c.BOLD))
            output.append(c.colored("-" * 40, c.GREEN))
            for rel in metadata.relationships:
                output.append(f"• {rel.source_interface} → {rel.target_interface}")
                output.append(f"  Type: {rel.relationship_type}")
                if rel.dependency_type:
                    output.append(f"  Dependency: {rel.dependency_type}")
                if rel.scale_factor:
                    output.append(f"  Scale Factor: {rel.scale_factor}")
            output.append("")
        
        # Warnings and Diagnostics
        if metadata.parsing_warnings:
            output.append(c.colored("⚠️  PARSING WARNINGS", c.YELLOW + c.BOLD))
            output.append(c.colored("-" * 40, c.YELLOW))
            for warning in metadata.parsing_warnings:
                output.append(f"• {warning}")
            output.append("")
        
        # Summary Statistics
        output.append(c.colored("📊 SUMMARY STATISTICS", c.BLUE + c.BOLD))
        output.append(c.colored("-" * 40, c.BLUE))
        output.append(f"✅ Parse Time: {result['parse_time_ms']:.2f}ms")
        output.append(f"📊 Total Components: {len(metadata.parameters) + len(metadata.interfaces)}")
        output.append(f"🏷️  Total Pragmas: {len(metadata.pragmas)}")
        output.append(f"⚙️  Parameters: {len(metadata.parameters)} total, {len(metadata.exposed_parameters)} exposed")
        output.append(f"🔌 Interfaces: {len(metadata.interfaces)} total")
        
        # Interface type breakdown
        interface_types = {}
        for interface in metadata.interfaces:
            itype = interface.interface_type.value
            interface_types[itype] = interface_types.get(itype, 0) + 1
        if interface_types:
            output.append("   Interface Types:")
            for itype, count in interface_types.items():
                output.append(f"     {itype}: {count}")
        
        output.append("")
        
        # Final Kernel Metadata Summary
        output.append(c.colored("🎯 COMPREHENSIVE KERNEL METADATA DISPLAY", c.GREEN + c.BOLD))
        output.append(c.colored("=" * 80, c.GREEN))
        output.append("")
        
        # Core Information
        output.append("┌─ Core Information ───────────────────────────────────────────")
        output.append(f"│ Module Name: {metadata.name}")
        output.append(f"│ Class Name: {metadata.get_class_name()}")
        output.append(f"│ Source File: {metadata.source_file}")
        output.append("└──────────────────────────────────────────────────────────────")
        output.append("")
        
        # All Parameters Section
        output.append("┌─ All RTL Parameters ─────────────────────────────────────────")
        if metadata.parameters:
            output.append("│ Name                 │ Type               │ Default Value")
            output.append("├──────────────────────┼────────────────────┼─────────────────")
            for param in metadata.parameters:
                name = param.name
                param_type = str(getattr(param, 'param_type', 'N/A'))
                default_val = str(getattr(param, 'default_value', 'N/A'))
                output.append(f"│ {name:<20} │ {param_type:<18} │ {default_val}")
        else:
            output.append("│ No parameters found")
        output.append("└──────────────────────┴────────────────────┴─────────────────")
        output.append("")
        
        # Exposed Parameters Section
        output.append("┌─ Exposed Parameters ─────────────────────────────────────────")
        if metadata.exposed_parameters:
            for param in metadata.exposed_parameters:
                # Check if this is an alias
                alias_name = None
                if metadata.linked_parameters and metadata.linked_parameters.get('aliases'):
                    for rtl, alias in metadata.linked_parameters['aliases'].items():
                        if rtl == param:
                            alias_name = alias
                            break
                
                if alias_name:
                    output.append(f"│ {param} → {alias_name}")
                else:
                    output.append(f"│ {param}")
        else:
            output.append("│ No exposed parameters")
        output.append("└──────────────────────────────────────────────────────────────")
        output.append("")
        
        # Linked Parameters Section
        output.append("┌─ Linked Parameters ──────────────────────────────────────────")
        if metadata.linked_parameters:
            has_content = False
            
            if metadata.linked_parameters.get('aliases'):
                output.append("│ Aliases:")
                for rtl_name, alias_name in metadata.linked_parameters['aliases'].items():
                    output.append(f"│   {rtl_name} → {alias_name}")
                has_content = True
                
            if metadata.linked_parameters.get('derived'):
                if has_content:
                    output.append("│")
                output.append("│ Derived Parameters:")
                for param_name, expression in metadata.linked_parameters['derived'].items():
                    output.append(f"│   {param_name} = {expression}")
                has_content = True
                
            if metadata.linked_parameters.get('axilite'):
                if has_content:
                    output.append("│")
                output.append("│ AXI-Lite Parameters:")
                for param_name, interface_name in metadata.linked_parameters['axilite'].items():
                    output.append(f"│   {param_name} → {interface_name}")
                has_content = True
                
            if not has_content:
                output.append("│ No linked parameters")
        else:
            output.append("│ No linked parameters")
        output.append("└──────────────────────────────────────────────────────────────")
        output.append("")
        
        # Interface Overview Table (keep closed for table format)
        output.append("┌─ Interface Overview ─────────────────────────────────────────┐")
        if metadata.interfaces:
            output.append("│ Name              │ Type    │ Datatype │ BDIM │ SDIM      │")
            output.append("├───────────────────┼─────────┼──────────┼──────┼───────────┤")
            
            for interface in metadata.interfaces:
                name = interface.name[:17] + "..." if len(interface.name) > 20 else interface.name
                itype = interface.interface_type.value.upper()[:7]
                
                # Datatype info
                if interface.datatype_metadata and interface.datatype_metadata.width:
                    dtype = str(interface.datatype_metadata.width)
                    if len(dtype) > 8:
                        dtype = dtype[:6] + ".."
                elif interface.datatype_constraints:
                    constraint = interface.datatype_constraints[0]
                    dtype = constraint.base_type.value
                    if len(dtype) > 8:
                        dtype = dtype[:6] + ".."
                else:
                    dtype = "AUTO"
                
                # BDIM info
                if interface.interface_type.value in ['input', 'output', 'weight']:
                    if interface.bdim_params:
                        bdim = f"{len(interface.bdim_params)}D"
                    elif interface.bdim_param and interface.bdim_param != f"{interface.name}_BDIM":
                        bdim = "✓"
                    else:
                        bdim = "-"
                else:
                    bdim = "N/A"
                
                # SDIM info
                if interface.interface_type.value in ['input', 'weight']:
                    if interface.sdim_params:
                        sdim = f"{len(interface.sdim_params)}D"
                    elif interface.sdim_param and interface.sdim_param != f"{interface.name}_SDIM":
                        sdim = "✓"
                    else:
                        sdim = "-"
                else:
                    sdim = "N/A"
                
                output.append(f"│ {name:<17} │ {itype:<7} │ {dtype:<8} │ {bdim:<4} │ {sdim:<9} │")
        else:
            output.append("│ No interfaces found                                          │")
        output.append("└───────────────────┴─────────┴──────────┴──────┴───────────┘")
        output.append("")
        
        # Detailed Interface Information
        if metadata.interfaces:
            output.append("┌─ Interface Details ──────────────────────────────────────────")
            for i, interface in enumerate(metadata.interfaces):
                if i > 0:
                    output.append("│")
                
                # Interface header
                header = f"{interface.name} ({interface.interface_type.value.upper()})"
                output.append(f"│ {header}")
                
                # Compiler name
                compiler_name = interface.compiler_name or 'N/A'
                output.append(f"│   Compiler Name: {compiler_name}")
                
                # Datatype information
                if interface.datatype_metadata:
                    dt = interface.datatype_metadata
                    output.append(f"│   Datatype Metadata: {dt.name}")
                    if dt.width:
                        output.append(f"│     Width: {dt.width}")
                    if dt.signed is not None:
                        output.append(f"│     Signed: {dt.signed}")
                    if dt.format:
                        output.append(f"│     Format: {dt.format}")
                elif interface.datatype_constraints:
                    constraint_desc = interface.get_constraint_description()
                    output.append(f"│   Datatype Constraints: {constraint_desc}")
                else:
                    output.append("│   Datatype: AUTO")
                
                # Dimension parameters
                if interface.bdim_params:
                    output.append(f"│   BDIM (indexed): {interface.bdim_params}")
                elif interface.bdim_param:
                    output.append(f"│   BDIM: {interface.bdim_param}")
                
                if interface.sdim_params:
                    output.append(f"│   SDIM (indexed): {interface.sdim_params}")
                elif interface.sdim_param:
                    output.append(f"│   SDIM: {interface.sdim_param}")
                
                # Shape parameters
                if interface.shape_params:
                    shape_info = interface.shape_params
                    if 'shape' in shape_info:
                        output.append(f"│   Shape: {shape_info['shape']}")
                    if 'rindex' in shape_info:
                        output.append(f"│   R-Index: {shape_info['rindex']}")
                
                # Description
                if interface.description:
                    output.append(f"│   Description: {interface.description}")
                    
            output.append("└──────────────────────────────────────────────────────────────")
            output.append("")
        
        # Internal Datatypes Section
        output.append("┌─ Internal Datatypes ─────────────────────────────────────────")
        if metadata.internal_datatypes:
            for i, dt in enumerate(metadata.internal_datatypes):
                if i > 0:
                    output.append("│")
                
                # Datatype name header
                output.append(f"│ {dt.name}:")
                
                # Parameters
                params = dt.get_all_parameters() if hasattr(dt, 'get_all_parameters') else []
                if params:
                    params_str = ', '.join(params)
                    output.append(f"│   Parameters: {params_str}")
                
                # Description
                if dt.description:
                    output.append(f"│   Description: {dt.description}")
        else:
            output.append("│ No internal datatypes")
        output.append("└──────────────────────────────────────────────────────────────")
        output.append("")
        
        # Relationships Section
        output.append("┌─ Interface Relationships ────────────────────────────────────")
        if metadata.relationships:
            for i, rel in enumerate(metadata.relationships):
                if i > 0:
                    output.append("│")
                output.append(f"│ {rel.source_interface} → {rel.target_interface}:")
                output.append(f"│   Type: {rel.relationship_type}")
                if rel.dependency_type:
                    output.append(f"│   Dependency: {rel.dependency_type}")
                if rel.scale_factor:
                    output.append(f"│   Scale Factor: {rel.scale_factor}")
        else:
            output.append("│ No interface relationships")
        output.append("└──────────────────────────────────────────────────────────────")
        output.append("")
        
        # Parsing Warnings Section
        output.append("┌─ Parsing Warnings ───────────────────────────────────────────")
        if metadata.parsing_warnings:
            for warning in metadata.parsing_warnings:
                output.append(f"│ • {warning}")
        else:
            output.append("│ No parsing warnings")
        output.append("└──────────────────────────────────────────────────────────────")
        output.append("")
        
        # Pragma Summary (keep table format)
        output.append("┌─ Pragma Summary ─────────────────────────────────────────────┐")
        if metadata.pragmas:
            pragma_counts = {}
            for pragma in metadata.pragmas:
                ptype = pragma.type.value
                pragma_counts[ptype] = pragma_counts.get(ptype, 0) + 1
            
            output.append("│ Type                           │ Count                       │")
            output.append("├────────────────────────────────┼─────────────────────────────┤")
            for ptype, count in sorted(pragma_counts.items()):
                output.append(f"│ {ptype:<30} │ {count:>27} │")
        else:
            output.append("│ No pragmas found                                             │")
        output.append("└────────────────────────────────┴─────────────────────────────┘")
        output.append("")
        
        # Key Features
        features = []
        if any(iface.interface_type.value == 'weight' for iface in metadata.interfaces):
            features.append("✓ Weight Interfaces")
        if metadata.linked_parameters and metadata.linked_parameters.get('aliases'):
            features.append("✓ Parameter Aliases")
        if metadata.linked_parameters and metadata.linked_parameters.get('derived'):
            features.append("✓ Derived Parameters")
        if any(hasattr(iface, 'bdim_params') and iface.bdim_params for iface in metadata.interfaces):
            features.append("✓ Multi-dimensional BDIM")
        if any(hasattr(iface, 'sdim_params') and iface.sdim_params for iface in metadata.interfaces):
            features.append("✓ Multi-dimensional SDIM")
        if metadata.internal_datatypes:
            features.append("✓ Internal Datatypes")
        
        if features:
            output.append("┌─ Key Features ───────────────────────────────────────────────")
            for feature in features:
                output.append(f"│ {feature}")
            output.append("└──────────────────────────────────────────────────────────────")
        
        output.append("")
        output.append(c.colored("=" * 80, c.HEADER))
        output.append(c.colored("DEMO COMPLETE - RTL PARSER METADATA INSPECTION FINISHED", c.HEADER + c.BOLD))
        output.append(c.colored("=" * 80, c.HEADER))
        
        return "\n".join(output)
    
    def format_json_output(self, result: Dict[str, Any]) -> str:
        """Generate JSON output for machine-readable inspection."""
        json_data = {
            'demo_info': {
                'success': result['success'],
                'parse_time_ms': result['parse_time_ms'],
                'file_info': result['file_info']
            }
        }
        
        if not result['success']:
            json_data['error'] = result['error']
            return json.dumps(json_data, indent=2)
        
        metadata = result['kernel_metadata']
        
        # Convert KernelMetadata to JSON-serializable format
        json_data['kernel_metadata'] = {
            'name': metadata.name,
            'class_name': metadata.get_class_name(),
            'source_file': str(metadata.source_file),
            'parameters': [
                {
                    'name': p.name,
                    'param_type': getattr(p, 'param_type', None),
                    'default_value': getattr(p, 'default_value', None),
                    'description': getattr(p, 'description', None)
                } for p in metadata.parameters
            ],
            'exposed_parameters': metadata.exposed_parameters,
            'interfaces': [
                {
                    'name': iface.name,
                    'interface_type': iface.interface_type.value,
                    'compiler_name': iface.compiler_name,
                    'datatype_constraints': [
                        {
                            'base_type': constraint.base_type.value,
                            'min_width': constraint.min_width,
                            'max_width': constraint.max_width
                        } for constraint in iface.datatype_constraints
                    ],
                    'datatype_metadata': {
                        'name': iface.datatype_metadata.name,
                        'width': iface.datatype_metadata.width,
                        'signed': iface.datatype_metadata.signed,
                        'format': iface.datatype_metadata.format,
                        'bias': iface.datatype_metadata.bias
                    } if iface.datatype_metadata else None,
                    'bdim_parameter': iface.bdim_param if iface.bdim_param else None,
                    'bdim_parameters': iface.bdim_params if hasattr(iface, 'bdim_params') and iface.bdim_params else None,
                    'sdim_parameter': iface.sdim_param if iface.sdim_param else None,
                    'sdim_parameters': iface.sdim_params if hasattr(iface, 'sdim_params') and iface.sdim_params else None,
                    'shape_params': iface.shape_params,
                    'description': iface.description
                } for iface in metadata.interfaces
            ],
            'pragmas': [
                {
                    'type': pragma.type.value,
                    'line_number': pragma.line_number,
                    'text': str(pragma),
                    'inputs': pragma.inputs
                } for pragma in metadata.pragmas
            ],
            'linked_parameters': metadata.linked_parameters,
            'internal_datatypes': [
                {
                    'name': dt.name,
                    'parameters': dt.get_all_parameters(),
                    'description': dt.description
                } for dt in metadata.internal_datatypes
            ],
            'relationships': [
                {
                    'source_interface': rel.source_interface,
                    'target_interface': rel.target_interface,
                    'relationship_type': rel.relationship_type,
                    'dependency_type': rel.dependency_type,
                    'scale_factor': rel.scale_factor
                } for rel in metadata.relationships
            ],
            'parsing_warnings': metadata.parsing_warnings
        }
        
        return json.dumps(json_data, indent=2)
    
    def format_markdown_output(self, result: Dict[str, Any]) -> str:
        """Generate Markdown output for human-readable documentation."""
        output = []
        
        # Header
        output.append("# RTL Parser Demo - Metadata Inspection Report")
        output.append("")
        output.append("## File Information")
        output.append("")
        file_info = result['file_info']
        output.append(f"- **File Path**: `{file_info['path']}`")
        output.append(f"- **File Size**: {file_info['size_bytes']:,} bytes")
        output.append(f"- **Parse Time**: {result['parse_time_ms']:.2f}ms")
        output.append(f"- **Parse Success**: {'✅ Yes' if result['success'] else '❌ No'}")
        output.append("")
        
        if not result['success']:
            output.append("## ❌ Parsing Failed")
            output.append("")
            output.append(f"**Error**: {result['error']}")
            return "\n".join(output)
            
        metadata = result['kernel_metadata']
        
        # Module Overview
        output.append("## 🏗️ Module Overview")
        output.append("")
        output.append(f"- **Module Name**: `{metadata.name}`")
        output.append(f"- **Generated Class Name**: `{metadata.get_class_name()}`")
        output.append(f"- **Source File**: `{metadata.source_file}`")
        output.append(f"- **Total Parameters**: {len(metadata.parameters)}")
        output.append(f"- **Exposed Parameters**: {len(metadata.exposed_parameters)}")
        output.append(f"- **Total Interfaces**: {len(metadata.interfaces)}")
        output.append(f"- **Total Pragmas**: {len(metadata.pragmas)}")
        output.append("")
        
        # Parameters
        output.append("## ⚙️ Parameters")
        output.append("")
        
        if metadata.parameters:
            output.append("### All Parameters")
            output.append("")
            output.append("| Name | Type | Default Value |")
            output.append("|------|------|---------------|")
            for param in metadata.parameters:
                param_type = getattr(param, 'param_type', 'N/A')
                default_val = getattr(param, 'default_value', 'N/A')
                output.append(f"| `{param.name}` | {param_type} | `{default_val}` |")
            output.append("")
            
            output.append("### Exposed Parameters")
            output.append("")
            if metadata.exposed_parameters:
                for exp_param in metadata.exposed_parameters:
                    output.append(f"- `{exp_param}`")
            else:
                output.append("*No exposed parameters*")
            output.append("")
        
        # Interfaces
        output.append("## 🔌 Interfaces")
        output.append("")
        
        if metadata.interfaces:
            for interface in metadata.interfaces:
                output.append(f"### {interface.name} ({interface.interface_type.value})")
                output.append("")
                
                if interface.compiler_name:
                    output.append(f"- **Compiler Name**: `{interface.compiler_name}`")
                    
                if interface.datatype_constraints:
                    constraint_desc = interface.get_constraint_description()
                    output.append(f"- **Datatype Constraints**: {constraint_desc}")
                
                if interface.datatype_metadata:
                    dt = interface.datatype_metadata
                    output.append(f"- **Datatype Metadata**: `{dt.name}`")
                    if dt.width:
                        output.append(f"  - Width Parameter: `{dt.width}`")
                    if dt.signed:
                        output.append(f"  - Signed Parameter: `{dt.signed}`")
                    if dt.format:
                        output.append(f"  - Format Parameter: `{dt.format}`")
                
                # Handle both single and indexed parameters
                if interface.bdim_params:
                    output.append(f"- **BDIM Parameters (indexed)**: `{interface.bdim_params}`")
                elif interface.bdim_param:
                    output.append(f"- **BDIM Parameter**: `{interface.bdim_param}`")
                else:
                    output.append(f"- **BDIM Parameter (default)**: `{interface.get_bdim_parameter_name()}`")
                
                if interface.sdim_params:
                    output.append(f"- **SDIM Parameters (indexed)**: `{interface.sdim_params}`")
                elif interface.sdim_param:
                    output.append(f"- **SDIM Parameter**: `{interface.sdim_param}`")
                else:
                    output.append(f"- **SDIM Parameter (default)**: `{interface.get_sdim_parameter_name()}`")
                
                if interface.shape_params:
                    shape_info = interface.shape_params
                    if 'shape' in shape_info:
                        output.append(f"- **Block Shape**: `{shape_info['shape']}`")
                    if 'rindex' in shape_info:
                        output.append(f"- **R-Index**: {shape_info['rindex']}")
                
                if interface.description:
                    output.append(f"- **Description**: {interface.description}")
                
                output.append("")
        
        # Pragmas
        output.append("## 📋 Pragmas")
        output.append("")
        
        if metadata.pragmas:
            pragma_types = {}
            for pragma in metadata.pragmas:
                pragma_type = pragma.type.value
                if pragma_type not in pragma_types:
                    pragma_types[pragma_type] = []
                pragma_types[pragma_type].append(pragma)
            
            for pragma_type, pragmas in pragma_types.items():
                output.append(f"### {pragma_type} Pragmas ({len(pragmas)})")
                output.append("")
                for pragma in pragmas:
                    output.append(f"- **Line {pragma.line_number}**: `{str(pragma)}`")
                output.append("")
        
        # Summary
        output.append("## 📊 Summary Statistics")
        output.append("")
        output.append(f"- **Parse Time**: {result['parse_time_ms']:.2f}ms")
        output.append(f"- **Total Components**: {len(metadata.parameters) + len(metadata.interfaces)}")
        output.append(f"- **Total Pragmas**: {len(metadata.pragmas)}")
        
        # Interface type breakdown
        interface_types = {}
        for interface in metadata.interfaces:
            itype = interface.interface_type.value
            interface_types[itype] = interface_types.get(itype, 0) + 1
        
        if interface_types:
            output.append("")
            output.append("### Interface Type Breakdown")
            output.append("")
            for itype, count in interface_types.items():
                output.append(f"- **{itype}**: {count}")
        
        output.append("")
        output.append("---")
        output.append("*Generated by RTL Parser Demo*")
        
        return "\n".join(output)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="RTL Parser Demo - Comprehensive metadata inspection for SystemVerilog files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rtl_parser_demo.py thresholding_axi.sv
  python rtl_parser_demo.py mvu_vvu_axi.sv --format json --output metadata.json
  python rtl_parser_demo.py ../kernels/*//*.sv --format markdown --output report.md
  python rtl_parser_demo.py kernel.sv --debug --format all

Supported output formats:
  console  - Rich terminal output with colors (default)
  json     - Machine-readable JSON format
  markdown - Human-readable Markdown documentation
  all      - Generate all formats (requires --output prefix)
        """
    )
    
    parser.add_argument('rtl_file', type=Path, 
                       help='SystemVerilog RTL file to parse and inspect')
    
    parser.add_argument('--format', choices=['console', 'json', 'markdown', 'all'], 
                       default='console',
                       help='Output format (default: console)')
    
    parser.add_argument('--output', type=str,
                       help='Output file path (for json/markdown) or prefix (for all)')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed parser logging')
    
    parser.add_argument('--no-color', action='store_true',
                       help='Disable color output for console format')
    
    parser.add_argument('--no-strict', action='store_true',
                       help='Disable strict validation (allows parsing files that don\'t meet all requirements)')
    
    return parser


def main():
    """Main entry point for RTL parser demo."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate input file
    if not args.rtl_file.exists():
        print(f"❌ Error: RTL file not found: {args.rtl_file}")
        return 1
    
    # Disable colors if requested
    if args.no_color:
        Colors.HEADER = Colors.BLUE = Colors.CYAN = Colors.GREEN = ''
        Colors.YELLOW = Colors.RED = Colors.BOLD = Colors.UNDERLINE = Colors.END = ''
    
    # Initialize demo
    demo = RTLParserDemo(debug=args.debug, strict=not args.no_strict)
    
    print(f"🔍 RTL Parser Demo - Parsing: {args.rtl_file.name}")
    if args.debug:
        print(f"📍 Full path: {args.rtl_file.absolute()}")
    print()
    
    # Parse RTL file
    result = demo.parse_rtl_file(args.rtl_file)
    
    # Generate output based on format
    if args.format == 'console' or args.format == 'all':
        console_output = demo.format_console_output(result)
        print(console_output)
        
        if args.format == 'console' and args.output:
            # Save console output to file
            with open(args.output, 'w') as f:
                # Strip ANSI color codes for file output
                import re
                clean_output = re.sub(r'\x1b\[[0-9;]*m', '', console_output)
                f.write(clean_output)
            print(f"📁 Console output saved to: {args.output}")
    
    if args.format == 'json' or args.format == 'all':
        json_output = demo.format_json_output(result)
        
        if args.output:
            output_path = args.output if args.format == 'json' else f"{args.output}.json"
            with open(output_path, 'w') as f:
                f.write(json_output)
            print(f"📁 JSON output saved to: {output_path}")
        else:
            print("\n" + "="*80)
            print("JSON OUTPUT:")
            print("="*80)
            print(json_output)
    
    if args.format == 'markdown' or args.format == 'all':
        markdown_output = demo.format_markdown_output(result)
        
        if args.output:
            output_path = args.output if args.format == 'markdown' else f"{args.output}.md"
            with open(output_path, 'w') as f:
                f.write(markdown_output)
            print(f"📁 Markdown output saved to: {output_path}")
        else:
            print("\n" + "="*80)
            print("MARKDOWN OUTPUT:")
            print("="*80)
            print(markdown_output)
    
    if not result['success']:
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())