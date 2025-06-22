############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Kernel Definition Factory

Factory for creating kernel definitions from various sources including
RTL files, specifications, and existing metadata.
"""

from typing import Dict, Any, List, Optional
import json
import yaml
from pathlib import Path

from brainsmith.core.dataflow import Shape
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata

from .kernel_definition import (
    KernelDefinition, InterfaceDefinition, ProtocolType,
    DatatypeConstraint, PerformanceModel, ResourceModel
)


class KernelDefinitionFactory:
    """
    Factory for creating kernel definitions from various sources.
    """
    
    @staticmethod
    def from_rtl_file(rtl_path: str, **kwargs) -> KernelDefinition:
        """
        Create kernel definition from RTL file using enhanced parser.
        
        Args:
            rtl_path: Path to RTL file
            **kwargs: Additional options for parser
            
        Returns:
            KernelDefinition instance
        """
        # Parse RTL file
        parser = RTLParser()
        kernel_metadata = parser.parse_file(rtl_path)
        
        # Convert to kernel definition
        return KernelDefinitionFactory.from_kernel_metadata(kernel_metadata)
    
    @staticmethod
    def from_kernel_metadata(kernel_metadata) -> KernelDefinition:
        """
        Create kernel definition from existing KernelMetadata.
        
        Args:
            kernel_metadata: KernelMetadata instance from RTL parser
            
        Returns:
            KernelDefinition instance
        """
        # Convert interfaces
        interfaces = []
        for intf_meta in kernel_metadata.interfaces:
            intf_def = KernelDefinitionFactory._convert_interface_metadata(intf_meta)
            interfaces.append(intf_def)
        
        # Extract constraints from pragmas
        constraints = []
        for pragma in kernel_metadata.pragmas:
            if hasattr(pragma, 'to_constraint'):
                constraints.append(pragma.to_constraint())
        
        # Create kernel definition
        kernel_def = KernelDefinition(
            name=kernel_metadata.kernel_name,
            version="1.0",
            description=f"Auto-generated from {kernel_metadata.kernel_name}",
            interfaces=interfaces,
            constraints=constraints,
            rtl_parameters=kernel_metadata.rtl_parameters,
            exposed_parameters=kernel_metadata.exposed_parameters,
            metadata={
                'original_metadata': kernel_metadata,
                'parameter_linkage': kernel_metadata.parameter_linkage
            }
        )
        
        # Add parameter defaults if available
        if hasattr(kernel_metadata, 'parameter_defaults'):
            kernel_def.parameter_defaults = kernel_metadata.parameter_defaults
        
        return kernel_def
    
    @staticmethod
    def _convert_interface_metadata(intf_meta: InterfaceMetadata) -> InterfaceDefinition:
        """Convert InterfaceMetadata to InterfaceDefinition."""
        # Convert datatype constraints
        datatype_constraints = []
        for constraint_group in intf_meta.datatype_constraint_groups:
            constraint = DatatypeConstraint(
                base_type=constraint_group.base_type,
                min_width=constraint_group.min_width,
                max_width=constraint_group.max_width
            )
            datatype_constraints.append(constraint)
        
        # Determine protocol from interface type
        protocol = ProtocolType.AXI_STREAM  # Default
        if intf_meta.interface_type == InterfaceType.CONFIG:
            protocol = ProtocolType.AXI_LITE
        
        # Create interface definition
        intf_def = InterfaceDefinition(
            name=intf_meta.name,
            type=intf_meta.interface_type,
            protocol=protocol,
            datatype_constraints=datatype_constraints,
            parameter_links={
                'BDIM': intf_meta.bdim_parameter,
                'SDIM': intf_meta.sdim_parameter,
                'datatype': intf_meta.datatype_parameter
            },
            metadata={
                'chunking_strategy': intf_meta.chunking_strategy,
                'compiler_name': intf_meta.compiler_name,
                'original_metadata': intf_meta
            }
        )
        
        return intf_def
    
    @staticmethod
    def from_specification(spec: Dict[str, Any]) -> KernelDefinition:
        """
        Create kernel definition from high-level specification.
        
        Args:
            spec: Dictionary specification with structure:
                {
                    'name': str,
                    'version': str (optional),
                    'description': str (optional),
                    'interfaces': [
                        {
                            'name': str,
                            'type': str,  # 'INPUT', 'OUTPUT', 'WEIGHT', etc.
                            'protocol': str (optional),
                            'datatype_constraints': [...] (optional)
                        }
                    ],
                    'constraints': [...] (optional),
                    'rtl_parameters': {...} (optional),
                    'exposed_parameters': [...] (optional),
                    'performance_model': {...} (optional),
                    'resource_model': {...} (optional)
                }
                
        Returns:
            KernelDefinition instance
        """
        # Create interfaces
        interfaces = []
        for intf_spec in spec.get('interfaces', []):
            intf_def = KernelDefinitionFactory._create_interface_from_spec(intf_spec)
            interfaces.append(intf_def)
        
        # Create kernel definition
        kernel_def = KernelDefinition(
            name=spec['name'],
            version=spec.get('version', '1.0'),
            description=spec.get('description', ''),
            interfaces=interfaces,
            constraints=spec.get('constraints', []),
            rtl_parameters=spec.get('rtl_parameters', {}),
            exposed_parameters=set(spec.get('exposed_parameters', [])),
            parameter_defaults=spec.get('parameter_defaults', {}),
            parameter_ranges=spec.get('parameter_ranges', {}),
            metadata=spec.get('metadata', {})
        )
        
        # Add performance model if specified
        if 'performance_model' in spec:
            kernel_def.performance_model = KernelDefinitionFactory._create_performance_model(
                spec['performance_model']
            )
        
        # Add resource model if specified
        if 'resource_model' in spec:
            kernel_def.resource_model = KernelDefinitionFactory._create_resource_model(
                spec['resource_model']
            )
        
        return kernel_def
    
    @staticmethod
    def _create_interface_from_spec(intf_spec: Dict[str, Any]) -> InterfaceDefinition:
        """Create InterfaceDefinition from specification."""
        # Parse interface type
        intf_type = InterfaceType[intf_spec['type']]
        
        # Parse protocol
        protocol = ProtocolType.AXI_STREAM  # Default
        if 'protocol' in intf_spec:
            protocol = ProtocolType[intf_spec['protocol']]
        
        # Parse datatype constraints
        datatype_constraints = []
        for constraint_spec in intf_spec.get('datatype_constraints', []):
            if isinstance(constraint_spec, dict):
                constraint = DatatypeConstraint(
                    base_type=constraint_spec['base_type'],
                    min_width=constraint_spec['min_width'],
                    max_width=constraint_spec['max_width']
                )
            else:
                # Simple format: "INT8-16" means INT with 8-16 bit width
                parts = constraint_spec.split('-')
                base_type = ''.join(c for c in parts[0] if c.isalpha())
                min_width = int(''.join(c for c in parts[0] if c.isdigit()))
                max_width = int(parts[1]) if len(parts) > 1 else min_width
                constraint = DatatypeConstraint(base_type, min_width, max_width)
            datatype_constraints.append(constraint)
        
        # Create interface definition
        intf_def = InterfaceDefinition(
            name=intf_spec['name'],
            type=intf_type,
            protocol=protocol,
            protocol_config=intf_spec.get('protocol_config', {}),
            datatype_constraints=datatype_constraints,
            parameter_links=intf_spec.get('parameter_links', {}),
            metadata=intf_spec.get('metadata', {})
        )
        
        return intf_def
    
    @staticmethod
    def _create_performance_model(perf_spec: Dict[str, Any]) -> PerformanceModel:
        """Create PerformanceModel from specification."""
        return PerformanceModel(
            base_latency=perf_spec.get('base_latency', 1),
            throughput_scaling=perf_spec.get('throughput_scaling', {}),
            pipeline_depth_formula=perf_spec.get('pipeline_depth_formula'),
            metadata=perf_spec.get('metadata', {})
        )
    
    @staticmethod
    def _create_resource_model(res_spec: Dict[str, Any]) -> ResourceModel:
        """Create ResourceModel from specification."""
        return ResourceModel(
            base_luts=res_spec.get('base_luts', 0),
            base_ffs=res_spec.get('base_ffs', 0),
            base_brams=res_spec.get('base_brams', 0),
            base_dsps=res_spec.get('base_dsps', 0),
            base_urams=res_spec.get('base_urams', 0),
            lut_scaling=res_spec.get('lut_scaling', {}),
            ff_scaling=res_spec.get('ff_scaling', {}),
            bram_scaling=res_spec.get('bram_scaling', {}),
            dsp_scaling=res_spec.get('dsp_scaling', {}),
            metadata=res_spec.get('metadata', {})
        )
    
    @staticmethod
    def from_yaml_file(yaml_path: str) -> KernelDefinition:
        """Load kernel definition from YAML file."""
        with open(yaml_path, 'r') as f:
            spec = yaml.safe_load(f)
        return KernelDefinitionFactory.from_specification(spec)
    
    @staticmethod
    def from_json_file(json_path: str) -> KernelDefinition:
        """Load kernel definition from JSON file."""
        with open(json_path, 'r') as f:
            spec = json.load(f)
        return KernelDefinitionFactory.from_specification(spec)
    
    @staticmethod
    def save_to_yaml(kernel_def: KernelDefinition, yaml_path: str):
        """Save kernel definition to YAML file."""
        spec = kernel_def.to_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False)
    
    @staticmethod
    def save_to_json(kernel_def: KernelDefinition, json_path: str):
        """Save kernel definition to JSON file."""
        spec = kernel_def.to_dict()
        with open(json_path, 'w') as f:
            json.dump(spec, f, indent=2)