############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Unit tests for indexed dimension parameter auto-linking.

Tests the new functionality for auto-linking multi-dimensional BDIM/SDIM
parameters using indexed naming convention (e.g., input_BDIM0, input_BDIM1).
"""

import pytest
from pathlib import Path

from brainsmith.tools.hw_kernel_gen.rtl_parser.parameter_linker import ParameterLinker
from brainsmith.tools.hw_kernel_gen.rtl_parser.rtl_data import Parameter
from brainsmith.tools.hw_kernel_gen.metadata import KernelMetadata, InterfaceMetadata
from brainsmith.tools.hw_kernel_gen.data import InterfaceType


class TestIndexedDimensionLinking:
    """Test cases for indexed BDIM/SDIM parameter linking."""
    
    def test_collect_single_bdim_parameter(self):
        """Test collecting single BDIM parameter (existing behavior)."""
        linker = ParameterLinker()
        parameters = [
            Parameter("s_axis_BDIM", default_value="64"),
            Parameter("OTHER_PARAM", default_value="32")
        ]
        
        single, indexed = linker.collect_dimension_parameters("s_axis", parameters, "BDIM")
        
        assert single == "s_axis_BDIM"
        assert indexed is None
    
    def test_collect_contiguous_indexed_bdim(self):
        """Test collecting contiguous indexed BDIM parameters."""
        linker = ParameterLinker()
        parameters = [
            Parameter("input_BDIM0", default_value="16"),
            Parameter("input_BDIM1", default_value="16"),
            Parameter("input_BDIM2", default_value="3"),
            Parameter("OTHER_PARAM", default_value="32")
        ]
        
        single, indexed = linker.collect_dimension_parameters("input", parameters, "BDIM")
        
        assert single is None
        assert indexed == ["input_BDIM0", "input_BDIM1", "input_BDIM2"]
    
    def test_collect_non_contiguous_indexed_bdim(self):
        """Test collecting non-contiguous indexed BDIM with gaps."""
        linker = ParameterLinker()
        parameters = [
            Parameter("input_BDIM0", default_value="32"),
            Parameter("input_BDIM2", default_value="64"),
            # Missing input_BDIM1
            Parameter("OTHER_PARAM", default_value="16")
        ]
        
        single, indexed = linker.collect_dimension_parameters("input", parameters, "BDIM")
        
        assert single is None
        assert indexed == ["input_BDIM0", "1", "input_BDIM2"]  # Gap filled with singleton
    
    def test_collect_sdim_parameters(self):
        """Test collecting SDIM parameters (same logic as BDIM)."""
        linker = ParameterLinker()
        parameters = [
            Parameter("weights_SDIM0", default_value="128"),
            Parameter("weights_SDIM1", default_value="256"),
            Parameter("weights_SDIM2", default_value="512")
        ]
        
        single, indexed = linker.collect_dimension_parameters("weights", parameters, "SDIM")
        
        assert single is None
        assert indexed == ["weights_SDIM0", "weights_SDIM1", "weights_SDIM2"]
    
    def test_mixed_case_support(self):
        """Test support for lowercase dimension type."""
        linker = ParameterLinker()
        parameters = [
            Parameter("data_bdim0", default_value="8"),
            Parameter("data_bdim1", default_value="8"),
            Parameter("data_bdim2", default_value="128")
        ]
        
        single, indexed = linker.collect_dimension_parameters("data", parameters, "BDIM")
        
        assert single is None
        assert indexed == ["data_bdim0", "data_bdim1", "data_bdim2"]
    
    def test_no_dimension_parameters_found(self):
        """Test when no dimension parameters are found."""
        linker = ParameterLinker()
        parameters = [
            Parameter("WIDTH", default_value="32"),
            Parameter("DEPTH", default_value="16")
        ]
        
        single, indexed = linker.collect_dimension_parameters("stream", parameters, "BDIM")
        
        assert single is None
        assert indexed is None
    
    def test_single_takes_precedence_over_indexed(self):
        """Test that single parameter takes precedence if both exist."""
        linker = ParameterLinker()
        parameters = [
            Parameter("input_BDIM", default_value="64"),  # Single parameter
            Parameter("input_BDIM0", default_value="32"), # Also has indexed
            Parameter("input_BDIM1", default_value="32")
        ]
        
        single, indexed = linker.collect_dimension_parameters("input", parameters, "BDIM")
        
        assert single == "input_BDIM"  # Single takes precedence
        assert indexed is None
    
    def test_apply_indexed_bdim_to_kernel_metadata(self):
        """Test applying indexed BDIM parameters to kernel metadata."""
        linker = ParameterLinker()
        
        kernel = KernelMetadata(
            name="test_module",
            source_file=Path("test.sv"),
            parameters=[
                Parameter("in0_BDIM0", default_value="16"),
                Parameter("in0_BDIM1", default_value="16"),
                Parameter("in0_BDIM2", default_value="3"),
                Parameter("OTHER_PARAM", default_value="32")
            ],
            interfaces=[
                InterfaceMetadata(
                    name="in0",
                    interface_type=InterfaceType.INPUT
                )
            ],
            exposed_parameters=["in0_BDIM0", "in0_BDIM1", "in0_BDIM2", "OTHER_PARAM"],
            pragmas=[],
            internal_datatypes=[]
        )
        
        # Apply parameter linking
        linker.apply_to_kernel_metadata(kernel)
        
        # Check interface has indexed BDIM parameters
        interface = kernel.interfaces[0]
        assert interface.bdim_params == ["in0_BDIM0", "in0_BDIM1", "in0_BDIM2"]
        
        # Check indexed parameters were removed from exposed
        assert "in0_BDIM0" not in kernel.exposed_parameters
        assert "in0_BDIM1" not in kernel.exposed_parameters
        assert "in0_BDIM2" not in kernel.exposed_parameters
        assert "OTHER_PARAM" in kernel.exposed_parameters
    
    def test_apply_non_contiguous_sdim_to_kernel_metadata(self):
        """Test applying non-contiguous SDIM parameters with gaps."""
        linker = ParameterLinker()
        
        kernel = KernelMetadata(
            name="test_module",
            source_file=Path("test.sv"),
            parameters=[
                Parameter("stream_SDIM0", default_value="1024"),
                Parameter("stream_SDIM2", default_value="2048"),
                # Missing stream_SDIM1
                Parameter("WIDTH", default_value="32")
            ],
            interfaces=[
                InterfaceMetadata(
                    name="stream",
                    interface_type=InterfaceType.INPUT
                )
            ],
            exposed_parameters=["stream_SDIM0", "stream_SDIM2", "WIDTH"],
            pragmas=[],
            internal_datatypes=[]
        )
        
        # Apply parameter linking
        linker.apply_to_kernel_metadata(kernel)
        
        # Check interface has indexed SDIM parameters with gap filled
        interface = kernel.interfaces[0]
        assert interface.sdim_params == ["stream_SDIM0", "1", "stream_SDIM2"]
        
        # Check only real parameters were removed from exposed
        assert "stream_SDIM0" not in kernel.exposed_parameters
        assert "stream_SDIM2" not in kernel.exposed_parameters
        assert "WIDTH" in kernel.exposed_parameters
    
    def test_pragma_overrides_indexed_parameters(self):
        """Test that pragma-set parameters override auto-linking."""
        linker = ParameterLinker()
        
        kernel = KernelMetadata(
            name="test_module",
            source_file=Path("test.sv"),
            parameters=[
                Parameter("in0_BDIM0", default_value="16"),
                Parameter("in0_BDIM1", default_value="16"),
                Parameter("TILE_H", default_value="8"),
                Parameter("TILE_W", default_value="8")
            ],
            interfaces=[
                InterfaceMetadata(
                    name="in0",
                    interface_type=InterfaceType.INPUT,
                    # Pragma already set these
                    bdim_params=["TILE_H", "TILE_W"]
                )
            ],
            exposed_parameters=["in0_BDIM0", "in0_BDIM1", "TILE_H", "TILE_W"],
            pragmas=[],
            internal_datatypes=[]
        )
        
        # Apply parameter linking
        linker.apply_to_kernel_metadata(kernel)
        
        # Check pragma values were preserved (not overridden)
        interface = kernel.interfaces[0]
        assert interface.bdim_params == ["TILE_H", "TILE_W"]
        
        # Indexed parameters should still be in exposed since pragma takes precedence
        assert "in0_BDIM0" in kernel.exposed_parameters
        assert "in0_BDIM1" in kernel.exposed_parameters
    
    def test_mixed_single_and_indexed_interfaces(self):
        """Test kernel with mix of single and indexed parameter interfaces."""
        linker = ParameterLinker()
        
        kernel = KernelMetadata(
            name="test_module",
            source_file=Path("test.sv"),
            parameters=[
                # Interface 1: uses indexed
                Parameter("in0_BDIM0", default_value="16"),
                Parameter("in0_BDIM1", default_value="16"),
                # Interface 2: uses single
                Parameter("weights_BDIM", default_value="64"),
                # Interface 3: uses default (no params)
                Parameter("OTHER", default_value="32")
            ],
            interfaces=[
                InterfaceMetadata(name="in0", interface_type=InterfaceType.INPUT),
                InterfaceMetadata(name="weights", interface_type=InterfaceType.WEIGHT),
                InterfaceMetadata(name="out0", interface_type=InterfaceType.OUTPUT)
            ],
            exposed_parameters=["in0_BDIM0", "in0_BDIM1", "weights_BDIM", "OTHER"],
            pragmas=[],
            internal_datatypes=[]
        )
        
        # Apply parameter linking
        linker.apply_to_kernel_metadata(kernel)
        
        # Check each interface got correct parameter style
        assert kernel.interfaces[0].bdim_params == ["in0_BDIM0", "in0_BDIM1"]
        assert kernel.interfaces[1].bdim_params == ["weights_BDIM"]
        assert kernel.interfaces[2].bdim_params is None  # No params found, remains None
        
        # Check exposed parameters
        assert "in0_BDIM0" not in kernel.exposed_parameters
        assert "in0_BDIM1" not in kernel.exposed_parameters
        assert "weights_BDIM" not in kernel.exposed_parameters
        assert "OTHER" in kernel.exposed_parameters
    
    def test_zero_based_indexing(self):
        """Test that indexing starts at 0 and handles it correctly."""
        linker = ParameterLinker()
        parameters = [
            Parameter("data_BDIM0", default_value="1"),  # Start at 0
            Parameter("data_BDIM1", default_value="32"),
            Parameter("data_BDIM2", default_value="32"),
            Parameter("data_BDIM3", default_value="3")
        ]
        
        single, indexed = linker.collect_dimension_parameters("data", parameters, "BDIM")
        
        assert single is None
        assert indexed == ["data_BDIM0", "data_BDIM1", "data_BDIM2", "data_BDIM3"]
        assert len(indexed) == 4  # 0-based means 4 elements for indices 0-3
    
    def test_large_gap_in_indices(self):
        """Test handling large gaps in indices."""
        linker = ParameterLinker()
        parameters = [
            Parameter("tensor_BDIM0", default_value="128"),
            Parameter("tensor_BDIM5", default_value="3"),  # Large gap
        ]
        
        single, indexed = linker.collect_dimension_parameters("tensor", parameters, "BDIM")
        
        assert single is None
        assert indexed == ["tensor_BDIM0", "1", "1", "1", "1", "tensor_BDIM5"]
        assert len(indexed) == 6  # Indices 0-5