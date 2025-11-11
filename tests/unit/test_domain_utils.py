# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for domain matching and derivation utilities.

Tests the core domain resolution functions that enable hierarchical
prefix matching and ONNX domain alignment.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from brainsmith.registry._domain_utils import (
    derive_domain_from_module,
    expand_short_form,
    get_subdomain_for_type,
    match_domain_to_source,
)


class TestGetSubdomainForType:
    """Test subdomain mapping for component types."""

    def test_kernel_subdomain(self):
        assert get_subdomain_for_type('kernel') == 'kernels'

    def test_backend_subdomain(self):
        # Backends use same domain as kernels
        assert get_subdomain_for_type('backend') == 'kernels'

    def test_step_subdomain(self):
        assert get_subdomain_for_type('step') == 'steps'

    def test_unknown_type(self):
        assert get_subdomain_for_type('unknown') is None


class TestMatchDomainToSource:
    """Test domain → source matching with hierarchical prefixes."""

    def test_exact_match(self):
        """Test exact domain match."""
        config = MagicMock()
        config.component_sources = {'brainsmith': Path('/path')}
        config.source_priority = ['brainsmith']

        with patch('brainsmith.settings.get_config', return_value=config):
            assert match_domain_to_source('brainsmith') == 'brainsmith'

    def test_prefix_match_kernels(self):
        """Test prefix match for brainsmith.kernels domain."""
        config = MagicMock()
        config.component_sources = {'brainsmith': Path('/path')}
        config.source_priority = ['brainsmith']

        with patch('brainsmith.settings.get_config', return_value=config):
            assert match_domain_to_source('brainsmith.kernels') == 'brainsmith'

    def test_prefix_match_steps(self):
        """Test prefix match for brainsmith.steps domain."""
        config = MagicMock()
        config.component_sources = {'brainsmith': Path('/path')}
        config.source_priority = ['brainsmith']

        with patch('brainsmith.settings.get_config', return_value=config):
            assert match_domain_to_source('brainsmith.steps') == 'brainsmith'

    def test_longest_prefix_wins(self):
        """Test that longest matching prefix is preferred."""
        config = MagicMock()
        config.component_sources = {
            'my_proj': Path('/path1'),
            'my_proj.experimental': Path('/path2'),
        }
        config.source_priority = ['my_proj.experimental', 'my_proj']

        with patch('brainsmith.settings.get_config', return_value=config):
            # Should match longer prefix
            result = match_domain_to_source('my_proj.experimental.kernels')
            assert result == 'my_proj.experimental'

    def test_finn_domain(self):
        """Test FINN's long domain."""
        config = MagicMock()
        config.component_sources = {'finn': Path('/path')}
        config.source_priority = ['finn']

        with patch('brainsmith.settings.get_config', return_value=config):
            result = match_domain_to_source('finn.custom_op.fpgadataflow.hls')
            assert result == 'finn'

    def test_no_match_uses_source_priority(self):
        """Test fallback to source_priority when no match found."""
        config = MagicMock()
        config.component_sources = {'brainsmith': Path('/path')}
        config.source_priority = ['brainsmith', 'custom']

        with patch('brainsmith.settings.get_config', return_value=config):
            result = match_domain_to_source('unknown.domain')
            assert result == 'brainsmith'  # First in priority

    def test_no_match_no_priority_uses_custom(self):
        """Test fallback to 'custom' when no match and no priority."""
        config = MagicMock()
        config.component_sources = {}
        config.source_priority = []

        with patch('brainsmith.settings.get_config', return_value=config):
            result = match_domain_to_source('unknown.domain')
            assert result == 'custom'


class TestDeriveDomainFromModule:
    """Test module path → domain derivation."""

    def test_brainsmith_kernels(self):
        """Test brainsmith kernel module."""
        module = "brainsmith.kernels.layernorm.layernorm_hls"
        assert derive_domain_from_module(module) == "brainsmith.kernels"

    def test_brainsmith_kernels_exact(self):
        """Test exact brainsmith.kernels module."""
        module = "brainsmith.kernels"
        assert derive_domain_from_module(module) == "brainsmith.kernels"

    def test_brainsmith_steps(self):
        """Test brainsmith step module."""
        module = "brainsmith.steps.build_dataflow_graph"
        assert derive_domain_from_module(module) == "brainsmith.steps"

    def test_brainsmith_steps_exact(self):
        """Test exact brainsmith.steps module."""
        module = "brainsmith.steps"
        assert derive_domain_from_module(module) == "brainsmith.steps"

    def test_finn_hls(self):
        """Test FINN HLS module."""
        module = "finn.custom_op.fpgadataflow.hls.mvau_hls"
        result = derive_domain_from_module(module)
        assert result == "finn.custom_op.fpgadataflow.hls"

    def test_finn_rtl(self):
        """Test FINN RTL module."""
        module = "finn.custom_op.fpgadataflow.rtl.mvau_rtl"
        result = derive_domain_from_module(module)
        assert result == "finn.custom_op.fpgadataflow.rtl"

    def test_finn_base(self):
        """Test FINN base module without language suffix."""
        module = "finn.custom_op.fpgadataflow.utils"
        result = derive_domain_from_module(module)
        # For modules without hls/rtl, use base domain
        assert result == "finn.custom_op.fpgadataflow"

    def test_finn_base_longer(self):
        """Test FINN base module with deeper nesting (more realistic)."""
        module = "finn.custom_op.fpgadataflow.utils.helper"
        result = derive_domain_from_module(module)
        # Should truncate to first 4 segments
        assert result == "finn.custom_op.fpgadataflow"

    def test_custom_source_kernels(self):
        """Test custom source with kernels category."""
        config = MagicMock()
        config.component_sources = {'acme': Path('/path')}

        with patch('brainsmith.settings.get_config', return_value=config):
            module = "acme.kernels.my_kernel"
            result = derive_domain_from_module(module)
            assert result == "acme.kernels"

    def test_custom_source_steps(self):
        """Test custom source with steps category."""
        config = MagicMock()
        config.component_sources = {'acme': Path('/path')}

        with patch('brainsmith.settings.get_config', return_value=config):
            module = "acme.steps.my_step"
            result = derive_domain_from_module(module)
            assert result == "acme.steps"

    def test_unique_module_name_pattern(self):
        """Test unique module names from discovery (source__category)."""
        config = MagicMock()
        config.component_sources = {'project': Path('/path')}

        with patch('brainsmith.settings.get_config', return_value=config):
            module = "project__kernels.my_kernel"
            result = derive_domain_from_module(module)
            # Should handle unique naming pattern
            assert "kernels" in result

    def test_hierarchical_source(self):
        """Test hierarchical source names."""
        # Mock discovered sources to include com.acme
        with patch('brainsmith.registry._state._discovered_sources', {'com.acme'}):
            module = "com.acme.kernels.custom_op"
            result = derive_domain_from_module(module)
            assert result == "com.acme.kernels"

    def test_fallback_two_segments(self):
        """Test fallback for unknown pattern with known category."""
        config = MagicMock()
        config.component_sources = {}

        with patch('brainsmith.settings.get_config', return_value=config):
            module = "unknown.kernels.my_op"
            result = derive_domain_from_module(module)
            assert result == "unknown.kernels"

    def test_fallback_single_segment(self):
        """Test fallback for single segment module."""
        config = MagicMock()
        config.component_sources = {}

        with patch('brainsmith.settings.get_config', return_value=config):
            module = "unknown_module"
            result = derive_domain_from_module(module)
            assert result == "unknown_module.kernels"


class TestExpandShortForm:
    """Test short form expansion for component references."""

    def test_kernel_short_form(self):
        """Test kernel short form expansion."""
        result = expand_short_form("brainsmith:LayerNorm", "kernel")
        assert result == "brainsmith.kernels:LayerNorm"

    def test_backend_short_form(self):
        """Test backend short form expansion (uses kernels)."""
        result = expand_short_form("brainsmith:LayerNorm_hls", "backend")
        assert result == "brainsmith.kernels:LayerNorm_hls"

    def test_step_short_form(self):
        """Test step short form expansion."""
        result = expand_short_form("brainsmith:build_dataflow_graph", "step")
        assert result == "brainsmith.steps:build_dataflow_graph"

    def test_already_full_form(self):
        """Test that full form is not modified."""
        result = expand_short_form("brainsmith.kernels:LayerNorm", "kernel")
        assert result == "brainsmith.kernels:LayerNorm"

    def test_no_colon_unchanged(self):
        """Test that unqualified names are unchanged."""
        result = expand_short_form("LayerNorm", "kernel")
        assert result == "LayerNorm"

    def test_finn_short_form(self):
        """Test FINN short form."""
        result = expand_short_form("finn:MVAU", "kernel")
        assert result == "finn.kernels:MVAU"

    def test_custom_source_short_form(self):
        """Test custom source short form."""
        result = expand_short_form("acme:CustomOp", "kernel")
        assert result == "acme.kernels:CustomOp"

    def test_unknown_type_unchanged(self):
        """Test that unknown types don't get expanded."""
        result = expand_short_form("brainsmith:Something", "unknown")
        assert result == "brainsmith:Something"

    def test_hierarchical_short_form(self):
        """Test hierarchical source name short form."""
        result = expand_short_form("com.acme:CustomOp", "kernel")
        # Has dot before colon, so treated as full form
        assert result == "com.acme:CustomOp"


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_round_trip_brainsmith(self):
        """Test module → domain → source round trip for brainsmith."""
        config = MagicMock()
        config.component_sources = {'brainsmith': Path('/path')}
        config.source_priority = ['brainsmith']

        with patch('brainsmith.settings.get_config', return_value=config):
            # Module → domain
            module = "brainsmith.kernels.layernorm.layernorm_hls"
            domain = derive_domain_from_module(module)
            assert domain == "brainsmith.kernels"

            # Domain → source
            source = match_domain_to_source(domain)
            assert source == "brainsmith"

    def test_round_trip_finn(self):
        """Test module → domain → source round trip for FINN."""
        config = MagicMock()
        config.component_sources = {'finn': Path('/path')}
        config.source_priority = ['finn']

        with patch('brainsmith.settings.get_config', return_value=config):
            # Module → domain
            module = "finn.custom_op.fpgadataflow.hls.mvau_hls"
            domain = derive_domain_from_module(module)
            assert domain == "finn.custom_op.fpgadataflow.hls"

            # Domain → source
            source = match_domain_to_source(domain)
            assert source == "finn"

    def test_short_form_to_domain(self):
        """Test short form expansion matches derived domain."""
        # Short form expansion
        expanded = expand_short_form("brainsmith:LayerNorm", "kernel")
        assert expanded == "brainsmith.kernels:LayerNorm"

        # Should match domain from module
        module = "brainsmith.kernels.layernorm.layernorm"
        domain = derive_domain_from_module(module)
        assert expanded.startswith(domain + ":")
