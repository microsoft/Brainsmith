# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MVAU Phase 3 parity tests: Memory modes (internal_decoupled, external).

Tests MVAU with different memory modes for weight access:
- internal_decoupled: Weights streamed via AXI stream (in1_V)
- external: Weights read from external memory

Each configuration tests both MV and MVTU modes.

Inherits 20 tests per configuration from DualKernelTest:
- 7 core parity tests (shapes, widths, datatypes)
- 5 HW estimation tests (cycles, resources)
- 8 golden execution tests (Python, cppsim, rtlsim)

Total: 4 configurations × 20 tests = 80 tests
"""

import pytest
from qonnx.core.datatype import DataType
from tests.parity.mvau_parity_base import MVAUParityBase


@pytest.mark.parity
@pytest.mark.mvau
@pytest.mark.phase3
class TestMVAUPhase3_Decoupled_MV(MVAUParityBase):
    """Test MVAU Phase 3: MV mode, internal_decoupled."""

    def get_mvau_config(self):
        return {
            'mw': 16,
            'mh': 16,
            'pe': 2,
            'simd': 2,
            'idt': DataType["INT4"],
            'wdt': DataType["INT4"],
            'odt': DataType["INT32"],
            'no_activation': 1,
            'mem_mode': "internal_decoupled",
            'act_val': 0,
        }


@pytest.mark.parity
@pytest.mark.mvau
@pytest.mark.phase3
class TestMVAUPhase3_Decoupled_MVTU(MVAUParityBase):
    """Test MVAU Phase 3: MVTU mode, internal_decoupled."""

    def get_mvau_config(self):
        return {
            'mw': 16,
            'mh': 16,
            'pe': 2,
            'simd': 2,
            'idt': DataType["INT4"],
            'wdt': DataType["INT4"],
            'odt': DataType["INT2"],
            'no_activation': 0,
            'mem_mode': "internal_decoupled",
            'act_val': 2,
        }


@pytest.mark.parity
@pytest.mark.mvau
@pytest.mark.phase3
class TestMVAUPhase3_External_MV(MVAUParityBase):
    """Test MVAU Phase 3: MV mode, external."""

    def get_mvau_config(self):
        return {
            'mw': 16,
            'mh': 16,
            'pe': 2,
            'simd': 2,
            'idt': DataType["INT4"],
            'wdt': DataType["INT4"],
            'odt': DataType["INT32"],
            'no_activation': 1,
            'mem_mode': "external",
            'act_val': 0,
        }


@pytest.mark.parity
@pytest.mark.mvau
@pytest.mark.phase3
class TestMVAUPhase3_External_MVTU(MVAUParityBase):
    """Test MVAU Phase 3: MVTU mode, external."""

    def get_mvau_config(self):
        return {
            'mw': 16,
            'mh': 16,
            'pe': 2,
            'simd': 2,
            'idt': DataType["INT4"],
            'wdt': DataType["INT4"],
            'odt': DataType["INT2"],
            'no_activation': 0,
            'mem_mode': "external",
            'act_val': 2,
        }
