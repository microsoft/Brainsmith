# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MVAU Phase 1 parity tests: MV mode (noActivation=1), internal_embedded.

Tests the simplest MVAU configuration:
- Matrix: 16×16
- PE=2, SIMD=2
- Datatypes: INT4 inputs/weights, INT32 accumulator output
- Mode: MV (no activation)
- Memory: internal_embedded

Inherits 20 tests from DualKernelTest:
- 7 core parity tests (shapes, widths, datatypes)
- 5 HW estimation tests (cycles, resources)
- 8 golden execution tests (Python, cppsim, rtlsim)
"""

import pytest
from qonnx.core.datatype import DataType
from tests.parity.mvau_parity_base import MVAUParityBase


@pytest.mark.parity
@pytest.mark.mvau
@pytest.mark.phase1
class TestMVAUPhase1Parity(MVAUParityBase):
    """Test MVAU Phase 1: MV mode (noActivation=1), internal_embedded."""

    def get_mvau_config(self):
        return {
            'mw': 16,
            'mh': 16,
            'pe': 2,
            'simd': 2,
            'idt': DataType["INT4"],
            'wdt': DataType["INT4"],
            'odt': DataType["INT32"],  # Accumulator output
            'no_activation': 1,
            'mem_mode': "internal_embedded",
            'act_val': 0,  # Not used in MV mode
        }
