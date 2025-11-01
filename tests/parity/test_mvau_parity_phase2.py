# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MVAU Phase 2 parity tests: MVTU mode (noActivation=0), internal_embedded.

Tests MVAU with multi-threshold activation:
- Matrix: 16×16
- PE=2, SIMD=2
- Datatypes: INT4 inputs/weights, INT2 quantized output
- Mode: MVTU (with multi-threshold activation)
- Memory: internal_embedded
- ActVal: 2 (output range 0-3)

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
@pytest.mark.phase2
class TestMVAUPhase2Parity(MVAUParityBase):
    """Test MVAU Phase 2: MVTU mode (noActivation=0), internal_embedded."""

    def get_mvau_config(self):
        return {
            'mw': 16,
            'mh': 16,
            'pe': 2,
            'simd': 2,
            'idt': DataType["INT4"],
            'wdt': DataType["INT4"],
            'odt': DataType["INT2"],  # Quantized output (ActVal=2 → 4 levels)
            'no_activation': 0,
            'mem_mode': "internal_embedded",
            'act_val': 2,  # 2 bits = 4 output levels (0-3)
        }
