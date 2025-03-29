############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import pytest
from op_test import OpTest
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto

"""
Below is an example of a test constructed using the OpTest class.
"""

@pytest.mark.parametrize("simd", [1, 2, 4], ids=["SIMD1", "SIMD2", "SIMD4"])
@pytest.mark.parametrize("idt", ["INT8", "INT9"])
@pytest.mark.parametrize("ifm_dim", [(1, 128, 384), (1, 12, 12, 128)])
class TestReduceSum(OpTest):

    @pytest.fixture
    def model(self, simd, idt, ifm_dim)->ModelWrapper:

        odt = idt
        model:ModelWrapper = self.create_model(
            inputs = [
                (dict(name='X', elem_type=TensorProto.FLOAT, shape=ifm_dim), idt),
            ],
            inits = [
            ],
            outputs= [
                (dict(name='Y', elem_type=TensorProto.FLOAT, shape=ifm_dim), odt),
            ],
            nodes= [
                dict(op_type="HWReduceSum",
                    inputs=['X'],
                    outputs=['Y'],
                    domain="finnbrainsmith.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    SIMD=simd,
                    preferred_impl_style="hls",
                    ifm_dim=ifm_dim,
                    data_type=idt,
                   ),
            ]
        )
        return model
