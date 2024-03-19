# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import unittest
import torch
import pytest
from parameterized import parameterized

import logging
import os
logging.basicConfig(level=logging.INFO)


class TestQuantFlow(unittest.TestCase):

    def setUp(self):
        os.environ["TORCHAO_AUTOTUNER_ENABLE"] = "1"

    def tearDown(self):
        del os.environ["TORCHAO_AUTOTUNER_ENABLE"]

    @parameterized.expand(
            [
                ("cuda", torch.bfloat16),
                ("cuda", torch.bfloat16),
                # TODO: ("cpu", torch.bfloat16),
                ("cuda", torch.float16),
                ("cuda", torch.float16),
                # TODO: ("cpu", torch.float16),
            ])
    def test_int_mm(self, device, dtype):
        from torchao.kernel import intmm_triton
        dtype = torch.bfloat16
        m, k, n = (128, 64, 16)
        x = torch.randn(m, k, dtype=dtype, device=device)
        w = torch.randn(n, k, dtype=dtype, device=device).t()
        x_int = x.to(dtype=torch.int8)
        w_int = w.to(dtype=torch.int8)
        out32_1 = intmm_triton.safe_int_mm(x_int, w_int)
        assert out32_1.dtype == torch.int32
        out32_2 = intmm_triton.int_matmul(x_int, w_int)
        assert out32_2.dtype == out32_1.dtype
        torch.testing.assert_allclose(out32_1, out32_2)

    @parameterized.expand(
            [
                ("cuda", torch.bfloat16),
                ("cuda", torch.bfloat16),
                # TODO: ("cpu", torch.bfloat16),
                ("cuda", torch.float16),
                ("cuda", torch.float16),
                # TODO: ("cpu", torch.float16),
            ])
    def test_int_scaled_mm(self, device, dtype):
        from torchao.kernel import intmm_triton
        dtype = torch.bfloat16
        m, k, n = (128, 64, 16)
        x = torch.randn(m, k, dtype=dtype, device=device)
        scales = x.sum(-1, keepdim=True)
        w = torch.randn(n, k, dtype=dtype, device=device).t()
        x_int = x.to(dtype=torch.int8)
        w_int = w.to(dtype=torch.int8)
        out32_1 = intmm_triton.safe_int_mm(x_int, w_int) * scales
        assert out32_1.dtype == torch.bfloat16
        out32_2 = intmm_triton.int_scaled_matmul(x_int, w_int, scales)
        assert out32_2.dtype == out32_1.dtype
        torch.testing.assert_allclose(out32_1, out32_2)

if __name__ == "__main__":
    unittest.main()
