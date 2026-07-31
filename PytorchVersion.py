import contextlib
import sys
import types

import torch

if not hasattr(torch.nn, "attention"):
    _fake = types.ModuleType("torch.nn.attention")

    class SDPBackend:
        FLASH_ATTENTION = "flash_attention"
        EFFICIENT_ATTENTION = "efficient_attention"
        MATH = "math"

    @contextlib.contextmanager
    def sdpa_kernel(_backends):
        yield

    _fake.SDPBackend = SDPBackend
    _fake.sdpa_kernel = sdpa_kernel
    sys.modules["torch.nn.attention"] = _fake
    torch.nn.attention = _fake
