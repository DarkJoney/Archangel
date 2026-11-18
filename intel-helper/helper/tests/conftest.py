"""Pick the device the tests run on.

By default the Intel GPU when there is one, else the CPU. Set
OMLX_HELPER_DEVICE=cpu to keep a test run off the GPU (it shares the iGPU
with everything else on a laptop).
"""

import os

import torch

if not os.environ.get("OMLX_HELPER_DEVICE"):
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    os.environ["OMLX_HELPER_DEVICE"] = "xpu" if has_xpu else "cpu"
