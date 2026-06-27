"""End-to-end HTDemucs inference with the MAX core + torch STFT/iSTFT bookends.

separate(mix) reproduces HTDemucs.forward: STFT -> normalize -> [MAX NN core] ->
de-normalize -> iSTFT -> recombine. The heavy conv/transformer compute runs in
MAX; the cheap framing/normalization runs in torch.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import ref_torch as R
from .htdemucs_max import build_core

EPS = 1e-5
SOURCES = 4


class MaxHTDemucs:
    def __init__(self, state_dict: dict[str, np.ndarray], length: int = 343980):
        import os
        from max.engine import InferenceSession
        from max.driver import CPU, Accelerator
        use_gpu = os.environ.get("DEMUCS_MAX_DEVICE", "gpu").lower() == "gpu"
        self.device = Accelerator() if use_gpu else CPU()
        self.host = CPU()
        self.device_label = "GPU" if use_gpu else "CPU"
        self.sd = state_dict
        self.length = length
        # derive graph input shapes from a dummy STFT
        dummy = torch.zeros(1, 2, length)
        z = R._spec(dummy)
        T = z.shape[-1]
        self.x_shape = (1, 4, z.shape[-2], T)
        self.xt_shape = (1, 2, length)
        t0 = time.time()
        g = build_core(state_dict, self.x_shape, self.xt_shape)
        self.build_s = time.time() - t0
        t0 = time.time()
        self.model = InferenceSession(devices=[self.device]).load(g)
        self.compile_s = time.time() - t0

    def _pre(self, mix: torch.Tensor):
        z = R._spec(mix)
        x = R.magnitude(z)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x_norm = (x - mean) / (EPS + std)
        meant = mix.mean(dim=(1, 2), keepdim=True)
        stdt = mix.std(dim=(1, 2), keepdim=True)
        xt_norm = (mix - meant) / (EPS + stdt)
        return z, x_norm, mean, std, xt_norm, meant, stdt

    def _post(self, z, x_pre, mean, std, xt_pre, meant, stdt, length):
        B, Fq, T = 1, z.shape[-2], z.shape[-1]
        x = torch.from_numpy(x_pre).view(B, SOURCES, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        zout = R.from_cac(x)
        xrec = R._ispec(zout, length)
        xt = torch.from_numpy(xt_pre).view(B, SOURCES, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        return xt + xrec

    def separate(self, mix: torch.Tensor, timing: dict | None = None):
        length = mix.shape[-1]
        if length != self.length:
            mix = torch.nn.functional.pad(mix, (0, self.length - length))
        z, x_norm, mean, std, xt_norm, meant, stdt = self._pre(mix)
        from max.driver import Buffer
        x_buf = Buffer.from_numpy(x_norm.numpy().astype(np.float32)).to(self.device)
        xt_buf = Buffer.from_numpy(xt_norm.numpy().astype(np.float32)).to(self.device)
        t0 = time.time()
        out = self.model.execute(x_buf, xt_buf)
        # .to(host).to_numpy() forces the device->host copy, which synchronizes the
        # GPU so the timing reflects real compute (execute() returns asynchronously).
        x_pre = out[0].to(self.host).to_numpy()
        xt_pre = out[1].to(self.host).to_numpy()
        if timing is not None:
            timing["core_s"] = time.time() - t0
        y = self._post(z, x_pre, mean, std, xt_pre, meant, stdt, self.length)
        return y[..., :length]


def load_state_dict(path="reference/state_dict.npz") -> dict[str, np.ndarray]:
    return dict(np.load(Path(path)))
