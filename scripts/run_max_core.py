"""Build the MAX HTDemucs core, run it on the reference normalized inputs, and
compare against the torch-dumped pre-iSTFT activations (decoder.3 / tdecoder.3).

Run: uv run python scripts/run_max_core.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from demucs_mojo.htdemucs_max import build_core  # noqa: E402
from max.engine import InferenceSession  # noqa: E402
from max.driver import CPU  # noqa: E402

REF = Path("reference")
EPS = 1e-5


def normalize_inputs():
    io = np.load(REF / "io.npz")
    acts = dict(np.load(REF / "activations.npz"))
    mix = io["input"].astype(np.float32)  # (1,2,L)
    # freq input = magnitude(STFT(mix)) then normalize; reuse torch ref for STFT.
    import torch
    from demucs_mojo import ref_torch as R
    mt = torch.from_numpy(mix)
    z = R._spec(mt)
    x = R.magnitude(z).numpy()
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    std = x.std(axis=(1, 2, 3), ddof=1, keepdims=True)
    x_norm = (x - mean) / (EPS + std)
    meant = mix.mean(axis=(1, 2), keepdims=True)
    stdt = mix.std(axis=(1, 2), ddof=1, keepdims=True)
    xt_norm = (mix - meant) / (EPS + stdt)
    return x_norm.astype(np.float32), xt_norm.astype(np.float32), acts


def main():
    x_norm, xt_norm, acts = normalize_inputs()
    print("x_norm", x_norm.shape, "xt_norm", xt_norm.shape)
    sd = dict(np.load(REF / "state_dict.npz"))

    t0 = time.time()
    g = build_core(sd, x_norm.shape, xt_norm.shape)
    print(f"graph built in {time.time()-t0:.1f}s")
    t0 = time.time()
    sess = InferenceSession(devices=[CPU()])
    model = sess.load(g)
    print(f"compiled in {time.time()-t0:.1f}s")

    t0 = time.time()
    out = model.execute(x_norm, xt_norm)
    x_pre = out[0].to_numpy()
    xt_pre = out[1].to_numpy()
    print(f"executed in {time.time()-t0:.2f}s")

    ref_x = acts["decoder.3.out0"]
    ref_xt = acts["tdecoder.3.out0"]
    print("x_pre", x_pre.shape, "ref", ref_x.shape)
    print("xt_pre", xt_pre.shape, "ref", ref_xt.shape)
    dx = np.abs(x_pre - ref_x).max()
    dxt = np.abs(xt_pre - ref_xt).max()
    print(f"freq-branch pre-iSTFT max abs diff:  {dx:.3e}")
    print(f"time-branch pre       max abs diff:  {dxt:.3e}")


if __name__ == "__main__":
    main()
