"""Isolated component tests: build a tiny MAX graph for one sub-function with real
weights and compare to the dumped torch activation. Fast compile, localizes bugs.

Run: PYTHONPATH=ext/demucs:src uv run python scripts/comp_test.py <which>
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
import demucs_mojo.htdemucs_max as M  # noqa: E402
from demucs_mojo import ref_torch as R  # noqa: E402
from max.dtype import DType  # noqa: E402
from max.graph import DeviceRef, Graph, TensorType, ops  # noqa: E402
from max.engine import InferenceSession  # noqa: E402
from max.driver import CPU  # noqa: E402

REF = Path("reference")
DEV = DeviceRef.CPU()
sd = dict(np.load(REF / "state_dict.npz"))
acts = dict(np.load(REF / "activations.npz"))
io = np.load(REF / "io.npz")


def x_norm():
    mix = torch.from_numpy(io["input"]).float()
    x = R.magnitude(R._spec(mix)).numpy()
    mean = x.mean((1, 2, 3), keepdims=True)
    std = x.std((1, 2, 3), ddof=1, keepdims=True)
    return ((x - mean) / (1e-5 + std)).astype(np.float32)


def run(build, *inputs):
    bd = M.Builder(dict(sd))
    its = [TensorType(DType.float32, a.shape, device=DEV) for a in inputs]
    g = Graph("c", input_types=its)
    with g:
        outs = build(bd, [t.tensor for t in g.inputs])
        g.output(*outs) if isinstance(outs, (list, tuple)) else g.output(outs)
    m = InferenceSession(devices=[CPU()]).load(g)
    res = m.execute(*inputs)
    return [r.to_numpy() for r in res]


def chk(name, got, ref):
    print(f"{name:24s} got{got.shape} ref{ref.shape} diff={np.abs(got-ref).max():.3e}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "enc0"
    if which == "enc0":
        xn = x_norm()
        out = run(lambda bd, ins: M.henc(bd, "encoder.0", ins[0], freq=True), xn)[0]
        chk("encoder.0", out, acts["encoder.0"])
    elif which == "tenc0":
        mix = io["input"].astype(np.float32)
        meant = mix.mean((1, 2), keepdims=True)
        stdt = mix.std((1, 2), ddof=1, keepdims=True)
        xtn = ((mix - meant) / (1e-5 + stdt)).astype(np.float32)
        out = run(lambda bd, ins: M.henc(bd, "tencoder.0", ins[0], freq=False), xtn)[0]
        chk("tencoder.0", out, acts["tencoder.0"])
    elif which == "dec0":
        # decoder.0 needs post-downsample x and skip enc3; reconstruct from acts.
        xin = acts["crosstransformer.out0"].astype(np.float32)  # (1,512,8,336)
        skip = acts["encoder.3"].astype(np.float32)

        def downskip(bd, x, sk):
            b, c, f, t = 1, 512, 8, 336
            xf = ops.reshape(x, (b, c, f * t))
            xf = M.conv1x1(bd, xf, "channel_downsampler.weight", "channel_downsampler.bias")
            x = ops.reshape(xf, (b, -1, f, t))
            return x + sk

        def build_pre(bd, ins):
            x, sk = ins
            xx = downskip(bd, x, sk)
            y = M.conv2d_t(bd, xx, "decoder.0.rewrite.weight", "decoder.0.rewrite.bias",
                           stride=(1, 1), pad=(1, 1))
            y = M.glu(y, axis=1)
            sh = [int(d) for d in y.shape]
            B, C, Fr, T = sh
            y = ops.reshape(ops.permute(y, [0, 2, 1, 3]), (B * Fr, C, T))
            y = M.dconv(bd, "decoder.0.dconv", y)
            y = ops.permute(ops.reshape(y, (B, Fr, C, T)), [0, 2, 1, 3])
            return y
        out = run(build_pre, xin, skip)[0]
        chk("decoder.0.pre", out, acts["decoder.0.out1"])
    elif which == "dec0ct":
        pre = acts["decoder.0.out1"].astype(np.float32)  # (1,384,8,336)

        def build(bd, ins):
            y = ins[0]
            z = M.conv_transpose_freq(bd, y, "decoder.0.conv_tr.weight",
                                      "decoder.0.conv_tr.bias", stride_h=4)
            z = z[:, :, 2:-2, :]
            return M.gelu(z)
        out = run(build, pre)[0]
        chk("decoder.0 conv_tr", out, acts["decoder.0.out0"])
    elif which == "tdec0":
        xin = acts["crosstransformer.out1"].astype(np.float32)  # (1,512,1344)
        skip = acts["tencoder.3"].astype(np.float32)

        def build(bd, ins):
            x, sk = ins
            x = M.conv1x1(bd, x, "channel_downsampler_t.weight", "channel_downsampler_t.bias")
            return M.hdec(bd, "tdecoder.0", x, sk, 1344, freq=False, last=False)
        out = run(build, xin, skip)[0]
        chk("tdecoder.0", out, acts["tdecoder.0.out0"])


if __name__ == "__main__":
    main()
