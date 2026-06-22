"""From-scratch HTDemucs forward built ONLY from the raw state_dict tensors and
the verified spec (docs/htdemucs_spec.md) — it does NOT call the demucs module.

Purpose: an executable, debuggable blueprint that reproduces the torch reference
output and every dumped intermediate. Each torch op here maps 1:1 to a MAX graph
op, so this doubles as the translation template for the MAX port.

Run: PYTHONPATH=ext/demucs uv run python -m demucs_mojo.ref_torch   (validation)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REF = Path("reference")
NFFT = 4096
HOP = 1024
SOURCES = 4
AUDIO = 2
EPS = 1e-5


# ----------------------------- STFT / iSTFT --------------------------------
def spectro(x):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = torch.stft(x, NFFT, HOP, window=torch.hann_window(NFFT),
                   win_length=NFFT, normalized=True, center=True,
                   return_complex=True, pad_mode="reflect")
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, length):
    *other, freqs, frames = z.shape
    z = z.reshape(-1, freqs, frames)
    x = torch.istft(z, NFFT, HOP, window=torch.hann_window(NFFT),
                    win_length=NFFT, normalized=True, length=length, center=True)
    _, n = x.shape
    return x.view(*other, n)


def _spec(x):
    hl, nfft = HOP, NFFT
    le = int(math.ceil(x.shape[-1] / hl))
    pad = hl // 2 * 3
    x = F.pad(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
    z = spectro(x)[..., :-1, :]
    z = z[..., 2:2 + le]
    return z


def _ispec(z, length):
    hl = HOP
    z = F.pad(z, (0, 0, 0, 1))
    z = F.pad(z, (2, 2))
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    x = ispectro(z, le)
    x = x[..., pad:pad + length]
    return x


def magnitude(z):  # cac: complex -> channels
    B, C, Fr, T = z.shape
    m = torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(B, C * 2, Fr, T)
    return m


def from_cac(m):  # (B,S,2C,Fr,T) -> complex (B,S,C,Fr,T)
    B, S, C2, Fr, T = m.shape
    out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
    return torch.view_as_complex(out.contiguous())


# ----------------------------- primitives ----------------------------------
def glu(x, dim=1):
    a, b = x.chunk(2, dim=dim)
    return a * torch.sigmoid(b)


class W:
    """Thin accessor over the numpy state_dict."""
    def __init__(self, sd):
        self.sd = sd

    def __call__(self, key):
        return torch.from_numpy(self.sd[key]).float()


def dconv(w, prefix, y):
    # y: (Bf, C, T). Two residual layers.
    for d in range(2):
        p = f"{prefix}.layers.{d}"
        h = F.conv1d(y, w(f"{p}.0.weight"), w(f"{p}.0.bias"),
                     dilation=2 ** d, padding=2 ** d)
        h = F.group_norm(h, 1, w(f"{p}.1.weight"), w(f"{p}.1.bias"))
        h = F.gelu(h)
        h = F.conv1d(h, w(f"{p}.3.weight"), w(f"{p}.3.bias"))
        h = F.group_norm(h, 1, w(f"{p}.4.weight"), w(f"{p}.4.bias"))
        h = glu(h, dim=1)
        h = w(f"{p}.6.scale")[:, None] * h
        y = y + h
    return y


def henc(w, prefix, x, freq, inject=None):
    if not freq:
        le = x.shape[-1]
        if le % 4 != 0:
            x = F.pad(x, (0, 4 - le % 4))
    if freq:
        y = F.conv2d(x, w(f"{prefix}.conv.weight"), w(f"{prefix}.conv.bias"),
                     stride=(4, 1), padding=(2, 0))
    else:
        y = F.conv1d(x, w(f"{prefix}.conv.weight"), w(f"{prefix}.conv.bias"),
                     stride=4, padding=2)
    if inject is not None:
        y = y + inject
    y = F.gelu(y)  # norm1 = Identity
    # dconv (fold Fr into batch for freq)
    if freq:
        B, C, Fr, T = y.shape
        y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
        y = dconv(w, f"{prefix}.dconv", y)
        y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
    else:
        y = dconv(w, f"{prefix}.dconv", y)
    # rewrite 1x1 + GLU (norm2 = Identity)
    if freq:
        z = F.conv2d(y, w(f"{prefix}.rewrite.weight"), w(f"{prefix}.rewrite.bias"))
    else:
        z = F.conv1d(y, w(f"{prefix}.rewrite.weight"), w(f"{prefix}.rewrite.bias"))
    return glu(z, dim=1)


def hdec(w, prefix, x, skip, length, freq, last):
    # additive skip; rewrite (k3) + GLU; conv_tr; crop; gelu unless last.
    x = x + skip
    if freq:
        y = F.conv2d(x, w(f"{prefix}.rewrite.weight"), w(f"{prefix}.rewrite.bias"),
                     padding=(1, 1))
    else:
        y = F.conv1d(x, w(f"{prefix}.rewrite.weight"), w(f"{prefix}.rewrite.bias"),
                     padding=1)
    y = glu(y, dim=1)
    # DConv (dconv_mode=3 -> decoder has it too), fold Fr into batch for freq
    if freq:
        B, Cc, Fr, T = y.shape
        y = y.permute(0, 2, 1, 3).reshape(-1, Cc, T)
        y = dconv(w, f"{prefix}.dconv", y)
        y = y.view(B, Fr, Cc, T).permute(0, 2, 1, 3)
    else:
        y = dconv(w, f"{prefix}.dconv", y)
    pre = y
    if freq:
        z = F.conv_transpose2d(y, w(f"{prefix}.conv_tr.weight"),
                               w(f"{prefix}.conv_tr.bias"), stride=(4, 1))
        z = z[..., 2:-2, :]
    else:
        z = F.conv_transpose1d(y, w(f"{prefix}.conv_tr.weight"),
                               w(f"{prefix}.conv_tr.bias"), stride=4)
        z = z[..., 2:2 + length]
    if not last:
        z = F.gelu(z)
    return z, pre


# ----------------------------- transformer ---------------------------------
def create_2d_sin_embedding(d_model, height, width, max_period=10000.0):
    pe = torch.zeros(d_model, height, width)
    d = d_model // 2
    div = torch.exp(torch.arange(0.0, d, 2) * -(math.log(max_period) / d))
    pos_w = torch.arange(width).float()[:, None]
    pos_h = torch.arange(height).float()[:, None]
    pe[0:d:2] = torch.sin(pos_w * div).t()[:, None, :].expand(-1, height, -1)
    pe[1:d:2] = torch.cos(pos_w * div).t()[:, None, :].expand(-1, height, -1)
    pe[d::2] = torch.sin(pos_h * div).t()[:, :, None].expand(-1, -1, width)
    pe[d + 1::2] = torch.cos(pos_h * div).t()[:, :, None].expand(-1, -1, width)
    return pe[None, :]  # (1, d_model, height, width)


def create_sin_embedding(length, dim, max_period=10000.0):
    pos = torch.arange(length).float()[:, None]
    half = dim // 2
    adim = torch.arange(half).float()[None, :]
    phase = pos / (max_period ** (adim / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)[:, None]  # (T,1,dim)


def layernorm(w, prefix, x):
    return F.layer_norm(x, (x.shape[-1],), w(f"{prefix}.weight"), w(f"{prefix}.bias"), 1e-5)


def mygroupnorm(w, prefix, x):  # (B,N,C) -> transpose -> GN(1) -> back
    xt = x.transpose(1, 2)
    xt = F.group_norm(xt, 1, w(f"{prefix}.weight"), w(f"{prefix}.bias"), 1e-5)
    return xt.transpose(1, 2)


def mha(w, prefix, q, k, v, nheads=8):
    # in_proj packed [Wq;Wk;Wv]
    ipw = w(f"{prefix}.in_proj_weight")
    ipb = w(f"{prefix}.in_proj_bias")
    dim = q.shape[-1]
    wq, wk, wv = ipw.split(dim, dim=0)
    bq, bk, bv = ipb.split(dim, dim=0)
    Q = q @ wq.t() + bq
    K = k @ wk.t() + bk
    V = v @ wv.t() + bv
    B, Nq, _ = Q.shape
    Nk = K.shape[1]
    hd = dim // nheads
    Q = Q.view(B, Nq, nheads, hd).transpose(1, 2)
    K = K.view(B, Nk, nheads, hd).transpose(1, 2)
    V = V.view(B, Nk, nheads, hd).transpose(1, 2)
    att = (Q @ K.transpose(-2, -1)) / math.sqrt(hd)
    att = att.softmax(dim=-1)
    o = att @ V  # (B,nh,Nq,hd)
    o = o.transpose(1, 2).reshape(B, Nq, dim)
    return o @ w(f"{prefix}.out_proj.weight").t() + w(f"{prefix}.out_proj.bias")


def ffn(w, prefix, x):
    h = F.gelu(x @ w(f"{prefix}.linear1.weight").t() + w(f"{prefix}.linear1.bias"))
    return h @ w(f"{prefix}.linear2.weight").t() + w(f"{prefix}.linear2.bias")


def self_layer(w, prefix, x):
    g1 = w(f"{prefix}.gamma_1.scale")
    g2 = w(f"{prefix}.gamma_2.scale")
    n1 = layernorm(w, f"{prefix}.norm1", x)
    x = x + g1 * mha(w, f"{prefix}.self_attn", n1, n1, n1)
    n2 = layernorm(w, f"{prefix}.norm2", x)
    x = x + g2 * ffn(w, prefix, n2)
    return mygroupnorm(w, f"{prefix}.norm_out", x)


def cross_layer(w, prefix, q, k):
    g1 = w(f"{prefix}.gamma_1.scale")
    g2 = w(f"{prefix}.gamma_2.scale")
    nq = layernorm(w, f"{prefix}.norm1", q)
    nk = layernorm(w, f"{prefix}.norm2", k)
    x = q + g1 * mha(w, f"{prefix}.cross_attn", nq, nk, nk)
    n3 = layernorm(w, f"{prefix}.norm3", x)
    x = x + g2 * ffn(w, prefix, n3)
    return mygroupnorm(w, f"{prefix}.norm_out", x)


def crosstransformer(w, x, xt):
    B, C, Fr, T1 = x.shape
    pe2d = create_2d_sin_embedding(C, Fr, T1)  # (1,C,Fr,T1)
    pe2d = pe2d.reshape(1, C, Fr * T1)  # placeholder; reorder below
    # rearrange "b c fr t1 -> b (t1 fr) c"
    x = x.permute(0, 3, 2, 1).reshape(B, T1 * Fr, C)
    pe2d = create_2d_sin_embedding(C, Fr, T1)[0].permute(2, 1, 0).reshape(T1 * Fr, C)[None]
    x = layernorm(w, "crosstransformer.norm_in", x) + pe2d

    Bt, Ct, T2 = xt.shape
    xt = xt.permute(0, 2, 1)  # (B,T2,C)
    pe = create_sin_embedding(T2, Ct)[:, 0, :][None]  # (1,T2,C)
    xt = layernorm(w, "crosstransformer.norm_in_t", xt) + pe

    for idx in range(5):
        if idx % 2 == 0:  # self
            x = self_layer(w, f"crosstransformer.layers.{idx}", x)
            xt = self_layer(w, f"crosstransformer.layers_t.{idx}", xt)
        else:  # cross
            old_x = x
            x = cross_layer(w, f"crosstransformer.layers.{idx}", x, xt)
            xt = cross_layer(w, f"crosstransformer.layers_t.{idx}", xt, old_x)
    x = x.reshape(B, T1, Fr, C).permute(0, 3, 2, 1)
    xt = xt.permute(0, 2, 1)
    return x, xt


# ----------------------------- full forward --------------------------------
def forward(w, mix, dbg=None):
    length = mix.shape[-1]
    z = _spec(mix)
    x = magnitude(z)
    B, C, Fq, T = x.shape
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (EPS + std)

    xt = mix
    meant = xt.mean(dim=(1, 2), keepdim=True)
    stdt = xt.std(dim=(1, 2), keepdim=True)
    xt = (xt - meant) / (EPS + stdt)

    saved, saved_t, lengths, lengths_t = [], [], [], []
    for idx in range(4):
        lengths.append(x.shape[-1])
        lengths_t.append(xt.shape[-1])
        xt = henc(w, f"tencoder.{idx}", xt, freq=False)
        saved_t.append(xt)
        x = henc(w, f"encoder.{idx}", x, freq=True)
        if idx == 0:
            frs = torch.arange(x.shape[-2])
            emb = (w("freq_emb.embedding.weight") * 10.0)[frs].t()[None, :, :, None]
            x = x + 0.2 * emb.expand_as(x)
            if dbg is not None:
                dbg["freq_emb"] = (w("freq_emb.embedding.weight") * 10.0)
        saved.append(x)
        if dbg is not None:
            dbg[f"encoder.{idx}"] = x
            dbg[f"tencoder.{idx}"] = xt

    # transformer with channel up/down sampling (bottom_channels=512)
    b, c, f, t = x.shape
    xf = x.reshape(b, c, f * t)
    xf = F.conv1d(xf, w("channel_upsampler.weight"), w("channel_upsampler.bias"))
    x = xf.reshape(b, -1, f, t)
    xt = F.conv1d(xt, w("channel_upsampler_t.weight"), w("channel_upsampler_t.bias"))

    x, xt = crosstransformer(w, x, xt)
    if dbg is not None:
        dbg["crosstransformer.out0"] = x
        dbg["crosstransformer.out1"] = xt

    xf = x.reshape(b, x.shape[1], f * t)
    xf = F.conv1d(xf, w("channel_downsampler.weight"), w("channel_downsampler.bias"))
    x = xf.reshape(b, -1, f, t)
    xt = F.conv1d(xt, w("channel_downsampler_t.weight"), w("channel_downsampler_t.bias"))

    for idx in range(4):
        skip = saved.pop(-1)
        x, pre = hdec(w, f"decoder.{idx}", x, skip, lengths.pop(-1),
                      freq=True, last=(idx == 3))
        skip_t = saved_t.pop(-1)
        length_t = lengths_t.pop(-1)
        xt, _ = hdec(w, f"tdecoder.{idx}", xt, skip_t, length_t,
                     freq=False, last=(idx == 3))
        if dbg is not None:
            dbg[f"decoder.{idx}.out0"] = x
            dbg[f"tdecoder.{idx}.out0"] = xt

    S = SOURCES
    x = x.view(B, S, -1, Fq, T)
    x = x * std[:, None] + mean[:, None]
    zout = from_cac(x)
    x = _ispec(zout, length)

    xt = xt.view(B, S, -1, length)
    xt = xt * stdt[:, None] + meant[:, None]
    return xt + x


def main():
    sd = dict(np.load(REF / "state_dict.npz"))
    w = W(sd)
    io = np.load(REF / "io.npz")
    acts = dict(np.load(REF / "activations.npz"))
    mix = torch.from_numpy(io["input"]).float()
    dbg = {}
    with torch.no_grad():
        y = forward(w, mix, dbg)
    ref = torch.from_numpy(io["output"]).float()
    print("output max abs diff:", (y - ref).abs().max().item())
    print("--- intermediate checks ---")
    for k in sorted(dbg):
        if k in acts:
            a = torch.from_numpy(acts[k]).float()
            d = dbg[k]
            if d.shape == a.shape:
                print(f"  {k:28s} diff={ (d-a).abs().max().item():.3e}  shape={tuple(a.shape)}")
            else:
                print(f"  {k:28s} SHAPE MISMATCH got {tuple(d.shape)} want {tuple(a.shape)}")


if __name__ == "__main__":
    main()
