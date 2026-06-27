"""HTDemucs neural-network core in the MAX graph API.

Scope: the graph spans the full conv/transformer network — normalized freq input
`x_norm` (B,4,2048,T) and normalized time input `xt_norm` (B,2,L) through both
U-Net branches + cross-domain transformer to the pre-iSTFT outputs `x_pre`
(B,16,2048,T) and `xt_pre` (B,8,L). The STFT, per-example normalization, CaC
reshape, iSTFT and branch recombination are done in torch around the graph (see
runner.py) — those are cheap and fiddly; the heavy compute is all here.

Mirrors the verified blueprint in ref_torch.py op-for-op. Conv helpers keep
tensors in torch layouts (NCHW / NCL) and convert to MAX's NHWC + RSCF internally.
conv2d_transpose won't lower on CPU, so transposed convs use a dilate+conv trick.
"""
from __future__ import annotations

import math
import os

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

# Device selection: DEMUCS_MAX_DEVICE=cpu|gpu (default gpu).
USE_GPU = os.environ.get("DEMUCS_MAX_DEVICE", "gpu").lower() == "gpu"
DEV = DeviceRef.GPU() if USE_GPU else DeviceRef.CPU()
NHEADS = 8


# ------------------------------- weight helpers ----------------------------
class Builder:
    def __init__(self, sd: dict[str, np.ndarray]):
        self.sd = sd

    def k(self, key):  # raw numpy tensor
        return self.sd[key]

    def const(self, arr):
        return ops.constant(np.ascontiguousarray(arr, dtype=np.float32),
                            DType.float32, device=DEV)


# ------------------------------- primitives --------------------------------
def gelu(x):
    return ops.gelu(x, approximate="none")


def glu(x, axis):
    n = x.shape[axis]
    half = int(n) // 2
    a, b = ops.split(x, [half, half], axis=axis)
    return a * ops.sigmoid(b)


def groupnorm1(x, gamma_np, beta_np, eps=1e-5):
    """GroupNorm(num_groups=1): normalize over all non-batch dims, affine per channel
    (channel = axis 1). x: (B, C, *S)."""
    shp = [int(d) for d in x.shape]
    B, C = shp[0], shp[1]
    flat = ops.reshape(x, (B, -1))
    mu = ops.mean(flat, axis=1)                       # (B,1)
    cen = flat - mu
    var = ops.mean(cen * cen, axis=1)                 # (B,1)
    xn = cen * ops.rsqrt(var + eps)
    xn = ops.reshape(xn, shp)
    gshape = [1, C] + [1] * (len(shp) - 2)
    g = ops.constant(np.ascontiguousarray(gamma_np.reshape(gshape), np.float32), DType.float32, device=DEV)
    b = ops.constant(np.ascontiguousarray(beta_np.reshape(gshape), np.float32), DType.float32, device=DEV)
    return xn * g + b


def layernorm_last(x, gamma_np, beta_np, eps=1e-5):
    g = ops.constant(np.ascontiguousarray(gamma_np, np.float32), DType.float32, device=DEV)
    b = ops.constant(np.ascontiguousarray(beta_np, np.float32), DType.float32, device=DEV)
    return ops.layer_norm(x, g, b, eps)


def conv2d_t(bd: Builder, x, w_key, b_key, stride, pad):
    """torch-style conv2d. x: NCHW. w (O,I,kH,kW). pad=(pH,pW) symmetric."""
    w = bd.k(w_key)
    O = w.shape[0]
    wr = np.transpose(w, (2, 3, 1, 0))                # RSCF (kH,kW,I,O)
    xt = ops.permute(x, [0, 2, 3, 1])                 # NHWC
    y = ops.conv2d(xt, bd.const(wr), stride=stride,
                   padding=(pad[0], pad[0], pad[1], pad[1]))
    bias = bd.k(b_key).reshape(1, 1, 1, O)
    y = y + bd.const(bias)
    return ops.permute(y, [0, 3, 1, 2])               # NCHW


def conv1x1(bd: Builder, x, w_key, b_key):
    """1x1 conv (pure channel mix) via matmul. x: (N,C,*S). w (O,I[,1[,1]])."""
    w = bd.k(w_key)
    O, I = w.shape[0], w.shape[1]
    wmat = w.reshape(O, I)
    shp = [int(d) for d in x.shape]
    nd = len(shp)
    perm = [0] + list(range(2, nd)) + [1]              # move C to last
    xt = ops.permute(x, perm)
    M = 1
    for d in shp[2:]:
        M *= d
    xf = ops.reshape(xt, (shp[0] * M, I))
    y = ops.matmul(xf, bd.const(wmat.T)) + bd.const(bd.k(b_key).reshape(1, O))
    outshp = [shp[0]] + shp[2:] + [O]
    y = ops.reshape(y, outshp)
    inv = [0, nd - 1] + list(range(1, nd - 1))         # move C back to axis 1
    return ops.permute(y, inv)


def conv1d_t(bd: Builder, x, w_key, b_key, stride, pad, dilation=1):
    """torch-style conv1d. x: NCL. w (O,I,k). Inflates dilation into a dense kernel."""
    x4 = ops.unsqueeze(x, 2)                           # (N,C,1,L)
    w = bd.k(w_key)
    if dilation > 1:                                    # inflate: k -> (k-1)*dil+1
        k = w.shape[2]
        knew = (k - 1) * dilation + 1
        wi = np.zeros((w.shape[0], w.shape[1], knew), np.float32)
        wi[:, :, ::dilation] = w
        w = wi
    O = w.shape[0]
    wr = np.transpose(w[:, :, None, :], (2, 3, 1, 0))  # (1,k,I,O)
    xt = ops.permute(x4, [0, 2, 3, 1])                 # NHWC (N,1,L,C)
    y = ops.conv2d(xt, bd.const(wr), stride=(1, stride),
                   padding=(0, 0, pad, pad))
    y = y + bd.const(bd.k(b_key).reshape(1, 1, 1, O))
    y = ops.permute(y, [0, 3, 1, 2])                   # (N,O,1,Lout)
    return ops.squeeze(y, 2)


def conv_transpose_1axis(bd: Builder, x, w_key, b_key, stride):
    """ConvTranspose1d over (N,C,L) via dilate + flipped conv. w (Cin,Cout,k).
    The conv is oriented along the H axis with a (k,1) kernel — MAX's CPU conv2d
    mishandles wide (1,k) multichannel kernels, so we keep the spatial dim on H."""
    w = bd.k(w_key)
    Cin, Cout, k = w.shape
    shp = [int(d) for d in x.shape]
    N, C, L = shp
    # to NHWC with L on H: (N,C,L) -> (N,C,L,1) -> (N,L,1,C)
    x4 = ops.unsqueeze(x, 3)
    xt = ops.permute(x4, [0, 2, 3, 1])                    # (N,L,1,C)
    # dilate along H: (N,L,1,C) -> (N,L,1,1,C) pad-> (N,L,stride,1,C) -> (N,L*stride,1,C)
    xe = ops.unsqueeze(xt, 2)
    xe = ops.pad(xe, [0, 0, 0, 0, 0, stride - 1, 0, 0, 0, 0])
    xe = ops.reshape(xe, (N, L * stride, 1, C))
    if stride > 1:
        xe = xe[:, : (L - 1) * stride + 1, :, :]
    # conv: kernel (k,1), pad H by k-1, weight flipped along k & swapped to (Cout,Cin,k)
    wf = np.transpose(w[:, :, ::-1], (1, 0, 2)).copy()    # (Cout,Cin,k)
    wr = np.transpose(wf[:, :, :, None], (2, 3, 1, 0))    # (k,1,Cin,Cout)
    y = ops.conv2d(xe, bd.const(wr), stride=(1, 1), padding=(k - 1, k - 1, 0, 0))
    y = y + bd.const(bd.k(b_key).reshape(1, 1, 1, Cout))
    y = ops.permute(y, [0, 3, 1, 2])                      # (N,Cout,Hout,1)
    return ops.squeeze(y, 3)                              # (N,Cout,(L-1)*stride+k)


def conv_transpose_freq(bd: Builder, x, w_key, b_key, stride_h):
    """ConvTranspose2d for freq branch: kernel (kH,1) stride (stride_h,1) on
    (N,C,Fr,T). T is independent (kernel W=1) -> fold T into batch, do 1D along Fr."""
    shp = [int(d) for d in x.shape]
    N, C, Fr, T = shp
    # weight (Cin,Cout,kH,1) -> drop trailing 1 -> (Cin,Cout,kH)
    w = bd.k(w_key)
    bd.sd["__tmp_wfreq"] = w[:, :, :, 0]
    xr = ops.permute(x, [0, 3, 1, 2])                     # (N,T,C,Fr)
    xr = ops.reshape(xr, (N * T, C, Fr))
    y = conv_transpose_1axis(bd, xr, "__tmp_wfreq", b_key, stride_h)
    Fout = int(y.shape[-1])
    y = ops.reshape(y, (N, T, w.shape[1], Fout))
    return ops.permute(y, [0, 2, 3, 1])                   # (N,Cout,Fout,T)


# ------------------------------- DConv -------------------------------------
def dconv(bd: Builder, prefix, y):
    for d in range(2):
        p = f"{prefix}.layers.{d}"
        h = conv1d_t(bd, y, f"{p}.0.weight", f"{p}.0.bias", stride=1,
                     pad=2 ** d, dilation=2 ** d)
        h = groupnorm1(h, bd.k(f"{p}.1.weight"), bd.k(f"{p}.1.bias"))
        h = gelu(h)
        h = conv1x1(bd, h, f"{p}.3.weight", f"{p}.3.bias")
        h = groupnorm1(h, bd.k(f"{p}.4.weight"), bd.k(f"{p}.4.bias"))
        h = glu(h, axis=1)
        scale = bd.k(f"{p}.6.scale").reshape(1, -1, 1)
        h = h * bd.const(scale)
        y = y + h
    return y


# ------------------------------- enc / dec ---------------------------------
def henc(bd: Builder, prefix, x, freq):
    if freq:
        y = conv2d_t(bd, x, f"{prefix}.conv.weight", f"{prefix}.conv.bias",
                     stride=(4, 1), pad=(2, 0))
    else:
        le = int(x.shape[-1])
        if le % 4 != 0:                                # pad to multiple of stride
            x = ops.pad(x, [0, 0, 0, 0, 0, 4 - le % 4])
        y = conv1d_t(bd, x, f"{prefix}.conv.weight", f"{prefix}.conv.bias",
                     stride=4, pad=2)
    y = gelu(y)
    if freq:
        shp = [int(d) for d in y.shape]
        B, C, Fr, T = shp
        y = ops.reshape(ops.permute(y, [0, 2, 1, 3]), (B * Fr, C, T))
        y = dconv(bd, f"{prefix}.dconv", y)
        y = ops.permute(ops.reshape(y, (B, Fr, C, T)), [0, 2, 1, 3])
        z = conv1x1(bd, y, f"{prefix}.rewrite.weight", f"{prefix}.rewrite.bias")
    else:
        y = dconv(bd, f"{prefix}.dconv", y)
        z = conv1x1(bd, y, f"{prefix}.rewrite.weight", f"{prefix}.rewrite.bias")
    return glu(z, axis=1)


def hdec(bd: Builder, prefix, x, skip, length, freq, last):
    x = x + skip
    if freq:
        y = conv2d_t(bd, x, f"{prefix}.rewrite.weight", f"{prefix}.rewrite.bias",
                     stride=(1, 1), pad=(1, 1))
    else:
        y = conv1d_t(bd, x, f"{prefix}.rewrite.weight", f"{prefix}.rewrite.bias",
                     stride=1, pad=1)
    y = glu(y, axis=1)
    if freq:
        shp = [int(d) for d in y.shape]
        B, C, Fr, T = shp
        y = ops.reshape(ops.permute(y, [0, 2, 1, 3]), (B * Fr, C, T))
        y = dconv(bd, f"{prefix}.dconv", y)
        y = ops.permute(ops.reshape(y, (B, Fr, C, T)), [0, 2, 1, 3])
        z = conv_transpose_freq(bd, y, f"{prefix}.conv_tr.weight",
                                f"{prefix}.conv_tr.bias", stride_h=4)
        z = z[:, :, 2:-2, :]
    else:
        y = dconv(bd, f"{prefix}.dconv", y)
        z = conv_transpose_1axis(bd, y, f"{prefix}.conv_tr.weight",
                                 f"{prefix}.conv_tr.bias", stride=4)
        z = z[:, :, 2:2 + length]
    if not last:
        z = gelu(z)
    return z


# ------------------------------- transformer -------------------------------
def create_2d_sin_embedding(d_model, height, width, max_period=10000.0):
    pe = np.zeros((d_model, height, width), np.float32)
    d = d_model // 2
    div = np.exp(np.arange(0.0, d, 2) * -(math.log(max_period) / d))
    pos_w = np.arange(width)[:, None]
    pos_h = np.arange(height)[:, None]
    pe[0:d:2] = np.sin(pos_w * div).T[:, None, :]
    pe[1:d:2] = np.cos(pos_w * div).T[:, None, :]
    pe[d::2] = np.sin(pos_h * div).T[:, :, None]
    pe[d + 1::2] = np.cos(pos_h * div).T[:, :, None]
    return pe[None]


def create_sin_embedding(length, dim, max_period=10000.0):
    pos = np.arange(length)[:, None]
    half = dim // 2
    adim = np.arange(half)[None, :]
    phase = pos / (max_period ** (adim / (half - 1)))
    return np.concatenate([np.cos(phase), np.sin(phase)], axis=-1)[:, None].astype(np.float32)


def linear(bd: Builder, prefix, x, wkey="weight", bkey="bias"):
    """x (B,N,C) -> (B,N,out). torch weight (out,in)."""
    shp = [int(d) for d in x.shape]
    B, N, C = shp
    w = bd.k(f"{prefix}.{wkey}")
    out = w.shape[0]
    xf = ops.reshape(x, (B * N, C))
    y = ops.matmul(xf, bd.const(w.T))                 # (B*N, out)
    y = y + bd.const(bd.k(f"{prefix}.{bkey}").reshape(1, out))
    return ops.reshape(y, (B, N, out))


def mha(bd: Builder, prefix, q, k, v):
    dim = int(q.shape[-1])
    ipw = bd.k(f"{prefix}.in_proj_weight")
    ipb = bd.k(f"{prefix}.in_proj_bias")
    wq, wk, wv = ipw[:dim], ipw[dim:2 * dim], ipw[2 * dim:]
    bq, bk, bv = ipb[:dim], ipb[dim:2 * dim], ipb[2 * dim:]
    B, Nq = int(q.shape[0]), int(q.shape[1])
    Nk = int(k.shape[1])
    hd = dim // NHEADS

    def proj(t, wm, bm, N):
        tf = ops.reshape(t, (B * N, dim))
        o = ops.matmul(tf, bd.const(wm.T)) + bd.const(bm.reshape(1, dim))
        o = ops.reshape(o, (B, N, NHEADS, hd))
        return ops.permute(o, [0, 2, 1, 3])           # (B,h,N,hd)

    Q, K, V = proj(q, wq, bq, Nq), proj(k, wk, bk, Nk), proj(v, wv, bv, Nk)
    Q = ops.reshape(Q, (B * NHEADS, Nq, hd))
    K = ops.reshape(K, (B * NHEADS, Nk, hd))
    V = ops.reshape(V, (B * NHEADS, Nk, hd))
    att = ops.matmul(Q, ops.permute(K, [0, 2, 1])) * (1.0 / math.sqrt(hd))
    att = ops.softmax(att, axis=-1)
    o = ops.matmul(att, V)                            # (B*h,Nq,hd)
    o = ops.permute(ops.reshape(o, (B, NHEADS, Nq, hd)), [0, 2, 1, 3])
    o = ops.reshape(o, (B, Nq, dim))
    return linear(bd, f"{prefix}.out_proj", o)


def ffn(bd: Builder, prefix, x):
    h = gelu(linear(bd, f"{prefix}.linear1", x))
    return linear(bd, f"{prefix}.linear2", h)


def gscale(bd: Builder, prefix, x):
    s = bd.k(f"{prefix}.scale").reshape(1, 1, -1)
    return x * bd.const(s)


def self_layer(bd: Builder, prefix, x):
    n1 = layernorm_last(x, bd.k(f"{prefix}.norm1.weight"), bd.k(f"{prefix}.norm1.bias"))
    x = x + gscale(bd, f"{prefix}.gamma_1", mha(bd, f"{prefix}.self_attn", n1, n1, n1))
    n2 = layernorm_last(x, bd.k(f"{prefix}.norm2.weight"), bd.k(f"{prefix}.norm2.bias"))
    x = x + gscale(bd, f"{prefix}.gamma_2", ffn(bd, prefix, n2))
    return mygroupnorm(bd, f"{prefix}.norm_out", x)


def cross_layer(bd: Builder, prefix, q, k):
    nq = layernorm_last(q, bd.k(f"{prefix}.norm1.weight"), bd.k(f"{prefix}.norm1.bias"))
    nk = layernorm_last(k, bd.k(f"{prefix}.norm2.weight"), bd.k(f"{prefix}.norm2.bias"))
    x = q + gscale(bd, f"{prefix}.gamma_1", mha(bd, f"{prefix}.cross_attn", nq, nk, nk))
    n3 = layernorm_last(x, bd.k(f"{prefix}.norm3.weight"), bd.k(f"{prefix}.norm3.bias"))
    x = x + gscale(bd, f"{prefix}.gamma_2", ffn(bd, prefix, n3))
    return mygroupnorm(bd, f"{prefix}.norm_out", x)


def mygroupnorm(bd: Builder, prefix, x):
    # (B,N,C) -> (B,C,N) -> GN(1) -> back
    xt = ops.permute(x, [0, 2, 1])
    xt = groupnorm1(xt, bd.k(f"{prefix}.weight"), bd.k(f"{prefix}.bias"))
    return ops.permute(xt, [0, 2, 1])


def crosstransformer(bd: Builder, x, xt):
    shp = [int(d) for d in x.shape]
    B, C, Fr, T1 = shp
    pe2d = create_2d_sin_embedding(C, Fr, T1)[0]               # (C,Fr,T1)
    pe2d = np.transpose(pe2d, (2, 1, 0)).reshape(T1 * Fr, C)[None]
    x = ops.reshape(ops.permute(x, [0, 3, 2, 1]), (B, T1 * Fr, C))
    x = layernorm_last(x, bd.k("crosstransformer.norm_in.weight"),
                       bd.k("crosstransformer.norm_in.bias")) + bd.const(pe2d)

    Ct, T2 = int(xt.shape[1]), int(xt.shape[2])
    xt = ops.permute(xt, [0, 2, 1])                           # (B,T2,C)
    pe = create_sin_embedding(T2, Ct)[:, 0, :][None]
    xt = layernorm_last(xt, bd.k("crosstransformer.norm_in_t.weight"),
                        bd.k("crosstransformer.norm_in_t.bias")) + bd.const(pe)

    for idx in range(5):
        if idx % 2 == 0:
            x = self_layer(bd, f"crosstransformer.layers.{idx}", x)
            xt = self_layer(bd, f"crosstransformer.layers_t.{idx}", xt)
        else:
            old_x = x
            x = cross_layer(bd, f"crosstransformer.layers.{idx}", x, xt)
            xt = cross_layer(bd, f"crosstransformer.layers_t.{idx}", xt, old_x)
    x = ops.permute(ops.reshape(x, (B, T1, Fr, C)), [0, 3, 2, 1])
    xt = ops.permute(xt, [0, 2, 1])
    return x, xt


# ------------------------------- full core ---------------------------------
def build_core(sd, x_shape, xt_shape):
    """Graph: (x_norm, xt_norm) -> (x_pre, xt_pre)."""
    bd = Builder(dict(sd))
    g = Graph("htdemucs_core", input_types=[
        TensorType(DType.float32, x_shape, device=DEV),
        TensorType(DType.float32, xt_shape, device=DEV),
    ])
    with g:
        x, xt = g.inputs
        x, xt = x.tensor, xt.tensor
        saved, saved_t, lengths_t = [], [], []
        for idx in range(4):
            lengths_t.append(int(xt.shape[-1]))
            xt = henc(bd, f"tencoder.{idx}", xt, freq=False)
            saved_t.append(xt)
            x = henc(bd, f"encoder.{idx}", x, freq=True)
            if idx == 0:
                emb = (bd.k("freq_emb.embedding.weight") * 10.0)  # (512,48)
                Fr = int(x.shape[-2])
                emb = emb.T[None, :, :, None]                      # (1,48,512,1)
                x = x + 0.2 * bd.const(emb)
            saved.append(x)

        # channel upsample 384->512
        b, c, f, t = [int(d) for d in x.shape]
        xf = ops.reshape(x, (b, c, f * t))
        xf = conv1x1(bd, xf, "channel_upsampler.weight", "channel_upsampler.bias")
        x = ops.reshape(xf, (b, -1, f, t))
        xt = conv1x1(bd, xt, "channel_upsampler_t.weight", "channel_upsampler_t.bias")

        x, xt = crosstransformer(bd, x, xt)

        xf = ops.reshape(x, (b, int(x.shape[1]), f * t))
        xf = conv1x1(bd, xf, "channel_downsampler.weight", "channel_downsampler.bias")
        x = ops.reshape(xf, (b, -1, f, t))
        xt = conv1x1(bd, xt, "channel_downsampler_t.weight", "channel_downsampler_t.bias")

        for idx in range(4):
            skip = saved.pop(-1)
            x = hdec(bd, f"decoder.{idx}", x, skip, None, freq=True, last=(idx == 3))
            skip_t = saved_t.pop(-1)
            length_t = lengths_t.pop(-1)
            xt = hdec(bd, f"tdecoder.{idx}", xt, skip_t, length_t, freq=False, last=(idx == 3))

        g.output(x, xt)
    return g
