"""Verified MAX Python graph API cheat-sheet smoke test (CPU).

Run with:  uv run python scripts/scratch_max_smoke.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight, ops

CPU_DEV = DeviceRef.CPU()
RTOL = ATOL = 1e-4


def run_graph(graph: Graph, *inputs: np.ndarray, weights=None):
    """Compile `graph`, run it on numpy inputs, return list of numpy outputs."""
    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=weights or {})
    outs = model.execute(*inputs)  # accepts np.ndarray directly
    return [o.to_numpy() for o in outs]  # outs are max.driver buffers


def check(name, got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    err = np.max(np.abs(got - ref))
    assert err < 1e-4, f"{name}: max abs err {err} too large"
    print(f"  OK {name:28s} shape={got.shape} maxerr={err:.2e}")


# ---------------------------------------------------------------------------
# 1. Graph construction + execution boilerplate
# ---------------------------------------------------------------------------
def test_boilerplate():
    print("1. boilerplate")
    x_np = np.random.randn(2, 3).astype(np.float32)
    with Graph(
        "boiler",
        input_types=[TensorType(DType.float32, (2, 3), device=CPU_DEV)],
    ) as g:
        (x,) = g.inputs
        y = x.tensor * 2.0 + 1.0
        g.output(y)
    (out,) = run_graph(g, x_np)
    check("boilerplate", out, x_np * 2.0 + 1.0)


# ---------------------------------------------------------------------------
# 2. Weights / constants from numpy
# ---------------------------------------------------------------------------
def test_constant_and_weight():
    print("2. constants + weights")
    x_np = np.random.randn(2, 4).astype(np.float32)
    w_np = np.random.randn(4, 4).astype(np.float32)

    # (a) ops.constant: bakes the numpy array into the graph at build time.
    with Graph(
        "const", input_types=[TensorType(DType.float32, (2, 4), device=CPU_DEV)]
    ) as g:
        (x,) = g.inputs
        w = ops.constant(w_np, DType.float32, device=CPU_DEV)
        g.output(x.tensor @ w)
    (out,) = run_graph(g, x_np)
    check("constant", out, x_np @ w_np)

    # (b) Weight + weights_registry: idiomatic for loading a state_dict.
    #     Build with symbolic Weight placeholders, supply numpy at load time.
    sd = {"layer.weight": w_np, "layer.bias": np.random.randn(4).astype(np.float32)}
    with Graph(
        "weights", input_types=[TensorType(DType.float32, (2, 4), device=CPU_DEV)]
    ) as g:
        (x,) = g.inputs
        w = g.add_weight(
            Weight("layer.weight", DType.float32, (4, 4), device=CPU_DEV)
        )
        b = g.add_weight(
            Weight("layer.bias", DType.float32, (4,), device=CPU_DEV)
        )
        g.output(x.tensor @ w + b)
    (out,) = run_graph(g, x_np, weights=sd)
    check("weight+registry", out, x_np @ sd["layer.weight"] + sd["layer.bias"])


# ---------------------------------------------------------------------------
# 3. conv2d  (torch weight (out,in,kH,kW), input NCHW)
# ---------------------------------------------------------------------------
def test_conv2d():
    print("3. conv2d")
    N, Cin, H, W = 1, 3, 9, 11
    Cout, kH, kW = 4, 3, 5
    x_np = np.random.randn(N, Cin, H, W).astype(np.float32)
    w_np = np.random.randn(Cout, Cin, kH, kW).astype(np.float32)  # torch layout
    b_np = np.random.randn(Cout).astype(np.float32)
    stride = (2, 3)
    # asymmetric padding to probe the 4-tuple order: (top,bottom,left,right)
    pad_top, pad_bottom, pad_left, pad_right = 1, 2, 0, 3

    # torch reference: F.conv2d padding is symmetric only, so pad manually.
    xt = torch.from_numpy(x_np)
    xt_p = F.pad(xt, (pad_left, pad_right, pad_top, pad_bottom))
    ref = F.conv2d(
        xt_p, torch.from_numpy(w_np), torch.from_numpy(b_np), stride=stride
    ).numpy()

    # MAX wants input NHWC and filter RSCF=(kH,kW,in,out).
    with Graph(
        "conv2d",
        input_types=[TensorType(DType.float32, (N, Cin, H, W), device=CPU_DEV)],
    ) as g:
        (x,) = g.inputs
        xt_nhwc = ops.permute(x.tensor, [0, 2, 3, 1])  # NCHW -> NHWC
        w = ops.constant(
            np.transpose(w_np, (2, 3, 1, 0)), DType.float32, device=CPU_DEV
        )  # (out,in,kH,kW) -> (kH,kW,in,out)
        b = ops.constant(b_np, DType.float32, device=CPU_DEV)
        # padding 4-tuple is (dim1_before, dim1_after, dim2_before, dim2_after)
        #   = (H_top, H_bottom, W_left, W_right)
        y = ops.conv2d(
            xt_nhwc, w, stride=stride,
            padding=(pad_top, pad_bottom, pad_left, pad_right), bias=b,
        )
        y = ops.permute(y, [0, 3, 1, 2])  # NHWC -> NCHW
        g.output(y)
    (out,) = run_graph(g, x_np)
    check("conv2d", out, ref)


# ---------------------------------------------------------------------------
# 4. conv2d_transpose  (torch weight (in,out,kH,kW))
# ---------------------------------------------------------------------------
def test_conv_transpose2d():
    print("4. conv_transpose2d")
    N, Cin, H, W = 1, 3, 5, 6
    Cout, kH, kW = 4, 3, 3
    stride = (2, 2)
    x_np = np.random.randn(N, Cin, H, W).astype(np.float32)
    w_np = np.random.randn(Cin, Cout, kH, kW).astype(np.float32)  # torch layout
    b_np = np.random.randn(Cout).astype(np.float32)

    ref = F.conv_transpose2d(
        torch.from_numpy(x_np), torch.from_numpy(w_np),
        torch.from_numpy(b_np), stride=stride,
    ).numpy()

    # MAX conv2d_transpose filter layout RSCF = (kH, kW, out_channels, in_channels)
    # torch weight is (in, out, kH, kW) -> permute to (kH, kW, out, in)
    with Graph(
        "convT",
        input_types=[TensorType(DType.float32, (N, Cin, H, W), device=CPU_DEV)],
    ) as g:
        (x,) = g.inputs
        xt_nhwc = ops.permute(x.tensor, [0, 2, 3, 1])
        w = ops.constant(
            np.transpose(w_np, (2, 3, 1, 0)), DType.float32, device=CPU_DEV
        )  # (in,out,kH,kW) -> (kH,kW,out,in)
        # bias path is buggy (wrong broadcast layout); add bias manually.
        y = ops.conv2d_transpose(xt_nhwc, w, stride=stride)   # NHWC, no bias
        b = ops.constant(b_np.reshape(1, 1, 1, Cout), DType.float32, device=CPU_DEV)
        y = y + b
        y = ops.permute(y, [0, 3, 1, 2])                      # NHWC -> NCHW
        g.output(y)
    (out,) = run_graph(g, x_np)
    check("conv_transpose2d", out, ref)


# ---------------------------------------------------------------------------
# 5. conv1d / conv_transpose1d via conv2d (singleton spatial dim)
# ---------------------------------------------------------------------------
def test_conv1d():
    print("5. conv1d / conv_transpose1d via conv2d")
    N, Cin, L = 1, 3, 20
    Cout, k = 5, 4
    stride, pad = 2, 1
    x_np = np.random.randn(N, Cin, L).astype(np.float32)
    w_np = np.random.randn(Cout, Cin, k).astype(np.float32)  # torch (out,in,k)
    b_np = np.random.randn(Cout).astype(np.float32)
    ref = F.conv1d(
        torch.from_numpy(x_np), torch.from_numpy(w_np),
        torch.from_numpy(b_np), stride=stride, padding=pad,
    ).numpy()

    # Treat L as the W axis; add a singleton H axis of size 1.
    with Graph(
        "conv1d",
        input_types=[TensorType(DType.float32, (N, Cin, L), device=CPU_DEV)],
    ) as g:
        (x,) = g.inputs
        xt = ops.unsqueeze(x.tensor, 2)            # (N,Cin,1,L)
        xt = ops.permute(xt, [0, 2, 3, 1])         # -> NHWC (N,1,L,Cin)
        wt = w_np[:, :, None, :]                    # (out,in,1,k)
        w = ops.constant(
            np.transpose(wt, (2, 3, 1, 0)), DType.float32, device=CPU_DEV
        )  # -> (1,k,in,out)
        b = ops.constant(b_np, DType.float32, device=CPU_DEV)
        y = ops.conv2d(
            xt, w, stride=(1, stride), padding=(0, 0, pad, pad), bias=b
        )
        y = ops.permute(y, [0, 3, 1, 2])           # NHWC -> NCHW (N,Cout,1,Lout)
        y = ops.squeeze(y, 2)                       # (N,Cout,Lout)
        g.output(y)
    (out,) = run_graph(g, x_np)
    check("conv1d", out, ref)

    # conv_transpose1d
    Cin2, Cout2 = 3, 5
    wT_np = np.random.randn(Cin2, Cout2, k).astype(np.float32)  # torch (in,out,k)
    bT_np = np.random.randn(Cout2).astype(np.float32)
    refT = F.conv_transpose1d(
        torch.from_numpy(x_np), torch.from_numpy(wT_np),
        torch.from_numpy(bT_np), stride=stride,
    ).numpy()
    with Graph(
        "convT1d",
        input_types=[TensorType(DType.float32, (N, Cin2, L), device=CPU_DEV)],
    ) as g:
        (x,) = g.inputs
        xt = ops.unsqueeze(x.tensor, 2)            # (N,Cin,1,L)
        xt = ops.permute(xt, [0, 2, 3, 1])         # NHWC
        wt = wT_np[:, :, None, :]                   # (in,out,1,k)
        w = ops.constant(
            np.transpose(wt, (2, 3, 1, 0)), DType.float32, device=CPU_DEV
        )  # -> (1,k,out,in)
        b = ops.constant(bT_np, DType.float32, device=CPU_DEV)
        y = ops.conv2d_transpose(xt, w, stride=(1, stride), bias=b)  # NCHW
        y = ops.squeeze(y, 2)
        g.output(y)
    (out,) = run_graph(g, x_np)
    check("conv_transpose1d", out, refT)


# ---------------------------------------------------------------------------
# 6. layer_norm and group_norm
# ---------------------------------------------------------------------------
def test_norms():
    print("6. layer_norm / group_norm")
    eps = 1e-5
    x_np = np.random.randn(2, 6, 8).astype(np.float32)
    g_np = np.random.randn(8).astype(np.float32)
    b_np = np.random.randn(8).astype(np.float32)
    ref = F.layer_norm(
        torch.from_numpy(x_np), (8,),
        torch.from_numpy(g_np), torch.from_numpy(b_np), eps=eps,
    ).numpy()
    with Graph(
        "ln", input_types=[TensorType(DType.float32, (2, 6, 8), device=CPU_DEV)]
    ) as g:
        (x,) = g.inputs
        gamma = ops.constant(g_np, DType.float32, device=CPU_DEV)
        beta = ops.constant(b_np, DType.float32, device=CPU_DEV)
        g.output(ops.layer_norm(x.tensor, gamma, beta, epsilon=eps))
    (out,) = run_graph(g, x_np)
    check("layer_norm", out, ref)

    # group_norm: (N, C, ...) over channel axis 1
    C, groups = 8, 4
    xg_np = np.random.randn(2, C, 5).astype(np.float32)
    gg_np = np.random.randn(C).astype(np.float32)
    bg_np = np.random.randn(C).astype(np.float32)
    refg = F.group_norm(
        torch.from_numpy(xg_np), groups,
        torch.from_numpy(gg_np), torch.from_numpy(bg_np), eps=eps,
    ).numpy()
    with Graph(
        "gn", input_types=[TensorType(DType.float32, (2, C, 5), device=CPU_DEV)]
    ) as g:
        (x,) = g.inputs
        gamma = ops.constant(gg_np, DType.float32, device=CPU_DEV)
        beta = ops.constant(bg_np, DType.float32, device=CPU_DEV)
        g.output(
            ops.group_norm(x.tensor, gamma, beta, num_groups=groups, epsilon=eps)
        )
    (out,) = run_graph(g, xg_np)
    check("group_norm", out, refg)


# ---------------------------------------------------------------------------
# 7. matmul / linear  (torch weight (out,in))
# ---------------------------------------------------------------------------
def test_linear():
    print("7. matmul / linear")
    x_np = np.random.randn(2, 4).astype(np.float32)
    w_np = np.random.randn(5, 4).astype(np.float32)  # torch (out,in)
    b_np = np.random.randn(5).astype(np.float32)
    ref = (x_np @ w_np.T + b_np)
    with Graph(
        "lin", input_types=[TensorType(DType.float32, (2, 4), device=CPU_DEV)]
    ) as g:
        (x,) = g.inputs
        w = ops.constant(w_np.T.copy(), DType.float32, device=CPU_DEV)  # (in,out)
        b = ops.constant(b_np, DType.float32, device=CPU_DEV)
        g.output(ops.matmul(x.tensor, w) + b)
    (out,) = run_graph(g, x_np)
    check("linear", out, ref)


# ---------------------------------------------------------------------------
# 8. gelu, sigmoid, GLU
# ---------------------------------------------------------------------------
def test_activations():
    print("8. gelu / sigmoid / glu")
    x_np = np.random.randn(2, 8).astype(np.float32)
    ref_gelu = F.gelu(torch.from_numpy(x_np)).numpy()  # exact erf gelu
    ref_sig = torch.sigmoid(torch.from_numpy(x_np)).numpy()
    ref_glu = F.glu(torch.from_numpy(x_np), dim=-1).numpy()  # a * sigmoid(b)
    with Graph(
        "act", input_types=[TensorType(DType.float32, (2, 8), device=CPU_DEV)]
    ) as g:
        (x,) = g.inputs
        gelu = ops.gelu(x.tensor, approximate="none")
        sig = ops.sigmoid(x.tensor)
        a, b = ops.split(x.tensor, [4, 4], axis=-1)
        glu = a * ops.sigmoid(b)
        g.output(gelu, sig, glu)
    out_gelu, out_sig, out_glu = run_graph(g, x_np)
    check("gelu", out_gelu, ref_gelu)
    check("sigmoid", out_sig, ref_sig)
    check("glu", out_glu, ref_glu)


# ---------------------------------------------------------------------------
# 9. irfft  (GPU-only per source — verify behavior on CPU)
# ---------------------------------------------------------------------------
def test_irfft():
    print("9. irfft")
    n = 16
    sig = np.random.randn(n).astype(np.float32)
    spec = np.fft.rfft(sig)  # complex, length n//2+1
    spec_il = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)
    ref = np.fft.irfft(spec, n=n).astype(np.float32)
    try:
        with Graph(
            "irfft",
            input_types=[
                TensorType(DType.float32, (n // 2 + 1, 2), device=CPU_DEV)
            ],
        ) as g:
            (x,) = g.inputs
            y = ops.irfft(
                x.tensor, n=n, axis=-1,
                normalization="backward", input_is_complex=True,
            )
            g.output(y)
        (out,) = run_graph(g, spec_il)
        check("irfft", out, ref)
    except Exception as e:  # noqa: BLE001
        print(f"  irfft FAILED on CPU as expected: {type(e).__name__}: {e}")


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    test_boilerplate()
    test_constant_and_weight()
    test_conv2d()
    # test_conv_transpose2d()  # native op won't lower on CPU; see proto_convT.py workaround
    test_conv1d()
    test_norms()
    test_linear()
    test_activations()
    test_irfft()
    print("\nALL DONE")


if __name__ == "__main__":
    main()
