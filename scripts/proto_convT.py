"""Verify a conv2d-based transposed-conv (since conv2d_transpose won't lower on CPU).
Also probe whether the native op compiles on the Metal GPU.

Run: uv run python scripts/proto_convT.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from max.graph import Graph, TensorType, DeviceRef, ops
from max.dtype import DType
from max.engine import InferenceSession
from max.driver import CPU, Accelerator

CPU_DEV = DeviceRef.CPU()


def run(graph, x_np, dev):
    sess = InferenceSession(devices=[dev])
    model = sess.load(graph)
    return [o.to_numpy() for o in model.execute(x_np)]


def convT_via_conv(x, w_np, stride_h):
    """ConvTranspose2d along H (kernel (kH,1), stride (stride_h,1), no pad) via conv2d.
    x: NHWC TensorValue. w_np torch layout (Cin, Cout, kH, 1)."""
    Cin, Cout, kH, kW = w_np.shape
    # dilate H by inserting (stride_h-1) zeros between rows.
    # x is NHWC: (N,H,W,Cin). Work on H axis=1.
    N = x.shape[0]
    H = x.shape[1]
    Wd = x.shape[2]
    xe = ops.unsqueeze(x, 2)                       # (N,H,1,W,Cin)
    xe = ops.pad(xe, [0, 0, 0, 0, 0, stride_h - 1, 0, 0, 0, 0])  # pad axis2 -> (N,H,s,W,Cin)
    newH = H * stride_h
    xe = ops.reshape(xe, (N, newH, Wd, Cin))       # (N, H*s, W, Cin)
    # crop trailing s-1 -> (H-1)*s + 1
    xe = xe[:, : (H - 1) * stride_h + 1, :, :]
    # regular conv: pad H by kH-1 each side, kernel = flip(w) along H, swap in/out
    wf = w_np[:, :, ::-1, :].copy()                # flip kH
    wf = np.transpose(wf, (1, 0, 2, 3))            # (Cout, Cin, kH, 1) torch conv layout
    wmax = np.transpose(wf, (2, 3, 1, 0))          # RSCF (kH,1,Cin,Cout)
    wc = ops.constant(np.ascontiguousarray(wmax), DType.float32, device=CPU_DEV)
    y = ops.conv2d(xe, wc, stride=(1, 1), padding=(kH - 1, kH - 1, 0, 0))
    return y                                        # NHWC (N, outH, W, Cout)


def main():
    np.random.seed(0)
    N, Cin, H, W = 1, 6, 8, 5
    Cout, kH = 4, 8
    stride_h = 4
    x_np = np.random.randn(N, Cin, H, W).astype(np.float32)
    w_np = np.random.randn(Cin, Cout, kH, 1).astype(np.float32)
    ref = F.conv_transpose2d(torch.from_numpy(x_np), torch.from_numpy(w_np),
                             stride=(stride_h, 1)).numpy()  # (N,Cout,outH,W)
    print("ref shape", ref.shape)

    with Graph("cT", input_types=[TensorType(DType.float32, (N, Cin, H, W), device=CPU_DEV)]) as g:
        (x,) = g.inputs
        xnhwc = ops.permute(x.tensor, [0, 2, 3, 1])
        y = convT_via_conv(xnhwc, w_np, stride_h)
        y = ops.permute(y, [0, 3, 1, 2])           # -> NCHW
        g.output(y)
    out = run(g, x_np, CPU())[0]
    print("conv2d-based convT shape", out.shape, "maxerr", np.abs(out - ref).max())

    # probe native op on GPU
    try:
        gdev = DeviceRef.GPU()
        with Graph("cTg", input_types=[TensorType(DType.float32, (N, Cin, H, W), device=gdev)]) as g2:
            (x,) = g2.inputs
            xnhwc = ops.permute(x.tensor, [0, 2, 3, 1])
            wmax = np.transpose(w_np, (2, 3, 1, 0))
            wc = ops.constant(np.ascontiguousarray(wmax), DType.float32, device=gdev)
            yy = ops.conv2d_transpose(xnhwc, wc, stride=(stride_h, 1))
            g2.output(ops.permute(yy, [0, 3, 1, 2]))
        out2 = run(g2, x_np, Accelerator())[0]
        print("native GPU convT shape", out2.shape, "maxerr", np.abs(out2 - ref).max())
    except Exception as e:
        print("native GPU convT FAILED:", repr(e)[:300])


if __name__ == "__main__":
    main()
