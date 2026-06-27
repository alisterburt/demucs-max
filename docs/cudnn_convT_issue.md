# `conv2d_transpose` aborts on GPU: `ABORT: invalid cudnnStatus_t entry` (CUDA 13 / cuDNN 9.20)

## Summary

`ops.conv2d_transpose` hard-aborts the process on an NVIDIA GPU with the message
`ABORT: invalid cudnnStatus_t entry` for *any* input shape I tried, including a
tiny one. Regular `ops.conv2d` on the same device/session works correctly, so
this is specific to the transposed-conv lowering. The abort message strongly
suggests MAX is receiving a `cudnnStatus_t` value it cannot map to its known
enum — i.e. a cuDNN version-skew or an ungracefully-surfaced cuDNN error.

## Environment

- MAX / modular: **26.2.0** (`max`, `max-core`, `max-mojo-libs`, `modular` all 26.2.0)
- GPU: **NVIDIA A10** (driver 580.105.08)
- CUDA: **13.0** (`cuda-toolkit` 13.0.2, `nvidia-cuda-runtime` 13.0.96)
- cuDNN: **9.20.0.48** (`nvidia-cudnn-cu13`, pulled in by torch 2.12.1+cu130)
- cuBLAS: `nvidia-cublas` 13.1.1.3
- OS: Linux x86_64, Python 3.13
- MAX does not declare a cuDNN dependency; it dynamically loads the
  `libcudnn.so.9` present in the environment (here, cuDNN 9.20 for CUDA 13).

## Repro

```python
import numpy as np, torch
import torch.nn.functional as F
from max.graph import Graph, TensorType, DeviceRef, ops
from max.dtype import DType
from max.engine import InferenceSession
from max.driver import Accelerator, CPU, Buffer

dev, host, GDEV = Accelerator(), CPU(), DeviceRef.GPU()

# small ConvTranspose2d along H: (Cin,Cout,kH,1), stride (4,1), no pad
N, Cin, H, W, Cout, kH, s = 1, 6, 8, 5, 4, 8, 4
x = np.random.randn(N, Cin, H, W).astype(np.float32)
w = np.random.randn(Cin, Cout, kH, 1).astype(np.float32)
ref = F.conv_transpose2d(torch.from_numpy(x), torch.from_numpy(w), stride=(s, 1)).numpy()

with Graph("cTg", input_types=[TensorType(DType.float32, (N, Cin, H, W), device=GDEV)]) as g:
    (xx,) = g.inputs
    xn = ops.permute(xx.tensor, [0, 2, 3, 1])                       # NHWC
    wc = ops.constant(np.ascontiguousarray(np.transpose(w, (2, 3, 1, 0))),
                      DType.float32, device=GDEV)                   # RSCF
    yy = ops.conv2d_transpose(xn, wc, stride=(s, 1))
    g.output(ops.permute(yy, [0, 3, 1, 2]))

m = InferenceSession(devices=[dev]).load(g)
b = Buffer.from_numpy(x).to(dev)
out = [o.to(host).to_numpy() for o in m.execute(b)][0]
print(np.abs(out - ref).max())
```

### Actual

```
ABORT: invalid cudnnStatus_t entry
```

The process aborts (no Python traceback — it is a hard abort from MAX's native
CUDA/cuDNN layer). The same abort occurs for larger, "realistic" shapes
(e.g. `(1, 384, 1344, 1)` with `Cout=192, kH=8, stride=4`).

### Expected

The op runs and matches `torch.nn.functional.conv_transpose2d` to fp32 tolerance,
as the equivalent `conv2d` does.

## Notes / analysis

- Plain `ops.conv2d` (forward) works correctly and at full GPU utilization on the
  same machine/session, so cuDNN is loaded and functioning; only the transposed
  path aborts.
- `ABORT: invalid cudnnStatus_t entry` reads like MAX converting an integer
  status returned by cuDNN into its `cudnnStatus_t` enum and finding the value
  out of range. cuDNN 9.x reorganized/extended `cudnnStatus_t` (new,
  non-contiguous status codes). A MAX build that predates cuDNN 9.20's status
  set would not recognize a newly-returned code — most likely an underlying
  error status (e.g. `CUDNN_STATUS_NOT_SUPPORTED`) for the transpose descriptor.
- Two things would help users here:
  1. Map unknown `cudnnStatus_t` values to a descriptive error instead of
     aborting (surface the raw integer + the failing op), so this is debuggable.
  2. Confirm the supported cuDNN/CUDA matrix for MAX 26.2 — is CUDA 13 /
     cuDNN 9.20 supported for `conv2d_transpose`, and if not, what is?

## Workaround

We reformulated the transposed convolution as a stride-1 `conv2d` plus a
channel→space ("pixel shuffle" / sub-pixel) reshape, which uses only the working
`conv2d` op and avoids `conv2d_transpose` entirely.
