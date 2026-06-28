# `ops.conv2d` on GPU runs at ~0.1% of fp32 peak (no cuDNN/tensor cores?)

## Summary

Forward `ops.conv2d` on an NVIDIA A10 sustains only **~30 GMAC/s in fp32**
(~0.1% of the card's ~31 TFLOP/s fp32 peak) across every shape I measured —
including "nice" square, high-channel convs. The same model in PyTorch (cuDNN)
runs the full network ~16× faster. bf16 gives ~3.7× on the dominant shape but is
still far from tensor-core throughput. This looks like `conv2d` is using an
unoptimized kernel rather than cuDNN / tensor cores.

## Environment

- MAX / modular **26.2.0**, NVIDIA **A10** (driver 580.105.08)
- CUDA **13.0**, cuDNN **9.20.0.48** (`nvidia-cudnn-cu13`)
- Linux x86_64, Python 3.13

## Measurements

Bare `ops.conv2d` (NHWC in/out, constant filter, no surrounding permutes),
stride 1, 3×3, timed over 30 iters incl. device→host sync:

| in→out ch | spatial | dtype | latency | throughput |
|---|---|---|---|---|
| 48→96   | 512×336 | fp32 | 218 ms | 33 GMAC/s |
| 384→768 | 8×336   | fp32 | 225 ms | 32 GMAC/s |
| 256→256 | 64×64   | fp32 |  96 ms | 25 GMAC/s |
| 512→512 | 32×32   | fp32 | 378 ms |  6 GMAC/s |
| 48→96   | 512×336 | bf16 |  59 ms | 122 GMAC/s |

GPU is at 100% utilization during these — it is busy, just inefficient. The
throughput ceiling (~30 GMAC/s fp32, ~120 GMAC/s bf16) is far below what cuDNN
delivers on this card, suggesting `conv2d` is not dispatching to cuDNN /
tensor-core kernels. (Note `conv2d_transpose` *does* appear to hit cuDNN — it
aborts with a cuDNN status error; see the companion issue.)

## Repro

```python
import time, numpy as np
from max.graph import Graph, TensorType, DeviceRef, ops
from max.dtype import DType
from max.engine import InferenceSession
from max.driver import Accelerator, CPU, Buffer

dev, host, GDEV = Accelerator(), CPU(), DeviceRef.GPU()
N, H, W, I, O, k = 1, 512, 336, 48, 96, 3
wr = np.random.randn(k, k, I, O).astype(np.float32)
with Graph("c", input_types=[TensorType(DType.float32, (N, H, W, I), device=GDEV)]) as g:
    (x,) = g.inputs
    g.output(ops.conv2d(x.tensor, ops.constant(wr, DType.float32, device=GDEV),
                        stride=(1, 1), padding=(1, 1, 1, 1)))
m = InferenceSession(devices=[dev]).load(g)
b = Buffer.from_numpy(np.random.randn(N, H, W, I).astype(np.float32)).to(dev)
for _ in range(5): [o.to(host).to_numpy() for o in m.execute(b)]
t0 = time.time()
for _ in range(30): [o.to(host).to_numpy() for o in m.execute(b)]
dt = (time.time() - t0) / 30
print(f"{dt*1e3:.1f} ms  {N*H*W*O*I*k*k/1e9/dt:.0f} GMAC/s")
```

## Ask

- Is forward `conv2d` expected to dispatch to cuDNN / tensor-core kernels on
  NVIDIA GPUs in 26.2? If so, why is throughput ~0.1% of peak here?
- Any flag / dtype / layout needed to get the tuned conv path?
