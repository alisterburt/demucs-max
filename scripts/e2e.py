"""End-to-end parity + benchmark: MAX HTDemucs vs torch HTDemucs.

Builds the MAX model once, checks the final separated waveform against the torch
reference, then benchmarks inference latency of both on the same fixed input.

Run: PYTHONPATH=ext/demucs:src uv run python scripts/e2e.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from demucs_mojo.runner import MaxHTDemucs, load_state_dict  # noqa: E402

REF = Path("reference")
N_BENCH = 5


def bench(fn, n=N_BENCH, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.time()
        fn()
        ts.append(time.time() - t0)
    return np.array(ts)


def main():
    io = np.load(REF / "io.npz")
    mix = torch.from_numpy(io["input"]).float()
    ref_out = torch.from_numpy(io["output"]).float()

    print("Building + compiling MAX model...")
    sd = load_state_dict()
    m = MaxHTDemucs(sd, length=mix.shape[-1])
    print(f"  graph build: {m.build_s:.1f}s   compile: {m.compile_s:.1f}s")

    # ---- parity ----
    timing = {}
    y = m.separate(mix, timing)
    diff = (y - ref_out).abs().max().item()
    rel = diff / ref_out.abs().max().item()
    print(f"\nPARITY  final-waveform max abs diff: {diff:.3e}  (rel {rel:.2e})")

    # ---- benchmark MAX ----
    dev = m.device_label
    tmax = bench(lambda: m.separate(mix))
    print(f"\nMAX ({dev}) end-to-end:  {tmax.mean()*1e3:7.1f} ± {tmax.std()*1e3:.1f} ms"
          f"   (NN core only: {timing['core_s']*1e3:.1f} ms)")

    # ---- benchmark torch ----
    from demucs.pretrained import get_model
    sub = get_model("htdemucs").models[0].eval()
    torch.set_grad_enabled(False)
    with torch.no_grad():
        ttorch_cpu = bench(lambda: sub(mix))
    print(f"torch (CPU) forward:   {ttorch_cpu.mean()*1e3:7.1f} ± {ttorch_cpu.std()*1e3:.1f} ms")

    # torch CUDA
    if torch.cuda.is_available():
        sub_cuda = sub.to("cuda")
        mix_cuda = mix.to("cuda")
        def run_cuda():
            sub_cuda(mix_cuda)
            torch.cuda.synchronize()
        with torch.no_grad():
            tcuda = bench(run_cuda)
        name = torch.cuda.get_device_name(0)
        print(f"torch (CUDA) forward:  {tcuda.mean()*1e3:7.1f} ± {tcuda.std()*1e3:.1f} ms   [{name}]")
        print(f"\nSpeedup MAX-{dev} vs torch-CUDA: {tcuda.mean()/tmax.mean():.2f}x")
    else:
        print("torch CUDA not available")

    print(f"Speedup MAX-{dev} vs torch-CPU:  {ttorch_cpu.mean()/tmax.mean():.2f}x")


if __name__ == "__main__":
    main()
