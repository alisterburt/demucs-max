# demucs-max

Reimplementation of **HTDemucs v4** (Hybrid Transformer Demucs, the default
`htdemucs` music source-separation model) in **Modular MAX**, with a numerical
parity check and a speed comparison against the original PyTorch model.

## Status

- ✅ Full conv/transformer network ported to the MAX graph API (`max.graph`).
- ✅ Loads the real pretrained checkpoint; **reproduces torch output to 1.5e-4**
  (relative 4.6e-5) end-to-end on CPU.
- ✅ Benchmarked on CPU (Apple M4 Pro).
- ✅ **Runs on NVIDIA GPU** (A10, CUDA 13). Device selectable via
  `DEMUCS_MAX_DEVICE=cpu|gpu` (default `gpu`).

### Benchmark — CPU (input = 343980 samples ≈ 7.8 s @ 44.1 kHz, batch 1)

| backend | latency |
|---|---|
| **MAX (CPU, M4 Pro)** | **1965 ± 42 ms** |
| torch (CPU) | 2001 ± 52 ms |
| torch (MPS / Apple GPU) | 407 ± 6 ms |

MAX-CPU matches torch-CPU (1.02×). MAX one-time graph compile is ~240 s
(dominated by constant-folding the 42 M weights; can be cut with runtime weights).
MAX caches the compiled graph, so subsequent runs start in <1 s.

### Benchmark — GPU (NVIDIA A10, CUDA 13)

| backend | latency | parity (rel) |
|---|---|---|
| MAX (GPU) — initial port | 2349 ± 9 ms | 1.25e-3 |
| MAX (GPU) — after sub-pixel convT | 1915 ± 3 ms | 1.25e-3 |
| **MAX (GPU) — bf16 convs** (`DEMUCS_MAX_CONV_DTYPE=bf16`) | **1482 ± 4 ms** | 1.56e-2 |
| torch (CPU) | 525 ± 19 ms | — |
| torch (CUDA, A10) | **122 ± 0.3 ms** | — |

The port runs correctly on GPU (parity rel 1.25e-3, consistent with Ampere TF32
matmuls — inaudible) but is ~16× slower than torch-CUDA. Two graph-level wins
landed: the transposed convs were reformulated as a sub-pixel (pixel-shuffle)
`conv2d` (−18%, exact parity), and an opt-in bf16 conv mode (−23% more, but
parity drops to rel 1.6e-2).

**The residual gap is inside MAX, not the port.** Profiling traces it to MAX's
fp32 `conv2d` GPU kernel, which sustains only ~30 GMAC/s (~0.1% of the A10's fp32
peak) on every shape tested — it does not appear to use cuDNN/tensor cores, while
torch's 122 ms is exactly cuDNN doing so. See `docs/benchmark_results.txt` for the
full profile and `docs/conv2d_perf_issue.md` / `docs/cudnn_convT_issue.md` for the
two MAX issues this surfaced.

## How it's structured

The MAX graph spans the **heavy compute**: both U-Net branches (spectral +
waveform), the cross-domain transformer, and channel up/down-sampling —
normalized inputs → pre-iSTFT outputs. The cheap, fiddly bookends (STFT,
per-example normalization, complex-as-channels reshape, iSTFT, branch
recombination) run in torch around the graph. See `docs/htdemucs_spec.md` for the
checkpoint-verified architecture spec.

```
src/demucs_mojo/
  htdemucs_max.py   # the MAX graph: HEncLayer, HDecLayer, DConv, transformer
  ref_torch.py      # from-scratch torch forward built ONLY from the state_dict
                    #   (verified blueprint + oracle; reproduces torch to 2.9e-6)
  runner.py         # end-to-end: torch STFT/normalize -> MAX core -> torch iSTFT
scripts/
  dump_reference.py # dump pretrained weights + reference IO + activations
  e2e.py            # parity + benchmark vs torch
  comp_test.py      # per-layer isolation tests against dumped activations
tests/
  test_ref_torch.py # fast regression (no MAX compile)
```

## Key implementation notes (MAX CPU workarounds)

MAX's CPU backend had several gaps that shaped the port:

- **`conv2d_transpose` won't lower on CPU** (and *aborts* on GPU/cuDNN, see
  `docs/cudnn_convT_issue.md`) → transposed convs implemented as a sub-pixel
  (pixel-shuffle) `conv2d`: a stride-1 conv producing `Cout*stride` channels,
  then reshaping those channels into the spatial axis. No zero-inflation.
- **`irfft` is GPU-only** → iSTFT done in torch (forward STFT validated as a
  conv with DFT×window kernels; see `scripts/proto_stft.py`).
- **1×1 convs fail to compile** → implemented as channel matmuls.
- **dilated convs fail** → kernel inflated with zero gaps into a dense kernel.
- **wide `(1,k)` multichannel conv kernels are numerically wrong** → transposed
  convs oriented along the H axis with `(k,1)` kernels instead.
- `group_norm` isn't exported → `num_groups=1` done manually via mean/rsqrt.

## Running

First download *demucs* to ext/demucs

```bash
mkdir ext
git clone org-16943930@github.com:facebookresearch/demucs.git ext/demucs
```

Everything uses `uv` (Python 3.13). The pretrained checkpoint downloads on first
model load.

```bash
# 1. dump the torch reference (weights + IO + activations)
PYTHONPATH=ext/demucs uv run python scripts/dump_reference.py

# 2. fast regression test (no MAX compile)
PYTHONPATH=ext/demucs:src uv run pytest tests/test_ref_torch.py

# 3. end-to-end parity + benchmark (compiles the MAX graph, ~4 min)
PYTHONPATH=ext/demucs:src uv run python scripts/e2e.py
```

## Next steps

- **Close the GPU gap with torch-CUDA (~16×).** Root-caused to MAX's slow fp32
  `conv2d` GPU kernel (~0.1% of peak, no cuDNN/tensor cores) — see
  `docs/conv2d_perf_issue.md`. This is a MAX-internal fix; graph-level
  reformulation can't close it (verified: the convs are already the minimal
  formulation, and un-folding the DConv was *slower*). bf16 convs recover ~23%
  but cost accuracy.
- Move the STFT/iSTFT in-graph with native `irfft` (currently in torch).
- Cut compile time with runtime weights (`weights_registry`) instead of baked
  constants.
- `conv2d_transpose` aborts on GPU (cuDNN status-enum skew) — see
  `docs/cudnn_convT_issue.md`; can be dropped once fixed upstream (though the
  sub-pixel `conv2d` form is competitive anyway).
