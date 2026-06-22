# demucs-mojo

Reimplementation of **HTDemucs v4** (Hybrid Transformer Demucs, the default
`htdemucs` music source-separation model) in **Modular MAX**, with a numerical
parity check and a speed comparison against the original PyTorch model.

## Status

- ✅ Full conv/transformer network ported to the MAX graph API (`max.graph`).
- ✅ Loads the real pretrained checkpoint; **reproduces torch output to 1.5e-4**
  (relative 4.6e-5) end-to-end.
- ✅ Benchmarked on CPU (Apple M4 Pro).

### Benchmark (input = 343980 samples ≈ 7.8 s @ 44.1 kHz, batch 1)

| backend | latency |
|---|---|
| **MAX (CPU, M4 Pro)** | **1965 ± 42 ms** |
| torch (CPU) | 2001 ± 52 ms |
| torch (MPS / Apple GPU) | 407 ± 6 ms |

MAX-CPU matches torch-CPU (1.02×). torch on the Apple GPU (MPS) is ~5× faster —
the eventual GPU target for the MAX model is the natural next step. MAX one-time
graph compile is ~240 s (dominated by constant-folding the 42 M weights; can be
cut with runtime weights).

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

- **`conv2d_transpose` won't lower on CPU** → transposed convs implemented via
  input-dilation + a flipped regular `conv2d`.
- **`irfft` is GPU-only** → iSTFT done in torch (forward STFT validated as a
  conv with DFT×window kernels; see `scripts/proto_stft.py`).
- **1×1 convs fail to compile** → implemented as channel matmuls.
- **dilated convs fail** → kernel inflated with zero gaps into a dense kernel.
- **wide `(1,k)` multichannel conv kernels are numerically wrong** → transposed
  convs oriented along the H axis with `(k,1)` kernels instead.
- `group_norm` isn't exported → `num_groups=1` done manually via mean/rsqrt.

## Running

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

- Run on a GPU server (MAX GPU backend) — the native `conv2d_transpose`/`irfft`
  GPU kernels exist, so the workarounds above can be dropped and the full
  pipeline (incl. STFT/iSTFT) moved in-graph.
- Cut compile time with runtime weights (`weights_registry`) instead of baked
  constants.
- Optimize the spectral DConv (currently `B·Fr` tiny batched convs).
