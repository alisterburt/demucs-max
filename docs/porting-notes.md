# Porting HTDemucs v4 to MAX — issues & workarounds

A running log of every non-trivial issue hit while porting the pretrained
`htdemucs` model from PyTorch to the Modular MAX graph API (`modular==26.4.0`),
and how each was resolved. Grouped into: checkpoint/architecture surprises, MAX
graph-API gotchas, and MAX CPU op limitations.

Environment: Apple M4 Pro (20 cores, Metal 3), Python 3.13 via `uv`, torch 2.12,
`modular` 26.4.0. Target device for this work: **CPU** (GPU server is a later step).

---

## 1. Checkpoint vs. code defaults (architecture surprises)

The biggest early trap: **the constructor defaults in `htdemucs.py` are not the
config of the released checkpoint.** Three values differ, and each silently
produces a structurally wrong model if you trust the source defaults. All were
caught by diffing against the dumped `state_dict` and intermediate activations.

| thing | code default | real checkpoint | how it bit us |
|---|---|---|---|
| `bottom_channels` | `0` | **`512`** | Default implies no channel up/down-sampling and a transformer at 384 dims. Real model has `channel_upsampler/downsampler` 1×1 convs and runs the transformer at **512** dims (FFN 2048). The first activation dump showed `crosstransformer` at 512 channels — the giveaway. |
| `dconv_mode` | `1` (encoder only) | **`3`** (encoder **and** decoder) | We initially omitted DConv from the decoder. The decoder output was off by ~2.3 while the encoder/transformer matched perfectly. The clue: each `decoder.N.*` group had 22 weight tensors, not the 4 a DConv-less layer would have. |
| transformer layer types | inferred cross-first | self at **0,2,4**, cross at **1,3** | `cross_first=False` ⇒ `classic_parity=0` ⇒ `idx % 2 == 0` is the *classic self-attention* layer. The state_dict keys (`layers.0.self_attn` vs `layers.1.cross_attn`) are authoritative — read those, don't infer. |

**Lesson:** for a pretrained port, derive the config from the checkpoint
(`state_dict` keys + shapes + the model object's attributes), never from the
library's `__init__` defaults. The verified config lives in
`docs/htdemucs_spec.md`.

Other architecture details that needed care (all handled, none were "bugs"):

- **Complex-as-channels (CaC):** `cac=True` and `wiener_iters=0` mean the entire
  Wiener-filter path is dead code. `_magnitude` interleaves `[re, im]` as adjacent
  channels; `_mask` reverses it. Channel order is `[c0_re, c0_im, c1_re, c1_im]`.
- **Two independent normalizations** (freq over `(C,Fr,T)`, time over `(C,T)`),
  **unbiased std**, eps `1e-5`, and the two reconstructed waveforms are **summed**.
- **Additive skip connections** (not concat) and **center-cropped** transposed
  convs in the decoder.
- **Time-encoder pads to a multiple of stride** before each conv. Skipping this
  drifts the length by 1 sample per layer and the decoder crop then runs off the
  end of the tensor. (This was a real bug we hit — `rmo.slice stop index out of
  range`.)
- **Positional embeddings are recomputed every forward** (not in the state_dict);
  the spectral one uses time-major/freq-minor token ordering.

---

## 2. MAX graph-API gotchas

Things that work but aren't obvious from the API surface:

- **Imports:** `from max.graph import Graph, TensorType, DeviceRef, ops`;
  `from max.engine import InferenceSession`; `from max.driver import CPU,
  Accelerator`. `DeviceRef.CPU()` for graph types, `CPU()` (driver) for the
  session.
- **conv2d layout:** input is **NHWC**, filter is **RSCF = (kH, kW, in, out)**.
  A torch weight `(out, in, kH, kW)` becomes `np.transpose(w, (2,3,1,0))`.
- **conv2d padding is a 4-tuple `(top, bottom, left, right)`** — i.e.
  `(dim1_before, dim1_after, dim2_before, dim2_after)`. Supports asymmetric pad
  (torch `F.conv2d` does not).
- **`ops.pad` only supports `mode="constant"`** in this build (the docstring
  mentions reflect/edge, but the installed signature is constant-only). Reflect
  padding for the STFT therefore has to live in the torch bookend.
- **`ops.mean(x, axis)` reduces a single axis and keeps the dim** (`(B,…)` →
  `(B,1)`). For a multi-axis reduction (GroupNorm), flatten first then reduce.
- **`execute()` takes numpy arrays directly** (via DLPack) and returns a list of
  `max.driver.Buffer`; call `.to_numpy()` on each. There is no `Tensor` export in
  `max.driver` — it's `Buffer`.
- **Baked constants compile slowly.** Injecting all 42 M weights as
  `ops.constant` makes graph compile take ~240 s (constant-folding into packed
  conv layouts). Functionally fine, but `weights_registry` (runtime weights) is
  the fix for iteration speed.

---

## 3. MAX CPU op limitations (the real workarounds)

These are CPU-backend gaps in `modular==26.4.0`. Each was isolated with a minimal
repro and given a workaround in `src/demucs_mojo/htdemucs_max.py`. Several GPU
kernels exist where the CPU ones don't, so most of these should disappear on the
GPU target.

### 3.1 `conv2d_transpose` won't lower on CPU
**Symptom:** compile error `Failed to infer parameter num_groups` /
`Could not lower this operation to Mojo`. The op's own source has a
`TODO(GEX-2043): Add support for GPU kernel for conv_transpose`.

**Workaround:** implement transposed convolution with a regular `conv2d` —
dilate the input (insert `stride-1` zeros between samples), pad by `k-1`, and
convolve with the kernel **flipped along the spatial axis and with in/out
channels swapped**. Verified to 1e-6 vs `F.conv_transpose1d/2d`
(`scripts/proto_convT.py`).

### 3.2 Wide `(1, k)` multichannel conv kernels are numerically WRONG
**Symptom:** the nastiest one — it *compiles and runs*, but silently produces
wrong numbers. A transposed conv with `Cin=Cout=1` was correct; multichannel with
a small kernel was correct; **multichannel + a wide `(1,8)` kernel was off by
~0.6.** Tall `(8,1)` kernels (the forward encoder) are always correct.

**Workaround:** orient all 1-D and transposed convs along the **H axis** with
`(k, 1)` kernels instead of `(1, k)`. This is why `conv_transpose_1axis` moves the
active spatial dimension onto H before convolving.

> This took the longest to find because the forward path used `(1,k)` kernels in
> `conv1d` *and matched* — the bug only surfaces for the dilated/transposed case
> with many channels and a wide kernel. Lesson: when one tensor branch is far more
> wrong than another that shares most ops, bisect by feeding *dumped* intermediate
> activations into single isolated layers (`scripts/comp_test.py`).

### 3.3 1×1 convolutions fail to compile
**Symptom:** `error: 'mo.layout.transform' op … Could not find a mojo kernel
registered for layout_transform_RSCF_to_KNkni`.

**Workaround:** a 1×1 conv is pure channel mixing, so implement it as a matmul
over the channel dimension (`conv1x1`). Applies to the encoder/decoder `rewrite`
convs, DConv's 1×1, and the channel up/down-samplers.

### 3.4 Dilated convolutions fail to compile
**Symptom:** separate compile failure for `dilation > 1` (DConv's second residual
layer uses dilation 2).

**Workaround:** "inflate" the kernel — a kernel of size `k` with dilation `d`
becomes a dense kernel of size `(k-1)*d + 1` with the taps spread out and zeros in
the gaps, then convolve with `dilation=1`.

### 3.5 `irfft` is GPU-only
**Symptom:** `ValueError: IRFFT is currently only supported on GPU`.

**Workaround:** keep the iSTFT in torch for the CPU target. The forward STFT was
validated as a conv with DFT×window kernels, and the iSTFT as an inverse-DFT
matmul + overlap-add (both < 1e-6 vs `torch.stft/istft`, `scripts/proto_stft.py`),
so they can move in-graph on GPU where `irfft` works.

### 3.6 `ops.group_norm` not exported
**Symptom:** `module 'max.graph.ops' has no attribute 'group_norm'` (the file
exists internally but isn't surfaced).

**Workaround:** every GroupNorm in HTDemucs uses `num_groups=1`, which is just
"normalize over all channels and spatial positions per sample." Implemented
directly with flatten → mean/variance → `rsqrt` → per-channel affine.

### 3.7 `conv2d_transpose` bias kwarg broadcasts wrong
**Symptom:** passing `bias=` to `conv2d_transpose` adds it against the wrong axis
(the output isn't in the layout the bias-add code assumes).

**Workaround:** moot once we replaced the op entirely (3.1), but in general: add
bias manually as a reshaped constant after the op.

---

## 4. Strategy notes (what made the port tractable)

- **Dump a reference first.** `scripts/dump_reference.py` saves the pretrained
  weights, a fixed seeded input/output, and **every intermediate activation** via
  forward hooks. This turned every parity question into a direct array diff.
- **Write a from-scratch torch forward built only from the `state_dict`**
  (`ref_torch.py`) before touching MAX. It reproduces torch to 2.9e-6, proved we
  understood the architecture and weight layout, and became the exact op-for-op
  blueprint for the MAX translation. Every MAX op has a one-line torch analogue
  there.
- **Isolate components.** When the full graph was wrong, `comp_test.py` built a
  *single-layer* graph fed with dumped activations — fast to compile and it pins
  the bug to one function instead of re-running the 240 s full compile.
- **Validate numeric tricks in numpy first.** The STFT-as-conv and
  iSTFT-as-matmul equivalences were proven in `proto_stft.py` and the
  transposed-conv-as-conv identity in `proto_convT.py` before going anywhere near
  MAX.

---

## 5. Open items / would-be-nicer

- Move STFT/iSTFT and normalization in-graph (needs GPU for `irfft`, or an
  inverse-DFT matmul on CPU).
- Use `weights_registry` instead of baked constants to cut the ~240 s compile.
- The spectral DConv folds frequency into the batch (`B·Fr` tiny 1-D convs) — a
  likely throughput bottleneck worth profiling.
- Re-evaluate all CPU workarounds on the GPU backend; most should be removable.
