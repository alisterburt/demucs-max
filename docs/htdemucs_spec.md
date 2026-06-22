# HTDemucs (v4) — checkpoint-verified port spec

Reference for porting the **official `htdemucs`** checkpoint to MAX. Every value
below was verified against the loaded model + dumped `state_dict.npz`
(533 tensors, ~42M params). Source: `ext/demucs/demucs/{htdemucs,hdemucs,transformer,spec,demucs}.py`.

## Verified configuration (overrides the code defaults!)

| param | value | param | value |
|---|---|---|---|
| sources | `[drums,bass,other,vocals]` (S=4) | audio_channels | 2 |
| channels | 48 (chain 48,96,192,384) | growth | 2 |
| depth | 4 | nfft | 4096 / hop 1024 |
| **dconv_mode** | **3** ⚠ (encoder AND decoder have DConv; not 1) | | |
| cac | True | wiener_iters | 0 (Wiener = dead code) |
| **bottom_channels** | **512** ⚠ (not 0!) | transformer dim | **512** |
| t_layers | 5 | t_heads | 8 (head_dim 64) |
| FFN hidden | 2048 (=512×4) | norm_starts | 4 |
| segment | 39/5 = 7.8s | samplerate | 44100 |
| **training_length** | **343980** | freq_emb scale | 0.2 |

**Concrete shapes at training_length** (verified from activations dump):
`le` (freq time-frames) = **336**, time-branch bottleneck `T2` = **1344**, Fr after enc0 = 512.

### Corrections vs first-pass spec
1. **bottom_channels=512** → transformer runs at 512 ch. There ARE `channel_upsampler`
   (384→512, Conv1d 1×1), `channel_downsampler` (512→384), and `_t` variants. Applied
   around the transformer (htdemucs.py:585-599).
2. **Transformer layer layout** (`classic_parity=0`, cross_first=False → `idx%2==0` is classic/self):
   - layer **0,2,4 = SELF** (`MyTransformerEncoderLayer`, norms: norm1,norm2,norm_out)
   - layer **1,3 = CROSS** (`CrossTransformerEncoderLayer`, norms: norm1,norm2,norm3,norm_out)
3. FFN hidden = 2048, attn in_proj = (1536,512)=(3·512,512).

## Forward pass (htdemucs.py:527-660)

1. **Pad to training_length** (inference): right-zero-pad if shorter; crop output to orig length at end.
2. **STFT** `_spec` (420-440): reflect-pad by `pad=1536` left, `pad+le*hl-L` right; `le=ceil(L/hl)`;
   `spectro` = `torch.stft(nfft=4096, hop=1024, hann_window(4096), win_length=4096, normalized=True,
   center=True, return_complex=True, pad_mode=reflect)`; drop Nyquist row `[...,:-1,:]`; crop frames `[...,2:2+le]`.
   → `z` complex `(B,2,2048,le)`.
3. **CaC magnitude** `_magnitude` (452-461): `view_as_real(z).permute(0,1,4,2,3).reshape(B,4,2048,le)`.
   Channel order per audio ch = `[re, im]` → 4 channels `[c0_re,c0_im,c1_re,c1_im]`.
4. **Normalize** (unbiased std, eps 1e-5): freq `x=(x-mean)/(1e-5+std)` over dims (1,2,3);
   time `xt=(mix-meant)/(1e-5+stdt)` over (1,2). Save mean/std for de-norm.
5. **Encoder** (4 layers each branch, interleaved). No injection (no tenc empty). After enc0 add
   `0.2 * freq_emb`. Freq: Conv2d k(8,1) s(4,1) p(2,0), Fr÷4 each layer. Time: Conv1d k8 s4 p2, T÷4.
   Save skips. Shapes: freq→(B,384,8,336), time→(B,384,1344).
6. **Transformer** (bottleneck): upsample 384→512 (channel_upsampler on freq via `(f t)` flatten,
   channel_upsampler_t on time), run `crosstransformer(x,xt)`, downsample 512→384.
7. **Decoder** (4 layers each, parallel, offset=0). Additive skips; ConvTranspose center-cropped.
   Freq last layer→16 ch, time last→8 ch. GELU except last layer.
8. **Recombine**: freq `x.view(B,4,4,2048,le)`, de-norm, `_mask` CaC→complex `(B,4,2,2048,le)`,
   `_ispec`→waveform `(B,4,2,Lp)`. time `xt.view(B,4,2,Lp)`, de-norm. `out = xt + x`. Crop to orig len.
   → **(B,4,2,L)**, sources order `[drums,bass,other,vocals]`.

## Building blocks

- **HEncLayer.forward** (hdemucs.py:111): `y=conv(x)`; (+inject); `y=gelu(norm1(y))` (norm1=Identity);
  DConv (fold Fr into batch for freq layers: `(B,C,Fr,T)→(B*Fr,C,T)`); `z=glu(norm2(rewrite(y)),dim=1)`.
  rewrite = 1×1 conv chout→2·chout. GLU(a,b)=a·sigmoid(b).
- **HDecLayer.forward** (hdemucs.py:304): `x=x+skip` (additive!); `y=glu(norm1(rewrite(x)),dim=1)`
  rewrite=Conv2d(chin,2chin,(3,3),p(1,1)) for freq / Conv1d k3 p1 for time; **then DConv(chin)** (dconv_mode=3,
  same Fr-fold for freq); `z=norm2(conv_tr(y))`; crop freq `z[...,2:-2,:]` or time `z[...,2:2+length]`;
  `gelu` unless last. returns (z, pre=y).
- **DConv** (demucs.py:86): 2 residual layers, each `Conv1d(C,C/8,3,dil=2^d,pad=2^d) → GN(1) → GELU
  → Conv1d(C/8,2C,1) → GN(1) → GLU(1) → LayerScale(C,init1e-3)`. `x = x + layer(x)`.
- **ScaledEmbedding** (hdemucs.py:43): `Embedding(512,48)`; forward `embedding(x)*scale(=10)`. Smooth
  transforms baked into checkpoint weight. Used: `x += 0.2 * freq_emb(arange(512)).T[None,:,:,None]`.
- **norms in enc/dec are Identity** (norm_starts=4 > all layer indices). Only DConv + transformer have norm params.

## Transformer (transformer.py:526-719), dim=512, heads=8

- `norm_in`, `norm_in_t` = LayerNorm(512). Pre-norm (norm_first=True) + post `norm_out`=MyGroupNorm(1,512) every layer.
- Token flatten: spectral `rearrange "b c fr t1 -> b (t1 fr) c"` (**time-major, freq-minor**), N=336·8=2688.
  time `(B,T2=1344,512)`. Add pos-emb (weight 1.0) after norm_in.
- **Pos-emb (fixed buffers, NOT params)**: spectral `create_2d_sin_embedding` (37-70, d split 256/256
  width/height, sin even/cos odd); time `create_sin_embedding` (19-34, cat[cos,sin], shift=0). max_period=10000.
- **Self layer** (0,2,4): `x = x + gamma_1(SA(norm1(x)))`; `x = x + gamma_2(FF(norm2(x)))`; `x = norm_out(x)`.
  SA = MultiheadAttention(512,8). FF = linear2(gelu(linear1(·))), hidden 2048.
- **Cross layer** (1,3): `x = q + gamma_1(CA(norm1(q), norm2(k)))`; `x = x + gamma_2(FF(norm3(x)))`;
  `x = norm_out(x)`. In forward: spectral updates with time as k/v, time updates with **pre-update** spectral as k/v.
- `gamma_1,gamma_2` = LayerScale(512, init 1e-4, channel_last → `scale*x`).
- **MyGroupNorm** (258): transpose (B,N,C)↔(B,C,N) around GroupNorm(1) → normalizes jointly over C & all tokens.
- `in_proj_weight (1536,512)` = stacked [Wq;Wk;Wv]; split for MAX. scale 1/sqrt(64).

## Weight key map (533 tensors)

- `encoder.{0-3}`: `.conv.{weight(chout,chin,8,1),bias}`, `.rewrite.{weight(2chout,chout,1,1),bias}`,
  `.dconv.layers.{0,1}.{0(conv),1(GN),3(conv1x1),4(GN),6.scale(LayerScale)}`. chout=[48,96,192,384], hidden=chout/8.
- `tencoder.{0-3}`: same but Conv1d (kernel 8, no trailing 1-dim). chin=[2,48,96,192].
- `decoder.{0-3}`: `.conv_tr.{weight(chin,chout,8,1),bias}`, `.rewrite.{weight(2chin,chin,3,3),bias}`.
  chin→chout: 384→192,192→96,96→48,48→16(last). No dconv, no norm.
- `tdecoder.{0-3}`: `.conv_tr.{weight(chin,chout,8),bias}`, `.rewrite.{weight(2chin,chin,3),bias}`.
  chout last = 8.
- `freq_emb.embedding.weight (512,48)`.
- `channel_upsampler.{weight(512,384,1),bias}`, `channel_downsampler.{weight(384,512,1),bias}`, + `_t` variants.
- `crosstransformer.norm_in{,_t}.{weight,bias}(512)`.
- self layers `{0,2,4}` (and `_t`): `.self_attn.{in_proj_weight(1536,512),in_proj_bias,out_proj.weight(512,512),
  out_proj.bias}`, `.linear1.{weight(2048,512),bias}`, `.linear2.{weight(512,2048),bias}`,
  `.norm1.{w,b}`, `.norm2.{w,b}`, `.norm_out.{w,b}`, `.gamma_1.scale`, `.gamma_2.scale`.
- cross layers `{1,3}` (and `_t`): `.cross_attn.{...}`, linear1/2, `.norm1/2/3.{w,b}`, norm_out, gamma_1/2.

## Gotchas
- All GELU = exact erf variant; GLU(a,b)=a·sigmoid(b) split on channel dim.
- STFT/iSTFT use `normalized=True` (scale 1/sqrt(nfft)). Match framing exactly (§forward step 2/8).
- Two independent waveform reconstructions are **summed**, not averaged.
- Pos-embeddings recomputed each forward; not in state_dict.
- rescale_module already baked into checkpoint — load weights verbatim.
