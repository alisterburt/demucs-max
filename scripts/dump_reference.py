"""Dump a torch HTDemucs reference: fixed input, output, intermediate activations,
and the state_dict. Used as ground truth for validating the MAX port.

Run:  PYTHONPATH=ext/demucs uv run python scripts/dump_reference.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from demucs.pretrained import get_model

OUT = Path("reference")
OUT.mkdir(exist_ok=True)
SEED = 0


def main() -> None:
    torch.manual_seed(SEED)
    model = get_model("htdemucs")
    model.eval()
    sub = model.models[0]  # the single HTDemucs inside the BagOfModels
    sub.eval()

    # Fixed seeded input at the model's training segment length.
    training_length = int(sub.segment * sub.samplerate)
    print(f"segment={sub.segment} samplerate={sub.samplerate} "
          f"training_length={training_length}")
    x = torch.randn(1, sub.audio_channels, training_length)

    # Capture intermediate activations by name via forward hooks.
    acts: dict[str, np.ndarray] = {}

    def save(name):
        def hook(_mod, _inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                for i, o in enumerate(t):
                    if isinstance(o, torch.Tensor):
                        acts[f"{name}.out{i}"] = o.detach().cpu().numpy()
            elif isinstance(t, torch.Tensor):
                acts[name] = t.detach().cpu().numpy()
        return hook

    handles = []
    for i, m in enumerate(sub.encoder):
        handles.append(m.register_forward_hook(save(f"encoder.{i}")))
    for i, m in enumerate(sub.tencoder):
        handles.append(m.register_forward_hook(save(f"tencoder.{i}")))
    for i, m in enumerate(sub.decoder):
        handles.append(m.register_forward_hook(save(f"decoder.{i}")))
    for i, m in enumerate(sub.tdecoder):
        handles.append(m.register_forward_hook(save(f"tdecoder.{i}")))
    if sub.crosstransformer is not None:
        handles.append(sub.crosstransformer.register_forward_hook(
            save("crosstransformer")))
    if getattr(sub, "freq_emb", None) is not None:
        handles.append(sub.freq_emb.register_forward_hook(save("freq_emb")))

    with torch.no_grad():
        y = sub(x)

    for h in handles:
        h.remove()

    print("output shape", tuple(y.shape))
    for k in sorted(acts):
        print(f"  act {k}: {acts[k].shape}")

    np.savez(OUT / "io.npz", input=x.numpy(), output=y.numpy())
    np.savez(OUT / "activations.npz", **acts)

    # state_dict as a flat npz (float32 numpy). Keeps it dependency-light.
    sd = {k: v.detach().cpu().numpy() for k, v in sub.state_dict().items()}
    np.savez(OUT / "state_dict.npz", **sd)
    print(f"saved {len(sd)} params to {OUT/'state_dict.npz'}")


if __name__ == "__main__":
    main()
