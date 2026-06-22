"""Fast regression test: the from-scratch torch forward (built only from the raw
state_dict) must reproduce the demucs reference output. No MAX compile needed.

Requires the reference dump (scripts/dump_reference.py) to have been run.
Run: PYTHONPATH=ext/demucs:src uv run pytest tests/test_ref_torch.py
"""
from pathlib import Path

import numpy as np
import pytest
import torch

REF = Path("reference")


@pytest.mark.skipif(not (REF / "io.npz").exists(), reason="reference dump missing")
def test_from_scratch_matches_reference():
    from demucs_mojo import ref_torch as R
    sd = dict(np.load(REF / "state_dict.npz"))
    io = np.load(REF / "io.npz")
    mix = torch.from_numpy(io["input"]).float()
    ref = torch.from_numpy(io["output"]).float()
    with torch.no_grad():
        y = R.forward(R.W(sd), mix)
    assert y.shape == ref.shape
    assert (y - ref).abs().max().item() < 1e-3
