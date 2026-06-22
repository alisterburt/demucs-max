"""Prototype: reproduce torch.stft (as used by demucs spectro) via framing + DFT
matmul / conv, and reproduce torch.istft via irfft + overlap-add. Validate exactly.

Run: uv run python scripts/proto_stft.py
"""
import numpy as np
import torch

NFFT = 4096
HOP = 1024


def torch_spectro(x):  # mirrors demucs.spec.spectro defaults
    z = torch.stft(x, NFFT, HOP, window=torch.hann_window(NFFT),
                   win_length=NFFT, normalized=True, center=True,
                   return_complex=True, pad_mode="reflect")
    return z  # (B, 2049, frames) complex


def dft_basis(nfft):
    n = np.arange(nfft)
    k = np.arange(nfft // 2 + 1)
    ang = 2 * np.pi * np.outer(k, n) / nfft  # (F, nfft)
    return np.cos(ang), -np.sin(ang)         # real, imag of e^{-i ang}


def np_stft_conv(x_np):
    """STFT via conv1d (cross-correlation) with window*DFT kernels. center=True."""
    B, L = x_np.shape
    pad = NFFT // 2
    xp = np.stack([np.pad(row, (pad, pad), mode="reflect") for row in x_np])  # center pad
    win = np.hanning(NFFT + 1)[:-1]  # torch hann_window(periodic=True) == np.hanning(N+1)[:-1]
    cos_b, sin_b = dft_basis(NFFT)
    # kernels: (F, nfft) = basis * window
    kc = cos_b * win[None, :]
    ks = sin_b * win[None, :]
    nframes = 1 + (xp.shape[1] - NFFT) // HOP
    re = np.zeros((B, NFFT // 2 + 1, nframes))
    im = np.zeros((B, NFFT // 2 + 1, nframes))
    for b in range(B):
        for f in range(nframes):
            seg = xp[b, f * HOP: f * HOP + NFFT]
            re[b, :, f] = kc @ seg
            im[b, :, f] = ks @ seg
    z = (re + 1j * im) / np.sqrt(NFFT)  # normalized=True
    return z


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 343980)
    zt = torch_spectro(x).numpy()
    zc = np_stft_conv(x.numpy())
    print("torch z", zt.shape, "conv z", zc.shape)
    err = np.abs(zt - zc).max()
    print("max abs diff (STFT):", err)
    assert err < 1e-3, err

    # --- iSTFT via overlap-add using irfft (numpy as proxy for MAX irfft) ---
    # mirror demucs.spec.ispectro: istft normalized=True, center=True, length given.
    z = zt  # (B, 2049, T)
    length = x.shape[-1]
    win = np.hanning(NFFT + 1)[:-1]
    B, F, T = z.shape
    frames = np.fft.irfft(z * np.sqrt(NFFT), n=NFFT, axis=1)  # undo normalize, (B,NFFT,T)
    frames = frames * win[None, :, None]
    out_len = NFFT + HOP * (T - 1)
    sig = np.zeros((B, out_len))
    wsum = np.zeros((B, out_len))
    for f in range(T):
        sig[:, f * HOP:f * HOP + NFFT] += frames[:, :, f]
        wsum[:, f * HOP:f * HOP + NFFT] += win[None, :] ** 2
    sig = sig / np.maximum(wsum, 1e-8)
    sig = sig[:, NFFT // 2: NFFT // 2 + length]  # remove center padding
    xi = torch.istft(torch.from_numpy(z), NFFT, HOP, window=torch.hann_window(NFFT),
                     win_length=NFFT, normalized=True, length=length, center=True).numpy()
    ierr = np.abs(sig - xi).max()
    print("max abs diff (iSTFT):", ierr)
    assert ierr < 1e-3, ierr
    print("OK: STFT-via-conv and iSTFT-via-irfft+OLA match torch.")


if __name__ == "__main__":
    main()
