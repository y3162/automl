"""STFT reconstruction, phase, feature matching and multi-scale Slaney mel losses."""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


def _reflect_pad(waveform: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return waveform
    length = int(waveform.shape[-1])
    if length <= pad:
        raise ValueError(
            f"waveform length {length} cannot be reflect-padded by {pad} samples"
        )
    left = waveform[..., 1 : pad + 1].flip(-1)
    right = waveform[..., -pad - 1 : -1].flip(-1)
    return torch.cat((left, waveform, right), dim=-1)


class _FoldedReflect(torch.autograd.Function):
    """Index reflection whose backward matches ``reflection_pad1d``.

    ``slice`` / ``flip`` / ``cat`` send the center samples and the reflected
    edges back to the waveform as separate gradient nodes. Another loss on
    that waveform can accumulate between those nodes, which moves only the
    reflected boundary. ``F.pad(..., mode="reflect")`` adds the reflected
    samples before the gradient joins the waveform. Folding both copies here
    keeps that order without CUDA ``reflection_pad1d_backward``.
    """

    @staticmethod
    def forward(ctx, waveform: torch.Tensor, pad: int) -> torch.Tensor:
        ctx.pad = int(pad)
        return _reflect_pad(waveform, int(pad))

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        pad = int(ctx.pad)
        if pad == 0:
            return grad, None
        folded = grad[..., pad:-pad].clone()
        folded[..., 1 : pad + 1] = folded[..., 1 : pad + 1] + grad[..., :pad].flip(-1)
        folded[..., -pad - 1 : -1] = folded[..., -pad - 1 : -1] + grad[..., -pad:].flip(-1)
        return folded, None


def _folded_reflect(waveform: torch.Tensor, pad: int) -> torch.Tensor:
    return _FoldedReflect.apply(waveform, int(pad))


def centered_stft(
    waveform: torch.Tensor,
    n_fft: int,
    hop: int,
    window: torch.Tensor,
    *,
    normalized: bool = False,
) -> torch.Tensor:
    """Hann or custom-window STFT matching ``center=True`` without ``reflection_pad1d``."""
    pad = n_fft // 2
    if int(waveform.shape[-1]) <= pad:
        waveform = torch.nn.functional.pad(waveform, (0, pad + 1 - int(waveform.shape[-1])))
    framed = _reflect_pad(waveform, pad)
    return torch.stft(
        framed,
        n_fft,
        hop_length=hop,
        win_length=int(window.shape[-1]),
        window=window,
        center=False,
        normalized=normalized,
        return_complex=True,
    )


def semamba_mag_phase_stft(
    waveform: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 0.3,
    *,
    center: bool = True,
    addeps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SEMamba++ magnitude and phase.

    The dataset path uses ``abs`` and ``angle``. The consistency path adds
    ``1e-10`` inside the square root and both ``atan2`` arguments. Centering
    uses an index reflection so the values match ``center=True`` without the
    non-deterministic CUDA reflection backward. The backward folds the
    reflected edges into one gradient, matching ``reflection_pad1d`` when this
    spectrum is combined with other waveform losses. The Hann window is built
    on CPU and moved, matching the reference training code. A window created
    directly on CUDA differs by about 1e-7 and can flip phase bins by 2π.
    """
    window = torch.hann_window(win_size, dtype=waveform.dtype).to(waveform.device)
    framed = _folded_reflect(waveform, n_fft // 2) if center else waveform
    spectrum = torch.stft(
        framed,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=False,
        normalized=False,
        return_complex=True,
    )
    if addeps:
        epsilon = 1e-10
        magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + epsilon)
        phase = torch.atan2(spectrum.imag + epsilon, spectrum.real + epsilon)
    else:
        magnitude = spectrum.abs()
        phase = spectrum.angle()
    magnitude = magnitude.pow(compress_factor)
    compressed = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    return magnitude, phase, compressed


def semamba_mag_phase_istft(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 0.3,
    *,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse of ``semamba_mag_phase_stft`` using the same CPU-built Hann window."""
    magnitude = magnitude.pow(1.0 / compress_factor)
    spectrum = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    window = torch.hann_window(win_size, dtype=spectrum.real.dtype).to(spectrum.device)
    return torch.istft(
        spectrum,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        length=length,
    )


def anti_wrapping(value: torch.Tensor) -> torch.Tensor:
    return torch.abs(value - torch.round(value / (2 * math.pi)) * (2 * math.pi))


def _difference_matrix(size: int, like: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(size, size, device=like.device, dtype=like.dtype)
    return torch.triu(ones, diagonal=1) - torch.triu(ones, diagonal=2) - torch.eye(size, device=like.device, dtype=like.dtype)


def least_squares_discriminator(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return torch.mean((1 - real_score).square()) + torch.mean(fake_score.square())


def least_squares_generator(fake_score: torch.Tensor) -> torch.Tensor:
    return torch.mean((1 - fake_score).square())


def feature_matching(
    real_features: list[torch.Tensor],
    fake_features: list[torch.Tensor],
    scale: float = 2.0,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sum or mean of per-map L1 distances, then ``scale``.

    SEMamba++ sums the map means and multiplies by 2. Averaging would change
    the loss coefficient by the number of maps.
    """
    if len(real_features) != len(fake_features) or not real_features:
        raise ValueError("feature lists must be non-empty and the same length")
    total = real_features[0].new_zeros(())
    for real, fake in zip(real_features, fake_features, strict=True):
        total = total + F.l1_loss(fake, real.detach())
    if reduction == "sum":
        return total * scale
    if reduction == "mean":
        return total / len(real_features) * scale
    raise ValueError(f"unsupported feature-matching reduction {reduction}")


def summed_slaney_log_mel(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    *,
    n_mels: tuple[int, ...] = (40, 80, 160, 320),
    windows: tuple[int, ...] = (256, 512, 1024, 2048),
    clamp: float = 1e-5,
) -> torch.Tensor:
    """Sum of mean L1 distances between log10 Slaney mel spectrograms.

    This is the multi-scale mel term used by SEMamba++: four windows, hop
    equal to one quarter of the window, Hann, centered STFT, log10, and no
    extra magnitude term. The four scales are added, not averaged.
    """
    if len(n_mels) != len(windows):
        raise ValueError("n_mels and windows must have the same length")
    if reference.ndim == 3:
        reference = reference.squeeze(1)
        estimate = estimate.squeeze(1)
    total = reference.new_zeros(())
    for bands, window in zip(n_mels, windows, strict=True):
        hop = window // 4
        ref = _slaney_log_mel(reference, sample_rate, bands, window, hop, clamp)
        est = _slaney_log_mel(estimate, sample_rate, bands, window, hop, clamp)
        frames = min(ref.shape[-1], est.shape[-1])
        # The reference adds the same L1 term a second time with weight zero.
        # Keep that graph and operation order for the backward comparison.
        total = total + F.l1_loss(ref[..., :frames], est[..., :frames])
        total = total + 0.0 * F.l1_loss(ref[..., :frames], est[..., :frames])
    return total


def _slaney_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int,
    window: int,
    hop: int,
    clamp: float,
) -> torch.Tensor:
    # The published mel loss builds both the window and the librosa basis on
    # CPU, then moves them to the signal device. GPU construction and a
    # mathematically equivalent transposed multiplication change gradients.
    # A zero-width reflection pad is an identity. Its CUDA backward has no
    # deterministic implementation, so that path unsqueezes without the kernel.
    # Other devices keep the pad so the graph stays the published one.
    if waveform.is_cuda and torch.are_deterministic_algorithms_enabled():
        waveform = waveform.unsqueeze(1)
    else:
        waveform = F.pad(waveform.unsqueeze(1), (0, 0), mode="reflect")
    hann = torch.hann_window(window).to(waveform.device)
    if waveform.is_cuda and torch.are_deterministic_algorithms_enabled():
        spec = centered_stft(waveform.squeeze(1), window, hop, hann)
    else:
        spec = torch.stft(waveform.squeeze(1), window, hop_length=hop,
                          window=hann, center=True, return_complex=True)
    spec = spec.abs().unsqueeze(1)
    basis = _official_slaney_basis(sample_rate, window, n_mels).to(spec.device)
    mel = (spec.transpose(2, -1) @ basis.T).transpose(-1, 2)
    return torch.log(mel.clamp(min=clamp).pow(1.0)) / torch.log(torch.tensor(10.0))


@lru_cache(maxsize=32)
def _official_slaney_basis(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    from librosa.filters import mel

    return torch.from_numpy(mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=None))
