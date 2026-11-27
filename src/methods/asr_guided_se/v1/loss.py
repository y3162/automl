"""SEMamba++ objectives and recognition-guided reductions."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .config import GuidanceV1
from .error_mask import centered_token_ids, crop_error_mask, overlapping_tokens
from .spectral import (
    _difference_matrix, _slaney_log_mel, anti_wrapping, feature_matching,
    least_squares_discriminator, least_squares_generator, summed_slaney_log_mel,
)


def mean_tdt(per_sample: torch.Tensor) -> torch.Tensor:
    """Mean over the micro-batch. A scalar is already reduced."""

    if per_sample.ndim == 0:
        return per_sample
    return per_sample.mean()


def guidance_terms(
    guidance: GuidanceV1,
    audio: torch.Tensor,
    spans: Sequence[tuple[int, int, int, int]],
    *,
    full_references: Iterable[torch.Tensor],
    recognize: Callable[[torch.Tensor, str], Any],
    asr: Any,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Align enhanced and clean tokens and compute the configured crop-level TDT objective."""
    content = torch.tensor([span[1] for span in spans], device=audio.device, dtype=torch.long)
    if not guidance.changes_loss:
        return None, content, None
    masks = []
    targets = []
    hop = int(asr.samples_per_encoder_frame)
    for row, (reference, span) in enumerate(zip(full_references, spans, strict=True)):
        start, length, window, full_length = span
        clean_tokens = recognize(reference, "clean").tokens
        if guidance.tdt_loss_enabled:
            targets.append(centered_token_ids(clean_tokens, start, length, hop))
        enhanced_tokens = recognize(audio[row, :length], "enhanced").tokens
        masks.append(crop_error_mask(
            overlapping_tokens(clean_tokens, start, length, hop), enhanced_tokens, crop_start=start, content_length=length,
            window=window, hop=hop, full_length=full_length,
        ))
    sample_error = torch.tensor(masks, device=audio.device, dtype=torch.bool)
    tdt = None
    if guidance.tdt_loss_enabled:
        tdt = mean_tdt(asr.loss_from_ids(audio, content, targets, deterministic_tdt=False))
    return sample_error, content, tdt


def frame_weight_and_valid(
    sample_error: torch.Tensor,
    content_length: torch.Tensor,
    n_frames: int,
    hop: int,
    alpha: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``w = 1 + alpha * m``. A frame center ``j * hop`` is an error only inside the content."""

    if sample_error.ndim != 2:
        raise ValueError(f"sample_error must be [B, T], got {tuple(sample_error.shape)}")
    if content_length.ndim != 1 or content_length.shape[0] != sample_error.shape[0]:
        raise ValueError("content_length must be [B] aligned with sample_error")
    if n_frames < 1 or hop <= 0 or alpha < 0:
        raise ValueError(f"invalid frame weight n_frames={n_frames} hop={hop} alpha={alpha}")
    device = sample_error.device
    content = content_length.to(device=device, dtype=torch.long)
    centers = torch.arange(n_frames, device=device) * hop
    n_valid = torch.where(
        content > 0,
        torch.div(content, hop, rounding_mode="floor") + 1,
        torch.zeros_like(content),
    )
    frame_index = torch.arange(n_frames, device=device).view(1, -1)
    valid = frame_index < n_valid.view(-1, 1)
    inside = centers.view(1, -1) < content.view(-1, 1)
    in_tensor = centers < sample_error.shape[1]
    if sample_error.shape[1] == 0:
        belongs = torch.zeros(sample_error.shape[0], n_frames, dtype=torch.bool, device=device)
    else:
        index = centers.clamp(max=sample_error.shape[1] - 1)
        belongs = sample_error[:, index] & inside & in_tensor.view(1, -1) & valid
    weight = belongs.to(dtype=dtype) * float(alpha) + 1.0
    return weight, valid


def weighted_mean(
    values: torch.Tensor,
    weight: torch.Tensor,
    valid: torch.Tensor,
    time_dim: int,
) -> torch.Tensor:
    """``sum(valid * w * l) / sum(valid * w)``. Padding is in neither sum."""

    if values.shape[0] != weight.shape[0] or values.shape[time_dim] != weight.shape[1]:
        raise ValueError(
            f"time axis mismatch values {tuple(values.shape)} dim {time_dim} weight {tuple(weight.shape)}"
        )
    gate = weight.to(dtype=values.dtype) * valid.to(device=values.device, dtype=values.dtype)
    view = [1] * values.ndim
    view[0] = gate.shape[0]
    view[time_dim] = gate.shape[1]
    broadcast = gate.view(*view).expand_as(values)
    present = broadcast > 0
    masked = torch.where(present, values, torch.zeros_like(values))
    numer = (masked * broadcast).sum()
    denom = broadcast.sum()
    zero = numer * 0
    return torch.where(denom > 0, numer / denom.clamp_min(torch.finfo(values.dtype).tiny), zero)


def _phase_maps(
    reference: torch.Tensor,
    generated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frequency = _difference_matrix(int(reference.shape[-2]), reference)
    time = _difference_matrix(int(reference.shape[-1]), reference)
    group_reference = torch.matmul(reference.transpose(1, 2), frequency)
    group_generated = torch.matmul(generated.transpose(1, 2), frequency)
    angular_reference = torch.matmul(reference, time)
    angular_generated = torch.matmul(generated, time)
    return (
        anti_wrapping(reference - generated),
        anti_wrapping(group_reference - group_generated),
        anti_wrapping(angular_reference - angular_generated),
    )


_MEL_BANDS = (40, 80, 160, 320)
_MEL_WINDOWS = (256, 512, 1024, 2048)


def discriminator_loss(cqt, mrd, clean, enhanced):
    def family(module):
        real, fake, _, _ = module(clean, enhanced.detach())
        return sum(least_squares_discriminator(r, f) for r, f in zip(real, fake, strict=True))

    return family(cqt) + family(mrd)


def _weighted_mel(reference, estimate, sample_error, content, alpha):
    total = reference.new_zeros(())
    for bands, window in zip(_MEL_BANDS, _MEL_WINDOWS, strict=True):
        hop = window // 4
        ref = _slaney_log_mel(reference, 16000, bands, window, hop, 1e-5)
        est = _slaney_log_mel(estimate, 16000, bands, window, hop, 1e-5)
        frames = min(ref.shape[-1], est.shape[-1])
        difference = (ref[..., :frames] - est[..., :frames]).abs()
        weight, valid = frame_weight_and_valid(sample_error, content, frames, hop, alpha, difference.dtype)
        total = total + weighted_mean(difference, weight, valid, -1)
    return total


def generator_loss(cqt_outputs, mrd_outputs, clean, parts, *, guidance, sample_error=None, content=None, tdt=None, hop=100):
    """Return the ordered seven SEMamba++ terms and the optional unweighted TDT NLL."""
    _, fake_q, real_qf, fake_qf = cqt_outputs
    _, fake_r, real_rf, fake_rf = mrd_outputs
    adversarial = sum(least_squares_generator(f) for f in fake_q) + sum(least_squares_generator(f) for f in fake_r)
    matching = feature_matching(real_qf, fake_qf, reduction="sum") + feature_matching(real_rf, fake_rf, reduction="sum")
    if content is None:
        mel = summed_slaney_log_mel(clean, parts["audio"], 16000)
        magnitude = F.mse_loss(parts["clean_mag"], parts["mag"])
        ip, gd, af = _phase_maps(parts["clean_phase"], parts["phase"])
        phase = ip.mean() + gd.mean() + af.mean()
        complex_loss = F.mse_loss(parts["clean_complex"], parts["complex"]) * 2
        consistency = F.mse_loss(parts["complex"], parts["recomputed"]) * 2
    else:
        alpha = guidance.error_weight_alpha if guidance.error_weight_enabled else 0.0
        if sample_error is None or not guidance.error_weight_enabled:
            if guidance.error_weight_enabled:
                raise ValueError("Error weighting requires a recognition error mask")
            sample_error = clean.new_zeros((len(clean), 1), dtype=torch.bool)
        weight, valid = frame_weight_and_valid(sample_error, content, parts["clean_mag"].shape[2],
                                               hop, alpha, clean.dtype)

        def reduce(values, time_dim=2):
            return weighted_mean(values, weight, valid, time_dim)

        mel = _weighted_mel(clean, parts["audio"], sample_error, content, alpha)
        magnitude = reduce((parts["clean_mag"] - parts["mag"]).square())
        ip, gd, af = _phase_maps(parts["clean_phase"], parts["phase"])
        phase = reduce(ip) + reduce(gd, 1) + reduce(af)
        complex_loss = reduce((parts["clean_complex"] - parts["complex"]).square()) * 2
        consistency = reduce((parts["complex"] - parts["recomputed"]).square()) * 2
    total = adversarial + matching + mel * 0.1 + magnitude * 0.9 + phase * 0.3 + complex_loss * 0.1 + consistency * 0.1
    terms = dict(adversarial=adversarial, matching=matching, mel=mel, magnitude=magnitude,
                 phase=phase, complex=complex_loss, consistency=consistency)
    if tdt is not None:
        total = total + tdt
        terms["tdt"] = tdt
    return total, terms
