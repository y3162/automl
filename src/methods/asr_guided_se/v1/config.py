"""The three v1 switches. Projection width, the ASR id, and TDT's coefficient are fixed."""

from __future__ import annotations

import math
from dataclasses import dataclass


GUIDANCE_DIM = 128
STFT_HOP = 100
ASR_HOP = 1280


@dataclass(frozen=True)
class GuidanceV1:
    """ASR feature, error-mask and TDT settings for one experiment."""

    asr_features_enabled: bool = False
    encoder_layer: int = 15
    error_weight_enabled: bool = False
    error_weight_alpha: float = 1.0
    tdt_loss_enabled: bool = False

    def __post_init__(self) -> None:
        _flag("asr_features_enabled", self.asr_features_enabled)
        _flag("error_weight_enabled", self.error_weight_enabled)
        _flag("tdt_loss_enabled", self.tdt_loss_enabled)
        if isinstance(self.encoder_layer, bool) or not isinstance(self.encoder_layer, int) or self.encoder_layer < 0:
            raise ValueError("encoder_layer must be an integer >= 0")
        if (
            isinstance(self.error_weight_alpha, bool)
            or not isinstance(self.error_weight_alpha, (int, float))
            or not math.isfinite(float(self.error_weight_alpha))
            or float(self.error_weight_alpha) < 0
        ):
            raise ValueError("error_weight_alpha must be a finite number >= 0")
        object.__setattr__(self, "error_weight_alpha", float(self.error_weight_alpha))

    @property
    def uses_asr(self) -> bool:
        return self.asr_features_enabled or self.error_weight_enabled or self.tdt_loss_enabled

    @property
    def changes_loss(self) -> bool:
        return self.error_weight_enabled or self.tdt_loss_enabled


def _flag(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
