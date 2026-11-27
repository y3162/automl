"""AutoML model construction and named module checkpoints for ASR-guided SE v1."""

from dataclasses import asdict
from io import BytesIO

import torch

from automl.model import Model

from . import semambapp
from .config import GuidanceV1


_BACKBONES = {"semambapp": semambapp}


def _backbone(name):
    if not isinstance(name, str) or name not in _BACKBONES:
        raise ValueError(f"Unknown speech backbone {name!r}. Available: {', '.join(_BACKBONES)}")
    return _BACKBONES[name]


def save_tensors(state):
    buffer = BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def load_tensors(state):
    return torch.load(BytesIO(state), map_location="cpu", weights_only=True)


class SpeechEnhancementModel(Model):
    """Expose the selected backbone's named modules to the trainer and checkpoint store."""

    def __init__(self, architecture=None, guidance=None, seed=42, *, backbone="semambapp", asr=None):
        implementation = _backbone(backbone)
        self.guidance = GuidanceV1(**(guidance or {}))
        config = implementation.Architecture(**(architecture or {}))
        modules = implementation.build_modules(config, self.guidance, seed, asr=asr)
        self._module_names = tuple(modules)
        for name, module in modules.items():
            setattr(self, name, module)

    @classmethod
    def resolve_settings(cls, settings, root):
        unknown = set(settings) - {"backbone", "architecture", "guidance", "seed"}
        if unknown:
            raise ValueError(f"Unknown speech model settings {sorted(unknown)}")
        backbone = settings.get("backbone", "semambapp")
        config = _backbone(backbone).Architecture(**settings.get("architecture", {}))
        guidance = GuidanceV1(**settings.get("guidance", {}))
        seed = settings.get("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")
        return dict(backbone=backbone, architecture=asdict(config), guidance=asdict(guidance), seed=seed)

    def to(self, device):
        for name in self._module_names:
            getattr(self, name).to(device)
        return self

    def state_dict(self):
        return save_tensors({name: getattr(self, name).state_dict() for name in self._module_names})

    def load_state_dict(self, state):
        states = load_tensors(state) if isinstance(state, bytes) else state
        if set(states) != set(self._module_names):
            raise ValueError(f"Expected checkpoint states for {', '.join(self._module_names)}")
        for name in self._module_names:
            getattr(self, name).load_state_dict(states[name], strict=True)
