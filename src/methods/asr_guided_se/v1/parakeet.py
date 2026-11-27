"""Frozen Parakeet TDT v2 with timed hypotheses and waveform gradients."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

TDT_V2_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
TDT_V2_DECODING = {
    "strategy": "greedy_batch",
    "compute_timestamps": False,
    "tdt_include_token_duration": True,
    "use_cuda_graph_decoder": False,
}


def apply_tdt_v2_decoding(model):
    from omegaconf import OmegaConf, open_dict

    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.compute_timestamps = False
        cfg.tdt_include_token_duration = True
        if cfg.get("greedy") is None:
            cfg.greedy = OmegaConf.create({})
        with open_dict(cfg.greedy):
            cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(cfg, verbose=False)
    return dict(TDT_V2_DECODING)


@dataclass(frozen=True)
class Token:
    token_id: int
    token: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class EncodeResult:
    encoded: torch.Tensor
    encoded_length: torch.Tensor
    layer_outputs: dict[int, torch.Tensor]


@dataclass(frozen=True)
class Recognition:
    text: str
    encoded_length: int
    samples_per_encoder_frame: int
    tokens: tuple[Token, ...]


def _checkpoint_path(model_id):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_id, filename="parakeet-tdt-0.6b-v2.nemo",
                           revision="c00e38779b028d48624b08ab3d9909952db43901")


def clip_encoded_time(
    encoded: torch.Tensor,
    encoded_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make encoded T equal ``max(encoded_length)`` so the TDT length check passes."""

    if encoded.ndim != 3:
        raise ValueError(f"encoded must be [B, D, T], got {tuple(encoded.shape)}")
    time = encoded.size(-1)
    clipped_length = encoded_length.to(dtype=torch.int64).clamp(min=0, max=time)
    cut = int(clipped_length.max().item()) if clipped_length.numel() else 0
    if time != cut:
        encoded = encoded[:, :, :cut]
    return encoded, clipped_length


def checkpoint_features(featurizer, waveform, lengths):
    """Apply the v2 checkpoint's centered, reflect-padded filterbank.

    NeMo 3 changed preemphasis at the end of padded rows, STFT padding and
    the frame count. The saved Hann window and mel filters still apply.
    """
    from nemo.collections.asr.parts.preprocessing.features import normalize_batch

    features_length = torch.div(lengths, featurizer.hop_length, rounding_mode="floor") + 1
    if torch.any(features_length < 2):
        raise ValueError("Parakeet per-feature normalization needs at least two frames")
    x = torch.cat((waveform[:, :1], waveform[:, 1:] - featurizer.preemph * waveform[:, :-1]), dim=1)
    window = featurizer.window.to(x.device, torch.float32)
    explicit_pad = x.is_cuda and torch.are_deterministic_algorithms_enabled()
    if explicit_pad:
        x = _index_reflect(x, featurizer.n_fft // 2)
    spectrum = torch.stft(x, featurizer.n_fft, featurizer.hop_length, featurizer.win_length,
                          window, center=not explicit_pad, pad_mode="reflect", return_complex=True)
    magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-5)
    power = magnitude.pow(featurizer.mag_power)
    mel = torch.matmul(featurizer.fb.to(power.dtype), power)
    guard = featurizer.log_zero_guard_value_fn(mel)
    mel = torch.log(mel + guard) if featurizer.log_zero_guard_type == "add" else torch.log(mel.clamp_min(guard))
    normalized, _, _ = normalize_batch(mel, features_length, normalize_type=featurizer.normalize)
    return normalized, features_length


def checkpoint_subsampling(encoder, features, lengths):
    """Run the v2 convolutions over padded rows, then restore their true lengths."""
    from nemo.collections.asr.parts.submodules.subsampling import calc_length

    subsampling = encoder.pre_encode
    # NeMo 3 masks every convolution using its input lengths. The v2 checkpoint
    # first convolved the full padded batch and masked in the Conformer.
    padded_lengths = torch.full_like(lengths, features.size(-1))
    embeddings, _ = subsampling(x=features.transpose(1, 2), lengths=padded_lengths)
    output_lengths = calc_length(
        lengths, subsampling._left_padding + subsampling._right_padding,
        subsampling._kernel_size, subsampling._stride, subsampling._ceil_mode, subsampling._sampling_num,
    )
    return embeddings, output_lengths


class FrozenParakeet(nn.Module):
    """One frozen Parakeet TDT v2 checkpoint."""

    def __init__(self, model_id: str = TDT_V2_MODEL_ID) -> None:
        super().__init__()
        if model_id != TDT_V2_MODEL_ID:
            raise ValueError(f"FrozenParakeet only loads {TDT_V2_MODEL_ID}, got {model_id}")
        # Numba discovers its toolkit during NeMo import. The isolated worker
        # uses python -S, so locate the venv by its executable.
        toolkit = Path(sys.executable).parent.parent / "cuda"
        if (toolkit / "nvvm").is_dir():
            os.environ.setdefault("CUDA_HOME", str(toolkit))
        import nemo.collections.asr as nemo_asr

        self.model_id = model_id
        self.model = nemo_asr.models.ASRModel.restore_from(_checkpoint_path(model_id), map_location="cpu")
        self.model.preprocessor.featurizer.use_grads = True
        self.effective_decoding = apply_tdt_v2_decoding(self.model)
        self._read_tdt_checkpoint()
        hop = int(self.model.preprocessor.featurizer.hop_length)
        subsampling = int(self.model.encoder.subsampling_factor)
        self.samples_per_encoder_frame = hop * subsampling
        self.encoder_dim = int(self.model.encoder.d_model)
        self.encoder_layers = int(len(self.model.encoder.layers))
        self._tdt_loss = None
        self._deterministic_tdt_loss = None
        self._tdt_enabled = False
        self.eval()

    def _read_tdt_checkpoint(self) -> None:
        loss = getattr(self.model, "loss", None)
        blank = getattr(loss, "_blank", None)
        tdt_kwargs = getattr(getattr(self.model, "cfg", None), "loss", None)
        tdt_kwargs = getattr(tdt_kwargs, "tdt_kwargs", None)
        missing = [
            name
            for name, present in (
                ("blank", blank is not None),
                ("durations", tdt_kwargs is not None and "durations" in tdt_kwargs),
                ("sigma", tdt_kwargs is not None and "sigma" in tdt_kwargs),
                ("omega", tdt_kwargs is not None and "omega" in tdt_kwargs),
            )
            if not present
        ]
        if missing:
            raise RuntimeError(
                "Parakeet TDT v2 checkpoint is missing "
                + ", ".join(missing)
                + "; refusing to hardcode TDT settings"
            )
        self.blank_id = int(blank)
        self.durations = [int(duration) for duration in tdt_kwargs.durations]
        self.sigma = float(tdt_kwargs.sigma)
        self.omega = float(tdt_kwargs.omega)
        if not self.durations:
            raise RuntimeError("Parakeet TDT v2 durations are empty")

    def train(self, mode: bool = True) -> nn.Module:
        """Stay in eval with frozen parameters. ``mode`` does not unfreeze them."""

        del mode
        super().train(False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self._tdt_enabled:
            self._set_decoder_backward_enabled(True)
        else:
            self.model.decoder.eval()
        return self

    def enable_tdt_loss(self) -> None:
        """Build ``TDTLossNumba`` from the checkpoint. A second call is a no-op."""

        if self._tdt_enabled:
            return
        from nemo.collections.asr.parts.numba.rnnt_loss import TDTLossNumba

        kwargs = {
            "blank": self.blank_id,
            "durations": list(self.durations),
            "reduction": "none",
            "sigma": self.sigma,
            "fastemit_lambda": 0.0,
            "clamp": -1.0,
        }
        try:
            self._tdt_loss = TDTLossNumba(**kwargs, omega=self.omega)
            self._deterministic_tdt_loss = (
                self._tdt_loss if self.omega == 0.0 else TDTLossNumba(**kwargs, omega=0.0)
            )
        except Exception as exc:
            raise RuntimeError(
                "TDTLossNumba could not be built. Another TDT loss is not substituted."
            ) from exc
        self._tdt_enabled = True
        self.train(False)

    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> EncodeResult:
        layer_indices = self._resolve_layer_indices(layers)
        batch_size = int(waveforms.shape[0])
        valid = lengths > 0
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            encoded = waveforms.new_zeros(batch_size, self.encoder_dim, 0) + waveforms.sum() * 0
            encoded_length = torch.zeros(batch_size, device=lengths.device, dtype=torch.int64)
            return EncodeResult(
                encoded=encoded,
                encoded_length=encoded_length,
                layer_outputs={index: encoded for index in layer_indices},
            )
        idx = None
        waveforms_valid = waveforms
        lengths_valid = lengths
        if n_valid != batch_size:
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms_valid = waveforms.index_select(0, idx)
            lengths_valid = lengths.index_select(0, idx)
        features, feature_lengths = checkpoint_features(
            self.model.preprocessor.featurizer,
            waveforms_valid.to(dtype=torch.float32),
            lengths_valid,
        )
        layer_outputs_valid: dict[int, torch.Tensor] = {}
        handles = []

        def capture_layer(layer_index: int):
            def hook(_module, _args, out):
                layer_outputs_valid[layer_index] = _layer_tensor(out, self.encoder_dim)

            return hook

        try:
            for layer_index in layer_indices:
                handles.append(self.model.encoder.layers[layer_index].register_forward_hook(capture_layer(layer_index)))
            embeddings, encoder_lengths = checkpoint_subsampling(self.model.encoder, features, feature_lengths)
            encoded_valid, encoded_length_valid = self.model.encoder(
                audio_signal=embeddings, length=encoder_lengths, bypass_pre_encode=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        encoded_length_valid = encoded_length_valid.to(dtype=torch.int64)
        encoded_valid, encoded_length_valid = clip_encoded_time(encoded_valid, encoded_length_valid)
        cut = encoded_valid.size(-1)
        for layer_index, layer_tensor in list(layer_outputs_valid.items()):
            if layer_tensor.size(-1) != cut:
                layer_outputs_valid[layer_index] = layer_tensor[:, :, :cut]
        missing = [index for index in layer_indices if index not in layer_outputs_valid]
        if missing:
            raise RuntimeError(f"encoder layers did not produce outputs: {missing}")
        if idx is None:
            return EncodeResult(
                encoded=encoded_valid,
                encoded_length=encoded_length_valid,
                layer_outputs=layer_outputs_valid,
            )
        encoded = _scatter_batch(encoded_valid, idx, batch_size)
        encoded_length = encoded_length_valid.new_zeros(batch_size).scatter(0, idx, encoded_length_valid)
        layer_outputs = {
            layer_index: _scatter_batch(layer_tensor, idx, batch_size)
            for layer_index, layer_tensor in layer_outputs_valid.items()
        }
        return EncodeResult(encoded=encoded, encoded_length=encoded_length, layer_outputs=layer_outputs)

    def encode_layer(self, waveforms, lengths, layer):
        result = self.encode(waveforms, lengths, layers=(layer,))
        return result.layer_outputs[layer], result.encoded_length

    def recognize(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> list[Recognition]:
        with torch.no_grad():
            encoded = self.encode(waveforms, lengths)
        return self.recognize_encoded(encoded.encoded, encoded.encoded_length)

    def recognize_encoded(
        self,
        encoded: torch.Tensor,
        encoded_length: torch.Tensor,
    ) -> list[Recognition]:
        batch_size = int(encoded.shape[0])
        valid = encoded_length > 0
        if not bool(valid.any()):
            return [self._empty_recognition() for _ in range(batch_size)]
        idx = None
        encoded_valid = encoded
        encoded_length_valid = encoded_length
        if not bool(valid.all()):
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            encoded_valid = encoded.index_select(0, idx)
            encoded_length_valid = encoded_length.index_select(0, idx)
        encoded_lengths = encoded_length_valid.detach().cpu().tolist()
        decoder_training = bool(self.model.decoder.training)
        self.model.decoder.eval()
        try:
            with torch.no_grad():
                predictions = self.model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded_valid.detach(),
                    encoded_lengths=encoded_length_valid,
                    return_hypotheses=True,
                )
        finally:
            self._set_decoder_backward_enabled(decoder_training and self._tdt_enabled)
        timed = [
            self._timed_recognition(prediction, int(encoded_lengths[i]))
            for i, prediction in enumerate(predictions)
        ]
        if idx is None:
            return timed
        results = [self._empty_recognition() for _ in range(batch_size)]
        for source, destination in enumerate(idx.tolist()):
            results[destination] = timed[source]
        return results

    def loss_from_ids(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        token_ids: Sequence[Sequence[int]],
        *,
        deterministic_tdt: bool = False,
    ) -> torch.Tensor:
        """Per-sample TDT NLL. Empty targets and zero lengths stay connected to ``waveforms``."""

        if not self._tdt_enabled or self._tdt_loss is None or self._deterministic_tdt_loss is None:
            raise RuntimeError("TDT loss is not enabled")
        batch_size = int(waveforms.shape[0])
        if len(token_ids) != batch_size:
            raise ValueError(f"token_ids length {len(token_ids)} != batch {batch_size}")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        nonempty = [len(ids) > 0 for ids in token_ids]
        valid = (lengths > 0) & torch.tensor(nonempty, device=lengths.device, dtype=torch.bool)
        if int(valid.sum().item()) == 0:
            return waveforms.new_zeros(batch_size) + waveforms.sum() * 0
        idx = None
        token_id_lists = [list(ids) for ids in token_ids]
        waveforms_valid = waveforms
        lengths_valid = lengths
        if int(valid.sum().item()) != batch_size:
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            waveforms_valid = waveforms.index_select(0, idx)
            lengths_valid = lengths.index_select(0, idx)
            token_id_lists = [token_id_lists[int(i)] for i in idx.tolist()]
        target_lengths_cpu = torch.tensor([len(ids) for ids in token_id_lists], dtype=torch.int64)
        max_length = int(target_lengths_cpu.max().item())
        targets_cpu = torch.zeros((len(token_id_lists), max_length), dtype=torch.int64)
        for row, ids in enumerate(token_id_lists):
            if ids:
                targets_cpu[row, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
        target_lengths = target_lengths_cpu.to(device=waveforms.device)
        targets = targets_cpu.to(device=waveforms.device)
        self._set_decoder_backward_enabled(True)
        encoded = self.encode(waveforms_valid, lengths_valid)
        decoder_outputs, _, _ = self.model.decoder(targets=targets, target_length=target_lengths)
        logits = self.model.joint.joint(
            encoded.encoded.transpose(1, 2),
            decoder_outputs.transpose(1, 2),
        )
        tdt_loss = self._deterministic_tdt_loss if deterministic_tdt else self._tdt_loss
        nll_valid = tdt_loss(
            acts=logits,
            labels=targets.to(dtype=torch.int64),
            act_lens=encoded.encoded_length.to(dtype=torch.int64),
            label_lens=target_lengths.to(dtype=torch.int64),
        )
        if idx is None:
            return nll_valid
        return nll_valid.new_zeros(batch_size).scatter(0, idx, nll_valid)

    def ids_to_text(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        return self.model.tokenizer.ids_to_text(list(token_ids))

    def _resolve_layer_indices(self, layers: Sequence[int] | None) -> list[int]:
        if not layers:
            return []
        resolved: list[int] = []
        seen: set[int] = set()
        for index in layers:
            layer_index = int(index)
            if layer_index < 0:
                layer_index += self.encoder_layers
            if not 0 <= layer_index < self.encoder_layers:
                raise ValueError(
                    f"encoder layer index {index} is out of range for {self.encoder_layers} layers"
                )
            if layer_index not in seen:
                seen.add(layer_index)
                resolved.append(layer_index)
        return resolved

    def _empty_recognition(self) -> Recognition:
        return Recognition(
            text="",
            encoded_length=0,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=(),
        )

    def _timed_recognition(self, prediction: Any, encoded_length: int) -> Recognition:
        token_ids = _as_ints(getattr(prediction, "y_sequence", None))
        starts = _as_ints(getattr(prediction, "timestamp", None))
        raw_durations = getattr(prediction, "token_duration", None)
        durations = _as_ints(raw_durations)
        # NeMo 2.2.1 batched_hyps_to_hypotheses encodes an all-zero row as [].
        if (
            isinstance(raw_durations, list)
            and not raw_durations
            and self.effective_decoding["tdt_include_token_duration"]
            and len(token_ids) == len(starts)
        ):
            durations = [0] * len(token_ids)
        if not (len(token_ids) == len(starts) == len(durations)):
            raise RuntimeError(
                "y_sequence/timestamp/token_duration lengths differ: "
                f"{len(token_ids)}, {len(starts)}, {len(durations)}"
            )
        kept: list[tuple[int, int, int]] = []
        for token_id, start, duration in zip(token_ids, starts, durations, strict=True):
            if token_id >= self.blank_id:
                continue
            start_offset = max(int(start), 0)
            if start_offset >= encoded_length:
                continue
            end_offset = min(start_offset + max(int(duration), 1), encoded_length)
            kept.append((token_id, start_offset, end_offset))
        token_texts = self.model.decoding.decode_ids_to_tokens([token_id for token_id, _, _ in kept])
        if len(token_texts) != len(kept):
            raise RuntimeError("token text count does not match the kept token ids")
        tokens = tuple(
            Token(token_id=token_id, token=str(token_text), start_offset=start_offset, end_offset=end_offset)
            for (token_id, start_offset, end_offset), token_text in zip(kept, token_texts, strict=True)
        )
        text = getattr(prediction, "text", None)
        if not isinstance(text, str):
            text = self.ids_to_text([token.token_id for token in tokens])
        return Recognition(
            text=text,
            encoded_length=encoded_length,
            samples_per_encoder_frame=self.samples_per_encoder_frame,
            tokens=tokens,
        )

    def _set_decoder_backward_enabled(self, enabled: bool) -> None:
        if enabled:
            self.model.decoder.train()
            for module in self.model.decoder.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            return
        self.model.decoder.eval()


def _index_reflect(waveform: torch.Tensor, pad: int) -> torch.Tensor:
    """Reflect ``pad`` samples on each side by indexing, not ``reflection_pad1d``."""

    if pad == 0:
        return waveform
    length = int(waveform.shape[-1])
    if length <= pad:
        raise RuntimeError(f"waveform length {length} cannot be reflect-padded by {pad} samples")
    left = waveform[..., 1 : pad + 1].flip(-1)
    right = waveform[..., -pad - 1 : -1].flip(-1)
    return torch.cat((left, waveform, right), dim=-1)


def _layer_tensor(out: Any, encoder_dim: int) -> torch.Tensor:
    value = out[0] if isinstance(out, tuple) else out
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise RuntimeError(f"encoder layer output must be a rank-3 tensor, got {type(value).__name__}")
    if value.shape[-1] == encoder_dim:
        return value.transpose(1, 2)
    if value.shape[1] == encoder_dim:
        return value
    raise RuntimeError(
        f"encoder layer output shape {tuple(value.shape)} does not contain d_model {encoder_dim}"
    )


def _scatter_batch(valid: torch.Tensor, idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    index = idx.view(-1, *([1] * (valid.ndim - 1))).expand_as(valid)
    shape = (batch_size, *valid.shape[1:])
    return valid.new_zeros(shape).scatter(0, index, valid)


def _as_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        if value.numel() == 0:
            return []
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]
