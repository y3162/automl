"""Regression values from the audio-db CPU reference GAN update."""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnAudio")
pytest.importorskip("torchaudio")

from automl.experiment.config import load_config, load_implementation
from methods.asr_guided_se.v1 import semambapp
from methods.asr_guided_se.v1.data import WaveBatch
from methods.asr_guided_se.v1.error_mask import centered_token_ids, crop_error_mask
from methods.asr_guided_se.v1.loss import frame_weight_and_valid, weighted_mean
from methods.asr_guided_se.v1.model import SpeechEnhancementModel, load_tensors
from methods.asr_guided_se.v1.parakeet import checkpoint_features, checkpoint_subsampling
from methods.asr_guided_se.v1.trainer import GANUpdate


class FrozenFixture(torch.nn.Module):
    samples_per_encoder_frame = 1280
    encoder_dim = 8
    encoder_layers = 16

    def __init__(self):
        super().__init__()
        self.source = torch.nn.Parameter(torch.ones(8))
        self.seen = []

    def encode_layer(self, waveforms, lengths, layer):
        self.seen.append((layer, torch.is_grad_enabled()))
        hidden = (self.source.view(1, -1, 1) + float(layer)).expand(len(waveforms), 8, 2).contiguous()
        return hidden, torch.full_like(lengths, 2)

    def recognize(self, waveforms, lengths):
        return [SimpleNamespace(tokens=[SimpleNamespace(token_id=7, start_offset=0, end_offset=2)])]

    def enable_tdt_loss(self):
        pass

    def loss_from_ids(self, waveforms, lengths, token_ids, *, deterministic_tdt):
        assert token_ids == [[7]]
        return waveforms.square().mean()


def test_training_config_selects_the_versioned_backbone():
    root = Path(__file__).resolve().parents[2]
    config = load_config("configs/asr_guided_se/v1/train.yaml", root)["components"]["model"]
    component = load_implementation(config["implementation"])
    assert component is SpeechEnhancementModel
    settings = component.resolve_settings(config["settings"], root)
    assert settings["backbone"] == "semambapp"
    assert component.resolve_settings({}, root)["backbone"] == settings["backbone"]
    with pytest.raises(ValueError, match="Unknown speech backbone"):
        component.resolve_settings({"backbone": "unknown"}, root)


def test_reference_gan_update_with_accumulation():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        asr = FrozenFixture()
        architecture = dict(hid_feature=8, num_tfmamba=1, d_state=4, d_conv=2)
        guidance = dict(asr_features_enabled=True, encoder_layer=3, tdt_loss_enabled=True, error_weight_enabled=True)
        model = SpeechEnhancementModel(architecture, guidance, seed=1234, backbone="semambapp", asr=asr)
        assert isinstance(model.generator, semambapp.GuidedSEMambaPPGenerator)
        update = GANUpdate(model)
        rng = torch.Generator().manual_seed(3)
        noisy, clean = (torch.randn(1, 2048, generator=rng) for _ in range(2))
        batch = WaveBatch(clean, noisy, [clean[0]], [(0, 2048, 2048, 2048)])
        metrics = update.step([batch, batch], accumulation=2)
        assert asr.seen == [(3, False)] * 4
        assert not asr.training and asr.source.grad is None
        expected = dict(generator=36.0371628, discriminator=5.8139176, tdt=0.00566724,
                        adversarial=5.6360559, matching=26.891161, mel=4.7970333,
                        magnitude=0.9903367, phase=4.9529495, complex=5.2288241, consistency=1.2450476)
        assert metrics == pytest.approx(expected, rel=2e-5, abs=1e-7)
        state = load_tensors(model.state_dict())
        assert set(state) == {"generator", "mrd", "cqt"}
        assert set(state["generator"]) == set(model.generator.state_dict())
        assert state["generator"]["generator.encoder_layer_index"].item() == 3
        assert state["generator"]["generator.asr_fuse.bias"][:4].tolist() == pytest.approx(
            [0.08472943, 0.05746167, -0.07238643, 0.04355210], abs=2e-7)
        restored = SpeechEnhancementModel(architecture, guidance, seed=0, asr=FrozenFixture())
        restored.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(restored.generator(clean, noisy)["audio"], model.generator(clean, noisy)["audio"],
                                       rtol=0, atol=0)
        update.end_epoch()
        device = torch.device("cpu")
        training_state = load_tensors(update.training_state(device))
        expected = update.step([batch])
        resumed = GANUpdate(restored)
        resumed.restore_training(training_state, device)
        assert resumed.step([batch]) == expected
        for name in state:
            torch.testing.assert_close(getattr(restored, name).state_dict(), getattr(model, name).state_dict(),
                                       rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)


def test_errors_are_local_to_clean_token_times_and_content():
    clean = [dict(token_id=7, start_offset=0, end_offset=2), dict(token_id=9, start_offset=2, end_offset=4)]
    hypothesis = [dict(token_id=7, start_offset=0, end_offset=1)]
    mask = crop_error_mask(clean, hypothesis, crop_start=10, content_length=25, window=40, hop=10, full_length=44)
    assert mask == [False] * 10 + [True] * 15 + [False] * 15
    assert centered_token_ids(clean, 10, 20, 10) == [7]
    weights, valid = frame_weight_and_valid(torch.tensor([mask]), torch.tensor([25]), 5, 10, 1, torch.float32)
    values = torch.arange(5., requires_grad=True)
    result = weighted_mean(values[None, :], weights, valid, 1)
    result.backward()
    torch.testing.assert_close(values.grad, torch.tensor([.2, .4, .4, 0., 0.]))


def test_parakeet_v2_convolves_the_entire_padded_batch():
    subsampling = pytest.importorskip("nemo.collections.asr.parts.submodules.subsampling")
    module = subsampling.ConvSubsampling("dw_striding", 8, feat_in=16, feat_out=8, conv_channels=2)
    features = torch.randn(2, 16, 33)
    features[1, :, 17:] = 0
    features.requires_grad_()
    embeddings, lengths = checkpoint_subsampling(SimpleNamespace(pre_encode=module), features, torch.tensor([33, 17]))
    # NeMo 2 used a plain Sequential here, including convolution biases in padded frames.
    convolved = torch.nn.Sequential.forward(module.conv, features.transpose(1, 2).unsqueeze(1))
    expected = module.out(convolved.transpose(1, 2).flatten(2))
    assert lengths.tolist() == [5, 3]
    torch.testing.assert_close(embeddings, expected, atol=0, rtol=0)
    actual_grad, = torch.autograd.grad(embeddings.sum(), features)
    expected_grad, = torch.autograd.grad(expected.sum(), features)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


def test_parakeet_v2_filterbank_and_waveform_gradient():
    features = pytest.importorskip("nemo.collections.asr.parts.preprocessing.features")
    featurizer = features.FilterbankFeatures(n_fft=512, n_window_size=320, n_window_stride=160, nfilt=4,
                                             pad_to=0, dither=0., use_grads=True).eval()
    wave = ((torch.arange(1024) * 31) % 167 - 83).float() / 83
    batch = torch.stack((wave, torch.nn.functional.pad(wave[:521], (0, 503)))).requires_grad_()
    mel, lengths = checkpoint_features(featurizer, batch, torch.tensor([1024, 521]))
    assert lengths.tolist() == [7, 4]
    assert torch.count_nonzero(mel[1, :, 4:]) == 0
    assert mel[1, :, 3].tolist() == pytest.approx([-0.873455, -1.4119275, -1.4897461, -1.4779474], abs=2e-6)
    (mel * torch.linspace(-0.5, 0.5, mel.numel()).reshape_as(mel)).sum().backward()
    assert batch.grad[1, 519:523].tolist() == pytest.approx(
        [0.00016557425, 0.0061284853, 0.0057882136, 0.0045900596], abs=3e-7)
