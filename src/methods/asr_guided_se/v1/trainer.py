"""Single-device FP32 SEMamba++ training with ASR-guided objectives."""

import math
import random
from itertools import chain

import numpy as np
import torch

from automl.experiment.runtime import Interrupted
from automl.trainer import Trainer

from .data import VoiceBankPairs, crop_batch
from .loss import discriminator_loss, generator_loss, guidance_terms
from .model import SpeechEnhancementModel, load_tensors, save_tensors
from .random import mix_seed, seed_training_stream, set_determinism


class GANUpdate:
    def __init__(self, model, *, learning_rate=0.0002, gamma=0.99, valid_frame_aggregation=False, grad_clip=None):
        self.model = model
        self.valid_frame_aggregation = valid_frame_aggregation
        self.grad_clip = grad_clip
        self.asr = model.generator.recognizer() if model.guidance.uses_asr else None
        if model.guidance.tdt_loss_enabled:
            self.asr.enable_tdt_loss()
        adamw = dict(lr=learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
        self.optim_g = torch.optim.AdamW(model.generator.parameters(), **adamw)
        self.optim_d = torch.optim.AdamW(chain(model.mrd.parameters(), model.cqt.parameters()), **adamw)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma)

    @torch.no_grad()
    def _recognize(self, waveform, kind):
        waveform = waveform.detach().to(next(self.model.generator.parameters()).device)
        lengths = waveform.new_tensor([waveform.numel()], dtype=torch.long)
        return self.asr.recognize(waveform.unsqueeze(0), lengths)[0]

    def _generator_loss(self, batch, parts):
        model = self.model
        audio = parts["audio"]
        cqt_outputs = model.cqt(batch.clean, audio)
        mrd_outputs = model.mrd(batch.clean, audio)
        mask = content = tdt = None
        if model.guidance.uses_asr or self.valid_frame_aggregation:
            mask, content, tdt = guidance_terms(
                model.guidance, audio, batch.spans, full_references=batch.references,
                recognize=self._recognize, asr=self.asr,
            )
        return generator_loss(cqt_outputs, mrd_outputs, batch.clean, parts, guidance=model.guidance,
                              sample_error=mask, content=content, tdt=tdt, hop=model.generator.hop)

    def step(self, batches, accumulation=1):
        """Apply D to detached estimates, then backpropagate G through the updated, frozen D."""
        model = self.model
        counts = [len(batch.clean) for batch in batches]
        if not 0 < len(counts) <= accumulation or min(counts) < 1:
            raise ValueError("Expected a nonempty optimizer window of micro-batches")
        uniform = len(counts) == accumulation and len(set(counts)) == 1
        total_count = sum(counts)

        def backward(loss, count):
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError("Non-finite SEMamba++ loss")
            scaled = loss / accumulation if uniform else loss * (count / total_count)
            scaled.backward()

        for module in (model.generator, model.mrd, model.cqt):
            module.train()
        self.optim_d.zero_grad(set_to_none=True)
        keep_graph = len(batches) == 1
        retained = None
        d_total = 0
        for batch, count in zip(batches, counts, strict=True):
            with torch.set_grad_enabled(keep_graph):
                parts = model.generator(batch.clean, batch.noisy, batch.lengths)
            if keep_graph:
                retained = parts
            d_loss = discriminator_loss(model.cqt, model.mrd, batch.clean, parts["audio"])
            backward(d_loss, count)
            d_total = d_total + d_loss.detach()
            del parts, d_loss
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.mrd.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(model.cqt.parameters(), self.grad_clip)
        self.optim_d.step()

        self.optim_g.zero_grad(set_to_none=True)
        metrics = {"discriminator": d_total}
        try:
            model.mrd.requires_grad_(False)
            model.cqt.requires_grad_(False)
            for batch, count in zip(batches, counts, strict=True):
                parts = retained if keep_graph else model.generator(batch.clean, batch.noisy, batch.lengths)
                g_loss, terms = self._generator_loss(batch, parts)
                backward(g_loss, count)
                for name, value in dict(generator=g_loss, **terms).items():
                    metrics[name] = metrics.get(name, 0) + value.detach()
                del parts, g_loss, terms
        finally:
            model.mrd.requires_grad_(True)
            model.cqt.requires_grad_(True)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), self.grad_clip)
        self.optim_g.step()
        return {k: float(v) / len(batches) for k, v in metrics.items()}

    def end_epoch(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def training_state(self, device):
        numpy_rng = np.random.get_state()
        return save_tensors({
            "optim_g": self.optim_g.state_dict(), "optim_d": self.optim_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(), "scheduler_d": self.scheduler_d.state_dict(),
            "python": random.getstate(), "numpy": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        })

    def restore_training(self, state, device):
        for key in ("optim_g", "optim_d", "scheduler_g", "scheduler_d"):
            getattr(self, key).load_state_dict(state[key])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"], device)


class GuidedSEMambaPPTrainer(Trainer):
    supports_resume = True

    def __init__(self, epochs=100, max_steps=None, batch_size=8, micro_batch_size=8, window=48000,
                 seed=42, device="cuda", learning_rate=0.0002, gamma=0.99, grad_clip=None,
                 valid_frame_aggregation=False, save_every=None, selection_metric=None):
        self.epochs, self.max_steps = epochs, max_steps
        self.batch_size, self.micro_batch_size, self.window = batch_size, micro_batch_size, window
        self.seed, self.device = seed, torch.device(device)
        self.learning_rate, self.gamma, self.grad_clip = learning_rate, gamma, grad_clip
        self.valid_frame_aggregation = valid_frame_aggregation
        self.save_every, self.selection_metric = save_every, selection_metric

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        for key in ("epochs", "batch_size", "micro_batch_size", "window", "max_steps", "save_every"):
            if values[key] is None and key in {"max_steps", "save_every"}:
                continue
            if type(values[key]) is not int or values[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if values["batch_size"] % values["micro_batch_size"]:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        if values["window"] <= 1024:
            raise ValueError("The longest mel window needs more than 1024 waveform samples")
        for key in ("learning_rate", "gamma", "grad_clip"):
            if key == "grad_clip" and values[key] is None:
                continue
            if type(values[key]) not in {float, int} or not math.isfinite(values[key]) or values[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if type(values["valid_frame_aggregation"]) is not bool or type(values["seed"]) is not int:
            raise ValueError("Expected a boolean valid_frame_aggregation and an integer seed")
        if torch.device(values["device"]).type not in {"cuda", "cpu"}:
            raise ValueError("device must be cpu or cuda")
        if values["selection_metric"] not in {None, "wer"}:
            raise ValueError("selection_metric must be wer or null")
        return values

    @classmethod
    def validate_components(cls, implementations):
        for key, cls_ in (("model", SpeechEnhancementModel), ("dataset", VoiceBankPairs)):
            if key not in implementations or not issubclass(implementations[key], cls_):
                raise ValueError(f"{key} must implement {cls_.__name__}")

    def validate_resume(self, training_state, progress):
        if set(progress) != {"step", "epoch", "offset"} or any(type(n) is not int or n < 0 for n in progress.values()):
            raise ValueError("Invalid step, epoch or sample offset in speech checkpoint")
        state = load_tensors(training_state)
        if set(state) != {"optim_g", "optim_d", "scheduler_g", "scheduler_d", "python", "numpy", "torch", "cuda"}:
            raise ValueError("Missing optimizer, scheduler or random state in speech checkpoint")
        if self.device.type == "cuda" and state["cuda"] is None:
            raise ValueError("The checkpoint has no CUDA random state")

    def train(self, components, context, *, start_method):
        cuda = self.device.type == "cuda"
        set_determinism(True, cuda=cuda)
        if cuda and self.device.index is not None:
            torch.cuda.set_device(self.device)
        version, dataset = context.open_dataset(components)
        ids = dataset.sample_ids("train")
        if not ids:
            raise ValueError("The training split is empty")
        model = context.prepare_model(components).to(self.device)
        update = GANUpdate(model, learning_rate=self.learning_rate, gamma=self.gamma, grad_clip=self.grad_clip,
                           valid_frame_aggregation=self.valid_frame_aggregation)
        seed_training_stream(self.seed, cuda=cuda)
        step = epoch = offset = position = 0
        if start_method == "resume":
            initial = context.initial_states()
            self.validate_resume(initial["training"], initial["progress"])
            update.restore_training(load_tensors(initial["training"]), self.device)
            step, epoch, offset = (initial["progress"][k] for k in ("step", "epoch", "offset"))
            if offset > len(ids):
                raise ValueError("Checkpoint sample offset exceeds training split size")

        def states():
            return dict(model_state=model.state_dict(), training_state=update.training_state(self.device),
                        progress=dict(step=step, epoch=epoch, offset=offset))

        context.start_training(**states())
        latest = None
        generator = torch.Generator().manual_seed(mix_seed("per_rank", self.seed, 0))
        base_order = torch.randperm(len(ids), generator=generator).tolist()
        while epoch < self.epochs:
            order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(epoch)).tolist()
            while offset < len(ids) and (self.max_steps is None or step < self.max_steps):
                if context.stop_requested():
                    context.save_checkpoint(**states(), usage_range=position, reuse_identical=True)
                    raise Interrupted("Speech training stopped between optimizer updates")
                batch_ids = [ids[base_order[i]] for i in order[offset:offset + self.batch_size]]
                batches = [crop_batch([dataset.get_sample(i) for i in batch_ids[j:j + self.micro_batch_size]],
                                      self.window, self.seed, epoch, self.device)
                           for j in range(0, len(batch_ids), self.micro_batch_size)]
                metrics = update.step(batches, self.batch_size // self.micro_batch_size)
                step += 1
                position += 1
                offset += len(batch_ids)
                context.add(usage_id=str(step), position=position, batch_id=step, dataset_version=version,
                            split="train", sample_ids=batch_ids,
                            details={"epoch": epoch, "spans": [s for batch in batches for s in batch.spans]})
                context.log("training_progress", step=step, epoch=epoch, **metrics)
                latest = None
                if self.save_every and step % self.save_every == 0 and offset < len(ids):
                    latest = context.save_checkpoint(**states(), usage_range=position)
            if offset < len(ids):
                break
            metric = score = None
            if "validation" in components:
                try:
                    report = context.evaluate(components, model, component="validation", progress={"step": step})
                except Interrupted:
                    context.save_checkpoint(**states(), usage_range=position, reuse_identical=True)
                    raise
                if not report["complete"]:
                    raise ValueError("Speech validation did not complete")
                if self.selection_metric:
                    score = report[self.selection_metric]
                    metric = dict(name=self.selection_metric, direction="min", evaluation_id=context.last_evaluation)
            update.end_epoch()
            epoch += 1
            offset = 0
            latest = context.save_checkpoint(**states(), usage_range=position, score=score, metric=metric)
            if self.max_steps is not None and step >= self.max_steps:
                break
        if latest is None:
            latest = context.save_checkpoint(**states(), usage_range=position, reuse_identical=True)
        if self.selection_metric and epoch and "validation" in components:
            latest, _ = context.best_checkpoint(direction="min")
        return {"checkpoint": latest, "step": step, "epoch": epoch, "usage_range": position}
