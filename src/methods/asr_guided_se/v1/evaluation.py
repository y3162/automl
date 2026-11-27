"""Full-utterance VoiceBank evaluation and corpus WER checkpoint selection."""

from itertools import islice

import kaldialign
import torch

from automl.evaluator import Evaluator
from automl.experiment.runtime import Interrupted

from .wer_normalization.normalizer import EnglishTextNormalizer


class WordErrorRate(Evaluator):
    def __init__(self, split="test", device="cuda", limit=None):
        self.split, self.device, self.limit = split, device, limit

    @classmethod
    def selection_metrics(cls, settings):
        return ("wer",)

    @torch.no_grad()
    def evaluate(self, components, context, model, *, predictions=None):
        version, pairs = context.open_dataset(components)
        generator = model.generator.to(self.device)
        training = generator.training
        generator.eval()
        asr = generator.recognizer()
        normalize = EnglishTextNormalizer()
        apostrophes = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
        errors = words = count = 0
        inputs = []
        try:
            for sample in islice(pairs.iter_samples(split=self.split), self.limit):
                if context.stop_requested():
                    raise Interrupted("WER evaluation stopped between utterances")
                clean, noisy = (v.unsqueeze(0).to(self.device) for v in sample.data)
                lengths = clean.new_tensor([clean.shape[-1]], dtype=torch.long)
                enhanced = generator(clean, noisy, lengths)["audio"]
                text = asr.recognize(enhanced, lengths)[0].text
                reference = normalize((sample.metadata["transcript"] or "").translate(apostrophes)).split()
                if not reference:
                    raise ValueError(f"Missing reference words for {sample.sample_id}")
                counts = kaldialign.edit_distance(reference, normalize(text.translate(apostrophes)).split())
                edits = counts["total"]
                context.record_result(sample.sample_id, {"dataset_version": version, "split": self.split},
                                      {"hypothesis": text, "errors": edits, "words": len(reference)})
                inputs.append(dict(kind="result", result_id=sample.sample_id, run_id=context.run_id,
                                   evaluation_id=context.evaluation_id))
                errors += edits
                words += len(reference)
                count += 1
        finally:
            generator.train(training)
        complete = words > 0 and (self.limit is None or count == self.limit)
        wer = errors / words if words else None
        context.publish_aggregate("wer", wer, {"dataset_version": version, "split": self.split},
                                  not complete, inputs)
        return {"complete": complete, "wer": wer, "utterances": count, "reference_words": words,
                "aggregate_ids": ["wer"]}
