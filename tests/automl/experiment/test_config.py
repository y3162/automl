import pytest

from automl.evaluator import Evaluator
from automl.experiment.config import Components, load_config, resolve_config
from automl.experiment.errors import ExperimentError
from automl.trainer import Trainer


class SelectTrainer(Trainer):
    def __init__(self, selection_metric="accuracy"):
        self.selection_metric = selection_metric

    def train(self, components, context, *, start_method):
        return {}


class SelectEvaluator(Evaluator):
    def __init__(self, metric="accuracy"):
        self.metric = metric

    @classmethod
    def selection_metrics(cls, settings):
        return (settings["metric"],)

    def evaluate(self, components, context, model, *, predictions=None):
        return {"complete": True}


def test_repeated_implementation_switch_discards_saved_settings(tmp_path):
    # If the YAML chain were flattened first, the inherited B-only setting
    # would incorrectly return when the child switches from A back to B.
    base = {"plan": ["evaluate"], "components": {"model": {
        "implementation": "task.B", "settings": {"retired": True},
    }}}
    (tmp_path / "parent.yaml").write_text(
        "components:\n  model:\n    implementation: task.A\n    settings: {a: 1}\n"
    )
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "child.yaml").write_text(
        "extends: parent.yaml\ncomponents:\n  model:\n    implementation: task.B\n    settings: {b: 2}\n"
    )
    resolved = load_config("configs/child.yaml", tmp_path, base=base)
    assert resolved["components"]["model"]["settings"] == {"b": 2}
    assert base["components"]["model"]["settings"] == {"retired": True}


def test_dictionary_and_list_overrides_apply_at_the_right_level(tmp_path):
    base = {"plan": ["train", "evaluate"], "components": {"model": {
        "implementation": "task.Model", "settings": {
            "strategy": {"implementation": "x", "scale": 4}, "widths": [1, 2, 3],
        },
    }}}
    (tmp_path / "child.yaml").write_text(
        "plan: [evaluate]\ncomponents:\n  model:\n    settings:\n"
        "      strategy: {implementation: y}\n      widths: [7]\n"
    )
    resolved = load_config("child.yaml", tmp_path, base=base)
    assert resolved["plan"] == ["evaluate"]
    assert resolved["components"]["model"]["settings"] == {
        "strategy": {"implementation": "y", "scale": 4}, "widths": [7],
    }


def test_duplicate_key_reports_file_line_and_name(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("components:\n  model:\n    settings:\n      width: 4\n      width: 9\n")
    with pytest.raises(ExperimentError) as rejected:
        load_config(path, tmp_path)
    assert rejected.value.code == "invalid_config"
    assert rejected.value.details == {"file": str(path), "line": 5, "key": "width"}


def test_yaml_merge_key_allows_explicit_override_but_checks_its_source(tmp_path):
    path = tmp_path / "merge.yaml"
    path.write_text("components: {model: {<<: {settings: {}}, settings: {width: 4}}}")
    assert load_config(path, tmp_path)["components"]["model"]["settings"] == {"width": 4}
    path.write_text("components: {model: {<<: {settings: {}, settings: {width: 4}}}}")
    with pytest.raises(ExperimentError) as rejected:
        load_config(path, tmp_path)
    assert rejected.value.details == {"file": str(path), "line": 1, "key": "settings"}


def test_cyclic_or_missing_parent_identifies_path(tmp_path):
    path = tmp_path / "child.yaml"
    path.write_text("extends: parent.yaml")
    with pytest.raises(ExperimentError, match="parent.yaml"):
        load_config(path, tmp_path)
    (tmp_path / "parent.yaml").write_text("extends: child.yaml")
    with pytest.raises(ExperimentError, match="cycle"):
        load_config(path, tmp_path)


def test_selection_metric_is_checked_against_resolved_evaluator_settings(tmp_path):
    raw = {"plan": ["train"], "components": {
        "trainer": {"implementation": f"{__name__}.SelectTrainer"},
        "validation": {"implementation": f"{__name__}.SelectEvaluator"},
    }}
    config = resolve_config(raw, tmp_path)
    assert config["components"]["trainer"]["settings"] == {"selection_metric": "accuracy"}
    assert Components(config, tmp_path).create("validation", Evaluator).metric == "accuracy"

    raw["components"]["validation"]["settings"] = {"metric": "loss"}
    with pytest.raises(ExperimentError, match="unknown selection_metric") as rejected:
        resolve_config(raw, tmp_path)
    assert rejected.value.details["available"] == ["loss"]
