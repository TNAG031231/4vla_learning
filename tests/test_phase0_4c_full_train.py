from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_full_train as cli
from src.phase0 import phase0_4c_full_train as training
from src.phase0.phase0_4_temporal_dataset import build_record, collect_history
from src.phase0.phase0_4c_evaluation import aggregate_metrics, compare_reload, paired_gap, prediction_metrics
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder, freeze_backbone
from src.phase0.qwen3vl_dataset_adapter import load_config as load_ego_config
from test_phase0_4_temporal_dataset import case


@pytest.fixture
def config():
    return training.load_config(ROOT / "configs/phase0_4c_full_train.yaml")


@pytest.fixture
def records(case):
    reader, source = case
    train = [build_record(row, collect_history(reader, row, 3), 3) for row in source]
    validation = deepcopy(train)
    for row in validation:
        row.update(split="validation", scene_token="validation-scene")
        row["sample_token"] += "-validation"
        row["historical_sample_tokens"][-1] = row["sample_token"]
    return {"train": train, "validation": validation}


@pytest.fixture
def data(records, monkeypatch):
    monkeypatch.setattr(training, "load_temporal_records", lambda *a, split: records[split])
    return training.prepare_data(ROOT, Path("unused"))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 16)
        self.lora_A = nn.Linear(1, 1)


class SyntheticRunner:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def predict(self, planner, sample, conditioning_type):
        self.calls.append((sample.split, conditioning_type, sample.sample_token))
        hidden = torch.ones(1, 3, planner.context_projection.in_features) * (
            1 if conditioning_type == "predicted_action" else 2)
        prediction = planner(hidden, torch.ones(1, 3, dtype=torch.long))
        return prediction, {"action_context": "longitudinal=keep; lateral=straight"}


def test_producer_intake_and_eligibility(records, data):
    train, validation, targets, summary = data
    assert len(train) == len(validation) == 4
    assert len(targets) == 8
    assert summary["source_provenance"]["train"]["source_combined_manifest_sha256"]
    assert summary["train"] == {"total_records": 4, "eligible_records": 4,
                                "excluded_records": 0, "exclusion_reason_counts": {}}
    ego = load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml")
    for split in ("train", "validation"):
        records[split][0].update(longitudinal_action=None, longitudinal_action_valid=False,
                                 factorized_action_joint_valid=False)
        samples, counts = training.eligible_samples(records[split], ego, split)
        assert len(samples) == (3 if split == "train" else 4)
        assert counts["excluded_records"] == (1 if split == "train" else 0)
    records["train"][1]["trajectory_valid_mask"][-1] = False
    with pytest.raises(ValueError, match="six valid"):
        training.eligible_samples(records["train"], ego, "train")


def test_full_config_and_architecture(config, tmp_path):
    planner = WaypointDecoder(2560, config)
    assert sum(p.numel() for p in planner.parameters()) == 2_765_058
    assert isinstance(planner.memory_norm, nn.LayerNorm)
    assert all(not layer.norm_first for layer in planner.decoder.layers)
    assert config.learning_rate == 1e-4 and config.train_subset_size == 0
    values = yaml.safe_load((ROOT / "configs/phase0_4c_full_train.yaml").read_text())
    for key, value in (("memory_normalization", False), ("planner_dimension", 128),
                       ("output_relative_dir", "phase_0_4/two_turn_planner_tiny_overfit_v0_4a"),
                       ("learning_rate", float("nan")), ("gradient_accumulation_steps", 0)):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**values, key: value}))
        with pytest.raises(ValueError):
            training.load_config(path)


def test_horizon_metrics_masks_and_invalid_predictions():
    target = torch.zeros(6, 2)
    prediction = torch.tensor([[float(i), 0.] for i in range(1, 7)])
    mask = torch.tensor([True, True, False, True, False, False])
    target[~mask] = float("nan")
    metrics = prediction_metrics(prediction, target, mask)
    assert metrics["ade_m"] == pytest.approx(7 / 3)
    assert metrics["fde_m"] == 4
    assert metrics["ade_1s_m"] == 1.5 and metrics["fde_1s_m"] == 2
    assert metrics["ade_2s_m"] == pytest.approx(7 / 3) and metrics["fde_2s_m"] == 4
    assert metrics["fde_3s_m"] is None
    invalids = [None, torch.zeros(5, 2), torch.full((6, 2), float("inf"))]
    rows = [{"conditioning_type": "predicted_action", **prediction_metrics(p, target, mask)}
            for p in [prediction, *invalids]]
    result = aggregate_metrics(rows, "predicted_action")
    assert result["sample_count"] == 4 and result["invalid_prediction_count"] == 3
    assert result["trajectory_valid_rate"] == .25 and result["invalid_prediction_rate"] == .75
    assert result["fde_3s_m_sample_count"] == 0
    with pytest.raises(ValueError, match="mixed"):
        aggregate_metrics(rows, "gt_action_diagnostic")
    with pytest.raises(ValueError, match="valid GT"):
        prediction_metrics(prediction, target, torch.zeros(6, dtype=torch.bool))


def test_freeze_optimizer_boundary(config):
    model = Backbone()
    freeze_backbone(model)
    planner = WaypointDecoder(16, config)
    optimizer = torch.optim.AdamW(planner.parameters())
    counts = training.verify_freeze(model, planner, optimizer)
    assert counts["qwen_trainable_parameters"] == counts["lora_trainable_parameters"] == 0
    assert counts["memory_norm_trainable"]
    for module in (model.base, model.lora_A):
        module.requires_grad_(True)
        with pytest.raises(ValueError, match="optimizer contract"):
            training.verify_freeze(model, planner, optimizer)
        freeze_backbone(model)
    optimizer.add_param_group({"params": model.parameters()})
    with pytest.raises(ValueError, match="optimizer contract"):
        training.verify_freeze(model, planner, optimizer)


def test_full_fit_artifacts_checkpoint_reload_and_paths(config, data, tmp_path):
    train, validation, targets, summary = data
    config = replace(config, planner_dimension=16, num_decoder_layers=1, dropout=0.,
                     gradient_accumulation_steps=3, checkpoint_interval=1, validation_interval=1)
    model = Backbone()
    freeze_backbone(model)
    original = deepcopy(model.state_dict())
    planner = WaypointDecoder(16, config)
    before = deepcopy(planner.state_dict())
    runner = SyntheticRunner()
    result = training.fit(model=model, planner=planner, runner=runner, train=train, validation=validation,
                          records=targets, config=config, output=tmp_path,
                          provenance={"run_kind": "synthetic_cpu_test", "data": summary})
    assert result["optimizer_steps"] == 2 and result["samples_consumed"] == 4
    assert result["reload_consistency"]["reload_consistency"]
    assert result["full_model_saved"] is False
    assert result["conditioning_gap"]["sample_count"] == 4
    assert all(torch.equal(original[n], p) for n, p in model.state_dict().items())
    assert any(not torch.equal(before[n], p) for n, p in planner.state_dict().items())
    assert all(split == ("train" if kind == "gt_action_teacher_forced_train" else "validation")
               for split, kind, _ in runner.calls)
    history = [json.loads(line) for line in (tmp_path / "training_history.jsonl").read_text().splitlines()]
    assert [r["samples_consumed"] for r in history] == [3, 4]
    for row in history:
        assert row["validation_metrics"]["conditioning_type"] == "predicted_action"
        assert {"train_loss", "learning_rate", "elapsed_seconds"} <= row.keys()
    for name, kind in (("predicted_action", "predicted_action"),
                       ("gt_action_diagnostic", "gt_action_diagnostic")):
        rows = [json.loads(line) for line in
                (tmp_path / f"validation_{name}_predictions.jsonl").read_text().splitlines()]
        metrics = json.loads((tmp_path / f"validation_{name}_metrics.json").read_text())
        assert aggregate_metrics(rows, kind) == metrics
    checkpoint = torch.load(tmp_path / result["best_checkpoint"]["checkpoint"], weights_only=True)
    assert "planner_state_dict" in checkpoint and "model_state_dict" not in checkpoint
    assert checkpoint["provenance"]["data"] == summary
    assert json.loads((tmp_path / "training_summary.json").read_text()) == result


def test_runner_action_paths_and_no_target_leakage(config, data, monkeypatch, tmp_path):
    train, validation, records, _ = data
    model = Backbone()
    freeze_backbone(model)
    captured, generated = [], []
    predicted = "longitudinal=decelerate; lateral=left"

    def generate(model, messages, processor, kwargs, device):
        generated.append(messages)
        return {"raw_output": predicted}

    def inputs(processor, messages, device):
        captured.append(messages)
        return {"attention_mask": torch.ones(1, 2, dtype=torch.long)}, {"action_context_matches": True}

    monkeypatch.setattr(training, "generate_action", generate)
    monkeypatch.setattr(training, "planning_inputs", inputs)
    monkeypatch.setattr(training, "contextual_hidden_states", lambda *a: torch.ones(1, 2, 16))
    runner = training.PlannerRunner(model, object(), SimpleNamespace(image_loader=lambda p: "image"),
                                    tmp_path, {}, "cpu")
    planner = WaypointDecoder(16, replace(config, planner_dimension=16, num_decoder_layers=1))
    runner.predict(planner, train[0], "gt_action_teacher_forced_train")
    assert not generated
    _, evidence = runner.predict(planner, validation[0], "predicted_action")
    assert evidence["action_context"] == predicted and len(generated) == 1
    runner.predict(planner, validation[0], "gt_action_diagnostic")
    assert len(generated) == 1
    assert captured[0][1]["content"][0]["text"] == captured[2][1]["content"][0]["text"]
    assert captured[1][1]["content"][0]["text"] == predicted
    assert all("future_waypoints" not in str(messages) and "trajectory_valid_mask" not in str(messages)
               for messages in captured + generated)
    for sample, kind in ((train[0], "predicted_action"), (validation[0], "gt_action_teacher_forced_train"),
                          (replace(validation[0], split="test"), "predicted_action")):
        with pytest.raises(ValueError, match="split mismatch"):
            runner.predict(planner, sample, kind)
    predicted = "not a structured action"
    prediction, evidence = runner.predict(planner, validation[0], "predicted_action")
    assert prediction is None and evidence["action_context"] is None and len(captured) == 3


def test_split_rejection_and_rerun_guard(config, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(training, "prepare_data", lambda *a: pytest.fail("data accessed"))
    for option in ("--train-split", "--validation-split"):
        with pytest.raises(SystemExit):
            cli.main([option, "test", "--dry-run"])
    assert cli.main(["--dry-run"]) == 0
    assert "dry_run_no_data_or_model_access" in capsys.readouterr().out
    kwargs = dict(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                  config=config, git_provenance=None)
    with pytest.raises(ValueError, match="validation evaluation only"):
        training.run_full(**kwargs, validation_split="test")
    monkeypatch.setattr(training, "validate_git_provenance", lambda p: SimpleNamespace(commit="synthetic"))
    output = tmp_path / config.output_relative_dir
    output.mkdir(parents=True)
    prior = tmp_path / "phase_0_4/two_turn_planner_tiny_overfit_v0_4a/training_summary.json"
    prior.parent.mkdir()
    prior.write_text("preserve")
    with pytest.raises(FileExistsError):
        training.run_full(**kwargs)
    assert prior.read_text() == "preserve"


def test_reload_detects_changes_and_gap_pairs():
    row = {"sample_token": "one", "conditioning_type": "predicted_action", "action_context": "action",
           **prediction_metrics(torch.ones(6, 2), torch.zeros(6, 2), torch.ones(6, dtype=torch.bool))}
    assert compare_reload([row], [deepcopy(row)])["reload_consistency"]
    changed = deepcopy(row)
    changed["predicted_waypoints"][0][0] += .1
    assert not compare_reload([row], [changed])["reload_consistency"]
    assert not compare_reload([row], [])["reload_consistency"]
    assert paired_gap([row], [{**row, "sample_token": "other"}])["sample_count"] == 0
    assert paired_gap([row], [row])["ade_m_gap"] == 0


def test_full_runtime_entrypoint(config, data, monkeypatch, tmp_path):
    monkeypatch.setattr(training, "prepare_data", lambda *a: data)
    monkeypatch.setattr(training, "validate_git_provenance", lambda p: SimpleNamespace(commit="synthetic"))
    model = Backbone()
    model.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560))
    loaded = []

    def adapter_loader(base, path):
        loaded.append(path)
        return base

    runtime = SimpleNamespace(
        device_selector=lambda d: "cpu", dtype_selector=lambda d: torch.bfloat16,
        processor_loader=lambda *a: object(), model_loader=lambda *a: model,
        adapter_loader=adapter_loader, package_version=lambda name: "synthetic")
    monkeypatch.setattr(training, "default_runtime_dependencies", lambda: runtime)
    monkeypatch.setattr(training, "PlannerRunner", lambda *a: SyntheticRunner())
    result = training.run_full(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                               config=config, git_provenance=None)
    output = tmp_path / config.output_relative_dir
    assert result["status"] == "full_training_completed"
    assert loaded == [tmp_path / config.selected_adapter_relative_path]
    assert result["provenance"]["freeze_policy"]["planner_trainable_parameters"] == 2_765_058
    assert result["provenance"]["conditioning_protocol"]["validation"] == "predicted_action"
    assert result["provenance"]["data"]["test_records_read"] == 0
    assert not result["provenance"]["data"]["test_evaluation_performed"]
    assert {"run_metadata.json", "resolved_config.json", "data_summary.json", "best_checkpoint.json",
            "training_summary.json", "training_history.jsonl", "reload_consistency.json",
            "final_sanity.json", "checkpoint_selection.json"} <= {p.name for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        training.run_full(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                          config=config, git_provenance=None)


@pytest.mark.parametrize("failure", ["reload", "invalid_action", "train_loss"])
def test_stop_conditions_keep_failure_artifacts(config, data, tmp_path, monkeypatch, failure):
    train, validation, targets, _ = data
    config = replace(config, planner_dimension=16, num_decoder_layers=1, dropout=0.)
    model = Backbone()
    freeze_backbone(model)
    runner = SyntheticRunner()
    if failure == "reload":
        monkeypatch.setattr(training, "compare_reload", lambda *a: {"reload_consistency": False})
        expected = "reload is inconsistent"
    elif failure == "invalid_action":
        predict = runner.predict
        runner.predict = lambda planner, sample, kind: (
            (None, {"action_context": None}) if kind == "predicted_action"
            else predict(planner, sample, kind))
        expected = "no valid predicted-action"
    else:
        runner.predict = lambda *a: (torch.full((1, 6, 2), float("nan")), {})
        expected = "nonfinite"
    with pytest.raises(ValueError, match=expected):
        training.fit(model=model, planner=WaypointDecoder(16, config), runner=runner,
                     train=train, validation=validation, records=targets, config=config,
                     output=tmp_path, provenance={"run_kind": "synthetic_cpu_test"})
    assert not (tmp_path / "training_summary.json").exists()
    if failure == "reload":
        assert not json.loads((tmp_path / "reload_consistency.json").read_text())["reload_consistency"]
    if failure == "invalid_action":
        metrics = json.loads((tmp_path / "validation_step_0001_metrics.json").read_text())
        assert metrics["invalid_prediction_count"] == len(validation)


def test_evaluation_and_optimization_reject_wrong_splits(config, data, tmp_path):
    train, validation, targets, _ = data
    model = Backbone()
    planner = WaypointDecoder(16, config)
    runner = SyntheticRunner()
    with pytest.raises(ValueError, match="validation samples only"):
        training.evaluate(planner, runner, train, targets, "predicted_action")
    with pytest.raises(ValueError, match="train samples only"):
        training.fit(model=model, planner=planner, runner=runner, train=validation, validation=validation,
                     records=targets, config=config, output=tmp_path, provenance={})
    with pytest.raises(ValueError, match="validation samples only"):
        training.fit(model=model, planner=planner, runner=runner, train=train, validation=train,
                     records=targets, config=config, output=tmp_path, provenance={})
    assert runner.calls == []
