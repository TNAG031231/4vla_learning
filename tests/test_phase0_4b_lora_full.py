from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phase0_4b_lora_full import main
from src.phase0 import phase0_4b_lora_full as full
from src.phase0 import phase0_4b_lora_smoke as smoke
from src.phase0.phase0_4_temporal_dataset import build_record, collect_history
from src.phase0.phase0_4b_protocol import ActionTarget, StructuredCollator, adapt_record
from src.phase0.qwen3vl_dataset_adapter import GitProvenance, load_config as load_ego_config
from test_phase0_3_development_projection import _mapping_payload
from test_phase0_4b_protocol import case, processor, samples


@pytest.fixture
def config():
    return full.load_config(ROOT / "configs/phase0_4b_lora_full.yaml")


def test_epoch_visits_each_record_once_with_incomplete_group(samples):
    train = [replace(samples[0], sample_token=f"sample-{i}") for i in range(14253)]
    groups = list(full.epoch_groups(train, 4, 17))
    assert len(groups) == 3564 and len(groups[-1]) == 1
    tokens = [s.sample_token for group in groups for s in group]
    assert len(tokens) == len(set(tokens)) == 14253
    assert tokens == [s.sample_token for group in full.epoch_groups(list(reversed(train)), 4, 17) for s in group]


def test_tail_gradient_uses_actual_group_size_and_no_cycling(samples, config):
    train = [replace(samples[0], sample_token=str(i)) for i in range(1, 6)]
    config = replace(config, learning_rate=0.1)
    visited = []
    class Collator:
        def __call__(self, group, expected_split):
            assert expected_split == "train"
            visited.extend(s.sample_token for s in group)
            return {"target": torch.tensor(float(group[0].sample_token))}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
        def forward(self, target):
            return SimpleNamespace(loss=(self.weight - target).square())
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = full.train_one_epoch(model=model, samples=train, collator=Collator(), optimizer=optimizer,
                                  config=config, device="cpu", on_step=lambda entry: None)
    expected = 0.0
    for step, group in enumerate(full.epoch_groups(train, 4, config.seed), 1):
        mean_target = sum(float(s.sample_token) for s in group) / len(group)
        expected -= full.learning_rate_at_step(step, 2, config) * 2 * (expected - mean_target)
    assert model.weight.item() == pytest.approx(expected)
    assert len(set(visited)) == len(visited) == 5
    assert history[-1]["sample_count"] == 1
    assert history[-1]["consumed_sample_count"] == 5


def test_both_invalid_consumed_without_loss_or_optimizer_decay(samples, config):
    invalid = replace(samples[0], target=ActionTarget(None, None, False, False))
    model = torch.nn.Linear(1, 1)
    initial = model.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    history = full.train_one_epoch(model=model, samples=[invalid], collator=lambda *a, **k: pytest.fail("empty loss"),
                                  optimizer=optimizer, config=config, device="cpu", on_step=lambda e: None)
    assert history[0]["consumed_sample_count"] == 1
    assert history[0]["loss"] is None and not history[0]["optimizer_step_performed"]
    assert torch.equal(model.weight, initial)


def test_milestones_and_scheduler(config):
    assert full.checkpoint_steps(3564, config.checkpoint_fractions) == [891, 1782, 2673, 3564]
    assert full.checkpoint_steps(7, config.checkpoint_fractions) == [2, 4, 5, 7]
    assert full.learning_rate_at_step(1, 3564, config) == pytest.approx(1e-4 / 107)
    assert full.learning_rate_at_step(107, 3564, config) == 1e-4
    assert full.learning_rate_at_step(108, 3564, config) == 1e-4
    assert 0 < full.learning_rate_at_step(3564, 3564, config) < 1e-9


@pytest.mark.parametrize("key,value", [("num_train_epochs", 2), ("local_files_only", False),
    ("lora_r", 16), ("gradient_accumulation_steps", 1),
    ("precision", "float16"), ("learning_rate", 0.01),
    ("output_relative_dir", "phase_0_4/structured_action_lora_smoke_v0_1")])
def test_config_rejects_protocol_changes(config, tmp_path, key, value):
    values = asdict(config)
    values[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="frozen"):
        full.load_config(path)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_optimization_and_cli_reject_non_train_before_access(samples, split):
    with pytest.raises(ValueError, match="train samples only"):
        list(full.epoch_groups([replace(samples[0], split=split)], 4, 7))
    with pytest.raises(SystemExit) as error:
        main(["--split", split])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="only permits train and validation"):
        smoke.load_temporal_samples(Path("missing"), Path("missing"), split="test")


def test_validation_observation_and_test_guard_before_image_access(case, samples, processor, tmp_path):
    reader, records = case
    row = build_record(records[3], collect_history(reader, records[3], 3), 3)
    row["split"] = "validation"
    ego = load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml")
    val = adapt_record(row, ego, expected_split="validation")
    assert val.observation == samples[3].observation
    collator = StructuredCollator(processor, lambda path: pytest.fail("image access"), tmp_path)
    with pytest.raises(ValueError):
        collator.messages(replace(val, split="test"), expected_split="validation")
    with pytest.raises(ValueError):
        collator([val], expected_split="validation")
    with pytest.raises(ValueError):
        adapt_record({"split": "test"}, ego, expected_split="test")


def test_real_producer_intake_and_split_mapping_contract(case, tmp_path, monkeypatch):
    reader, records = case
    mapping = _mapping_payload()
    source = smoke.load_source_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT)
    source = replace(source, source_contract=replace(source.source_contract,
                                                     expected_sample_counts={"train": 4, "validation": 4}))
    monkeypatch.setattr(smoke, "load_source_config", lambda *args: source)
    monkeypatch.setattr(smoke, "read_scene_mapping", lambda *args: mapping)
    directory = tmp_path / "phase_0_4/temporal_waypoint_v0_1"
    directory.mkdir(parents=True)
    rows = [build_record(row, collect_history(reader, row, 3), 3) for row in records]
    for row in rows:
        row["split_mapping_sha256"] = mapping["scene_split_mapping_sha256"]
    train_file = directory / "train.jsonl"
    train_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert len(smoke.load_temporal_samples(ROOT, tmp_path, split="train")) == 4
    selection = smoke.development.select_development_scenes(mapping, source.source_contract)
    for row in rows:
        row["split"] = "validation"
        row["scene_token"] = selection.scene_tokens_by_split["validation"][0]
    val_file = directory / "validation.jsonl"
    val_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert len(smoke.load_temporal_samples(ROOT, tmp_path, split="validation")) == 4
    rows[0]["scene_token"] = selection.scene_tokens_by_split["train"][0]
    val_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match="outside frozen split scenes"):
        smoke.load_temporal_samples(ROOT, tmp_path, split="validation")


def test_prepare_rejects_contamination(samples, config, tmp_path, monkeypatch):
    monkeypatch.setattr(full, "load_temporal_samples", lambda *a, split: [replace(samples[0], split=split)])
    with pytest.raises(ValueError, match="contamination"):
        full.prepare_data(ROOT, tmp_path, config)


@pytest.mark.parametrize("failure", [None, "reload_mismatch", "no_update", "nonfinite", "prediction_mismatch", "raw_output_mismatch"])
def test_synthetic_cpu_full_flow_releases_training_model_and_reloads_selected_adapter(
        samples, processor, config, tmp_path, monkeypatch, failure):
    train = [replace(samples[0], sample_token=f"train-{i}", scene_token="train-scene",
                     target=ActionTarget("keep", "straight", True, True)) for i in range(17)]
    validation = [replace(train[0], sample_token=f"val-{i}", scene_token="val-scene", split="validation")
                  for i in range(4)]
    monkeypatch.setattr(full, "load_temporal_samples", lambda *a, split: train if split == "train" else validation)
    refs, loaded_paths, calls = [], [], []
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(use_cache=True)
        def forward(self, **batch):
            assert self.training and (batch["labels"] != -100).any()
            calls.append("train")
            if failure == "nonfinite":
                return SimpleNamespace(loss=self.lora_B.sum() * float("nan"))
            if failure == "no_update":
                return SimpleNamespace(loss=self.lora_B.sum() * 0 + 1)
            return SimpleNamespace(loss=(self.lora_B - 1).square().mean())
        def generate(self, **batch):
            assert not self.training and "labels" not in batch
            calls.append("fresh_eval" if len(refs) == 2 else "milestone_eval")
            text = "longitudinal=keep; lateral=straight"
            if failure == "prediction_mismatch" and len(refs) == 2:
                text = "longitudinal=stop; lateral=right"
            if failure == "raw_output_mismatch":
                text = "invalid fresh" if len(refs) == 2 else "invalid selection"
            ids = processor.tokenizer.encode(text, add_special_tokens=False)
            return torch.cat((batch["input_ids"], torch.tensor([ids])), dim=1)
        def gradient_checkpointing_enable(self):
            pass
        def enable_input_require_grads(self):
            pass
        def save_pretrained(self, path):
            path.mkdir()
            torch.save(self.lora_B.detach(), path / "synthetic_adapter.pt")
    def loader(*args):
        assert args[-1] is True
        if refs:
            assert refs[0]() is None, "training model remains alive during fresh load"
        model = Model()
        refs.append(weakref.ref(model))
        return model
    def inject(model, config):
        model.lora_B = torch.nn.Parameter(torch.zeros(1))
        return model
    def reload(model, path):
        loaded_paths.append(path)
        model.lora_B = torch.nn.Parameter(torch.load(path / "synthetic_adapter.pt", weights_only=True))
        if failure == "reload_mismatch":
            with torch.no_grad():
                model.lora_B.zero_()
        return model
    runtime = SimpleNamespace(
        model_loader=loader, processor_loader=lambda *a: processor, lora_config_factory=lambda **kw: kw,
        lora_injector=inject, adapter_loader=reload,
        optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
        image_loader=lambda path: Image.new("RGB", (64, 64)), dtype_selector=lambda _: torch.bfloat16,
        device_selector=lambda _: "cpu", inference_context=nullcontext, package_version=lambda _: "synthetic",
    )
    monkeypatch.setattr(full, "default_runtime_dependencies", lambda: runtime)
    provenance = GitProvenance("a" * 40, "synthetic", False, True)
    if failure:
        messages = {"reload_mismatch": "reload weight report mismatch", "no_update": "did not update",
                    "nonfinite": "training loss must be finite",
                    "prediction_mismatch": "fresh reload validation differs",
                    "raw_output_mismatch": "fresh reload validation differs"}
        with pytest.raises(ValueError, match=messages[failure]):
            full.run_full(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                          config=config, git_provenance=provenance)
        if failure in ("prediction_mismatch", "raw_output_mismatch"):
            comparison = json.loads((tmp_path / config.output_relative_dir / "reload_consistency.json").read_text())
            assert not comparison["matched"] and len(comparison["sample_differences"]) == len(validation)
            assert comparison["metrics_match"] == (failure == "raw_output_mismatch")
        else:
            assert "fresh_eval" not in calls
        assert not (tmp_path / config.output_relative_dir / "training_summary.json").exists()
        return
    result = full.run_full(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                           config=config, git_provenance=provenance)
    assert result["train_records_consumed"] == calls.count("train") == 17
    assert result["optimizer_steps"] == 5
    assert calls.count("milestone_eval") == 16 and calls.count("fresh_eval") == 4
    assert result["selected_checkpoint"]["step"] == 1
    assert result["reload_validation_consistent"]
    checkpoints = json.loads((tmp_path / config.output_relative_dir / "milestone_checkpoints.json").read_text())
    for checkpoint in checkpoints:
        assert checkpoint["validation_sample_count"] == len(validation)
        rows = json.loads(Path(checkpoint["validation_predictions_path"]).read_text())
        assert [r["sample_token"] for r in rows] == [s.sample_token for s in validation]
        assert checkpoint["metrics"]["longitudinal_count"] == len(validation)
        assert checkpoint["metrics"]["lateral_count"] == len(validation)
        assert checkpoint["metrics"]["joint_count"] == len(validation)
    assert loaded_paths == [Path(result["selected_checkpoint"]["adapter_path"])]
    assert result["reload_status"] == "success" and result["lora_parameters_updated"]
    assert not result["full_model_saved"] and not result["test_evaluation_performed"]
    assert all(result[key] == 0 for key in ("test_records_read", "test_scene_traversal_attempts",
        "test_sample_records_read", "test_images_opened", "test_labels_read"))
    output = tmp_path / config.output_relative_dir
    history = [json.loads(line) for line in (output / "training_history.jsonl").read_text().splitlines()]
    assert history[-1]["sample_count"] == 1
    assert len(list(output.glob("adapter_step_*"))) == 4
    predictions = [json.loads(line) for line in (output / "validation_predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 4 and all(r["failure_category"] == "fully_correct" for r in predictions)
    assert json.loads((output / "training_summary.json").read_text())["epochs_completed"] == 1
    with pytest.raises(FileExistsError):
        full.run_full(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                      config=config, git_provenance=provenance)
    assert len(refs) == 2


@pytest.mark.parametrize("split", ["train", "test"])
def test_formal_evaluation_rejects_other_splits_before_model(samples, config, split):
    with pytest.raises(ValueError, match="validation samples only"):
        full.evaluate_validation(None, [replace(samples[0], split=split)], None, config, "cpu", None)


def test_cli_dry_run_uses_intake_without_runtime(config, samples, tmp_path, monkeypatch, capsys):
    from scripts import run_phase0_4b_lora_full as cli
    val = [replace(samples[0], split="validation", scene_token="val", sample_token="val")]
    monkeypatch.setattr(cli, "prepare_data", lambda *args: (samples, val, {"train_sample_tokens": []}))
    monkeypatch.setattr(cli, "run_full", lambda **kwargs: pytest.fail("training launched"))
    assert cli.main(["--dry-run", "--derived-root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["gpu_training_executed"] is False
