from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_no_action as cli
from src.baselines import ego_history_mlp as mlp
from src.phase0 import phase0_4b_protocol as protocol
from src.phase0 import phase0_4c_no_action as ablation
from src.phase0 import phase0_4c_convergence as convergence
from src.phase0 import phase0_4c_direct_waypoint as direct
from src.phase0 import phase0_4c_full_train as full
from src.phase0 import phase0_4c_two_turn_planner as two_turn
from src.phase0.qwen3vl_dataset_adapter import GitProvenance
from test_ego_history_mlp import case, records, artifacts
from test_phase0_4b_protocol import processor
from test_phase0_4c_two_turn_planner import tiny_qwen
from test_phase0_4c_direct_waypoint import Backbone


@pytest.fixture
def config():
    return ablation.load_config(ROOT / "configs/phase0_4c_no_action.yaml", ROOT)


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[1]["content"][0]["text"] == ablation.NEUTRAL_CONTEXT
        if not kwargs["tokenize"]:
            return repr(messages)
        return {"input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long)}


@pytest.fixture
def runtime(monkeypatch):
    models = []
    def load_model(*args):
        model = Backbone()
        models.append(model)
        return model
    runtime = SimpleNamespace(
        device_selector=lambda d: "cpu", dtype_selector=lambda d: torch.bfloat16,
        processor_loader=lambda *a: Processor(), model_loader=load_model,
        adapter_loader=lambda model, path: model,
        image_loader=lambda p: object(), package_version=lambda n: "synthetic")
    monkeypatch.setattr(convergence, "default_runtime_dependencies", lambda: runtime)
    monkeypatch.setattr(direct, "resolve_image_path", lambda root, path: root / path)
    return runtime, models


@pytest.fixture
def references(artifacts, records, runtime, config):
    root = artifacts[0]
    adapter = root / config.selected_adapter_relative_path
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic path fixture")
    historical = convergence.load_config(ROOT / "configs/phase0_4c_convergence.yaml")
    prepared = convergence.prepare_run(ROOT, root, root, historical, SimpleNamespace(commit="synthetic"))
    directory = root / "phase_0_4/action_conditioned_convergence_v0_1"
    directory.mkdir()
    ablation.write_json(directory / "run_metadata.json", prepared["provenance"])
    class ReferenceRunner:
        def predict(self, planner, sample, conditioning_type):
            value = 1. if conditioning_type == "predicted_action" else 2.
            return torch.full((1, 6, 2), value), {"action_context": "longitudinal=keep; lateral=straight"}
    for track in ("predicted_action", "gt_action_diagnostic"):
        _, rows = full.evaluate(torch.nn.Identity(), ReferenceRunner(), prepared["validation"], prepared["records"], track)
        full.write_predictions(root / "phase_0_4" / ablation.REFERENCE_FILES[track], rows)
    train, validation = records
    _, rows = mlp.evaluate(mlp.EgoHistoryMLP(3), validation, mlp.normalization_stats(train, 1e-6), 256)
    path = root / "phase_0_4" / ablation.REFERENCE_FILES["ego_history_mlp"]
    path.parent.mkdir()
    full.write_predictions(path, rows)
    return root


@pytest.fixture
def small(config):
    return replace(config, planner_dimension=16, num_decoder_layers=1,
                   expected_validation_count=4, expected_motion_unavailable_count=1,
                   gradient_accumulation_steps=3)


def test_protocol_and_architecture(config, tmp_path):
    historical = convergence.load_config(ROOT / "configs/phase0_4c_convergence.yaml")
    for key in ("seed", "learning_rate", "weight_decay", "gradient_accumulation_steps", "micro_batch_size",
                "selected_adapter_relative_path", "dropout", "memory_normalization", "smooth_l1_beta"):
        assert getattr(config, key) == getattr(historical, key)
    assert config.num_train_epochs == 2
    assert config.expected_validation_count == 3594 and config.expected_motion_unavailable_count == 94
    planner = two_turn.WaypointDecoder(2560, config)
    assert sum(p.numel() for p in planner.parameters()) == 2_765_058
    assert ablation.NEUTRAL_CONTEXT == config.neutral_context_text
    assert protocol.parse_action(ablation.NEUTRAL_CONTEXT) is None
    assert planner(torch.zeros(2, 3, 2560), torch.ones(2, 3, dtype=torch.long)).shape == (2, 6, 2)
    for key, value in (("num_train_epochs", 3), ("seed", 1), ("neutral_context_text", "longitudinal=keep; lateral=straight"),
                       ("conditioning_type", "vision_only"), ("learning_rate", .001), ("expected_motion_unavailable_count", 93)):
        values = asdict(config)
        values.pop("train_subset_size")
        values[key] = value
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(values))
        with pytest.raises(ValueError, match="frozen"):
            ablation.load_config(path, ROOT)


def test_matched_topology_and_no_sample_action_leakage(artifacts, processor, monkeypatch):
    root = artifacts[0]
    train, validation, records, _ = direct.prepare_data(ROOT, root)
    samples = train + validation
    for sample in samples:
        assert not hasattr(sample, "target")
        images = [Image.new("RGB", (64, 64)) for _ in sample.observation.image_paths]
        messages = ablation.neutral_messages(sample.observation, images)
        formal = two_turn.planning_messages(sample.observation, images, "longitudinal=stop; lateral=left")
        assert messages[0] == formal[0] and messages[2] == formal[2]
        assert messages[1]["content"][0]["text"] == ablation.NEUTRAL_CONTEXT
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert rendered.count(ablation.NEUTRAL_CONTEXT) == 1
        assert "longitudinal=stop; lateral=left" not in rendered
        assert "longitudinal=accelerate; lateral=right" not in rendered
        record = records[sample.sample_token]
        record.update(longitudinal_action="accelerate", lateral_action="right",
                      predicted_action="longitudinal=accelerate; lateral=right",
                      future_waypoints=[[987654., -987654.]] * 6)
        assert "987654" not in rendered
        assert messages == ablation.neutral_messages(sample.observation, images)
    formal_train, formal_val, _, _ = full.prepare_data(ROOT, root)
    assert [s.sample_token for s in train] == [s.sample_token for s in formal_train]
    assert [s.sample_token for s in validation] == [s.sample_token for s in formal_val]


def test_real_processor_tiny_qwen_forward_backward(config, artifacts, processor, tiny_qwen, monkeypatch):
    sample = direct.prepare_data(ROOT, artifacts[0])[0][0]
    monkeypatch.setattr(tiny_qwen, "generate", lambda **k: pytest.fail("action generation"))
    monkeypatch.setattr(protocol, "serialize_action", lambda *a: pytest.fail("action serialization"))
    monkeypatch.setattr(direct, "resolve_image_path", lambda root, path: root / path)
    runner = ablation.NeutralRunner(tiny_qwen, processor, SimpleNamespace(
        image_loader=lambda p: Image.new("RGB", (64, 64))), Path("unused"), "cpu")
    planner = two_turn.WaypointDecoder(32, replace(config, planner_dimension=16, num_decoder_layers=1))
    before = deepcopy(planner.state_dict())
    optimizer = torch.optim.AdamW(planner.parameters(), lr=config.learning_rate)
    prediction, _ = runner.predict(planner, sample)
    two_turn.masked_waypoint_loss(prediction, torch.ones_like(prediction),
                                 torch.ones(1, 6, dtype=torch.bool), beta=1.).backward()
    optimizer.step()
    assert prediction.shape == (1, 6, 2)
    assert all(not p.requires_grad and p.grad is None for p in tiny_qwen.parameters())
    assert any(not torch.equal(before[n], p) for n, p in planner.state_dict().items())
    assert runner.context_evidence["neutral_context_text"] == ablation.NEUTRAL_CONTEXT
    assert ablation.NEUTRAL_CONTEXT in runner.context_evidence["rendered_context"]
    with pytest.raises(ValueError):
        runner.predict(planner, replace(sample, split="test"))


def test_two_epochs_shadow_execution_reload_and_comparison(references, small, runtime):
    kwargs = dict(repository=ROOT, dataset_root=references, derived_root=references, config=small,
                  git_provenance=GitProvenance(commit="a" * 40, branch="synthetic", detached_head=False, worktree_clean=True))
    model_count = len(runtime[1])
    dry = ablation.run(**kwargs, dry_run=True)
    assert len(runtime[1]) == model_count
    assert dry["train_sample_count"] == dry["validation_sample_count"] == 4
    output = references / small.output_relative_dir
    assert not output.exists()
    result = ablation.run(**kwargs)
    model = runtime[1][-1]
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["freeze_policy"]["qwen_trainable_parameters"] == 0
    assert metadata["freeze_policy"]["lora_trainable_parameters"] == 0
    assert metadata["freeze_policy"]["planner_trainable_parameters"] > 0
    assert metadata["conditioning_protocol"] == {"train": ablation.TRACK, "validation": ablation.TRACK}
    summary = json.loads((output / "training_summary.json").read_text())
    assert summary["selected_epoch"] == 2 and summary["optimizer_steps"] == 4
    assert [e["epoch"] for e in summary["epochs"]] == [1, 2]
    assert summary["reload_consistency"]["reload_consistency"]
    history = [json.loads(line) for line in (output / "training_history.jsonl").read_text().splitlines()]
    train = direct.prepare_data(ROOT, references)[0]
    expected = [[s.sample_token for s in group] for epoch in range(2)
                for group in convergence.epoch_groups(train, 3, small.seed + epoch)]
    assert [r["sample_tokens"] for r in history] == expected
    for key in ("test_records_read", "test_images_opened", "test_labels_read"):
        assert summary["test_isolation"][key] == 0
    assert summary["test_isolation"]["test_evaluation_performed"] is False
    assert not (output / "epoch_3.pt").exists()
    assert result["models"][ablation.TRACK]["sample_count"] == 4
    assert set(result["models"]) == {ablation.TRACK, *ablation.REFERENCE_FILES}
    assert json.loads((output / "motion_unavailable_comparison.json").read_text())["sample_count"] == 1
    assert json.loads((output / "comparison.json").read_text()) == result
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        ablation.run(**kwargs)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize("failure", ["tokens", "duplicate", "scene", "target", "mask", "split", "provenance", "motion_count", "validation_count"])
def test_reference_intake_rejects_before_model_or_output(references, small, runtime, monkeypatch, failure):
    path = references / "phase_0_4" / ablation.REFERENCE_FILES["predicted_action"]
    rows = direct.read_predictions(path, "predicted_action")
    if failure == "tokens":
        rows[0]["sample_token"] = "wrong"
    elif failure == "duplicate":
        rows.append(rows[0])
    elif failure == "scene":
        rows[0]["scene_token"] = "wrong"
    elif failure == "target":
        rows[0]["target_waypoints"][0][0] += 1
    elif failure == "mask":
        rows[0]["trajectory_valid_mask"][0] = False
    elif failure == "split":
        rows[0]["split"] = "test"
    elif failure == "provenance":
        meta_path = path.parent / "run_metadata.json"
        metadata = json.loads(meta_path.read_text())
        metadata["data"]["source_provenance"] = {}
        ablation.write_json(meta_path, metadata)
    elif failure == "motion_count":
        small = replace(small, expected_motion_unavailable_count=94)
    else:
        small = replace(small, expected_validation_count=3594)
    full.write_predictions(path, rows)
    monkeypatch.setattr(convergence, "prepare_run", lambda *a, **k: pytest.fail("model loading"))
    with pytest.raises(ValueError):
        ablation.run(repository=ROOT, dataset_root=references, derived_root=references,
                     config=small, git_provenance=None, dry_run=True)
    assert not (references / small.output_relative_dir).exists()


def test_paired_direction_reordering_invalid_and_denominators(references, small):
    prepared, models, _ = ablation.intake(ROOT, references, small)
    temporal = {s.sample_token: prepared[2][s.sample_token] for s in prepared[1]}
    rows = direct.read_predictions(references / "phase_0_4" / ablation.REFERENCE_FILES["predicted_action"], "predicted_action")
    for row in rows:
        row["conditioning_type"] = ablation.TRACK
    models[ablation.TRACK] = ablation.aligned_metrics(list(reversed(rows)), ablation.TRACK, temporal)
    result = ablation.compare(models, sorted(temporal))
    pair = result["pairwise"]["no_action_minus_predicted_action"]["ade_3s_m"]
    assert pair["tie_count"] == 4 and pair["mean_delta_m"] == 0
    tokens = sorted(temporal)
    models[ablation.TRACK][tokens[0]]["ade_3s_m"] -= 1
    models[ablation.TRACK][tokens[1]]["ade_3s_m"] += 2
    models[ablation.TRACK][tokens[2]].update(prediction_valid=False, invalid_reason="nonfinite_trajectory")
    result = ablation.compare(models, tokens)
    pair = result["pairwise"]["no_action_minus_predicted_action"]["ade_3s_m"]
    assert (pair["win_count"], pair["tie_count"], pair["loss_count"]) == (1, 1, 1)
    assert pair["paired_sample_count"] == 3 and pair["excluded_sample_count"] == 1
    assert pair["mean_delta_m"] == pytest.approx(1 / 3)
    assert pair["win_rate"] == pytest.approx(1 / 3)
    assert result["models"][ablation.TRACK]["invalid_prediction_count"] == 1
    assert result["models"][ablation.TRACK]["sample_count"] == 4


def test_cli_and_split_isolation(config, tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "intake", lambda *a: pytest.fail("data access"))
    for key in ("train_split", "validation_split"):
        with pytest.raises(ValueError, match="train/validation"):
            ablation.run(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                         config=config, git_provenance=None, **{key: "test"})
    for key in ("--train-split", "--validation-split"):
        with pytest.raises(SystemExit):
            cli.main([key, "test", "--dry-run"])
    calls = []
    monkeypatch.setattr(cli, "run", lambda **k: calls.append(k) or {"synthetic": True})
    monkeypatch.setattr(cli, "collect_git_provenance", lambda *a: pytest.fail("dry-run git access"))
    assert cli.main(["--dry-run", "--dataset-root", str(tmp_path), "--derived-root", str(tmp_path)]) == 0
    assert calls[0]["dry_run"] and calls[0]["config"].num_train_epochs == 2


def test_initialization_matches_convergence_and_frozen_weights(references, small, monkeypatch):
    prepared = direct.prepare_data(ROOT, references)
    git = GitProvenance(commit="b" * 40, branch="synthetic", detached_head=False, worktree_clean=True)
    expected = convergence.prepare_run(ROOT, references, references, small, git)
    initial = deepcopy(expected["planner"].state_dict())
    backbone = deepcopy(expected["model"].state_dict())
    rng = torch.get_rng_state().clone()
    original_fit = ablation.fit
    def checked_fit(**kwargs):
        assert all(torch.equal(p, kwargs["planner"].state_dict()[n]) for n, p in initial.items())
        assert torch.equal(torch.get_rng_state(), rng)
        assert [s.sample_token for s in kwargs["train"]] == [s.sample_token for s in prepared[0]]
        result = original_fit(**kwargs)
        assert all(torch.equal(p, kwargs["model"].state_dict()[n]) for n, p in backbone.items())
        assert any(not torch.equal(p, kwargs["planner"].state_dict()[n]) for n, p in initial.items())
        return result
    monkeypatch.setattr(ablation, "fit", checked_fit)
    ablation.run(repository=ROOT, dataset_root=references, derived_root=references,
                 config=small, git_provenance=git)


def test_frozen_3594_population_and_94_subset_dry_run(references, config, monkeypatch):
    train, validation, records, data = direct.prepare_data(ROOT, references)
    source = records[validation[0].sample_token]
    expanded, samples = {}, []
    for i in range(3594):
        token = f"synthetic-validation-{i}"
        record = deepcopy(source)
        record["sample_token"] = token
        record["ego_motion_history"][-1]["availability"] = "unavailable" if i < 94 else "full"
        expanded[token] = record
        samples.append(replace(validation[0], sample_token=token))
    for track, name in ablation.REFERENCE_FILES.items():
        path = references / "phase_0_4" / name
        row = direct.read_predictions(path, track)[0]
        full.write_predictions(path, [{**row, "sample_token": token} for token in expanded])
    monkeypatch.setattr(direct, "prepare_data", lambda *a: (train, samples, expanded, data))
    monkeypatch.setattr(convergence, "prepare_run", lambda *a, **k: pytest.fail("model accessed"))
    result = ablation.run(repository=ROOT, dataset_root=references, derived_root=references,
                          config=config, git_provenance=None, dry_run=True)
    assert result["validation_sample_count"] == 3594
    assert result["motion_unavailable_count"] == 94
    assert not (references / config.output_relative_dir).exists()
