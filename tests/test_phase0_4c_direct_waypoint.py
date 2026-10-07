from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_direct_waypoint as cli
from src.baselines import ego_history_mlp as mlp
from src.phase0 import phase0_4c_direct_waypoint as direct
from src.phase0 import phase0_4c_full_train as formal
from src.phase0 import phase0_4b_protocol as protocol
from src.phase0 import phase0_4c_two_turn_planner as two_turn
from src.phase0.qwen3vl_dataset_adapter import GitProvenance
from test_ego_history_mlp import case, records, artifacts
from test_phase0_4b_protocol import processor
from test_phase0_4c_two_turn_planner import tiny_qwen


@pytest.fixture
def config():
    return direct.load_config(ROOT / "configs/phase0_4c_direct_waypoint.yaml", ROOT)


@pytest.fixture
def data(artifacts):
    return direct.prepare_data(ROOT, artifacts[0])


@pytest.fixture
def references(artifacts, records):
    root, _, _ = artifacts
    train, validation = records
    stats = mlp.normalization_stats(train, 1e-6)
    _, rows = mlp.evaluate(mlp.EgoHistoryMLP(3), validation, stats, 256)
    directory = root / "phase_0_4/ego_history_mlp_baseline_v0_1"
    directory.mkdir()
    direct.write_predictions(directory / "validation_predictions.jsonl", rows)
    return root


def test_frozen_config_architecture_and_prompt(config, data, tmp_path):
    expected = asdict(formal.load_config(ROOT / "configs/phase0_4c_full_train.yaml"))
    actual = asdict(config)
    for key in expected.keys() - {"protocol_version", "output_relative_dir"}:
        assert actual[key] == expected[key]
    assert direct.DIRECT_PROMPT == (
        "Based only on the provided past/current CAM_FRONT observations and ego states, "
        "prepare a representation for predicting the ego trajectory over the next 3.0 seconds "
        "at 0.5-second intervals. Observations are ordered oldest to current; "
        "unavailable history slots have no image.")
    assert "longitudinal=" not in direct.DIRECT_PROMPT and "factorized" not in direct.DIRECT_PROMPT
    planner = direct.WaypointDecoder(2560, config)
    assert sum(p.numel() for p in planner.parameters()) == 2_765_058
    assert isinstance(planner.memory_norm, nn.LayerNorm)
    for key, value in (("seed", 1), ("learning_rate", .001), ("num_train_epochs", 2),
                       ("memory_normalization", False), ("selected_adapter_relative_path", "other")):
        changed = {k: v for k, v in actual.items() if k != "train_subset_size"}
        changed[key] = value
        path = tmp_path / "changed.yaml"
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError, match="must match"):
            direct.load_config(path, ROOT)


def test_observations_identical_without_action_instruction(data):
    train, _, _, _ = data
    for sample in train:
        images = [object() for _ in sample.observation.image_paths]
        old = protocol.inference_messages(sample.observation, images)
        new = direct.direct_messages(sample.observation, images)
        assert len(new) == 1 and new[0]["role"] == "user"
        assert new[0]["content"][1:] == old[0]["content"][1:]
        assert new[0]["content"][0]["text"].split("History availability: ")[1] == (
            old[0]["content"][0]["text"].split("History availability: ")[1])
        assert not hasattr(sample, "target")
        assert protocol.TASK_PROMPT not in repr(new)
        assert two_turn.PLANNING_PROMPT not in repr(new)
    assert train[0].observation.history_valid_mask == (False, False, True)


def test_real_processor_tiny_qwen_direct_forward_no_generation(config, data, processor, tiny_qwen, monkeypatch):
    sample = data[0][0]
    def forbidden(*args, **kwargs):
        pytest.fail("action path invoked")
    for module, name in ((protocol, "serialize_action"), (protocol, "parse_action"),
                         (two_turn, "planning_messages"), (two_turn, "generate_action")):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(tiny_qwen, "generate", forbidden)
    monkeypatch.setattr(direct, "resolve_image_path", lambda root, path: root / path)
    messages = direct.direct_messages(sample.observation, [Image.new("RGB", (64, 64))])
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "longitudinal=" not in rendered and "lateral=" not in rendered
    runtime = SimpleNamespace(image_loader=lambda path: Image.new("RGB", (64, 64)))
    runner = direct.DirectRunner(tiny_qwen, processor, runtime, Path("unused"), "cpu")
    planner = direct.WaypointDecoder(32, replace(config, planner_dimension=16, num_decoder_layers=1))
    prediction, evidence = runner.predict(planner, sample)
    assert prediction.shape == (1, 6, 2)
    assert evidence["hidden_state_field"] == "hidden_states[-1]"
    assert evidence["hidden_state_shape"][:2] == evidence["attention_mask_shape"]
    assert evidence["hidden_state_shape"][-1] == 32
    assert evidence["waypoint_output_shape"] == [1, 6, 2]
    loss = direct.masked_waypoint_loss(prediction, torch.ones_like(prediction),
                                     torch.ones(1, 6, dtype=torch.bool), beta=1.)
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in tiny_qwen.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in planner.parameters())
    assert direct.masked_waypoint_loss is two_turn.masked_waypoint_loss


def test_same_eligibility_and_split_isolation(artifacts):
    root = artifacts[0]
    train, validation, _, summary = direct.prepare_data(ROOT, root)
    formal_train, formal_val, _, counts = formal.prepare_data(ROOT, root)
    assert summary["train"] == counts["train"] and summary["validation"] == counts["validation"]
    assert summary["train_sample_tokens"] == [s.sample_token for s in formal_train]
    assert [s.sample_token for s in validation] == [s.sample_token for s in formal_val]
    path = root / "phase_0_4/temporal_waypoint_v0_1/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(longitudinal_action=None, longitudinal_action_valid=False, factorized_action_joint_valid=False)
    direct.write_predictions(path, rows)
    train, _, _, summary = direct.prepare_data(ROOT, root)
    assert len(train) == 3 and summary["train"]["excluded_records"] == 1
    rows[1]["scene_token"] = "scene-560"
    direct.write_predictions(path, rows)
    with pytest.raises(ValueError, match="outside frozen split"):
        direct.prepare_data(ROOT, root)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 1)
        self.lora_A = nn.Linear(1, 1)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560))
        self.calls = 0

    def forward(self, **kwargs):
        assert not torch.is_grad_enabled() and not self.training
        assert kwargs["output_hidden_states"] and kwargs["return_dict"]
        assert kwargs["use_cache"] is False
        self.calls += 1
        return SimpleNamespace(hidden_states=(torch.ones(1, 3, 2560),))

    def generate(self, **kwargs):
        pytest.fail("Direct generated text")


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        assert "longitudinal=" not in repr(messages) and "lateral=" not in repr(messages)
        return {"input_ids": torch.ones(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3, dtype=torch.long)}


def test_run_training_reload_comparisons_and_freeze(config, references, monkeypatch):
    prepared = direct.prepare_data(ROOT, references)
    for record in prepared[2].values():
        record["longitudinal_action"] = record["lateral_action"] = object()
    monkeypatch.setattr(direct, "prepare_data", lambda *args: prepared)
    model = Backbone()
    original = deepcopy(model.state_dict())
    loads = []
    def adapter(base, path):
        assert path.name == "adapter_step_3564"
        return base
    runtime = SimpleNamespace(
        device_selector=lambda device: "cpu", dtype_selector=lambda precision: torch.bfloat16,
        processor_loader=lambda *args: Processor(), model_loader=lambda *args: model,
        adapter_loader=adapter, image_loader=lambda path: loads.append(path) or object(),
        package_version=lambda package: "synthetic_test")
    monkeypatch.setattr(direct, "default_runtime_dependencies", lambda: runtime)
    monkeypatch.setattr(direct, "resolve_image_path", lambda root, path: root / path)
    config = replace(config, planner_dimension=16, num_decoder_layers=1,
                     checkpoint_interval=1, validation_interval=1, gradient_accumulation_steps=3)
    reference_paths = [p for p in (references / "phase_0_4").rglob("*predictions.jsonl")]
    original_files = {p: p.read_bytes() for p in reference_paths}
    git = GitProvenance(commit="a" * 40, branch="test", detached_head=False, worktree_clean=True)
    result = direct.run(repository=ROOT, dataset_root=Path("unused"), derived_root=references,
                        config=config, git_provenance=git)
    assert result["optimizer_steps"] == 2 and result["samples_consumed"] == 4
    assert result["reload_consistency"]["reload_consistency"]
    assert result["reload_consistency"]["sample_count"] == 4
    assert all(torch.equal(original[n], p) for n, p in model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert original_files == {p: p.read_bytes() for p in reference_paths}
    output = references / config.output_relative_dir
    metadata = json.loads((output / "run_metadata.json").read_text())
    for key in ("qwen_trainable_parameters", "lora_trainable_parameters", "action_generation_calls",
                "action_tokens_inserted", "action_values_used_for_prediction", "action_values_used_for_loss"):
        assert metadata[key] == 0
    assert metadata["first_training_forward"]["hidden_state_shape"] == [1, 3, 2560]
    assert metadata["data"]["test_records_read"] == 0
    expected = {"training_summary.json", "training_history.jsonl", "checkpoint_selection.json",
                "validation_predictions.jsonl", "validation_metrics.json", "reload_consistency.json",
                "data_summary.json", "run_metadata.json", "resolved_config.json", "best_checkpoint.json",
                "comparison_to_action_conditioned_planner.json", "comparison_to_ego_history_mlp.json",
                "comparison_to_constant_velocity.json"}
    assert expected <= {p.name for p in output.iterdir()}
    assert json.loads((output / "comparison_to_constant_velocity.json").read_text())["jointly_valid_sample_count"] == 3
    assert json.loads((output / "comparison_to_ego_history_mlp.json").read_text())["jointly_valid_sample_count"] == 4
    rows = direct.read_predictions(output / "validation_predictions.jsonl", direct.TRACK)
    assert all(r["conditioning_type"] == direct.TRACK for r in rows)
    changed = deepcopy(rows)
    changed[0]["predicted_waypoints"][0][0] += .1
    assert not direct.compare_reload(rows, changed)["reload_consistency"]
    with pytest.raises(FileExistsError):
        direct.run(repository=ROOT, dataset_root=Path("unused"), derived_root=references,
                   config=config, git_provenance=git)


def test_optimizer_decoder_only_and_selection(config):
    model = Backbone()
    direct.freeze_backbone(model)
    planner = direct.WaypointDecoder(32, config)
    optimizer = torch.optim.AdamW(planner.parameters())
    assert direct.verify_freeze(model, planner, optimizer)["qwen_trainable_parameters"] == 0
    optimizer.add_param_group({"params": model.parameters()})
    with pytest.raises(ValueError, match="optimizer contract"):
        direct.verify_freeze(model, planner, optimizer)
    metrics = {"invalid_prediction_count": 0, "valid_prediction_count": 4, "ade_m": 1., "fde_m": 2.}
    assert direct.selection_key(metrics, 1) < direct.selection_key(metrics, 2)
    assert direct.selection_key(metrics, 9) < direct.selection_key({**metrics, "invalid_prediction_count": 1}, 1)
    assert direct.selection_key({**metrics, "ade_m": .5}, 9) < direct.selection_key(metrics, 1)
    assert direct.selection_key({**metrics, "fde_m": 1.}, 9) < direct.selection_key(metrics, 1)


@pytest.mark.parametrize("failure", ["duplicate_left", "duplicate_right", "scene", "target", "mask"])
def test_comparison_mismatch(artifacts, failure):
    reference = artifacts[1]
    left = [{**r, "conditioning_type": direct.TRACK} for r in deepcopy(reference)]
    if failure == "duplicate_left":
        left.append(left[-1])
    elif failure == "duplicate_right":
        reference.append(reference[-1])
    elif failure == "scene":
        reference[0]["scene_token"] = "wrong"
    elif failure == "target":
        reference[0]["target_waypoints"][0][0] += 1
    else:
        reference[0]["trajectory_valid_mask"][0] = False
    with pytest.raises(ValueError):
        direct.paired_comparison(left, reference, "constant_velocity")


def test_joint_comparison_and_unmatched(artifacts):
    reference = artifacts[1]
    left = [{**r, "conditioning_type": direct.TRACK} for r in deepcopy(reference)]
    result = direct.paired_comparison(left, reference[1:], "constant_velocity")
    assert result["jointly_valid_sample_count"] == 3
    assert result["direct_unmatched_tokens"] == [left[0]["sample_token"]]
    assert all(m["absolute_delta_m"] == 0 and m["sample_count"] == 3 for m in result["metrics"].values())


def test_test_rejection_and_reference_code_unchanged(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("access before test rejection")
    monkeypatch.setattr(direct, "prepare_data", forbidden)
    with pytest.raises(ValueError, match="only permits"):
        direct.run(repository=ROOT, dataset_root=Path("absent"), derived_root=Path("absent"),
                   config=config, git_provenance=None, validation_split="test")
    with pytest.raises(SystemExit):
        cli.main(["--validation-split", "test"])
    assert cli.main(["--dry-run"]) == 0
    subprocess.run(["git", "diff", "--exit-code", "5bc821d", "--", "src/baselines", "src/phase0/phase0_4c_full_train.py",
                    "src/phase0/phase0_4c_two_turn_planner.py", "src/phase0/phase0_4c_evaluation.py"],
                   cwd=ROOT, check=True, capture_output=True)
