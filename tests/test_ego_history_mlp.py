from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_ego_history_mlp as cli
from src.baselines import ego_history_mlp as mlp
from src.baselines import constant_velocity as cv
from src.phase0 import phase0_4b_lora_smoke as intake
from src.phase0.phase0_4_temporal_dataset import build_record, collect_history
from src.phase0.phase0_4c_full_train import eligible_samples, evaluate as evaluate_planner, write_predictions
from src.phase0.phase0_4c_two_turn_planner import masked_waypoint_loss
from src.phase0.qwen3vl_dataset_adapter import GitProvenance, load_config as load_ego_config
from test_phase0_3_development_projection import _mapping_payload
from test_phase0_4_temporal_dataset import case


@pytest.fixture
def config():
    return mlp.load_config(ROOT / "configs/phase0_4c_ego_history_mlp.yaml")


@pytest.fixture
def records(case):
    reader, source = case
    train = [build_record(r, collect_history(reader, r, 3), 3) for r in source]
    train = json.loads(json.dumps(train))
    validation = deepcopy(train)
    for r in validation:
        r.update(split="validation", scene_token="scene-560", sample_token=r["sample_token"] + "-val")
        r["historical_sample_tokens"][-1] = r["sample_token"]
    for r in train + validation:
        r["split_mapping_sha256"] = _mapping_payload()["scene_split_mapping_sha256"]
    return train, validation


@pytest.fixture
def artifacts(records, tmp_path, monkeypatch):
    train, validation = records
    source = intake.load_source_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT)
    source = replace(source, source_contract=replace(source.source_contract,
                                                    expected_sample_counts={"train": 4, "validation": 4}))
    monkeypatch.setattr(intake, "load_source_config", lambda *a: source)
    mapping = tmp_path / source.source_contract.scene_mapping_relative_path
    mapping.parent.mkdir(parents=True)
    mapping.write_text(json.dumps(_mapping_payload()))
    temporal = tmp_path / "phase_0_4/temporal_waypoint_v0_1"
    temporal.mkdir(parents=True)
    for split, rows in (("train", train), ("validation", validation)):
        write_predictions(temporal / f"{split}.jsonl", rows)
    cv_dir = tmp_path / "phase_0_4/constant_velocity_baseline_v0_1"
    cv_dir.mkdir()
    _, cv_rows = cv.evaluate(validation, [.5, 1., 1.5, 2., 2.5, 3.])
    write_predictions(cv_dir / "predictions.jsonl", cv_rows)
    samples, _ = eligible_samples(validation, load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml"), "validation")
    class Runner:
        def predict(self, model, sample, conditioning_type):
            return torch.ones(1, 6, 2), {"action_context": "longitudinal=keep; lateral=straight"}
    _, planner_rows = evaluate_planner(torch.nn.Identity(), Runner(), samples,
                                      {r["sample_token"]: r for r in validation}, "predicted_action")
    planner_dir = tmp_path / "phase_0_4/two_turn_planner_full_v0_1"
    planner_dir.mkdir()
    write_predictions(planner_dir / "validation_predicted_action_predictions.jsonl", planner_rows)
    return tmp_path, cv_rows, planner_rows


def test_features_padding_partial_missing_and_order(records):
    train, _ = records
    stats = mlp.normalization_stats(train, 1e-6)
    first = mlp.encode(train[0]["ego_motion_history"], train[0]["history_valid_mask"], stats).reshape(3, 7)
    assert torch.equal(first[:2], torch.zeros(2, 7))
    assert first[-1].tolist() == [0, 0, 0, 0, 0, 0, 1]
    partial = mlp.encode(train[1]["ego_motion_history"], train[1]["history_valid_mask"], stats).reshape(3, 7)
    assert partial[-1, [1, 3, 5, 6]].tolist() == [1, 0, 1, 1]
    history = deepcopy(train[-1]["ego_motion_history"])
    for index, motion in enumerate(history):
        motion["speed_mps"] = float(index * 5)
    result = mlp.encode(history, [True] * 3, stats).reshape(3, 7)
    assert result[:, 0].tolist() == sorted(result[:, 0].tolist())
    del history[-1]["speed_mps"]
    history[-1]["longitudinal_acceleration_mps2"] = None
    result = mlp.encode(history, [True] * 3, stats).reshape(3, 7)
    assert result[-1, :4].tolist() == [0, 0, 0, 0]
    history[-1]["yaw_rate_radps"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        mlp.encode(history, [True] * 3, stats)


def test_train_only_stats_population_std_and_no_validation_update(records):
    train, validation = records
    stats = mlp.normalization_stats(train, 1e-6)
    original = deepcopy(stats)
    speeds = [m["speed_mps"] for r in train for m in r["ego_motion_history"]
              if m is not None and m["speed_mps"] is not None]
    channel = stats["channels"]["speed_mps"]
    assert channel["count"] == len(speeds)
    assert channel["mean"] == pytest.approx(sum(speeds) / len(speeds))
    assert channel["std"] == pytest.approx(float(torch.tensor(speeds, dtype=torch.float64).std(correction=0)))
    assert stats["channels"]["yaw_rate_radps"]["scale"] >= 1e-6
    validation[-1]["ego_motion_history"][-1]["speed_mps"] = 99999
    mlp.encode(validation[-1]["ego_motion_history"], validation[-1]["history_valid_mask"], stats)
    assert stats == original
    with pytest.raises(ValueError, match="TRAIN only"):
        mlp.normalization_stats(validation, 1e-6)


def test_shape_loss_and_selection():
    model = mlp.EgoHistoryMLP(3)
    assert model(torch.zeros(2, 21)).shape == (2, 6, 2)
    assert [type(layer) for layer in model.layers] == [torch.nn.Linear, torch.nn.GELU,
                                                      torch.nn.Linear, torch.nn.GELU, torch.nn.Linear]
    prediction = torch.zeros(1, 6, 2, requires_grad=True)
    target = torch.ones(1, 6, 2)
    mask = torch.tensor([[True, False, False, False, False, False]])
    target[~mask] = float("nan")
    loss = masked_waypoint_loss(prediction, target, mask, beta=1.)
    assert float(loss.detach()) == .5
    loss.backward()
    assert torch.count_nonzero(prediction.grad[~mask]) == 0
    metrics = {"invalid_prediction_count": 0, "valid_prediction_count": 4, "ade_m": 1., "fde_m": 2.}
    assert mlp.selection_key(metrics, 1) < mlp.selection_key(metrics, 2)
    assert mlp.selection_key({**metrics, "fde_m": 1.}, 5) < mlp.selection_key(metrics, 1)
    assert mlp.selection_key({**metrics, "ade_m": .5, "fde_m": 99.}, 5) < mlp.selection_key(metrics, 1)


def test_no_forbidden_features(records):
    train, validation = records
    stats = mlp.normalization_stats(train, 1e-6)
    model = mlp.EgoHistoryMLP(3)
    _, first = mlp.evaluate(model, validation, stats, 256)
    changed = deepcopy(validation)
    for r in changed:
        for key in ("longitudinal_action", "lateral_action", "predicted_action", "nearby_agents",
                    "historical_cam_front_paths", "historical_relative_times_sec"):
            r[key] = object()
        r["future_waypoints"] = [[999., 999.]] * 6
    _, second = mlp.evaluate(model, changed, stats, 256)
    assert [r["predicted_waypoints"] for r in first] == [r["predicted_waypoints"] for r in second]


def test_full_producer_intake_training_reload_and_artifacts(config, artifacts, records):
    root, cv_rows, planner_rows = artifacts
    git = GitProvenance(commit="a" * 40, branch="test", detached_head=False, worktree_clean=True)
    result = mlp.run(repository=ROOT, derived_root=root, config=config, git_provenance=git)
    output = root / config["output_relative_dir"]
    assert result["epochs"] == 20 and result["optimizer_steps"] == 20
    assert result["reload_consistency"]["matched"]
    expected = {"training_summary.json", "training_history.jsonl", "best_checkpoint.json", "best_model.pt",
                "normalization_stats.json", "validation_predictions.jsonl", "validation_metrics.json",
                "data_summary.json", "run_metadata.json", "resolved_config.json", "reload_consistency.json",
                "comparison_to_constant_velocity.json", "comparison_to_action_conditioned_planner.json"}
    assert {p.name for p in output.iterdir()} == expected
    summary = json.loads((output / "data_summary.json").read_text())
    for key in ("images_opened", "qwen_model_loads", "lora_model_loads", "structured_action_inputs",
                "future_information_inputs", "test_records_read", "test_images_opened", "test_labels_read"):
        assert summary[key] == 0
    assert json.loads((output / "comparison_to_constant_velocity.json").read_text())["jointly_valid_sample_count"] == 3
    assert json.loads((output / "comparison_to_action_conditioned_planner.json").read_text())["jointly_valid_sample_count"] == 4
    rows = mlp.read_predictions(output / "validation_predictions.jsonl", mlp.TRACK)
    second = root / "repeat"
    second.mkdir()
    repeated, repeated_rows = mlp.fit(*records, config, second)
    assert repeated_rows == rows
    assert repeated["best_checkpoint"] == result["best_checkpoint"]
    with pytest.raises(FileExistsError):
        mlp.run(repository=ROOT, derived_root=root, config=config, git_provenance=git)
    corrupted = deepcopy(rows)
    corrupted[0]["predicted_waypoints"][0][0] += .01
    assert not mlp.reload_consistency(rows, corrupted, 1e-6, 1e-6)["matched"]


@pytest.mark.parametrize("failure", ["duplicate_left", "duplicate_right", "scene", "target", "mask"])
def test_comparison_rejects_mismatch(records, artifacts, failure):
    _, reference, _ = artifacts
    left = [{**r, "conditioning_type": mlp.TRACK} for r in deepcopy(reference)]
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
        mlp.paired_comparison(left, reference, "constant_velocity")


def test_joint_comparison_unmatched_and_sign(artifacts):
    _, reference, _ = artifacts
    left = [{**r, "conditioning_type": mlp.TRACK} for r in deepcopy(reference)]
    result = mlp.paired_comparison(left, reference[1:], "constant_velocity")
    assert result["jointly_valid_sample_count"] == 3
    assert result["mlp_unmatched_tokens"] == [left[0]["sample_token"]]
    assert all(m["absolute_delta_m"] == 0 for m in result["metrics"].values())
    for row in left[1:]:
        row["predicted_waypoints"] = row["target_waypoints"]
    result = mlp.paired_comparison(left, reference, "constant_velocity")
    assert all(m["absolute_delta_m"] <= 0 for m in result["metrics"].values())


def test_isolation_before_access(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("test data accessed")
    monkeypatch.setattr(mlp, "load_temporal_records", forbidden)
    with pytest.raises(ValueError, match="only permits"):
        mlp.prepare_data(ROOT, Path("absent"), 3, "train", "test")
    with pytest.raises(ValueError, match="only permits"):
        mlp.run(repository=ROOT, derived_root=Path("absent"), config=config,
                git_provenance=None, validation_split="test")
    with pytest.raises(SystemExit):
        cli.main(["--validation-split", "test"])
    assert cli.main(["--dry-run"]) == 0


def test_contract_violation_and_cv_unchanged(config, artifacts):
    root, _, _ = artifacts
    path = root / "phase_0_4/temporal_waypoint_v0_1/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["history_order"] = "current_to_oldest"
    write_predictions(path, rows)
    with pytest.raises(ValueError, match="policy mismatch"):
        mlp.prepare_data(ROOT, root, 3, "train", "validation")
    subprocess.run(["git", "diff", "--exit-code", "1f6910f", "--",
                    "src/baselines/constant_velocity.py", "configs/phase0_4c_constant_velocity.yaml",
                    "scripts/run_phase0_4c_constant_velocity.py", "tests/test_constant_velocity.py"],
                   cwd=ROOT, check=True, capture_output=True)
