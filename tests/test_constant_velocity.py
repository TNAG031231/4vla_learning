from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_constant_velocity as cli
from src.baselines import constant_velocity as cv
from src.phase0 import phase0_4b_lora_smoke as intake
from src.phase0.phase0_4_temporal_dataset import build_record, collect_history
from src.phase0.phase0_4c_evaluation import aggregate_metrics, prediction_metrics
from src.phase0.phase0_4c_full_train import eligible_samples, evaluate as evaluate_planner, write_predictions
from src.phase0.qwen3vl_dataset_adapter import GitProvenance, load_config as load_ego_config
from test_phase0_3_development_projection import _mapping_payload
from test_phase0_4_temporal_dataset import case


@pytest.fixture
def config():
    return cv.load_config(ROOT / "configs/phase0_4c_constant_velocity.yaml")


@pytest.fixture
def records(case):
    reader, sources = case
    rows = [build_record(row, collect_history(reader, row, 3), 3) for row in sources]
    for row in rows:
        row.update(split="validation", scene_token="scene-560")
        row["split_mapping_sha256"] = _mapping_payload()["scene_split_mapping_sha256"]
    return rows


@pytest.fixture
def frozen_data(records, tmp_path, monkeypatch):
    source = intake.load_source_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT)
    source = replace(source, source_contract=replace(
        source.source_contract, expected_sample_counts={"validation": len(records)}))
    monkeypatch.setattr(intake, "load_source_config", lambda *a: source)
    mapping_path = tmp_path / source.source_contract.scene_mapping_relative_path
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(json.dumps(_mapping_payload()))
    output = tmp_path / "phase_0_4/temporal_waypoint_v0_1"
    output.mkdir(parents=True)
    write_predictions(output / "validation.jsonl", records)
    return tmp_path


@pytest.fixture
def git():
    return GitProvenance(commit="a" * 40, branch="test-cv", detached_head=False, worktree_clean=True)


@pytest.mark.parametrize("speed", [0.0, 2.0, 13.25])
def test_equation_horizons_and_units(config, records, speed):
    row = records[-1]
    row["ego_motion_history"][-1]["speed_mps"] = speed
    prediction, reason = cv.predict(row["ego_motion_history"], row["history_valid_mask"],
                                    config["waypoint_times_sec"])
    assert reason is None
    assert prediction.shape == (6, 2)
    assert prediction[:, 0].tolist() == [speed * t for t in (.5, 1, 1.5, 2, 2.5, 3)]
    assert prediction[:, 1].tolist() == [0] * 6
    assert row["coordinate_metadata"]["future_ego_trajectory"]["x_axis"] == "forward"
    assert config["speed_semantics"] == "latest_available_past_interval_average_speed_magnitude"


@pytest.mark.parametrize("failure,reason", [
    ("entry", "missing_current_history_entry"),
    ("empty", "missing_current_history_entry"),
    ("truncated", "missing_current_history_entry"),
    ("mask", "invalid_current_history_mask"),
    ("empty_mask", "invalid_current_history_mask"),
    ("missing_speed", "missing_current_speed"),
    ("null", "missing_current_speed"),
    ("nan", "nonfinite_current_speed"),
    ("inf", "nonfinite_current_speed"),
])
def test_missing_motion_is_invalid_without_fallback(config, records, failure, reason):
    row = records[-1]
    if failure == "entry":
        row["ego_motion_history"][-1] = None
    elif failure == "empty":
        row["ego_motion_history"] = []
    elif failure == "truncated":
        row["ego_motion_history"].pop()
    elif failure == "mask":
        row["history_valid_mask"][-1] = False
    elif failure == "empty_mask":
        row["history_valid_mask"] = []
    elif failure == "missing_speed":
        del row["ego_motion_history"][-1]["speed_mps"]
    else:
        row["ego_motion_history"][-1]["speed_mps"] = {
            "null": None, "nan": float("nan"), "inf": float("inf")}[failure]
    metrics, rows = cv.evaluate([row], config["waypoint_times_sec"])
    assert rows[0]["predicted_waypoints"] is None
    assert metrics["invalid_reason_counts"] == {reason: 1}
    assert metrics["sample_count"] == metrics["invalid_prediction_count"] == 1
    assert metrics["trajectory_valid_rate"] == 0
    assert metrics["ade_m"] is None


def test_prediction_ignores_all_non_speed_inputs(config, records):
    original = records[-1]
    changed = deepcopy(original)
    changed["future_waypoints"] = [[999., -999.]] * 6
    changed["future_ego_trajectory"] = object()
    changed["longitudinal_action"] = changed["lateral_action"] = object()
    changed["predicted_action"] = object()
    changed["historical_cam_front_paths"] = object()
    changed["ego_motion_history"][0] = object()
    changed["ego_motion_history"][-1]["yaw_rate_radps"] = object()
    changed["ego_motion_history"][-1]["longitudinal_acceleration_mps2"] = object()
    _, first = cv.evaluate([original], config["waypoint_times_sec"])
    _, second = cv.evaluate([changed], config["waypoint_times_sec"])
    assert first[0]["predicted_waypoints"] == second[0]["predicted_waypoints"]
    assert first[0]["ade_m"] != second[0]["ade_m"]


def test_shared_metrics_and_mask_semantics(config, records):
    row = records[-1]
    row["trajectory_valid_mask"] = [True, True, False, True, False, False]
    row["future_waypoints"] = [[0., 0.]] * 6
    metrics, rows = cv.evaluate([row], config["waypoint_times_sec"])
    prediction, _ = cv.predict(row["ego_motion_history"], row["history_valid_mask"], config["waypoint_times_sec"])
    expected = prediction_metrics(prediction, torch.zeros(6, 2), torch.tensor(row["trajectory_valid_mask"]))
    assert all(rows[0][key] == value for key, value in expected.items())
    assert metrics == aggregate_metrics(rows, cv.TRACK)
    speed = row["ego_motion_history"][-1]["speed_mps"]
    assert metrics["ade_m"] == pytest.approx(speed * (0.5 + 1 + 2) / 3)
    assert metrics["fde_m"] == pytest.approx(speed * 2)
    assert metrics["fde_3s_m"] is None
    assert metrics["fde_3s_m_sample_count"] == 0


def test_real_intake_eligibility_and_artifacts(config, records, frozen_data, git):
    loaded, summary = cv.prepare_data(ROOT, frozen_data)
    samples, counts = eligible_samples(records, load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml"),
                                       "validation")
    assert summary["validation"] == counts
    assert [s.sample_token for s in samples] == [r["sample_token"] for r in loaded]
    result = cv.run(repository=ROOT, derived_root=frozen_data, config=config, git_provenance=git)
    assert result["metrics"]["sample_count"] == 4
    assert result["metrics"]["valid_prediction_count"] == 3
    assert result["metrics"]["invalid_reason_counts"] == {"missing_current_speed": 1}
    output = frozen_data / config["output_relative_dir"]
    assert {p.name for p in output.iterdir()} == {
        "predictions.jsonl", "metrics.json", "data_summary.json", "run_metadata.json", "resolved_config.json"}
    metadata = json.loads((output / "run_metadata.json").read_text())
    for key in ("speed_source", "speed_semantics", "direction_assumption", "motion_model"):
        assert metadata[key] == config[key]
    summary = json.loads((output / "data_summary.json").read_text())
    for key in ("test_records_read", "test_images_opened", "test_labels_read", "train_records_used_for_fitting",
                "images_opened", "model_loads"):
        assert summary[key] == 0
    assert summary["test_evaluation_performed"] is False
    restored = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert aggregate_metrics(restored, cv.TRACK) == result["metrics"]
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        cv.run(repository=ROOT, derived_root=frozen_data, config=config, git_provenance=git)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize("split", ["train", "test"])
def test_split_rejected_before_access(config, git, monkeypatch, split):
    def forbidden(*args, **kwargs):
        pytest.fail("data accessed before split rejection")
    monkeypatch.setattr(cv, "prepare_data", forbidden)
    with pytest.raises(ValueError, match="validation only"):
        cv.run(repository=ROOT, derived_root=Path("absent"), config=config, git_provenance=git, split=split)
    with pytest.raises(ValueError, match="validation only"):
        cv.evaluate([{"split": split}], config["waypoint_times_sec"])
    with pytest.raises(SystemExit):
        cli.main(["--split", split, "--config", "absent"])


def test_cli_dry_run_no_access(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run touched data or git")
    monkeypatch.setattr(cli, "run", forbidden)
    monkeypatch.setattr(cli, "collect_git_provenance", forbidden)
    assert cli.main(["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run_no_data_or_model_access"


@pytest.fixture
def planner_artifact(records, tmp_path):
    samples, _ = eligible_samples(records, load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml"),
                                  "validation")
    class Runner:
        def predict(self, planner, sample, conditioning_type):
            return torch.ones(1, 6, 2), {"action_context": "longitudinal=keep; lateral=straight"}
    _, rows = evaluate_planner(torch.nn.Identity(), Runner(), samples,
                               {r["sample_token"]: r for r in records}, "predicted_action")
    result = tmp_path / "planner"
    result.mkdir()
    write_predictions(result / "validation_predicted_action_predictions.jsonl", rows)
    return result, rows


def test_paired_comparison_uses_actual_producer_and_common_valid_set(config, records, planner_artifact):
    directory, reference = planner_artifact
    _, rows = cv.evaluate(records, config["waypoint_times_sec"])
    comparison = cv.compare_planner(rows, directory)
    assert comparison["matching_sample_count"] == 4
    assert comparison["jointly_valid_sample_count"] == 3
    expected = aggregate_metrics(reference[1:], "predicted_action")
    for key, values in comparison["metrics"].items():
        assert values["planner"] == expected[key]
        assert values["absolute_delta_m"] == pytest.approx(values["cv"] - values["planner"])
        assert values["relative_delta"] == pytest.approx(values["absolute_delta_m"] / values["planner"])
    path = directory / "validation_predicted_action_predictions.jsonl"
    write_predictions(path, list(reversed(reference[1:])))
    partial = cv.compare_planner(rows, directory)
    assert partial["cv_unmatched_count"] == 1
    assert partial["metrics"] == comparison["metrics"]
    write_predictions(path, [])
    empty = cv.compare_planner(rows, directory)
    assert empty["jointly_valid_sample_count"] == 0
    assert empty["metrics"]["ade_1s_m"]["relative_delta"] is None


@pytest.mark.parametrize("failure", ["duplicate", "split", "diagnostic", "target", "mask"])
def test_comparison_rejects_wrong_artifact(config, records, planner_artifact, failure):
    directory, reference = planner_artifact
    if failure == "duplicate":
        reference.append(reference[0])
    elif failure == "split":
        reference[0]["split"] = "test"
    elif failure == "diagnostic":
        reference[0]["conditioning_type"] = "gt_action_diagnostic"
    elif failure == "target":
        reference[0]["target_waypoints"][0][0] += 1
    else:
        reference[0]["trajectory_valid_mask"][0] = False
    write_predictions(directory / "validation_predicted_action_predictions.jsonl", reference)
    _, rows = cv.evaluate(records, config["waypoint_times_sec"])
    with pytest.raises(ValueError):
        cv.compare_planner(rows, directory)


def test_run_with_comparison(config, frozen_data, git, planner_artifact):
    directory, _ = planner_artifact
    result = cv.run(repository=ROOT, derived_root=frozen_data, config=config, git_provenance=git,
                    planner_result_dir=directory)
    saved = frozen_data / config["output_relative_dir"] / "comparison_to_action_conditioned_planner.json"
    assert json.loads(saved.read_text()) == result["comparison"]


def test_zero_reference_error_has_undefined_relative_delta(config, records, planner_artifact):
    directory, reference = planner_artifact
    for row in reference:
        row["predicted_waypoints"] = row["target_waypoints"]
    write_predictions(directory / "validation_predicted_action_predictions.jsonl", reference)
    _, rows = cv.evaluate(records, config["waypoint_times_sec"])
    result = cv.compare_planner(rows, directory)
    assert all(value["planner"] == 0 and value["relative_delta"] is None
               for value in result["metrics"].values())
