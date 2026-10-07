from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_phase0_4c_trajectory_capability as cli
from src.baselines import ego_history_mlp as mlp
from src.phase0 import phase0_4c_direct_waypoint as direct
from src.phase0 import phase0_4c_full_train as formal
from src.phase0 import phase0_4c_trajectory_stratification as analysis
from src.phase0.phase0_4c_evaluation import aggregate_metrics
from src.phase0.qwen3vl_dataset_adapter import GitProvenance, load_config as load_ego_config
from test_ego_history_mlp import case, records, artifacts


@pytest.fixture
def population(records, artifacts):
    root, cv_rows, _ = artifacts
    train, validation = records
    samples, _ = formal.eligible_samples(
        validation, load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml"), "validation")
    by_token = {r["sample_token"]: r for r in validation}
    offsets = dict(zip(by_token, (0., 4., 2., 1.), strict=True))

    class Runner:
        def predict(self, model, sample, conditioning_type=None):
            target = torch.tensor(by_token[sample.sample_token]["future_waypoints"], dtype=torch.float32)
            return (target + offsets[sample.sample_token]).unsqueeze(0), {
                "action_context": "longitudinal=keep; lateral=straight"}

    _, planner_rows = formal.evaluate(torch.nn.Identity(), Runner(), samples, by_token, "predicted_action")
    direct_samples = [direct.DirectSample(s.sample_token, s.scene_token, s.split, s.observation) for s in samples]
    _, direct_rows = direct.evaluate(torch.nn.Identity(), Runner(), direct_samples, by_token)

    class FixedPredictions(torch.nn.Module):
        def forward(self, features):
            return torch.tensor([r["future_waypoints"] for r in validation], dtype=torch.float32) + 2.

    _, mlp_rows = mlp.evaluate(FixedPredictions(), validation, mlp.normalization_stats(train, 1e-6), 256)
    predictions = {"mlp": mlp_rows, "planner": planner_rows, "direct": direct_rows, "cv": cv_rows}
    config = analysis.load_config(ROOT / "configs/phase0_4c_trajectory_stratification.yaml")
    config.update(expected_sample_count=4, expected_cv_valid_count=3, expected_cv_invalid_count=1, top_k=2)
    config["expected_metrics"] = {
        name: {k: aggregate_metrics(rows, analysis.TRACKS[name])[k] for k in mlp.METRICS[2:]}
        for name, rows in predictions.items() if name != "cv"}
    for name, (config_file, filename) in analysis.SOURCES.items():
        producer = yaml.safe_load((ROOT / "configs" / config_file).read_text())
        directory = root / producer["output_relative_dir"]
        directory.mkdir(exist_ok=True)
        formal.write_predictions(directory / filename, predictions[name])
        (directory / "run_metadata.json").write_text(json.dumps({
            "execution_git_commit": "a" * 40, "fixture": "producer_evaluation_with_fixed_predictions"}))
    return validation, predictions, config, root


def test_alignment_aggregate_conservation_and_ranking(population):
    records, predictions, config, _ = population
    predictions = {name: list(reversed(rows)) for name, rows in predictions.items()}
    summary, samples = analysis.analyze(records, predictions, config)
    assert len(samples) == 4
    assert summary["aggregate_reproduction"]["passed"]
    for name in analysis.TRACKS:
        expected = aggregate_metrics(predictions[name], analysis.TRACKS[name])
        for metric in mlp.METRICS:
            assert summary["overall"]["models"][name][metric] == pytest.approx(expected[metric])
    for axis, groups in summary["stratification"].items():
        assert sum(g["sample_count"] for g in groups) == 4
        assert all(sum(r["strata"][axis] == g["value"] for r in samples) == g["sample_count"] for g in groups)
    assert samples[0]["strata"]["valid_history_count"] == 1
    assert samples[0]["strata"]["motion_availability"] == "unavailable"
    for sample in samples:
        for metric in mlp.METRICS[2:]:
            assert sample["planner_minus_mlp"][metric] == pytest.approx(
                sample["models"]["planner"][metric] - sample["models"]["mlp"][metric])
    for key, sign in (("planner_strongest_relative_wins", -1), ("mlp_strongest_relative_wins", 1)):
        ranked = summary["ranking"][key]
        values = [r["planner_minus_mlp"]["ade_3s_m"] for r in ranked]
        assert all(v * sign > 0 for v in values)
        assert values == sorted(values, reverse=sign == 1)
        assert len(values) <= 2
    pair = summary["overall"]["pairwise"]["planner_minus_mlp"]["ade_3s_m"]
    assert pair["planner_win_count"] == 2
    assert pair["mlp_win_count"] == pair["tie_count"] == 1
    assert pair["planner_win_rate"] == .5
    assert pair["mean_planner_minus_mlp"] == pytest.approx(
        sum(s["planner_minus_mlp"]["ade_3s_m"] for s in samples) / 4)


def test_cv_missing_samples_remain_in_population(population):
    summary, samples = analysis.analyze(*population[:3])
    assert summary["overall"]["models"]["cv"]["invalid_prediction_count"] == 1
    missing = next(g for g in summary["stratification"]["cv_valid"] if g["value"] is False)
    assert missing["sample_count"] == 1
    assert all(missing["models"][m]["valid_prediction_count"] == 1 for m in ("mlp", "planner", "direct"))
    assert missing["models"]["cv"]["ade_3s_m"] is None
    for left in ("mlp", "planner", "direct"):
        direction = f"{left}_minus_cv"
        assert samples[0][direction]["ade_3s_m"] is None
        assert summary["overall"]["pairwise"][direction]["ade_3s_m"]["sample_count"] == 3
        assert missing["pairwise"][direction]["ade_3s_m"]["sample_count"] == 0


@pytest.mark.parametrize("model", list(analysis.TRACKS))
@pytest.mark.parametrize("failure", ["duplicate", "missing", "extra", "scene", "target", "mask", "split", "track"])
def test_prediction_contract_failures(population, model, failure):
    records, predictions, config, _ = population
    rows = predictions[model]
    if failure == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif failure == "missing":
        rows.pop()
    elif failure == "extra":
        rows.append({**rows[0], "sample_token": "extra"})
    elif failure == "scene":
        rows[0]["scene_token"] = "wrong"
    elif failure == "target":
        rows[0]["target_waypoints"][0][0] += 1
    elif failure == "mask":
        rows[0]["trajectory_valid_mask"][0] = False
    elif failure == "split":
        rows[0]["split"] = "test"
    else:
        rows[0]["conditioning_type"] = "gt_action_diagnostic"
    with pytest.raises(ValueError):
        analysis.analyze(records, predictions, config)


@pytest.mark.parametrize("failure", ["count", "cv_count", "metric", "duplicate_temporal", "invalid_primary", "cv_speed"])
def test_population_and_metric_gates(population, failure):
    records, predictions, config, _ = population
    if failure == "count":
        config["expected_sample_count"] += 1
    elif failure == "cv_count":
        config["expected_cv_valid_count"] += 1
    elif failure == "metric":
        config["expected_metrics"]["planner"]["ade_3s_m"] += 1
    elif failure == "duplicate_temporal":
        records.append(deepcopy(records[0]))
    elif failure == "invalid_primary":
        predictions["planner"][0].update(prediction_valid=False, predicted_waypoints=None)
    else:
        records[0]["ego_motion_history"][-1]["speed_mps"] = 1.
    with pytest.raises(ValueError):
        analysis.analyze(records, predictions, config)


def test_action_correctness_optional_and_invalid_gt(population):
    records, predictions, config, _ = population
    contexts = ("longitudinal=keep; lateral=straight", "longitudinal=keep; lateral=left",
                "longitudinal=stop; lateral=straight", "longitudinal=stop; lateral=right")
    for record, row, context in zip(records, predictions["planner"], contexts, strict=True):
        record.update(longitudinal_action="keep", longitudinal_action_valid=True,
                      lateral_action="straight", lateral_action_valid=True, factorized_action_joint_valid=True)
        row["action_context"] = context
    summary, samples = analysis.analyze(records, predictions, config)
    assert [s["strata"]["action_correctness"] for s in samples] == list(analysis.ACTION_GROUPS[:4])
    records[0].update(longitudinal_action=None, longitudinal_action_valid=False, factorized_action_joint_valid=False)
    summary, samples = analysis.analyze(records, predictions, config)
    assert samples[0]["strata"]["longitudinal"] == "invalid"
    assert samples[0]["strata"]["action_correctness"] == "invalid_gt_action"
    for row in predictions["planner"]:
        row.pop("action_context")
    summary, _ = analysis.analyze(records, predictions, config)
    assert summary["action_correctness_status"] == "not available from existing artifacts"
    assert "action_correctness" not in summary["stratification"]
    predictions["planner"][0]["action_context"] = contexts[0]
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze(records, predictions, config)


def test_persisted_artifacts_rerun_guard_and_no_model_or_test_access(population, monkeypatch):
    records, predictions, config, root = population
    opened = []
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        assert path.name not in ("test.jsonl", "train.jsonl")
        assert path.suffix not in (".jpg", ".png", ".pt", ".safetensors")
        opened.append(path)
        return original_open(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("model or training called by offline analysis")

    monkeypatch.setattr(Path, "open", tracked_open)
    for module, name in ((formal.PlannerRunner, "predict"), (direct.DirectRunner, "predict"),
                         (mlp.EgoHistoryMLP, "forward"), (formal, "fit"), (direct, "fit"), (mlp, "fit")):
        monkeypatch.setattr(module, name, forbidden)
    git = GitProvenance(commit="a" * 40, branch="fixture", detached_head=False, worktree_clean=True)
    summary = analysis.run(repository=ROOT, derived_root=root, config=config, git_provenance=git)
    output = root / config["output_relative_dir"]
    assert {p.name for p in output.iterdir()} == {"summary.json", "sample_analysis.jsonl"}
    assert json.loads((output / "summary.json").read_text()) == summary
    samples = [json.loads(line) for line in (output / "sample_analysis.jsonl").read_text().splitlines()]
    assert analysis.summarize(samples) == summary["overall"]
    assert summary["test_isolation"]["test_evaluation_performed"] is False
    assert all(summary["test_isolation"][k] == 0 for k in
               ("test_records_read", "test_images_opened", "test_labels_read"))
    assert all(v == 0 for v in summary["execution"].values())
    assert set(summary["artifact_provenance"]["sources"]) == set(analysis.TRACKS)
    before = {p: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        analysis.run(repository=ROOT, derived_root=root, config=config, git_provenance=git)
    assert before == {p: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize("split", ["train", "test"])
def test_split_rejected_before_access(population, monkeypatch, split):
    def forbidden(*args, **kwargs):
        pytest.fail("access before split guard")
    monkeypatch.setattr(analysis, "load_temporal_records", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="validation only"):
        analysis.run(repository=ROOT, derived_root=Path("absent"), config=population[2],
                     git_provenance=None, split=split)
    with pytest.raises(SystemExit):
        cli.main(["--split", split])


def test_cli_dry_run_without_artifact_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run accessed artifacts")
    monkeypatch.setattr(cli, "run", forbidden)
    monkeypatch.setattr(cli, "collect_git_provenance", forbidden)
    assert cli.main(["--dry-run"]) == 0
