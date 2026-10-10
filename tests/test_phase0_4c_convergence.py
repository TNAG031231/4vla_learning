from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_convergence as cli
from src.phase0 import phase0_4c_convergence as convergence
from src.phase0 import phase0_4c_full_train as full
from src.phase0.phase0_4c_evaluation import aggregate_metrics
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder, freeze_backbone
from test_phase0_4c_full_train import Backbone, SyntheticRunner, case, data, records


@pytest.fixture
def config():
    return convergence.load_config(ROOT / "configs/phase0_4c_convergence.yaml")


def inputs(config, data):
    train, validation, records, summary = data
    torch.manual_seed(config.seed)
    model = Backbone()
    freeze_backbone(model)
    return dict(model=model, planner=WaypointDecoder(16, config), runner=SyntheticRunner(),
                train=train, validation=validation, records=records, provenance={"data": summary})


@pytest.fixture
def synthetic(config, data, tmp_path):
    config = replace(config, planner_dimension=16, num_decoder_layers=1,
                     gradient_accumulation_steps=3, expected_validation_count=4,
                     expected_motion_unavailable_count=1)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    original = replace(config, num_train_epochs=1)
    full.fit(**inputs(original, data), config=original, output=baseline)
    metrics = json.loads((baseline / "validation_step_0002_metrics.json").read_text())
    return replace(config, epoch1_reference={k: metrics[k] for k in config.epoch1_reference}), baseline


@pytest.mark.parametrize("key,value", [
    ("num_train_epochs", 5), ("seed", 1), ("learning_rate", .001),
    ("weight_decay", .001), ("memory_normalization", False),
    ("planner_dimension", 128), ("validation_schedule", "step"),
    ("protocol_version", "direct"), ("validation_interval", 891),
    ("selected_adapter_relative_path", "other"), ("output_relative_dir", "old"),
    ("reproduction_rtol", float("nan")), ("epoch1_reference", {}),
])
def test_protocol_rejects_changes(config, tmp_path, key, value):
    values = asdict(config)
    values.pop("train_subset_size")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**values, key: value}))
    with pytest.raises(ValueError):
        convergence.load_config(path)


def test_production_protocol(config):
    assert convergence.EPOCH1_OPTIMIZER_STEP == 3564
    assert config.num_train_epochs == 3
    assert config.validation_schedule == "epoch_end"
    assert config.reproduction_rtol == config.extension_min_relative_improvement == .01
    assert config.expected_validation_count == 3594
    assert config.expected_motion_unavailable_count == 94
    assert config.epoch1_reference["ade_3s_m"] == 1.1878918724
    assert config.mlp_reference["fde_3s_m"] == 1.7724578323


def test_reproduction_and_extension_decisions(config):
    metrics = {**config.epoch1_reference, "sample_count": 3594, "invalid_prediction_count": 0}
    gate = convergence.reproduction_gate(metrics, config)
    assert gate["passed"] and all(v == 0 for v in gate["new_minus_old"].values())
    for key, value in (("ade_1s_m", metrics["ade_1s_m"] * 1.011),
                       ("fde_3s_m", metrics["fde_3s_m"] * .989),
                       ("invalid_prediction_count", 1), ("sample_count", 3500), ("ade_3s_m", None)):
        assert not convergence.reproduction_gate({**metrics, key: value}, config)["passed"]
    second = {"metrics": metrics}
    for factor, expected in ((.98, True), (.995, False), (1., False), (1.1, False)):
        third = {"metrics": {**metrics, "ade_3s_m": metrics["ade_3s_m"] * factor,
                             "fde_3s_m": metrics["fde_3s_m"] * factor}}
        assert convergence.extension_gate([second, second, third], config)["allowed"] is expected
    third["metrics"].update(ade_3s_m=0.1, fde_3s_m=.1, invalid_prediction_count=1)
    assert not convergence.extension_gate([second, second, third], config)["allowed"]


def test_three_epochs_reproduce_and_persist(synthetic, data, tmp_path):
    config, baseline = synthetic
    output = tmp_path / "convergence"
    output.mkdir()
    kwargs = inputs(config, data)
    original = deepcopy(kwargs["model"].state_dict())
    result = convergence.fit(**kwargs, config=config, output=output)
    assert [e["epoch"] for e in result["epochs"]] == [1, 2, 3]
    assert result["optimizer_steps"] == 6
    old = torch.load(baseline / "planner_step_0002.pt", weights_only=True)
    new = torch.load(output / "epoch_1.pt", weights_only=True)
    assert all(torch.equal(p, new["planner_state_dict"][n]) for n, p in old["planner_state_dict"].items())
    assert new["optimizer_state_dict"]["state"]
    assert new["epoch"] == 1 and new["optimizer_step"] == 2
    assert new["training_config"] == asdict(config)
    assert all(torch.equal(p, kwargs["model"].state_dict()[n]) for n, p in original.items())
    assert json.loads((output / "epoch1_reproduction.json").read_text())["passed"]
    validation_calls = [c for c in kwargs["runner"].calls if c[0] == "validation"]
    assert len(validation_calls) == 3 * 4 + 4  # Epoch-end full validation plus selected reload subset.
    assert {c[1] for c in validation_calls} == {"predicted_action"}
    assert {c[0] for c in kwargs["runner"].calls} == {"train", "validation"}
    for epoch in result["epochs"]:
        rows = [json.loads(line) for line in
                (output / f"epoch_{epoch['epoch']}_predictions.jsonl").read_text().splitlines()]
        assert epoch["metrics"] == aggregate_metrics(rows, "predicted_action")
        missing = epoch["motion_unavailable"]
        assert missing["metrics"]["sample_count"] == 1
        expected = [r for r in rows if r["sample_token"] == "sample-0-validation"]
        assert missing["metrics"] == aggregate_metrics(expected, "predicted_action")
        assert epoch["minus_mlp"]["ade_3s_m"] == epoch["metrics"]["ade_3s_m"] - config.mlp_reference["ade_3s_m"]
    best = min(result["epochs"], key=lambda e: (e["metrics"]["invalid_prediction_count"],
               e["metrics"]["ade_m"], e["metrics"]["fde_m"], e["optimizer_step"]))
    assert result["best_epoch"] == best["epoch"]
    assert result["Direct multi-epoch control"] == "NOT RUN"
    assert result["test_isolation"]["test_records_read"] == 0
    assert result["test_isolation"]["test_images_opened"] == 0
    assert result["test_isolation"]["test_labels_read"] == 0
    assert result["test_isolation"]["test_evaluation_performed"] is False
    assert result["reload_consistency"]["reload_consistency"]
    assert not (output / "epoch_4.pt").exists()


def test_reproduction_failure_stops_before_epoch2(synthetic, data, tmp_path):
    config, _ = synthetic
    config = replace(config, epoch1_reference={k: v * 2 for k, v in config.epoch1_reference.items()})
    output = tmp_path / "failure"
    output.mkdir()
    kwargs = inputs(config, data)
    with pytest.raises(ValueError, match="STOP before epoch 2"):
        convergence.fit(**kwargs, config=config, output=output)
    assert (output / "epoch_1.pt").exists()
    assert not (output / "epoch_2.pt").exists()
    assert len([c for c in kwargs["runner"].calls if c[0] == "train"]) == 4
    gate = json.loads((output / "epoch1_reproduction.json").read_text())
    assert not gate["passed"] and len(gate["new_minus_old"]) == 6


def assert_state_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_state_equal(first[key], second[key])
    elif isinstance(first, list):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_state_equal(a, b)
    else:
        assert first == second


def test_optimizer_and_dropout_resume_matches_uninterrupted(synthetic, data, tmp_path):
    config, _ = synthetic
    uninterrupted, split = tmp_path / "uninterrupted", tmp_path / "split"
    uninterrupted.mkdir()
    split.mkdir()
    # Internal five-epoch run is the continuity oracle, not a supported public configuration.
    full_config = replace(config, num_train_epochs=5)
    convergence.fit(**inputs(full_config, data), config=full_config, output=uninterrupted)
    convergence.fit(**inputs(config, data), config=config, output=split)
    saved = torch.load(split / "epoch_3.pt", weights_only=True)
    torch.manual_seed(1)
    convergence.fit(**inputs(config, data), config=config, output=split, resume=saved)
    expected = torch.load(uninterrupted / "epoch_5.pt", weights_only=True)
    actual = torch.load(split / "epoch_5.pt", weights_only=True)
    for key in ("planner_state_dict", "optimizer_state_dict", "torch_rng_state", "optimizer_step"):
        assert_state_equal(expected[key], actual[key])
    assert expected["epochs"] == actual["epochs"]
    assert actual["epoch"] == 5


def test_resume_gate_reads_producer_checkpoint(synthetic, data, tmp_path):
    config, _ = synthetic
    output = tmp_path / "resume"
    output.mkdir()
    convergence.fit(**inputs(config, data), config=config, output=output)
    path = output / "epoch_3.pt"
    saved = torch.load(path, weights_only=True)
    # Controlled metric changes isolate the gate using a real checkpoint producer shape.
    for epoch, factor in zip(saved["epochs"][1:], (1., .98)):
        for key in ("ade_3s_m", "fde_3s_m"):
            epoch["metrics"][key] = config.epoch1_reference[key] * factor
    torch.save(saved, path)
    assert convergence.load_resume(output, config)["epoch"] == 3
    with pytest.raises(ValueError, match="protocol"):
        convergence.load_resume(output, replace(config, seed=1))
    saved["epochs"][2]["metrics"]["ade_3s_m"] *= 2
    torch.save(saved, path)
    with pytest.raises(ValueError, match="STOP at epoch 3"):
        convergence.load_resume(output, config)
    full.save_checkpoint(path, inputs(config, data)["planner"], config, {}, 2)
    with pytest.raises((KeyError, ValueError)):
        convergence.load_resume(output, config)


def test_entrypoint_isolation_and_rerun_before_access(config, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(convergence, "prepare_run", lambda *a: pytest.fail("data/model accessed"))
    monkeypatch.setattr(convergence, "validate_git_provenance", lambda p: SimpleNamespace(commit="synthetic"))
    kwargs = dict(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                  config=config, git_provenance=None)
    for option in ("--train-split", "--validation-split"):
        with pytest.raises(SystemExit):
            cli.main([option, "test", "--dry-run"])
    assert cli.main(["--dry-run"]) == 0
    assert "dry_run_no_data_or_model_access" in capsys.readouterr().out
    for key in ("train_split", "validation_split"):
        with pytest.raises(ValueError, match="train/validation only"):
            convergence.run(**kwargs, **{key: "test"})
    output = tmp_path / config.output_relative_dir
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        convergence.run(**kwargs)
    with pytest.raises(FileNotFoundError):
        convergence.run(**kwargs, extend_to_five=True)


def test_public_runner_uses_local_preparation(synthetic, data, tmp_path, monkeypatch):
    config, _ = synthetic
    prepared = inputs(config, data)
    monkeypatch.setattr(convergence, "prepare_run", lambda *a: prepared)
    monkeypatch.setattr(convergence, "validate_git_provenance", lambda p: SimpleNamespace(commit="synthetic"))
    result = convergence.run(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                             config=config, git_provenance=None)
    assert result["epochs"][-1]["epoch"] == 3
    assert (tmp_path / config.output_relative_dir / "data_summary.json").exists()


@pytest.mark.parametrize("better_fde,expected_epoch", [(False, 1), (True, 3)])
def test_selection_invalid_priority_and_earliest_tie(synthetic, data, tmp_path, monkeypatch,
                                                    better_fde, expected_epoch):
    config, _ = synthetic
    output = tmp_path / "selection"
    output.mkdir()
    evaluate = full.evaluate
    calls, first = 0, None

    def controlled_metrics(*args):
        nonlocal calls, first
        metrics, rows = evaluate(*args)
        calls += 1
        if calls == 1:
            first = deepcopy(metrics)
        elif calls == 2:
            metrics.update(invalid_prediction_count=1, ade_m=0., fde_m=0.)
        elif calls == 3:
            metrics.update(invalid_prediction_count=0, ade_m=first["ade_m"],
                           fde_m=first["fde_m"] - (1 if better_fde else 0))
        return metrics, rows

    monkeypatch.setattr(full, "evaluate", controlled_metrics)
    result = convergence.fit(**inputs(config, data), config=config, output=output)
    assert result["best_epoch"] == expected_epoch


def test_subset_count_and_invalid_predictions(synthetic, data, tmp_path):
    config, _ = synthetic
    kwargs = inputs(config, data)
    _, rows = full.evaluate(kwargs["planner"], kwargs["runner"], kwargs["validation"],
                            kwargs["records"], "predicted_action")
    with pytest.raises(ValueError, match="subset count"):
        convergence.motion_subset(rows, kwargs["records"], replace(config, expected_motion_unavailable_count=94))
    missing = next(r for r in rows if r["sample_token"] == "sample-0-validation")
    missing.update(prediction_valid=False, invalid_reason="invalid_structured_action", predicted_waypoints=None)
    subset = convergence.motion_subset(rows, kwargs["records"], config)
    assert subset["metrics"]["sample_count"] == subset["metrics"]["invalid_prediction_count"] == 1
    assert subset["minus_mlp"]["ade_3s_m"] is None


def test_local_preparation_matches_historical_initialization(config, data, tmp_path, monkeypatch):
    def load_model(*args):
        model = Backbone()
        model.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560))
        return model

    runtime = SimpleNamespace(
        device_selector=lambda d: "cpu", dtype_selector=lambda d: torch.bfloat16,
        processor_loader=lambda *a: object(), model_loader=load_model,
        adapter_loader=lambda model, path: model, package_version=lambda name: "synthetic")
    git = SimpleNamespace(commit="synthetic")
    monkeypatch.setattr(full, "prepare_data", lambda *a: data)
    monkeypatch.setattr(full, "default_runtime_dependencies", lambda: runtime)
    monkeypatch.setattr(convergence, "default_runtime_dependencies", lambda: runtime)
    monkeypatch.setattr(full, "validate_git_provenance", lambda p: git)
    monkeypatch.setattr(full, "fit", lambda **kwargs: kwargs)
    historical = full.run_full(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                               config=config, git_provenance=git)
    local = convergence.prepare_run(ROOT, tmp_path, tmp_path, config, git)
    assert historical["provenance"] == local["provenance"]
    assert_state_equal(historical["planner"].state_dict(), local["planner"].state_dict())
    assert_state_equal(historical["model"].state_dict(), local["model"].state_dict())
    assert not any(p.requires_grad for p in local["model"].parameters())
    assert local["runner"].generation_kwargs == historical["runner"].generation_kwargs


@pytest.fixture
def failed_run(synthetic, data, tmp_path, monkeypatch):
    config, _ = synthetic
    config = replace(config, epoch1_reference={k: v * 2 for k, v in config.epoch1_reference.items()})
    kwargs = dict(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                  config=config, git_provenance=None)
    monkeypatch.setattr(convergence, "validate_git_provenance", lambda p: SimpleNamespace(commit="synthetic"))
    monkeypatch.setattr(convergence, "prepare_run", lambda *a: inputs(config, data))
    with pytest.raises(ValueError, match="STOP before epoch 2"):
        convergence.run(**kwargs)
    output = tmp_path / config.output_relative_dir
    # Scale the fixed step boundary only for the four-sample synthetic producer.
    monkeypatch.setattr(convergence, "EPOCH1_OPTIMIZER_STEP", 2)
    return config, output, kwargs


def snapshot(output):
    return {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}


def test_failed_continuation_exact_state_and_immutable_epoch1(failed_run, data, tmp_path, monkeypatch):
    config, output, kwargs = failed_run
    before = snapshot(output)
    oracle = tmp_path / "uninterrupted"
    oracle.mkdir()
    original_gate = convergence.reproduction_gate
    # The oracle bypasses only the STOP decision so the identical failed-reference run can reach epoch 3.
    with monkeypatch.context() as control:
        control.setattr(convergence, "reproduction_gate", lambda *a: {**original_gate(*a), "passed": True})
        convergence.fit(**inputs(config, data), config=config, output=oracle)
    prepared = inputs(config, data)
    monkeypatch.setattr(convergence, "prepare_run", lambda *a: prepared)
    result = convergence.run(**kwargs, continue_after_epoch1_reproduction_fail=True)
    assert len([c for c in prepared["runner"].calls if c[0] == "train"]) == 2 * len(data[0])
    for name in ("epoch_1.pt", "epoch_1_metrics.json", "epoch_1_predictions.jsonl",
                 "epoch1_reproduction.json", "resolved_config.json", "run_metadata.json"):
        assert (output / name).read_bytes() == before[name]
    assert (output / "training_history.jsonl").read_bytes().startswith(before["training_history.jsonl"])
    assert (output / "training_history.jsonl").read_bytes() == (oracle / "training_history.jsonl").read_bytes()
    for epoch in (2, 3):
        actual = torch.load(output / f"epoch_{epoch}.pt", weights_only=True)
        expected = torch.load(oracle / f"epoch_{epoch}.pt", weights_only=True)
        for key in ("planner_state_dict", "optimizer_state_dict", "torch_rng_state", "cuda_rng_state",
                    "optimizer_step", "epoch", "epochs"):
            assert_state_equal(actual[key], expected[key])
        for suffix in ("metrics.json", "predictions.jsonl"):
            assert (output / f"epoch_{epoch}_{suffix}").read_bytes() == (oracle / f"epoch_{epoch}_{suffix}").read_bytes()
    assert result["historical_epoch1_reproduction_passed"] is False
    assert result["continuation_after_failed_historical_reproduction"] is True
    assert result["convergence_evidence_scope"] == "within_run_epoch1_to_epoch3"
    assert result["Direct multi-epoch control"] == "NOT RUN"
    assert result["test_isolation"]["test_records_read"] == 0
    first, second, third = result["epochs"]
    assert third["minus_previous_epoch"]["ade_3s_m"] == third["metrics"]["ade_3s_m"] - second["metrics"]["ade_3s_m"]
    assert third["minus_epoch1"]["ade_m"] == third["metrics"]["ade_m"] - first["metrics"]["ade_m"]
    assert not (output / "epoch_4.pt").exists()
    with pytest.raises(ValueError, match="partial"):
        convergence.run(**kwargs, continue_after_epoch1_reproduction_fail=True)


@pytest.mark.parametrize("failure", [
    "optimizer_state_dict", "empty_optimizer", "planner_state_dict", "torch_rng_state", "cuda_rng_state",
    "epoch", "optimizer_step", "config", "sample_count", "valid_prediction_count", "invalid_prediction_count",
    "passed", "epoch_2.pt", "epoch_3.pt", "epoch_2_metrics.json", "epoch_3_predictions.jsonl",
    "training_summary.json", "continuation_metadata.json", "missing_file", "data_provenance",
])
def test_failed_continuation_rejects_before_mutation(failed_run, data, monkeypatch, failure):
    config, output, kwargs = failed_run
    path = output / "epoch_1.pt"
    saved = torch.load(path, weights_only=True)
    if failure in ("optimizer_state_dict", "planner_state_dict", "torch_rng_state", "cuda_rng_state"):
        del saved[failure]
    elif failure == "empty_optimizer":
        saved["optimizer_state_dict"]["state"] = {}
    elif failure in ("epoch", "optimizer_step"):
        saved[failure] += 1
    elif failure == "config":
        saved["training_config"]["seed"] += 1
    elif failure in ("sample_count", "valid_prediction_count", "invalid_prediction_count"):
        saved["epochs"][0]["metrics"][failure] += 1
        convergence.write_json(output / "epoch_1_metrics.json", saved["epochs"][0]["metrics"])
    elif failure == "passed":
        convergence.write_json(output / "epoch1_reproduction.json", {"passed": True})
    elif failure == "missing_file":
        (output / "epoch_1_metrics.json").unlink()
    elif failure == "data_provenance":
        saved["provenance"]["data"]["train"]["eligible_records"] += 1
    else:
        (output / failure).write_text("{}")
    torch.save(saved, path)
    before = snapshot(output)
    if failure == "data_provenance":
        monkeypatch.setattr(convergence, "prepare_run", lambda *a: inputs(config, data))
    else:
        monkeypatch.setattr(convergence, "prepare_run", lambda *a: pytest.fail("data/model accessed"))
    with pytest.raises(ValueError):
        convergence.run(**kwargs, continue_after_epoch1_reproduction_fail=True)
    assert snapshot(output) == before


def test_continuation_cli_conflict_and_missing_run(config, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        cli.main(["--continue-after-epoch1-reproduction-fail", "--extend-to-five", "--dry-run"])
    for option in ("--train-split", "--validation-split"):
        with pytest.raises(SystemExit):
            cli.main(["--continue-after-epoch1-reproduction-fail", option, "test"])
    monkeypatch.setattr(convergence, "prepare_run", lambda *a: pytest.fail("data/model accessed"))
    monkeypatch.setattr(convergence, "validate_git_provenance", lambda p: SimpleNamespace(commit="synthetic"))
    kwargs = dict(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                  config=config, git_provenance=None, continue_after_epoch1_reproduction_fail=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        convergence.run(**kwargs, extend_to_five=True)
    with pytest.raises(ValueError, match="requires epoch_1.pt"):
        convergence.run(**kwargs)
    with pytest.raises(ValueError, match="train/validation only"):
        convergence.run(**kwargs, validation_split="test")
    assert not (tmp_path / config.output_relative_dir).exists()


def test_failed_continuation_can_extend_without_erasing_failure(failed_run):
    config, output, kwargs = failed_run
    convergence.run(**kwargs, continue_after_epoch1_reproduction_fail=True)
    path = output / "epoch_3.pt"
    saved = torch.load(path, weights_only=True)
    # Isolate the extension gate with controlled validation metrics on producer-generated state.
    for epoch, factor in zip(saved["epochs"][1:], (1., .98)):
        for key in ("ade_3s_m", "fde_3s_m"):
            epoch["metrics"][key] = config.epoch1_reference[key] * factor
    torch.save(saved, path)
    assert convergence.load_resume(output, config)["epoch"] == 3
    unapproved = deepcopy(saved)
    unapproved["provenance"].pop("continuation_after_failed_historical_reproduction")
    torch.save(unapproved, path)
    with pytest.raises(ValueError, match="explicit continuation evidence"):
        convergence.load_resume(output, config)
    torch.save(saved, path)
    before = (output / "epoch1_reproduction.json").read_bytes()
    result = convergence.run(**kwargs, extend_to_five=True)
    assert result["epochs"][-1]["epoch"] == 5
    assert result["historical_epoch1_reproduction_passed"] is False
    assert result["continuation_after_failed_historical_reproduction"] is True
    assert result["provenance"]["historical_epoch1_reproduction_passed"] is False
    assert (output / "epoch1_reproduction.json").read_bytes() == before


@pytest.fixture
def diagnostic_run(failed_run):
    config, output, kwargs = failed_run
    convergence.run(**kwargs, continue_after_epoch1_reproduction_fail=True)
    summary = json.loads((output / "training_summary.json").read_text())
    # Isolate the required selection state on artifacts made by the actual producer.
    summary["best_epoch"] = 2
    summary["extension_gate"]["allowed"] = False
    convergence.write_json(output / "training_summary.json", summary)
    return config, output, kwargs


def test_diagnostic_reads_both_checkpoints_without_training(diagnostic_run, data, monkeypatch):
    config, output, kwargs = diagnostic_run
    before = snapshot(output)
    prepared = inputs(config, data)
    monkeypatch.setattr(convergence, "prepare_run", lambda *a: prepared)
    for owner, name in ((convergence, "fit"), (convergence, "train_group"),
                        (torch.optim, "AdamW"), (torch.Tensor, "backward")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("training called"))
    evaluate = full.evaluate
    loaded = []

    def checked_evaluate(planner, runner, samples, records, track):
        epoch = 2 + len(loaded)
        saved = torch.load(output / f"epoch_{epoch}.pt", weights_only=True)
        assert_state_equal(planner.state_dict(), saved["planner_state_dict"])
        assert not planner.training and not prepared["model"].training
        assert not torch.is_grad_enabled()
        assert not any(p.requires_grad for p in prepared["model"].parameters())
        loaded.append(epoch)
        return evaluate(planner, runner, samples, records, track)

    monkeypatch.setattr(full, "evaluate", checked_evaluate)
    result = convergence.diagnose_gt_action(**kwargs)
    assert loaded == [2, 3]
    assert result["status"] == "gt_action_epoch2_epoch3_diagnostic_completed"
    assert all((output / name).read_bytes() == content for name, content in before.items())
    assert len(snapshot(output)) == len(before) + 5
    assert {c[:2] for c in prepared["runner"].calls} == {("validation", "gt_action_diagnostic")}
    assert len(prepared["runner"].calls) == 2 * config.expected_validation_count
    for epoch in (2, 3):
        metrics = result[f"epoch_{epoch}"]["gt_action_diagnostic"]
        assert metrics["sample_count"] == metrics["valid_prediction_count"] == 4
        assert metrics["invalid_prediction_count"] == 0
        assert result["motion_unavailable"][f"epoch_{epoch}"]["gt_action_diagnostic"]["sample_count"] == 1
    for key in ("test_records_read", "test_images_opened", "test_labels_read"):
        assert result["test_isolation"][key] == 0
    assert result["test_isolation"]["test_evaluation_performed"] is False
    assert json.loads((output / "gt_action_diagnostic_comparison.json").read_text()) == result
    after = snapshot(output)
    with pytest.raises(FileExistsError):
        convergence.diagnose_gt_action(**kwargs)
    assert snapshot(output) == after


@pytest.mark.parametrize("failure", ["split", "existing", "step", "config", "provenance", "tokens", "target", "coverage"])
def test_diagnostic_intake_rejects_without_writes(diagnostic_run, data, monkeypatch, failure):
    config, output, kwargs = diagnostic_run
    prepared = inputs(config, data)
    monkeypatch.setattr(convergence, "prepare_run", lambda *a: prepared)
    if failure == "split":
        kwargs["validation_split"] = "test"
        monkeypatch.setattr(convergence, "prepare_run", lambda *a: pytest.fail("data/model accessed"))
    elif failure == "existing":
        (output / "epoch_3_gt_action_predictions.jsonl").write_text("partial")
        monkeypatch.setattr(convergence, "prepare_run", lambda *a: pytest.fail("data/model accessed"))
    elif failure in ("step", "config", "provenance"):
        path = output / "epoch_3.pt"
        saved = torch.load(path, weights_only=True)
        if failure == "step":
            saved["optimizer_step"] += 1
        elif failure == "config":
            saved["training_config"]["seed"] += 1
        else:
            saved["provenance"]["data"]["validation"]["eligible_records"] += 1
        torch.save(saved, path)
    elif failure in ("tokens", "target"):
        path = output / "epoch_3_predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if failure == "tokens":
            rows[0]["sample_token"] = "unknown"
        else:
            rows[0]["target_waypoints"][0][0] += 1
        full.write_predictions(path, rows)
    else:
        prepared["validation"] = prepared["validation"][:-1]
    monkeypatch.setattr(full, "evaluate", lambda *a: pytest.fail("evaluation called"))
    before = snapshot(output)
    with pytest.raises((ValueError, FileExistsError)):
        convergence.diagnose_gt_action(**kwargs)
    assert snapshot(output) == before


@pytest.mark.parametrize("option", ["--continue-after-epoch1-reproduction-fail", "--extend-to-five"])
def test_diagnostic_cli_excludes_training_modes(option):
    with pytest.raises(SystemExit):
        cli.main(["--diagnose-gt-action-epoch2-3", option, "--dry-run"])


def test_diagnostic_cli_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run", lambda **k: pytest.fail("training dispatch"))
    monkeypatch.setattr(cli, "collect_git_provenance", lambda *a: None)
    calls = []
    monkeypatch.setattr(cli, "diagnose_gt_action", lambda **k: calls.append(k) or {})
    assert cli.main(["--diagnose-gt-action-epoch2-3", "--dataset-root", str(tmp_path),
                     "--derived-root", str(tmp_path)]) == 0
    assert len(calls) == 1 and calls[0]["validation_split"] == "validation"
    with pytest.raises(SystemExit):
        cli.main(["--diagnose-gt-action-epoch2-3", "--validation-split", "test"])


@pytest.mark.parametrize("gt_factor,case", [(0.9, "case_a"), (1.005, "case_a"),
                                           (1.1, "case_b"), (None, "inconclusive")])
def test_diagnostic_comparison_deltas_and_interpretation(gt_factor, case):
    metrics = dict.fromkeys(convergence.DIAGNOSTIC_METRICS, 2.)
    second = {"predicted_action": metrics, "gt_action_diagnostic": metrics}
    gt = dict.fromkeys(convergence.DIAGNOSTIC_METRICS, 2 * (gt_factor or 1.1))
    if gt_factor is None:
        gt["fde_3s_m"] = 1.8
    third = {"predicted_action": dict.fromkeys(convergence.DIAGNOSTIC_METRICS, 3.),
             "gt_action_diagnostic": gt}
    result = convergence.diagnostic_comparison(second, third)
    assert result["diagnostic_interpretation"]["diagnostic"] == case
    assert not result["diagnostic_interpretation"]["threshold_is_experiment_gate"]
    changes = result["epoch2_to_epoch3"]
    for key in convergence.DIAGNOSTIC_METRICS:
        assert changes["predicted_action_delta"][key] == 1.
        assert changes["predicted_action_relative_change"][key] == .5
        assert changes["gt_action_delta"][key] == gt[key] - 2.
        assert result["conditioning_gap"]["epoch_3_predicted_minus_gt"][key] == 3. - gt[key]


def test_diagnostic_invalid_gt_output_does_not_complete(diagnostic_run, monkeypatch):
    _, output, kwargs = diagnostic_run
    evaluate = full.evaluate

    def invalid_output(*args):
        _, rows = evaluate(*args)
        rows[0].update(prediction_valid=False, invalid_reason="nonfinite_trajectory", predicted_waypoints=None)
        return aggregate_metrics(rows, "gt_action_diagnostic"), rows

    monkeypatch.setattr(full, "evaluate", invalid_output)
    before = snapshot(output)
    with pytest.raises(ValueError, match="GT-action validation coverage"):
        convergence.diagnose_gt_action(**kwargs)
    assert snapshot(output) == before
