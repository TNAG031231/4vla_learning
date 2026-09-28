from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_tiny_overfit as cli
from src.phase0 import phase0_4c_tiny_overfit as training
from src.phase0.phase0_4_temporal_dataset import build_record, collect_history
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder, freeze_backbone
from src.phase0.qwen3vl_dataset_adapter import load_config as load_ego_config
from test_phase0_4b_protocol import case, processor, samples


@pytest.fixture
def config():
    return training.load_config(ROOT / "configs/phase0_4c_tiny_overfit.yaml")


@pytest.fixture
def contexts():
    generator = torch.Generator().manual_seed(42)
    return [training.CachedContext(
        f"synthetic-{index}", torch.randn(1, 4 + index, 16, generator=generator).bfloat16(),
        torch.ones(1, 4 + index, dtype=torch.long),
        torch.tensor([[[0.2 * (step + 1), index * 0.05] for step in range(6)]]),
        torch.tensor([[True, True, True, True, index % 2 == 0, False]]),
    ) for index in range(8)]


def test_metrics_use_euclidean_meters_and_last_valid_point():
    prediction = torch.zeros(1, 6, 2)
    target = torch.tensor([[[3., 4.], [float("nan"), float("nan")], [0., 2.],
                            [6., 8.], [999., 999.], [999., 999.]]])
    mask = torch.tensor([[True, False, True, True, False, False]])
    assert training.trajectory_metrics(prediction, target, mask) == {
        "ade_m": pytest.approx(17 / 3), "fde_m": 10.,
    }


def test_subset_reuses_producer_contract_and_deterministic_selection(case, config):
    reader, source = case
    records = []
    for index in range(12):
        row = build_record(source[index % 4], collect_history(reader, source[index % 4], 3), 3)
        row["sample_token"] = f"synthetic-{index}"
        row["historical_sample_tokens"][-1] = row["sample_token"]
        records.append(row)
    records[-1].update(longitudinal_action=None, longitudinal_action_valid=False,
                       factorized_action_joint_valid=False)
    ego = load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml")
    selected = training.select_training_samples(records, ego, config)
    assert selected == training.select_training_samples(list(reversed(records)), ego, config)
    assert len(selected) == len({s.sample_token for s in selected}) == 8
    assert all(s.target.longitudinal_valid and s.target.lateral_valid for s in selected)
    with pytest.raises(ValueError, match="only permits train"):
        training.select_training_samples([{"split": "validation"}], ego, config)


def test_cache_runs_one_frozen_forward_per_sample_and_keeps_targets_out_of_context(
    samples, processor, tmp_path,
):
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(1, 16)
            self.lora_A = nn.Linear(1, 1)
            self.calls = 0

        def forward(self, **kwargs):
            assert not self.training and not torch.is_grad_enabled()
            assert kwargs["output_hidden_states"] and not kwargs["use_cache"]
            text = processor.tokenizer.decode(kwargs["input_ids"][0])
            assert "987654" not in text and "123456" not in text
            self.calls += 1
            shape = (*kwargs["input_ids"].shape, 16)
            return SimpleNamespace(hidden_states=(torch.zeros(shape), torch.ones(shape)))

    model = Backbone()
    freeze_backbone(model)
    selected = samples[:2]
    rows = {s.sample_token: {"future_waypoints": [[987654., 123456.]] * 6,
                             "trajectory_valid_mask": [True] * 6} for s in selected}
    contexts, evidence = training.cache_contexts(
        model=model, processor=processor, samples=selected, records=rows, dataset_root=tmp_path,
        device="cpu", runtime=SimpleNamespace(image_loader=lambda path: Image.new("RGB", (64, 64))),
    )
    assert model.calls == 2
    for context, item in zip(contexts, evidence, strict=True):
        assert context.hidden_states.device.type == context.attention_mask.device.type == "cpu"
        assert context.hidden_states.dtype == torch.bfloat16
        assert context.hidden_states.eq(1).all() and not context.hidden_states.requires_grad
        assert context.target[0, 0, 0] == 987654
        assert item["action_context_matches"]
        assert processor.tokenizer.decode(item["action_token_ids"]) == item["assistant_action_text"]
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_cpu_tiny_overfit_optimizer_and_fresh_reload(config, contexts, tmp_path, monkeypatch):
    config = replace(config, planner_dimension=16, num_heads=4, num_decoder_layers=1,
                     dropout=0., max_optimizer_steps=60)
    backbone = nn.ModuleDict({"qwen": nn.Linear(16, 16), "lora_A": nn.Linear(16, 2)})
    freeze_backbone(backbone)
    backbone_weights = {name: value.clone() for name, value in backbone.state_dict().items()}
    original_optimizer = torch.optim.AdamW
    optimizers = []

    def optimizer(parameters, **kwargs):
        parameters = list(parameters)
        assert not {id(p) for p in parameters} & {id(p) for p in backbone.parameters()}
        assert all(p.requires_grad for p in parameters)
        optimizers.append(original_optimizer(parameters, **kwargs))
        return optimizers[-1]

    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = training.fit_cached_contexts(
            contexts=contexts, config=config, hidden_size=16, device="cpu", output=tmp_path,
            provenance={"run_kind": "synthetic_cpu_test"},
        )
    finally:
        torch.set_num_threads(threads)
    assert result["status"] == "tiny_overfit_passed"
    assert result["planner_parameters_updated"] and result["reload_consistency"]
    assert result["final_metrics"]["loss"] <= .30 * result["initial_metrics"]["loss"]
    checkpoint = torch.load(tmp_path / "planner_state.pt", weights_only=True)
    assert set(checkpoint) == {"planner_state_dict", "planner_config", "training_config", "hidden_size",
                               "tiny_sample_tokens", "provenance"}
    assert checkpoint["tiny_sample_tokens"] == [c.sample_token for c in contexts]
    assert len(optimizers) == 1
    assert sum(p.numel() for g in optimizers[0].param_groups for p in g["params"]) == sum(
        p.numel() for p in checkpoint["planner_state_dict"].values())
    assert all(torch.equal(value, backbone_weights[name]) for name, value in backbone.state_dict().items())
    history = [json.loads(line) for line in (tmp_path / "training_history.jsonl").read_text().splitlines()]
    assert len(history) == config.max_optimizer_steps
    assert [row["optimizer_step"] for row in history] == list(range(1, config.max_optimizer_steps + 1))
    assert [row["samples_consumed"] for row in history] == [8 * i for i in range(1, 61)]
    assert result["samples_consumed"] == 480
    assert result["samples_per_optimizer_step"] == 8
    saved = json.loads((tmp_path / "training_summary.json").read_text())
    assert saved == result
    assert result["learning_gate_passed"]
    assert result["loss_ratio"] == pytest.approx(result["final_loss"] / result["initial_loss"])
    for label, metric in (("loss", "loss"), ("ADE", "ade_m"), ("FDE", "fde_m")):
        for stage in ("initial", "final"):
            assert result[f"{stage}_{label}"] == result[f"{stage}_metrics"][metric]
    predictions = [json.loads(line) for line in (tmp_path / "predictions_after.jsonl").read_text().splitlines()]
    for metric in ("loss", "ade_m", "fde_m"):
        assert result["final_metrics"][metric] == pytest.approx(sum(r[metric] for r in predictions) / 8)
    diagnostics = json.loads((tmp_path / "per_sample_metrics.json").read_text())
    before = [json.loads(line) for line in (tmp_path / "predictions_before.jsonl").read_text().splitlines()]
    assert [row["sample_token"] for row in diagnostics] == [c.sample_token for c in contexts]
    for label, metric in (("loss", "loss"), ("ADE", "ade_m"), ("FDE", "fde_m")):
        for row, initial, final in zip(diagnostics, before, predictions, strict=True):
            assert row[f"initial_{label}"] == initial[metric]
            assert row[f"final_{label}"] == final[metric]
        assert result[f"per_sample_improved_{label}_count"] == sum(
            row[f"final_{label}"] < row[f"initial_{label}"] for row in diagnostics)


@pytest.mark.parametrize("failure", ["learning", "reload"])
def test_failed_gate_saves_results(config, contexts, tmp_path, monkeypatch, failure):
    config = replace(config, planner_dimension=16, num_decoder_layers=1, max_optimizer_steps=1)
    if failure == "learning":
        monkeypatch.setattr(training, "train_planner", lambda *args, **kwargs: None)
    else:
        monkeypatch.setattr(training, "compare_reload", lambda *args: {"reload_consistency": False})
    result = training.fit_cached_contexts(
        contexts=contexts, config=config, hidden_size=16, device="cpu", output=tmp_path,
        provenance={"run_kind": "synthetic_cpu_test"},
    )
    assert result["status"] == "tiny_overfit_learning_gate_failed"
    assert (tmp_path / "planner_state.pt").exists()
    assert json.loads((tmp_path / "training_summary.json").read_text()) == result


def test_reload_comparison_checks_tokens_predictions_and_metrics():
    metrics = {"loss": 1., "ade_m": 2., "fde_m": 3.}
    predictions = [{"sample_token": "a", "predicted_waypoints": [[1., 2.]] * 6, **metrics}]
    assert training.compare_reload((metrics, predictions), (metrics, predictions))["reload_consistency"]
    for changed in ([{**predictions[0], "sample_token": "b"}],
                    [{**predictions[0], "predicted_waypoints": [[2., 3.]] * 6}],
                    [{**predictions[0], "fde_m": 7.}]):
        assert not training.compare_reload((metrics, predictions), (metrics, changed))["reload_consistency"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_cli_and_runner_reject_split_before_access(split, config, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_temporal_records", lambda *a, **kw: pytest.fail("dataset accessed"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--split", split, "--dry-run"])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="only permits train"):
        cli.run(dataset_root=tmp_path, derived_root=tmp_path, config=config, split=split)


def test_output_boundary_and_rerun_guard_before_access(config, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_temporal_records", lambda *a, **kw: pytest.fail("dataset accessed"))
    with pytest.raises(ValueError, match="outside repository"):
        cli.run(dataset_root=tmp_path, derived_root=ROOT, config=config)
    output = tmp_path / config.output_relative_dir
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.run(dataset_root=tmp_path, derived_root=tmp_path, config=config)


def test_cli_help_dry_run_and_exit_code(capsys, tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert cli.main(["--dry-run"]) == 0
    assert "dry_run_no_data_or_model_access" in capsys.readouterr().out
    monkeypatch.setattr(cli, "run", lambda **kwargs: {"status": "tiny_overfit_learning_gate_failed"})
    assert cli.main(["--dataset-root", str(tmp_path), "--derived-root", str(tmp_path)]) == 1


def test_config_reuses_frozen_architecture(config):
    assert config.train_subset_size == 8 and config.max_optimizer_steps == 200
    assert config.log_every_steps == 10
    assert config.output_relative_dir == "phase_0_4/two_turn_planner_tiny_overfit_v0_3"
    assert config.seed == 20260812
    assert (config.planner_dimension, config.num_decoder_layers, config.num_heads,
            config.num_waypoint_queries, config.dropout, config.smooth_l1_beta) == (256, 2, 4, 6, .1, 1.)
    assert (config.optimizer, config.learning_rate, config.weight_decay) == ("AdamW", .001, .0001)
    assert sum(p.numel() for p in WaypointDecoder(2560, config).parameters()) == 2_764_546


def test_full_subset_mean_gradient_and_1600_exposures(config, contexts, tmp_path, monkeypatch):
    class ScalarPlanner(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.25))
            self.lengths = []

        def forward(self, hidden, mask):
            self.lengths.append(hidden.shape[1])
            return self.value.expand(1, 6, 2)

    planner = ScalarPlanner()
    steps, means = [], []
    original_optimizer = torch.optim.AdamW

    class CheckedAdamW(original_optimizer):
        def step(self, closure=None):
            assert planner.lengths == [c.hidden_states.shape[1] for c in contexts]
            reference = planner.value.detach().clone().requires_grad_()
            losses = [training.masked_waypoint_loss(reference.expand(1, 6, 2), c.target,
                                                   c.valid_mask, beta=config.smooth_l1_beta)
                      for c in contexts]
            mean = torch.stack(losses).mean()
            expected_gradient, = torch.autograd.grad(mean, reference)
            torch.testing.assert_close(planner.value.grad, expected_gradient)
            means.append(float(mean.detach()))
            steps.append(len(planner.lengths))
            planner.lengths.clear()
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "AdamW", CheckedAdamW)
    history_path = tmp_path / "history.jsonl"
    training.train_planner(planner, contexts, config=config, device="cpu", history_path=history_path)
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(steps) == 200 and sum(steps) == 1600
    assert not planner.lengths
    assert history[-1]["optimizer_step"] == 200 and history[-1]["samples_consumed"] == 1600
    for index, row in enumerate(history):
        assert row["mean_step_loss"] == pytest.approx(means[index])
        assert row["running_mean_loss"] == pytest.approx(sum(means[:index + 1]) / (index + 1))


def test_prior_artifacts_untouched(config, contexts, tmp_path):
    previous = []
    for version in ("v0_1", "v0_2"):
        path = tmp_path / "phase_0_4" / f"two_turn_planner_tiny_overfit_{version}" / "training_summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(f"preserved {version}")
        previous.append((path, path.read_bytes(), path.stat().st_mtime_ns))
    output = tmp_path / config.output_relative_dir
    output.mkdir()
    training.fit_cached_contexts(
        contexts=contexts, config=replace(config, planner_dimension=16, num_decoder_layers=1,
                                          max_optimizer_steps=1),
        hidden_size=16, device="cpu", output=output, provenance={"run_kind": "synthetic_cpu_test"},
    )
    assert (output / "training_summary.json").exists()
    for path, content, modified_at in previous:
        assert path.read_bytes() == content and path.stat().st_mtime_ns == modified_at
