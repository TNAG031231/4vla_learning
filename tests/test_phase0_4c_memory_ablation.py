from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_tiny_overfit as cli
from src.phase0 import phase0_4c_memory_ablation as diagnostic
from src.phase0 import phase0_4c_tiny_overfit as training
from src.phase0.phase0_4c_two_turn_planner import PlannerConfig, WaypointDecoder
from test_phase0_4c_collapse_diagnostic import setup
from test_phase0_4c_tiny_overfit import config


def test_only_memory_layernorm_changes_architecture_and_initial_weights(config):
    ablation = training.load_config(ROOT / "configs/phase0_4c_tiny_overfit_v0_4a.yaml")
    assert ablation == replace(config, memory_normalization=True,
                              output_relative_dir="phase_0_4/two_turn_planner_tiny_overfit_v0_4a")
    torch.manual_seed(config.seed)
    baseline = WaypointDecoder(2560, config)
    torch.manual_seed(config.seed)
    planner = WaypointDecoder(2560, ablation)
    assert sum(p.numel() for p in planner.parameters() if p.requires_grad) == 2_765_058
    assert isinstance(planner.memory_norm, nn.LayerNorm) and planner.memory_norm.normalized_shape == (256,)
    assert all(p.requires_grad for p in planner.memory_norm.parameters())
    assert set(planner.state_dict()) - set(baseline.state_dict()) == {"memory_norm.weight", "memory_norm.bias"}
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, planner.state_dict()[key])
    assert all(not layer.norm_first and layer.self_attn.num_heads == 4 for layer in planner.decoder.layers)
    legacy = asdict(config)
    legacy.pop("memory_normalization")
    assert training.TinyOverfitConfig(**legacy) == config
    baseline.load_state_dict({k: v for k, v in planner.state_dict().items() if not k.startswith("memory_norm.")})


def test_feature_normalization_padding_and_diagnostic_rng_invariance(setup):
    _, contexts, config = setup
    planner = WaypointDecoder(16, replace(config, memory_normalization=True)).eval()
    c = contexts[0]
    captured = {}

    def capture(module, args, kwargs):
        captured.update(memory=args[1], mask=kwargs["memory_key_padding_mask"])

    handle = planner.decoder.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.no_grad():
        prediction = planner(c.hidden_states, c.attention_mask)
        assert prediction.shape == (1, 6, 2) and torch.isfinite(prediction).all()
        torch.testing.assert_close(captured["memory"], planner.memory_norm(planner.context_projection(c.hidden_states.float())))
        assert torch.equal(captured["mask"], ~c.attention_mask.bool())
        changed = c.hidden_states.clone()
        changed[:, -1] = 10000
        torch.testing.assert_close(planner(changed, c.attention_mask), prediction)
    handle.remove()
    _, predictions = training.evaluate(planner, contexts, beta=config.smooth_l1_beta, device="cpu")
    rng = torch.get_rng_state().clone()
    report = diagnostic.collect_diagnostics(planner, contexts, predictions, device="cpu")
    assert torch.equal(rng, torch.get_rng_state())
    assert report["planner_state_unchanged"] and not report["norm_first"]
    assert report["decoder_memory_after_layernorm"]["pooled_feature_rms"] == pytest.approx(1, abs=1e-3)
    assert len(report["raw_projected_memory"]) == 8
    assert "cross_attention_residual_before_norm" in report["layer2_stages"]
    assert report["qkv"]["interpretation"]["replay_fidelity_ok"]
    assert len(report["cross_query_attention"]["per_head"]) == 2
    assert all(p.grad is None for p in planner.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in planner.modules())
    json.dumps(report, allow_nan=False)


def test_diversity_distinguishes_point_collapse_from_temporal_and_sample_changes():
    rows = [{"sample_token": str(i), "predicted_waypoints": [[9.804, -.085]] * 6} for i in range(8)]
    collapse = diagnostic.trajectory_diversity(rows)
    assert all(s["temporal_path_length_m"] == 0 for s in collapse["per_sample"])
    assert collapse["across_samples_trajectory_l2"]["off_diagonal_l2"]["mean"] == 0
    for i, row in enumerate(rows):
        row["predicted_waypoints"] = [[float(t + i), 0.] for t in range(6)]
    spread = diagnostic.trajectory_diversity(rows)
    assert all(s["temporal_path_length_m"] == 5 and s["backward_longitudinal_steps"] == 0 for s in spread["per_sample"])
    assert spread["across_samples_trajectory_l2"]["off_diagonal_l2"]["mean"] > 0


def test_ablation_fit_uses_existing_training_gate_and_reload(setup, tmp_path):
    _, contexts, config = setup
    config = replace(config, memory_normalization=True, max_optimizer_steps=2)
    prior = tmp_path / "v0_3_reference.json"
    prior.write_text("preserved")
    content, mtime = prior.read_bytes(), prior.stat().st_mtime_ns
    result = training.fit_cached_contexts(contexts=contexts, config=config, hidden_size=16,
        device="cpu", output=tmp_path, provenance={"run_kind": "synthetic_cpu_test"})
    assert result["reload_consistency"] and result["planner_parameters_updated"]
    assert result["memory_norm_trainable"] and result["memory_norm_parameters"] == 16
    assert result["optimizer_steps"] == 2 and result["samples_consumed"] == 16
    assert (result["tiny_overfit_gate"] == "PASS") == all(result["gate_conditions"].values())
    for name in ("diagnostics_initial.json", "diagnostics_trained.json"):
        report = json.loads((tmp_path / name).read_text())
        assert report["planner_state_unchanged"] and report["diagnostic_optimizer_steps"] == 0
    saved = torch.load(tmp_path / "planner_state.pt", weights_only=True)
    assert saved["planner_config"]["memory_normalization"]
    assert "memory_norm.weight" in saved["planner_state_dict"]
    fresh = WaypointDecoder(16, PlannerConfig(**saved["planner_config"]))
    fresh.load_state_dict(saved["planner_state_dict"])
    assert prior.read_bytes() == content and prior.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("change", ["learning_rate", "max_optimizer_steps", "dropout", "output_relative_dir"])
def test_ablation_rejects_second_variable_before_data_access(config, tmp_path, monkeypatch, change):
    config = replace(config, memory_normalization=True,
                     output_relative_dir="phase_0_4/two_turn_planner_tiny_overfit_v0_4a")
    value = "wrong" if change == "output_relative_dir" else getattr(config, change) * 2
    config = replace(config, **{change: value})
    monkeypatch.setattr(cli, "load_temporal_records", lambda *a, **kw: pytest.fail("data accessed"))
    with pytest.raises(ValueError, match="only memory normalization"):
        cli.run(dataset_root=tmp_path, derived_root=tmp_path, config=config)


@pytest.mark.parametrize("mismatch", ["configuration", "sample identities/order"])
def test_ablation_reference_intake_before_model_access(setup, tmp_path, monkeypatch, mismatch):
    _, contexts, baseline = setup
    baseline = replace(baseline, max_optimizer_steps=1)
    source = tmp_path / baseline.output_relative_dir
    source.mkdir(parents=True)
    training.fit_cached_contexts(contexts=contexts, config=baseline, hidden_size=16,
        device="cpu", output=source, provenance={"run_kind": "synthetic_cpu_test"})
    checkpoint = source / "planner_state.pt"
    saved = torch.load(checkpoint, weights_only=True)
    if mismatch == "configuration":
        saved["training_config"]["learning_rate"] *= 2
    else:
        saved["tiny_sample_tokens"].reverse()
    torch.save(saved, checkpoint)
    snapshot = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()}
    monkeypatch.setattr(cli, "load_config", lambda path: baseline)
    monkeypatch.setattr(cli, "load_temporal_records", lambda *a, **kw: [])
    monkeypatch.setattr(cli, "select_training_samples", lambda *a: [
        SimpleNamespace(sample_token=c.sample_token) for c in contexts])
    monkeypatch.setattr(cli, "default_runtime_dependencies", lambda: pytest.fail("model accessed"))
    config = replace(baseline, memory_normalization=True,
                     output_relative_dir="phase_0_4/two_turn_planner_tiny_overfit_v0_4a")
    with pytest.raises(ValueError, match=mismatch):
        cli.run(dataset_root=tmp_path, derived_root=tmp_path, config=config)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in snapshot.items())
