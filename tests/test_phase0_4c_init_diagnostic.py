from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0 import phase0_4c_init_diagnostic as diagnostic
from src.phase0.phase0_4c_tiny_overfit import evaluate
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder
from test_phase0_4c_collapse_diagnostic import setup


def reference_predictions(contexts, config, path):
    torch.manual_seed(config.seed)
    initial = WaypointDecoder(16, config)
    _, predictions = evaluate(initial, contexts, beta=config.smooth_l1_beta, device="cpu")
    path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    return initial, predictions


def test_reconstruction_same_contexts_no_updates_and_summary_math(setup, tmp_path, monkeypatch):
    _, contexts, config = setup
    path = tmp_path / "predictions_before.jsonl"
    initial, _ = reference_predictions(contexts, config, path)
    trained = copy.deepcopy(initial)
    with torch.no_grad():
        trained.context_projection.weight.mul_(2)
        trained.decoder.layers[1].multihead_attn.in_proj_weight[8:16].add_(.1)
    snapshot = {k: v.clone() for k, v in trained.state_dict().items()}
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **kw: pytest.fail("backward called"))
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *a, **kw: pytest.fail("optimizer created"))
    monkeypatch.setattr(torch, "save", lambda *a, **kw: pytest.fail("checkpoint written"))
    monkeypatch.setattr(WaypointDecoder, "load_state_dict", lambda *a, **kw: pytest.fail("initial checkpoint loaded"))
    result = diagnostic.diagnose_initial_vs_trained(trained, contexts, config=config, hidden_size=16,
                                                    recorded_path=path, device="cpu")
    assert result["initialization_fidelity"]["max_absolute_difference"] == 0
    assert result["initialization_fidelity"]["mean_absolute_difference"] == 0
    assert result["comparison_performed"] and result["planner_state_unchanged"]
    assert result["parameter_changes"]["context_projection.weight"]["relative_delta_l2"] == pytest.approx(1)
    assert result["parameter_changes"]["decoder.layers.1.multihead_attn.Q.weight"]["delta_l2"] == 0
    assert result["parameter_changes"]["decoder.layers.1.multihead_attn.K.weight"]["delta_l2"] == pytest.approx(.8)
    assert result["parameter_changes"]["decoder.layers.1.norm2.bias"]["relative_delta_l2"] is None
    assert sum(v.numel() for v in diagnostic.parameter_groups(initial).values()) == sum(p.numel() for p in initial.parameters())
    for label, planner in (("initial", initial), ("trained", trained)):
        state = result[label]
        with torch.no_grad():
            values = torch.cat([planner.context_projection(c.hidden_states.float())[0, c.attention_mask[0].bool()]
                                for c in contexts]).double()
        assert state["memory"]["pooled_feature_rms"] == pytest.approx(float(values.square().mean().sqrt()))
        assert state["memory"]["pooled_valid_token_norms"]["median"] == pytest.approx(float(values.norm(dim=-1).quantile(.5)))
        for row in state["collapse_progression"]["per_sample"]:
            assert row["stages"]["layer1_output"] == row["stages"]["layer2_input"]
        assert len(state["qkv"]["per_sample"]) == 8
        assert all(h["token_norms"]["k"]["max_over_median"] >= 1
                   for s in state["qkv"]["per_sample"] for h in s["heads"])
    assert all(torch.equal(v, snapshot[k]) for k, v in trained.state_dict().items())
    assert all(p.grad is None for p in trained.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in trained.modules())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failure", ["missing", "order", "prediction", "shape", "nonfinite", "ragged", "malformed"])
def test_failed_reconstruction_stops_before_comparison(setup, tmp_path, monkeypatch, failure):
    trained, contexts, config = setup
    path = tmp_path / "predictions_before.jsonl"
    _, rows = reference_predictions(contexts, config, path)
    if failure == "missing":
        path = tmp_path / "absent.jsonl"
    else:
        if failure == "order":
            rows.reverse()
        elif failure == "prediction":
            rows[0]["predicted_waypoints"][0][0] += 1
        elif failure == "shape":
            for row in rows:
                row["predicted_waypoints"].pop()
        elif failure == "ragged":
            rows[0]["predicted_waypoints"].pop()
        else:
            rows[0]["predicted_waypoints"][0][0] = float("nan")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if failure == "malformed":
            path.write_text("{")
    monkeypatch.setattr(diagnostic, "state_diagnostic", lambda *a, **kw: pytest.fail("comparison started"))
    result = diagnostic.diagnose_initial_vs_trained(trained, contexts, config=config, hidden_size=16,
                                                    recorded_path=path, device="cpu")
    assert result["interpretation"]["code"] == "D" and not result["comparison_performed"]
    assert "initial" not in result and "parameter_changes" not in result
    if failure == "prediction":
        assert result["initialization_fidelity"]["max_absolute_difference"] == pytest.approx(1)
        assert result["initialization_fidelity"]["mean_absolute_difference"] == pytest.approx(1 / 96)


@pytest.mark.parametrize("case", ["A", "B", "C", "E", "bad_replay", "undefined_contraction"])
def test_interpretation_requires_bounded_evidence(setup, case):
    planner, contexts, _ = setup
    first = diagnostic.state_diagnostic(planner, contexts, device="cpu")
    last = copy.deepcopy(first)
    for state, pathological in ((first, case != "B"), (last, True)):
        qkv = state["qkv"]
        qkv["interpretation"]["evidence_sample_counts"] = {k: 8 if pathological and k in "CDE" else 0 for k in "ABCDEF"}
        qkv["interpretation"]["replay_fidelity_ok"] = case != "bad_replay"
        qkv["logit_aggregate_over_samples_heads_queries"]["top1_top2_margin"]["median"] = 100.
        for sample in qkv["per_sample"]:
            sample["norm2"]["contraction_ratios"]["pure_normalization"] = .05
    if case == "C":
        last["qkv"]["logit_aggregate_over_samples_heads_queries"]["top1_top2_margin"]["median"] = 300.
        for sample in last["qkv"]["per_sample"]:
            sample["norm2"]["contraction_ratios"]["pure_normalization"] = .01
    if case == "E":
        first["qkv"]["interpretation"]["evidence_sample_counts"]["C"] = 1
        first["qkv"]["interpretation"]["evidence_sample_counts"]["D"] = 0
        first["qkv"]["interpretation"]["evidence_sample_counts"]["E"] = 0
    if case == "undefined_contraction":
        first["qkv"]["per_sample"][0]["norm2"]["contraction_ratios"]["pure_normalization"] = None
    result = diagnostic.interpret_comparison(first, last, diagnostic.parameter_changes(planner, planner))
    assert result["code"] == ("E" if case in ("bad_replay", "undefined_contraction") else case)
