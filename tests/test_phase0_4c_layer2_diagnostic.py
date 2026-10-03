from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0 import phase0_4c_layer2_diagnostic as diagnostic
from src.phase0.phase0_4c_collapse_diagnostic import stage_statistics
from test_phase0_4c_collapse_diagnostic import setup


def test_attention_statistics_with_padding_and_known_distributions():
    weights = torch.tensor([[.5, .5, 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.]] * 2)
    result = diagnostic.attention_statistics(weights, torch.tensor([1, 1, 1, 0]), top_k=1)
    assert result["entropy_nats_per_query"] == pytest.approx([math.log(2), 0., 0.] * 2)
    assert result["normalized_entropy_per_query"][0] == pytest.approx(math.log(2) / math.log(3))
    assert result["top_token_indices"][1] == [0]
    assert result["top_token_indices"][2] == [2]
    assert all(3 not in indices for indices in result["top_token_indices"])
    assert result["pairwise_cosine"][1][2] == 0
    assert result["pairwise_cosine"][1][4] == 1
    assert result["pairwise_l1"][1][2] == 2
    assert result["pairwise_js_nats"][1][2] == pytest.approx(math.log(2))
    assert result["pairwise_js_nats"][1][4] == 0
    assert result["top_k_overlap_fraction"][1][4] == 1
    assert result["top_k_overlap_fraction"][1][2] == 0
    json.dumps(result, allow_nan=False)


def test_attention_single_valid_token_and_cross_sample_summary_comparison():
    first = diagnostic.attention_statistics(torch.tensor([[0., 1.]] * 6), torch.tensor([0, 1]))
    assert first["top_token_indices"] == [[1]] * 6
    assert first["normalized_entropy_per_query"] == [0.] * 6
    second = diagnostic.attention_statistics(torch.full((6, 4), .25), torch.ones(4), top_k=1)
    comparison = diagnostic.compare_sample_attention([first, second])
    assert comparison["mean_absolute_delta_normalized_entropy_per_query"] == [[0., 1.], [1., 0.]]
    assert comparison["mean_absolute_delta_top_k_mass_per_query"] == [[0., .75], [.75, 0.]]
    aggregate = diagnostic.aggregate_attention([first, second])
    assert aggregate["normalized_entropy_per_query"] == [.5] * 6
    assert aggregate["mean_pairwise_l1"] == 0


@pytest.mark.parametrize("norm_first", [False, True])
def test_stage_order_values_and_attention_replay_preserve_forward(setup, norm_first, monkeypatch):
    planner, contexts, _ = setup
    layer = planner.decoder.layers[1]
    layer.norm_first = norm_first
    planner.eval()
    before = {name: value.clone() for name, value in planner.state_dict().items()}
    expected_stages = []
    expected_predictions = []
    with torch.no_grad():
        for context in contexts:
            memory = planner.context_projection(context.hidden_states.float())
            mask = ~context.attention_mask.bool()
            x = planner.decoder.layers[0](planner.waypoint_queries.weight[None], memory,
                                          memory_key_padding_mask=mask)
            stages = {"layer2_input": stage_statistics(x)}
            for name, norm, block in (
                ("self_attention", layer.norm1, lambda x: layer._sa_block(x, None, None)),
                ("cross_attention", layer.norm2, lambda x: layer._mha_block(x, memory, None, mask)),
                ("ffn", layer.norm3, layer._ff_block),
            ):
                if norm_first:
                    normalized = norm(x)
                    stages[f"{name}_pre_norm"] = stage_statistics(normalized)
                    branch = block(normalized)
                    stages[f"{name}_branch_before_residual"] = stage_statistics(branch)
                    x = x + branch
                    if name != "ffn":
                        stages[f"after_{name}_residual"] = stage_statistics(x)
                else:
                    branch = block(x)
                    stages[f"{name}_branch_before_residual"] = stage_statistics(branch)
                    x = x + branch
                    stages[f"{name}_residual_before_norm"] = stage_statistics(x)
                    x = norm(x)
                    stages[f"after_{name}_residual_norm"] = stage_statistics(x)
            stages["layer2_output"] = stage_statistics(x)
            expected_stages.append(stages)
            expected_predictions.append(planner(context.hidden_states, context.attention_mask))
    attention_calls = []
    original = layer.multihead_attn.forward

    def attention_forward(*args, **kwargs):
        attention_calls.append(kwargs["need_weights"])
        return original(*args, **kwargs)

    monkeypatch.setattr(layer.multihead_attn, "forward", attention_forward)
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **kw: pytest.fail("backward called"))
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *a, **kw: pytest.fail("optimizer created"))
    monkeypatch.setattr(torch, "save", lambda *a, **kw: pytest.fail("checkpoint saved"))
    result = diagnostic.diagnose_layer2(planner, contexts, device="cpu")
    assert attention_calls == [False, True] * 8
    assert result["norm_first"] == norm_first
    assert result["planner_state_unchanged"]
    for row, stages, prediction in zip(result["per_sample"], expected_stages, expected_predictions, strict=True):
        assert row["execution_order"] == list(stages)
        for name, expected in stages.items():
            actual = row["stages"][name]
            assert actual["shape"] == expected["shape"]
            assert actual["off_diagonal_l2"] == pytest.approx(expected["off_diagonal_l2"])
            assert actual["off_diagonal_cosine"] == pytest.approx(expected["off_diagonal_cosine"])
        assert row["prediction_summary"] == stage_statistics(prediction)
        assert row["cross_attention"]["replay_output_close"]
        assert len(row["cross_attention"]["per_head"]) == 2
    for name, value in planner.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(p.grad is None for p in planner.parameters())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in planner.modules())
    assert result["attention_aggregate"]["head_mean"]["entropy_nats_per_query"] == pytest.approx(
        torch.tensor([s["cross_attention"]["head_mean"]["entropy_nats_per_query"]
                      for s in result["per_sample"]]).mean(0).tolist())
    json.dumps(result, allow_nan=False)
