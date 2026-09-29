from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0 import phase0_4c_qkv_diagnostic as diagnostic
from src.phase0.phase0_4c_collapse_diagnostic import stage_statistics
from test_phase0_4c_collapse_diagnostic import setup


def test_mask_accounting_and_inversion_error(setup):
    _, contexts, _ = setup
    context = contexts[0]
    result = diagnostic.mask_accounting(context, ~context.attention_mask.bool())
    assert result["hidden_sequence_length"] == 3
    assert result["attention_mask_shape"] == result["memory_key_padding_mask_shape"] == [1, 3]
    assert result["valid_tokens"] == result["visible_tokens"] == 2
    assert result["invalid_tokens"] == result["masked_tokens"] == 1
    assert result["valid_fraction"] == pytest.approx(2 / 3)
    assert result["equals_inverted_attention_mask"]
    wrong = diagnostic.mask_accounting(context, context.attention_mask.bool())
    assert not wrong["equals_inverted_attention_mask"] and wrong["visible_tokens"] == 1


def test_top_two_logits_exclude_masked_tokens_and_measure_saturation():
    values = torch.tensor([[10., 9., 1e8], [100., 0., 1e8]])
    result = diagnostic.logit_statistics(values, torch.tensor([True, True, False]))
    assert result[0]["top_token_indices"] == [0, 1]
    assert result[0]["top1_top2_margin"] == 1
    assert result[0]["mean"] == 9.5 and result[0]["std"] == .5
    assert result[0]["top1_probability"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert result[0]["top2_probability"] == pytest.approx(1 / (1 + math.exp(1)))
    assert result[1]["top1_probability"] == 1
    assert result[1]["entropy_nats"] == pytest.approx(0, abs=1e-30)
    assert result[1]["effective_token_count"] == 1
    assert result[1]["top1_probability_lower_bound_from_margin"] == 1
    single = diagnostic.logit_statistics(values, torch.tensor([True, False, False]))
    assert single[0]["second_largest"] is None and single[0]["top1_top2_margin"] is None


def test_layernorm_separates_geometry_and_affine():
    torch.manual_seed(1)
    z = torch.randn(1, 6, 8) + torch.arange(8) * 100
    norm = nn.LayerNorm(8)
    with torch.no_grad():
        norm.weight.fill_(.01)
        norm.bias.fill_(.5)
        report = diagnostic.layernorm_decomposition(z, norm, norm(z))
        reference = F.layer_norm(z, norm.normalized_shape, eps=norm.eps)
    assert report["reconstruction_close"]
    assert report["pre_affine"]["off_diagonal_l2"]["mean"] == pytest.approx(
        stage_statistics(reference)["off_diagonal_l2"]["mean"], rel=1e-4)
    assert report["post_affine"]["off_diagonal_l2"]["mean"] == pytest.approx(
        .01 * report["pre_affine"]["off_diagonal_l2"]["mean"], rel=1e-3)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_exact_projection_replay_matches_pytorch(batch_first, separate, bias):
    torch.manual_seed(2)
    module = nn.MultiheadAttention(8, 2, batch_first=batch_first, bias=bias,
                                   kdim=12 if separate else 8, vdim=10 if separate else 8).eval()
    q = torch.randn(2, 6, 8)
    k = torch.randn(2, 9, 12 if separate else 8)
    v = torch.randn(2, 9, 10) if separate else k
    if not batch_first:
        q = q.transpose(0, 1)
        if separate:
            k, v = k.transpose(0, 1), v.transpose(0, 1)
        else:
            k = v = k.transpose(0, 1)
    kwargs = {"key_padding_mask": torch.tensor([[False] * 8 + [True], [False] * 7 + [True] * 2]),
              "need_weights": False, "attn_mask": None, "is_causal": False}
    with torch.no_grad():
        real, _ = module(q, k, v, **kwargs)
        replay = diagnostic.replay_attention(module, (q, k, v), kwargs)
        _, weights = module(q, k, v, **(kwargs | {"need_weights": True, "average_attn_weights": False}))
    torch.testing.assert_close(replay["explicit_output"], real, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(replay["sdpa_output"], real, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(replay["probabilities"].view(2, 2, 6, 9), weights)
    assert replay["probabilities"][0, :, -1].eq(0).all()
    assert replay["scale"] == .5


class Tokenizer:
    all_special_ids = [9]

    def __call__(self, text, **kwargs):
        return {"input_ids": [3, 4]}

    def convert_ids_to_tokens(self, token_id):
        return "<|image_pad|>" if token_id == 9 else str(token_id)

    def decode(self, ids):
        return " ".join(map(str, ids))


def test_exact_token_mapping_and_unresolved_categories():
    tokenizer = Tokenizer()
    ids = [9, 3, 4, 5, 6, 7, 8]
    context = diagnostic.token_context(ids, tokenizer, ("ego",), [5, 6])
    valid = torch.ones(len(ids), dtype=torch.bool)
    assert diagnostic.token_position(0, valid, context, tokenizer)["category"] == "vision_token"
    assert diagnostic.token_position(1, valid, context, tokenizer)["category"] == "ego_history_text"
    assert diagnostic.token_position(3, valid, context, tokenizer)["category"] == "assistant_structured_action"
    last = diagnostic.token_position(6, valid, context, tokenizer)
    assert last["category"] == "text_context_unresolved"
    assert last["distance_from_final_valid_token"] == 0 and last["relative_valid_position"] == 1
    repeated = diagnostic.token_context(ids + [5, 6], tokenizer, (), [5, 6])
    assert repeated["spans"] == []
    assert repeated["span_match_counts"][0]["exact_match_count"] == 2


def test_qkv_probe_no_update_and_fidelity_gate(setup, monkeypatch):
    planner, contexts, _ = setup
    before = {name: value.clone() for name, value in planner.state_dict().items()}
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **kw: pytest.fail("backward called"))
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *a, **kw: pytest.fail("optimizer created"))
    monkeypatch.setattr(torch, "save", lambda *a, **kw: pytest.fail("checkpoint saved"))
    report = diagnostic.diagnose_qkv(planner, contexts, device="cpu")
    assert report["sample_count"] == 8 and report["planner_state_unchanged"]
    assert report["interpretation"]["replay_fidelity_ok"]
    assert report["q_projection_weights"]["numerical_rank"] == 8
    for sample in report["per_sample"]:
        assert sample["mask"]["equals_inverted_attention_mask"]
        assert sample["residual_sum_max_absolute_difference"] == 0
        for head in sample["heads"]:
            assert len(head["queries"]) == 6
            assert all(token["not_masked"] for q in head["queries"] for token in q["top_tokens"])
    assert all(torch.equal(value, before[name]) for name, value in planner.state_dict().items())
    assert all(p.grad is None for p in planner.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in planner.modules())
    json.dumps(report, allow_nan=False)
    report["per_sample"][0]["replay"]["explicit"]["close"] = False
    assert diagnostic.interpret(report["per_sample"])["code"] == "H"


def test_no_visible_tokens_are_reported_without_guessing_logits(setup):
    planner, contexts, _ = setup
    contexts = [replace(c, attention_mask=torch.zeros_like(c.attention_mask)) for c in contexts]
    report = diagnostic.diagnose_qkv(planner, contexts, device="cpu")
    assert report["interpretation"]["code"] == "A"
    assert report["interpretation"]["evidence_sample_counts"]["A"] == 8
    assert report["logit_aggregate_over_samples_heads_queries"] == {}
    assert all(s["mask"]["visible_tokens"] == 0 for s in report["per_sample"])
    json.dumps(report, allow_nan=False)


def test_zero_q_projection_is_measured_as_rank_zero(setup):
    planner, contexts, _ = setup
    module = planner.decoder.layers[1].multihead_attn
    with torch.no_grad():
        module.in_proj_weight[:module.embed_dim].zero_()
        module.in_proj_bias[:module.embed_dim].zero_()
    report = diagnostic.diagnose_qkv(planner, contexts, device="cpu")
    assert report["q_projection_weights"]["numerical_rank"] == 0
    assert report["interpretation"]["evidence_sample_counts"]["B"] == 8
