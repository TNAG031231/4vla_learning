from __future__ import annotations

import math

import torch
from torch import nn

from src.phase0.phase0_4c_collapse_diagnostic import pairwise_statistics, stage_statistics
from src.phase0.phase0_4c_tiny_overfit import CachedContext
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder


def attention_statistics(weights: torch.Tensor, valid_tokens: torch.Tensor, *, top_k: int = 10) -> dict:
    weights = weights.detach().cpu().double()
    token_indices = valid_tokens.detach().cpu().bool().nonzero().flatten()
    probabilities = weights[:, token_indices]
    entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
    k = min(top_k, len(token_indices))
    top_values, top_indices = probabilities.topk(k, dim=-1)
    cosine = pairwise_statistics(probabilities)
    l1 = torch.cdist(probabilities, probabilities, p=1)
    midpoint = (probabilities[:, None, :] + probabilities[None, :, :]) / 2
    js = (-torch.special.xlogy(midpoint, midpoint).sum(dim=-1)
          - (entropy[:, None] + entropy[None, :]) / 2).clamp_min(0)
    overlap = (top_indices[:, None, :, None] == top_indices[None, :, None, :]).sum(dim=(-1, -2)) / k
    off_diagonal = ~torch.eye(len(weights), dtype=torch.bool)
    return {
        "shape": list(weights.shape), "valid_tokens": len(token_indices), "top_k": k,
        "entropy_nats_per_query": entropy.tolist(),
        "normalized_entropy_per_query": (entropy / math.log(len(token_indices))).tolist()
        if len(token_indices) > 1 else [0.] * len(weights),
        "top_token_indices": token_indices[top_indices].tolist(),
        "top_token_probabilities": top_values.tolist(), "top_k_mass_per_query": top_values.sum(-1).tolist(),
        "valid_probability_mass_per_query": probabilities.sum(-1).tolist(),
        "pairwise_cosine": cosine["pairwise_cosine"], "pairwise_l1": l1.tolist(),
        "pairwise_js_nats": js.tolist(), "top_k_overlap_fraction": overlap.tolist(),
        "mean_pairwise_cosine": cosine["off_diagonal_cosine"]["mean"],
        "mean_pairwise_l1": float(l1[off_diagonal].mean()),
        "mean_pairwise_js_nats": float(js[off_diagonal].mean()),
        "mean_top_k_overlap_fraction": float(overlap[off_diagonal].mean()),
    }


def probe_sample(planner: WaypointDecoder, context: CachedContext, *, device: str) -> dict:
    layer = planner.decoder.layers[1]
    stages, captured = {}, {}

    def output_hook(name: str):
        def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
            stages[name] = stage_statistics(output)
        return hook

    def input_hook(name: str):
        def hook(module: nn.Module, args: tuple) -> None:
            stages[name] = stage_statistics(args[0])
        return hook

    def capture_attention(module: nn.Module, args: tuple, kwargs: dict, output: tuple) -> None:
        captured.update(args=args, kwargs=kwargs.copy(), output=output[0])

    handles = [layer.register_forward_pre_hook(input_hook("layer2_input"))]
    for name, dropout in (("self_attention", layer.dropout1), ("cross_attention", layer.dropout2),
                          ("ffn", layer.dropout3)):
        handles.append(dropout.register_forward_hook(output_hook(f"{name}_branch_before_residual")))
    if layer.norm_first:
        for name, norm in (("self_attention", layer.norm1), ("cross_attention", layer.norm2), ("ffn", layer.norm3)):
            handles.append(norm.register_forward_hook(output_hook(f"{name}_pre_norm")))
        handles.append(layer.norm2.register_forward_pre_hook(input_hook("after_self_attention_residual")))
        handles.append(layer.norm3.register_forward_pre_hook(input_hook("after_cross_attention_residual")))
    else:
        for name, norm in (("self_attention", layer.norm1), ("cross_attention", layer.norm2), ("ffn", layer.norm3)):
            handles.append(norm.register_forward_pre_hook(input_hook(f"{name}_residual_before_norm")))
            handles.append(norm.register_forward_hook(output_hook(f"after_{name}_residual_norm")))
    handles.append(layer.register_forward_hook(output_hook("layer2_output")))
    handles.append(layer.multihead_attn.register_forward_hook(capture_attention, with_kwargs=True))
    try:
        prediction = planner(context.hidden_states.to(device), context.attention_mask.to(device))
    finally:
        for handle in handles:
            handle.remove()
    kwargs = captured["kwargs"] | {"need_weights": True, "average_attn_weights": False}
    replay, weights = layer.multihead_attn(*captured["args"], **kwargs)
    difference = replay - captured["output"]
    valid = context.attention_mask[0]
    return {
        "sample_token": context.sample_token, "stages": stages, "execution_order": list(stages),
        "prediction_summary": stage_statistics(prediction),
        "cross_attention": {
            "weights_shape": list(weights.shape),
            "head_mean": attention_statistics(weights[0].mean(dim=0), valid),
            "per_head": [attention_statistics(head, valid) for head in weights[0]],
            "replay_output_mean_absolute_difference": float(difference.abs().mean()),
            "replay_output_max_absolute_difference": float(difference.abs().max()),
            "replay_output_close": torch.allclose(replay, captured["output"], atol=1e-5, rtol=1e-4),
            "replay_atol": 1e-5, "replay_rtol": 1e-4,
        },
    }


def aggregate_attention(summaries: list[dict]) -> dict:
    vector_fields = ("entropy_nats_per_query", "normalized_entropy_per_query", "top_k_mass_per_query")
    scalar_fields = ("mean_pairwise_cosine", "mean_pairwise_l1", "mean_pairwise_js_nats",
                     "mean_top_k_overlap_fraction")
    return {
        **{key: torch.tensor([s[key] for s in summaries], dtype=torch.float64).mean(dim=0).tolist()
           for key in vector_fields},
        **{key: sum(s[key] for s in summaries) / len(summaries) for key in scalar_fields},
    }


def compare_sample_attention(summaries: list[dict]) -> dict:
    result = {}
    for field in ("normalized_entropy_per_query", "top_k_mass_per_query",
                  "mean_pairwise_cosine", "mean_pairwise_js_nats"):
        values = torch.tensor([s[field] for s in summaries], dtype=torch.float64).reshape(len(summaries), -1)
        result[f"mean_absolute_delta_{field}"] = (values[:, None] - values[None, :]).abs().mean(-1).tolist()
    return result


def diagnose_layer2(planner: WaypointDecoder, contexts: list[CachedContext], *, device: str) -> dict:
    planner.eval()
    before = {name: value.detach().cpu().clone() for name, value in planner.state_dict().items()}
    with torch.no_grad():
        samples = [probe_sample(planner, context, device=device) for context in contexts]
    stages = {}
    for name in samples[0]["execution_order"]:
        values = [sample["stages"][name] for sample in samples]
        cosines = [v["off_diagonal_cosine"]["mean"] for v in values
                   if v["off_diagonal_cosine"]["mean"] is not None]
        stages[name] = {
            "shapes": [v["shape"] for v in values],
            "mean_pairwise_l2": sum(v["off_diagonal_l2"]["mean"] for v in values) / len(values),
            "min_pairwise_l2": min(v["off_diagonal_l2"]["min"] for v in values),
            "mean_pairwise_cosine": sum(cosines) / len(cosines) if cosines else None,
            "samples_with_defined_cosine": len(cosines),
        }
    unchanged = all(torch.equal(before[name], value.detach().cpu())
                    for name, value in planner.state_dict().items())
    if not unchanged:
        raise RuntimeError("layer-2 diagnostic changed planner state")
    summaries = [s["cross_attention"]["head_mean"] for s in samples]
    heads = planner.decoder.layers[1].multihead_attn.num_heads
    return {
        "status": "layer2_diagnostic_complete_not_a_phase_pass", "sample_count": len(contexts),
        "mode": "eval", "norm_first": planner.decoder.layers[1].norm_first,
        "execution_order": samples[0]["execution_order"], "stage_aggregates": stages,
        "per_sample": samples, "planner_state_unchanged": unchanged,
        "backward_calls": 0, "optimizer_steps": 0,
        "attention_aggregate": {
            "head_mean": aggregate_attention(summaries),
            "per_head": [aggregate_attention([s["cross_attention"]["per_head"][h] for s in samples])
                         for h in range(heads)],
        },
        "cross_sample_attention_summary_deltas": {
            "sample_tokens": [c.sample_token for c in contexts],
            "head_mean": compare_sample_attention(summaries),
            "per_head": [compare_sample_attention([s["cross_attention"]["per_head"][h] for s in samples])
                         for h in range(heads)],
        },
        "measurement_notes": [
            "Stage order is observed from hooks on the original forward; hooks never replace outputs.",
            "Branch outputs are after dropout1/2/3, before residual addition; dropout is identity in eval mode.",
            "Original attention keeps need_weights=False. A separate replay with captured Q/K/V and masks requests per-head weights; replay outputs never enter the planner.",
            "need_weights=True can use a different numerical kernel; inspect replay-output differences before interpreting attention summaries.",
            "Entropy and Jensen-Shannon divergence use natural logs; top-k overlap is intersection/k; pairwise summaries exclude the diagonal.",
            "Token indices are original positions within each sample. Equal-weight top-k ties can have arbitrary ordering.",
            "Across samples only summary deltas are compared, without token alignment. Similar summaries do not prove identical attention distributions or semantic focus.",
            "Locate the first separation drop using branch, residual, and norm stages; attention similarity alone does not establish a causal root cause.",
        ],
    }
