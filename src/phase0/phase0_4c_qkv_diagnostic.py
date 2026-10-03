from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.phase0.phase0_4c_collapse_diagnostic import pairwise_statistics, stage_statistics
from src.phase0.phase0_4c_tiny_overfit import CachedContext
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder


OUTCOMES = {
    "A": "MASK SEMANTICS / VALID-TOKEN PROBLEM", "B": "Q-PROJECTION COLLAPSE",
    "C": "DOMINANT KEY / LOGIT SATURATION", "D": "MEMORY / KV SCALE PATHOLOGY",
    "E": "LAYERNORM NORMALIZATION DOMINATES COLLAPSE",
    "F": "LAYERNORM AFFINE PARAMETERS DOMINATE COLLAPSE",
    "G": "MULTIPLE MECHANISMS CONTRIBUTE", "H": "CAUSE STILL NOT LOCALIZED",
}


def distribution(values: torch.Tensor) -> dict:
    values = values.detach().cpu().double().flatten()
    quantiles = torch.quantile(values, torch.tensor([0., .5, .95, .99, 1.], dtype=torch.float64))
    return dict(zip(("min", "median", "p95", "p99", "max"), quantiles.tolist(), strict=True)) | {
        "mean": float(values.mean()), "norm": float(values.norm()),
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0 else None


def mask_accounting(context: CachedContext, actual_mask: torch.Tensor) -> dict:
    valid = context.attention_mask.bool()
    ignored = actual_mask if actual_mask.dtype == torch.bool else torch.isneginf(actual_mask)
    return {
        "hidden_sequence_length": context.hidden_states.shape[1],
        "attention_mask_shape": list(valid.shape), "valid_tokens": int(valid.sum()),
        "invalid_tokens": int((~valid).sum()), "valid_fraction": float(valid.float().mean()),
        "memory_key_padding_mask_shape": list(actual_mask.shape),
        "memory_key_padding_mask_dtype": str(actual_mask.dtype),
        "visible_tokens": int((~ignored).sum()), "masked_tokens": int(ignored.sum()),
        "equals_inverted_attention_mask": torch.equal(ignored.cpu(), ~valid.cpu()),
        "semantics": "nn.MultiheadAttention boolean key_padding_mask: True=ignored, False=available",
    }


def project_qkv(module: nn.MultiheadAttention, args: tuple) -> tuple[torch.Tensor, ...]:
    query, key, value = args
    if module.batch_first:
        if key is value:
            if query is key:
                query = key = value = query.transpose(1, 0)
            else:
                query, key = query.transpose(1, 0), key.transpose(1, 0)
                value = key
        else:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))
    if module._qkv_same_embed_dim:
        q, k, v = F._in_projection_packed(query, key, value, module.in_proj_weight, module.in_proj_bias)
    else:
        biases = (None, None, None) if module.in_proj_bias is None else module.in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, module.q_proj_weight,
                                  module.k_proj_weight, module.v_proj_weight, *biases)
    batch, heads, dimension = query.shape[1], module.num_heads, module.head_dim
    split = tuple(x.view(x.shape[0], batch * heads, dimension).transpose(0, 1) for x in (q, k, v))
    return q, *split


def replay_attention(module: nn.MultiheadAttention, args: tuple, kwargs: dict) -> dict:
    if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError("v0.3 QKV probe requires the existing attention without appended K/V tokens")
    if kwargs.get("attn_mask") is not None or kwargs.get("is_causal", False):
        raise ValueError("v0.3 QKV probe requires the existing noncausal cross-attention padding mask only")
    projected_q, q, k, v = project_qkv(module, args)
    batch = args[0].shape[0 if module.batch_first else 1]
    mask = F._canonical_mask(kwargs["key_padding_mask"], "key_padding_mask", None, "attn_mask", q.dtype)
    mask = mask.view(batch, 1, 1, k.shape[1]).expand(-1, module.num_heads, -1, -1)
    flat_mask = mask.reshape(batch * module.num_heads, 1, k.shape[1])
    scale = math.sqrt(1.0 / module.head_dim)
    logits = torch.baddbmm(flat_mask, q * scale, k.transpose(-2, -1))
    probabilities = F.softmax(logits, dim=-1)
    explicit = torch.bmm(probabilities, v)
    sdpa = F.scaled_dot_product_attention(
        q.view(batch, module.num_heads, -1, module.head_dim),
        k.view(batch, module.num_heads, -1, module.head_dim),
        v.view(batch, module.num_heads, -1, module.head_dim), mask, dropout_p=0., is_causal=False,
    ).reshape(batch * module.num_heads, q.shape[1], module.head_dim)

    def output_projection(x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).contiguous().view(q.shape[1] * batch, module.embed_dim)
        x = F.linear(x, module.out_proj.weight, module.out_proj.bias).view(q.shape[1], batch, module.embed_dim)
        return x.transpose(1, 0) if module.batch_first else x

    return {"projected_q": projected_q, "q": q, "k": k, "v": v, "logits": logits,
            "probabilities": probabilities, "explicit_output": output_projection(explicit),
            "sdpa_output": output_projection(sdpa), "scale": scale}


def logit_statistics(logits: torch.Tensor, valid: torch.Tensor) -> list[dict]:
    mask = valid.to(logits.device).bool()
    indices = mask.nonzero().flatten()
    values = logits[:, indices]
    probabilities = logits.masked_fill(~mask[None], float("-inf")).softmax(-1)[:, indices]
    top, positions = values.topk(min(2, len(indices)), dim=-1)
    top_probabilities = probabilities.gather(-1, positions)
    entropy = -torch.special.xlogy(probabilities.double(), probabilities.double()).sum(-1)
    return [{
        "max": float(top[i, 0]), "second_largest": float(top[i, 1]) if len(indices) > 1 else None,
        "top1_top2_margin": float(top[i, 0] - top[i, 1]) if len(indices) > 1 else None,
        "mean": float(values[i].double().mean()), "std": float(values[i].double().std(correction=0)),
        "min": float(values[i].min()), "top1_probability": float(top_probabilities[i, 0]),
        "top2_probability": float(top_probabilities[i, 1]) if len(indices) > 1 else None,
        "entropy_nats": float(entropy[i]), "effective_token_count": float(entropy[i].exp()),
        "top_token_indices": indices[positions[i]].tolist(),
        "top1_probability_lower_bound_from_margin":
        1 / (1 + (len(indices) - 1) * math.exp(-float(top[i, 0] - top[i, 1]))) if len(indices) > 1 else 1.,
    } for i in range(len(logits))]


def aggregate_logits(rows: list[dict]) -> dict:
    return {key: distribution(torch.tensor([q[key] for q in rows if q[key] is not None], dtype=torch.float64))
            for key in ("max", "second_largest", "top1_top2_margin", "mean", "std", "min",
                        "top1_probability", "top2_probability", "entropy_nats", "effective_token_count")
            if any(q[key] is not None for q in rows)}


def layernorm_decomposition(z: torch.Tensor, norm: nn.LayerNorm, actual: torch.Tensor) -> dict:
    dims = tuple(range(z.ndim - len(norm.normalized_shape), z.ndim))
    variance, mean = torch.var_mean(z, dim=dims, correction=0, keepdim=True)
    normalized = (z - mean) * torch.rsqrt(variance + norm.eps)
    affine = normalized * norm.weight if norm.weight is not None else normalized
    if norm.bias is not None:
        affine = affine + norm.bias
    return {
        "eps": norm.eps, "variance_convention": "population variance (correction=0)",
        "z": stage_statistics(z), "pre_affine": stage_statistics(normalized),
        "post_affine": stage_statistics(affine), "actual_output": stage_statistics(actual),
        "reconstruction_max_absolute_difference": float((affine - actual).abs().max()),
        "reconstruction_close": torch.allclose(affine, actual, atol=1e-5, rtol=1e-4),
        "mean_per_query": mean.flatten().tolist(), "variance_per_query": variance.flatten().tolist(),
    }


def token_context(input_ids: list[int], tokenizer: object, frame_texts: tuple[str, ...], action_ids: list[int]) -> dict:
    candidates = [("assistant_structured_action", action_ids)] + [
        ("ego_history_text", tokenizer(text, add_special_tokens=False)["input_ids"]) for text in frame_texts
    ]
    spans, counts = [], []
    for category, ids in candidates:
        starts = [i for i in range(len(input_ids) - len(ids) + 1) if input_ids[i:i + len(ids)] == ids]
        counts.append({"category": category, "exact_match_count": len(starts)})
        if len(starts) == 1:
            spans.append({"category": category, "start": starts[0], "end": starts[0] + len(ids)})
    return {"input_ids": input_ids, "spans": spans, "span_match_counts": counts}


def token_position(index: int, valid: torch.Tensor, metadata: dict | None, tokenizer: object | None) -> dict:
    indices = valid.cpu().bool().nonzero().flatten().tolist()
    rank = indices.index(index) if index in indices else None
    result = {"absolute_index": index, "not_masked": bool(valid[index]), "valid_sequence_rank": rank,
              "relative_valid_position": rank / max(len(indices) - 1, 1) if rank is not None else None,
              "distance_from_final_valid_token": indices[-1] - index,
              "valid_tokens_after": len(indices) - rank - 1 if rank is not None else None}
    if metadata is None:
        return result | {"category": "unavailable", "token_mapping": "input IDs not provided"}
    token_id = metadata["input_ids"][index]
    name = tokenizer.convert_ids_to_tokens(token_id)
    owners = [s["category"] for s in metadata["spans"] if s["start"] <= index < s["end"]]
    if name in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"):
        category = "vision_token"
    elif owners:
        category = owners[0]
    elif token_id in tokenizer.all_special_ids:
        category = "special_token"
    else:
        category = "text_context_unresolved"
    return result | {"token_id": token_id, "token_name": name,
                     "decoded_text": tokenizer.decode([token_id]), "category": category}


def probe_sample(planner: WaypointDecoder, context: CachedContext, *, device: str,
                 metadata: dict | None, tokenizer: object | None) -> dict:
    layer = planner.decoder.layers[1]
    captured = {}

    def capture_attention(module: nn.Module, args: tuple, kwargs: dict, output: tuple) -> None:
        captured.update(args=args, kwargs=kwargs.copy(), output=output[0])

    def capture_norm(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        captured.update(z=args[0], norm_output=output)

    handles = [layer.multihead_attn.register_forward_hook(capture_attention, with_kwargs=True),
               layer.norm2.register_forward_hook(capture_norm)]
    try:
        planner(context.hidden_states.to(device), context.attention_mask.to(device))
    finally:
        for handle in handles:
            handle.remove()
    audit = mask_accounting(context, captured["kwargs"]["key_padding_mask"])
    if audit["visible_tokens"] == 0:
        return {"sample_token": context.sample_token, "mask": audit, "probe_skipped": "No visible memory tokens"}
    replay = replay_attention(layer.multihead_attn, captured["args"], captured["kwargs"])
    actual_mask = captured["kwargs"]["key_padding_mask"][0]
    valid = (~actual_mask if actual_mask.dtype == torch.bool else ~torch.isneginf(actual_mask)).cpu()
    valid_indices = valid.nonzero().flatten()
    query, memory, _ = captured["args"]
    memory_norms = memory[0].float().norm(dim=-1).cpu()
    memory_stats = distribution(memory_norms[valid])
    memory_stats["max_norm_token_index"] = int(valid_indices[memory_norms[valid].argmax()])
    heads = []
    for h, (q, k, v, logits) in enumerate(zip(replay["q"], replay["k"], replay["v"], replay["logits"], strict=True)):
        stats = logit_statistics(logits, valid)
        norms = {}
        for name, value in (("k", k), ("v", v)):
            magnitude = value.float().norm(dim=-1).cpu()
            norms[name] = distribution(magnitude[valid]) | {
                "max_norm_token_index": int(valid_indices[magnitude[valid].argmax()]),
                "top1_token_norm_per_query": [float(magnitude[s["top_token_indices"][0]]) for s in stats],
            }
        for t, stats_row in enumerate(stats):
            stats_row["top_tokens"] = [token_position(i, valid, metadata, tokenizer)
                                       for i in stats_row["top_token_indices"]]
            top_key = k[stats_row["top_token_indices"][0]]
            stats_row["top1_qk_cosine"] = float(F.cosine_similarity(q[t], top_key, dim=0))
        heads.append({"head": h, "q": pairwise_statistics(q), "token_norms": norms, "queries": stats,
                      "logit_aggregate_over_queries": aggregate_logits(stats),
                      "all_queries_same_top1": len({s["top_token_indices"][0] for s in stats}) == 1,
                      "saturated_all_queries": all(s["top1_probability"] >= .999 for s in stats)})
    fidelity = {}
    for name in ("explicit", "sdpa"):
        output = replay[f"{name}_output"]
        fidelity[name] = {"max_absolute_difference": float((output - captured["output"]).abs().max()),
                          "close": torch.allclose(output, captured["output"], atol=1e-5, rtol=1e-4)}
    residual_norm = query[0].norm(dim=-1)
    branch_norm = captured["output"][0].norm(dim=-1)
    norm = layernorm_decomposition(captured["z"], layer.norm2, captured["norm_output"])
    return {
        "sample_token": context.sample_token, "mask": audit,
        "q_before_projection": pairwise_statistics(query[0]),
        "q_after_projection": pairwise_statistics(replay["projected_q"][:, 0]),
        "projected_memory_norms": memory_stats, "heads": heads, "norm2": norm,
        "logit_aggregate_over_heads_queries": aggregate_logits([q for h in heads for q in h["queries"]]),
        "incoming_residual_norm_per_query": residual_norm.tolist(),
        "cross_attention_branch_norm_per_query": branch_norm.tolist(),
        "branch_to_residual_norm_ratio": [ratio(float(a), float(b)) for a, b in zip(branch_norm, residual_norm, strict=True)],
        "residual_sum_max_absolute_difference": float((query + captured["output"] - captured["z"]).abs().max()),
        "replay": fidelity, "scale": replay["scale"],
        "token_span_mapping": metadata["span_match_counts"] if metadata is not None else "unavailable",
    }


def interpret(samples: list[dict]) -> dict:
    counts = {key: 0 for key in "ABCDEF"}
    for sample in samples:
        mask = sample["mask"]
        if not mask["equals_inverted_attention_mask"] or mask["visible_tokens"] <= 1:
            counts["A"] += 1
    fidelity_ok = all("replay" in s and all(v["close"] for v in s["replay"].values())
                      and s["norm2"]["reconstruction_close"] for s in samples)
    for sample in samples:
        if "replay" not in sample:
            continue
        before, after = sample["q_before_projection"], sample["q_after_projection"]
        before_spread = ratio(before["off_diagonal_l2"]["mean"], sum(before["row_norms"]) / len(before["row_norms"]))
        after_spread = ratio(after["off_diagonal_l2"]["mean"], sum(after["row_norms"]) / len(after["row_norms"]))
        q_collapse = (before_spread is not None and before_spread > .001
                      and (max(after["row_norms"]) == 0 or (after_spread is not None
                           and after_spread <= .001 and after_spread <= .01 * before_spread)))
        counts["B"] += int(q_collapse)
        saturated = [h for h in sample["heads"] if h["saturated_all_queries"] and h["all_queries_same_top1"]]
        counts["C"] += int(bool(saturated))
        scale_outlier = any(
            values["median"] > 0 and max(values["top1_token_norm_per_query"]) >= 10 * values["median"]
            for h in saturated for values in h["token_norms"].values())
        ratios = [v for v in sample["branch_to_residual_norm_ratio"] if v is not None]
        counts["D"] += int(scale_outlier and bool(ratios) and min(ratios) >= 10)
        norm = sample["norm2"]
        geometric = ratio(norm["pre_affine"]["off_diagonal_l2"]["mean"], norm["z"]["off_diagonal_l2"]["mean"])
        affine = ratio(norm["post_affine"]["off_diagonal_l2"]["mean"], norm["pre_affine"]["off_diagonal_l2"]["mean"])
        counts["E"] += int(geometric is not None and geometric <= .1 and (affine is None or geometric <= affine))
        counts["F"] += int(affine is not None and geometric is not None and affine <= .1 and affine < geometric)
        sample["interpretation_evidence"] = {"q_relative_spread_before": before_spread,
            "q_relative_spread_after": after_spread, "normalization_l2_ratio": geometric,
            "affine_l2_ratio": affine, "saturated_same_top1_heads": len(saturated),
            "kv_scale_outlier_at_attended_token": scale_outlier}
    supported = [key for key in "BCDEF" if counts[key] > len(samples) / 2]
    code = "A" if counts["A"] else ("H" if not fidelity_ok or not supported else
                                      (supported[0] if len(supported) == 1 else "G"))
    return {
        "code": code, "outcome": OUTCOMES[code], "replay_fidelity_ok": fidelity_ok,
        "evidence_sample_counts": counts, "supported_mechanisms": {k: OUTCOMES[k] for k in supported},
        "thresholds": {"q_relative_spread_max": .001, "q_spread_retention_max": .01,
                       "saturated_top1_probability_min": .999, "kv_outlier_over_median_min": 10,
                       "branch_over_residual_min": 10, "norm_contraction_max": .1},
        "scope": "Heuristic evidence labels, not a causal proof or Phase PASS. A has priority; failed replay forces H. B-F require a strict majority of samples. Compare per-head Q diversity and K norms/logits before attributing saturation to Q collapse or key scale.",
    }


def diagnose_qkv(planner: WaypointDecoder, contexts: list[CachedContext], *, device: str,
                 token_contexts: dict[str, dict] | None = None, tokenizer: object | None = None) -> dict:
    planner.eval()
    layer = planner.decoder.layers[1]
    if layer.norm_first:
        raise ValueError("QKV norm2 residual decomposition requires the verified v0.3 post-norm checkpoint")
    before = {name: value.detach().cpu().clone() for name, value in planner.state_dict().items()}
    module = layer.multihead_attn
    weight = module.in_proj_weight[:module.embed_dim] if module._qkv_same_embed_dim else module.q_proj_weight
    singular = torch.linalg.svdvals(weight.detach().cpu().double())
    tolerance = float(singular.max()) * max(weight.shape) * torch.finfo(weight.dtype).eps
    with torch.no_grad():
        samples = [probe_sample(planner, c, device=device,
                                metadata=None if token_contexts is None else token_contexts[c.sample_token],
                                tokenizer=tokenizer) for c in contexts]
    interpretation = interpret(samples)
    queries = [q for s in samples if "heads" in s for h in s["heads"] for q in h["queries"]]
    aggregate = aggregate_logits(queries)
    unchanged = all(torch.equal(before[name], value.detach().cpu()) for name, value in planner.state_dict().items())
    if not unchanged:
        raise RuntimeError("QKV diagnostic changed planner state")
    return {
        "status": "qkv_diagnostic_complete_not_a_phase_pass", "sample_count": len(contexts),
        "norm_first": layer.norm_first, "planner_state_unchanged": unchanged,
        "backward_calls": 0, "optimizer_steps": 0, "per_sample": samples,
        "interpretation": interpretation, "logit_aggregate_over_samples_heads_queries": aggregate,
        "q_projection_weights": {"shape": list(weight.shape), "frobenius_norm": float(weight.detach().double().norm()),
                                 "singular_values": distribution(singular),
                                 "numerical_rank": int((singular > tolerance).sum()), "rank_tolerance": tolerance,
                                 "rank_tolerance_convention": "max(shape) * eps(weight dtype) * largest singular value"},
        "norm2_parameters": {"gamma": distribution(layer.norm2.weight), "beta": distribution(layer.norm2.bias)},
        "attention_contract": {"batch_first": module.batch_first, "num_heads": module.num_heads,
                               "head_dim": module.head_dim, "packed_projection": module._qkv_same_embed_dim,
                               "projection_bias": module.in_proj_bias is not None, "dtype": str(weight.dtype)},
        "measurement_notes": [
            "Projection uses installed PyTorch _in_projection_packed/_in_projection; explicit logits use Q * sqrt(1/head_dim), baddbmm with canonical additive padding mask, then softmax.",
            "Original need_weights=False forward is untouched. SDPA and explicit-softmax output reconstructions are both compared with it; fused kernels need not be bitwise identical.",
            "Replay and LayerNorm checks use atol=1e-5, rtol=1e-4; inspect absolute errors as well as flags.",
            "Logit moments exclude masked tokens; std uses correction=0. Quantile median is interpolated p50. Entropy is in nats.",
            "Token semantics use captured input IDs and unique exact token subsequences only; unresolved spans are reported, not guessed.",
            "No full Q/K/V, logits, probabilities, hidden states or input-ID sequences are persisted.",
        ],
    }
