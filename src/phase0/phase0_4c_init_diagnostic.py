from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.phase0.phase0_4c_collapse_diagnostic import stage_statistics
from src.phase0.phase0_4c_qkv_diagnostic import diagnose_qkv, distribution, ratio
from src.phase0.phase0_4c_tiny_overfit import CachedContext, TinyOverfitConfig, evaluate
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder


OUTCOMES = {
    "A": "PATHOLOGY PRESENT AT INITIALIZATION",
    "B": "PATHOLOGY CREATED DURING TRAINING",
    "C": "PATHOLOGY PRESENT INITIALLY AND AMPLIFIED BY TRAINING",
    "D": "INITIALIZATION COULD NOT BE RELIABLY RECONSTRUCTED",
    "E": "RESULT INCONCLUSIVE",
}


def initialization_fidelity(recorded_path: Path, predictions: list[dict]) -> dict:
    result = {"passed": False, "max_absolute_difference": None, "mean_absolute_difference": None,
              "atol": 1e-5, "rtol": 1e-5, "reference": str(recorded_path)}
    if not recorded_path.is_file():
        return result | {"reason": "predictions_before.jsonl is missing"}
    try:
        recorded = [json.loads(line) for line in recorded_path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        return result | {"reason": f"Invalid predictions_before.jsonl: {error}"}
    if not all(isinstance(row, dict) for row in recorded):
        return result | {"reason": "Recorded predictions must be JSON objects"}
    if [r.get("sample_token") for r in recorded] != [r["sample_token"] for r in predictions]:
        return result | {"reason": "Recorded sample tokens/order differ from the reconstructed contexts"}
    try:
        expected = torch.tensor([r.get("predicted_waypoints") for r in recorded], dtype=torch.float64)
    except (ValueError, TypeError) as error:
        return result | {"reason": f"Invalid recorded waypoint arrays: {error}"}
    actual = torch.tensor([r["predicted_waypoints"] for r in predictions], dtype=torch.float64)
    if expected.shape != actual.shape or not torch.isfinite(expected).all():
        return result | {"reason": "Recorded predictions have invalid shape or nonfinite values"}
    delta = (actual - expected).abs()
    passed = torch.allclose(actual, expected, atol=result["atol"], rtol=result["rtol"])
    return result | {
        "passed": passed, "max_absolute_difference": float(delta.max()),
        "mean_absolute_difference": float(delta.mean()), "bitwise_equal": torch.equal(actual, expected),
        "per_sample": [{"sample_token": row["sample_token"],
                        "max_absolute_difference": float(d.max()), "mean_absolute_difference": float(d.mean())}
                       for row, d in zip(predictions, delta, strict=True)],
        "reason": "Predictions match within the existing reload tolerance" if passed else
                  "Reconstructed initialization does not reproduce recorded predictions; comparison stopped",
    }


def parameter_groups(planner: WaypointDecoder) -> dict[str, torch.Tensor]:
    groups = {}
    for name, value in planner.named_parameters():
        value = value.detach().cpu().double()
        if ".in_proj_" in name:
            prefix, suffix = name.rsplit(".in_proj_", 1)
            for label, part in zip(("Q", "K", "V"), value.chunk(3), strict=True):
                groups[f"{prefix}.{label}.{suffix}"] = part
        else:
            groups[name] = value
    return groups


def parameter_changes(initial: WaypointDecoder, trained: WaypointDecoder) -> dict:
    before, after = parameter_groups(initial), parameter_groups(trained)
    result = {}
    for name, value in before.items():
        start, end = float(value.norm()), float(after[name].norm())
        delta = float((after[name] - value).norm())
        result[name] = {"initial_l2": start, "trained_l2": end, "delta_l2": delta,
                        "relative_delta_l2": ratio(delta, start), "initial_norm_zero": start == 0}
    return result


def state_diagnostic(planner: WaypointDecoder, contexts: list[CachedContext], *, device: str) -> dict:
    stages, memories, memory_norms = [], [], []

    def start_sample(module: nn.Module, args: tuple) -> None:
        stages.append({"input_waypoint_queries": stage_statistics(planner.waypoint_queries.weight[None])})

    def capture_stage(name: str, *, input_value: bool = False):
        def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
            stages[-1][name] = stage_statistics(args[0] if input_value else output)
        return hook

    def capture_memory(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        valid = contexts[len(memories)].attention_mask[0].bool().to(output.device)
        values = output[0, valid].detach().double()
        norms = values.norm(dim=-1).cpu()
        memory_norms.append(norms)
        memories.append({"sample_token": contexts[len(memories)].sample_token,
                         "valid_token_norms": distribution(norms), "feature_rms": float(values.square().mean().sqrt()),
                         "valid_tokens": len(norms)})

    layer = planner.decoder.layers[1]
    handles = [planner.register_forward_pre_hook(start_sample),
               planner.context_projection.register_forward_hook(capture_memory)]
    for name, module, input_value in (
        ("layer1_output", planner.decoder.layers[0], False),
        ("layer2_input", layer, True),
        ("after_layer2_self_attention_residual_norm", layer.norm1, False),
        ("after_layer2_cross_attention_residual_norm", layer.norm2, False),
        ("layer2_output", layer, False),
        ("waypoint_output", planner.waypoint_projection, False),
    ):
        handles.append(module.register_forward_hook(capture_stage(name, input_value=input_value)))
    try:
        qkv = diagnose_qkv(planner, contexts, device=device)
    finally:
        for handle in handles:
            handle.remove()
    aggregate = {
        name: {"shapes": [s[name]["shape"] for s in stages],
               "mean_pairwise_l2": sum(s[name]["off_diagonal_l2"]["mean"] for s in stages) / len(stages),
               "min_pairwise_l2": min(s[name]["off_diagonal_l2"]["min"] for s in stages)}
        for name in stages[0]
    }
    for sample in qkv["per_sample"]:
        for head in sample.get("heads", []):
            head["q_norms"] = distribution(torch.tensor(head["q"]["row_norms"], dtype=torch.float64))
            for value in head["token_norms"].values():
                value["max_over_median"] = ratio(value["max"], value["median"])
        if "norm2" in sample:
            norm = sample["norm2"]
            norm["contraction_ratios"] = {
                "pure_normalization": ratio(norm["pre_affine"]["off_diagonal_l2"]["mean"],
                                            norm["z"]["off_diagonal_l2"]["mean"]),
                "affine": ratio(norm["post_affine"]["off_diagonal_l2"]["mean"],
                                norm["pre_affine"]["off_diagonal_l2"]["mean"]),
                "total": ratio(norm["post_affine"]["off_diagonal_l2"]["mean"],
                               norm["z"]["off_diagonal_l2"]["mean"]),
            }
    return {
        "qkv": qkv, "memory": {"per_sample": memories,
            "pooled_valid_token_norms": distribution(torch.cat(memory_norms)),
            "pooled_feature_rms": (sum(m["feature_rms"] ** 2 * m["valid_tokens"] for m in memories)
                                   / sum(m["valid_tokens"] for m in memories)) ** .5},
        "collapse_progression": {"per_sample": [dict(sample_token=c.sample_token, stages=s)
                                                  for c, s in zip(contexts, stages, strict=True)],
                                 "aggregate": aggregate},
        "execution_order": "post-norm: self-attention -> residual -> norm1 -> cross-attention -> residual -> norm2 -> FFN -> residual -> norm3",
    }


def interpret_comparison(initial: dict, trained: dict, changes: dict) -> dict:
    first, last = initial["qkv"], trained["qkv"]
    interfaces = {"D": "layer2 projected memory / K,V scale", "C": "layer2 cross-attention logits / softmax",
                  "E": "layer2 norm2 pure normalization"}
    counts = [r["interpretation"]["evidence_sample_counts"] for r in (first, last)]
    supported = [{k for k in interfaces if c[k] > r["sample_count"] / 2}
                 for c, r in zip(counts, (first, last), strict=True)]
    valid = all(r["interpretation"]["replay_fidelity_ok"] and not c["A"]
                for r, c in zip((first, last), counts, strict=True))
    contraction_values = [[s["norm2"]["contraction_ratios"]["pure_normalization"]
                           for s in r["per_sample"] if "norm2" in s] for r in (first, last)]
    contractions_defined = all(len(values) == r["sample_count"] and all(v is not None for v in values)
                               for values, r in zip(contraction_values, (first, last), strict=True))
    code, amplification = "E", {}
    if valid and contractions_defined:
        margins = [r["logit_aggregate_over_samples_heads_queries"]["top1_top2_margin"]["median"]
                   for r in (first, last)]
        contractions = [sum(values) / len(values) for values in contraction_values]
        amplification = {"median_margin_trained_over_initial": ratio(margins[1], margins[0]),
                         "mean_normalization_contraction_trained_over_initial": ratio(contractions[1], contractions[0])}
        if not any(counts[0][k] for k in interfaces) and supported[1] == set(interfaces):
            code = "B"
        elif supported[0] and supported[0] <= supported[1]:
            amplified = margins[0] > 0 and margins[1] >= 2 * margins[0] and contractions[1] <= .5 * contractions[0]
            code = "C" if amplified else ("A" if supported[0] == supported[1] else "E")
    candidates = supported[1] - supported[0] if code == "B" else supported[0]
    nonzero = {k: v for k, v in changes.items() if v["relative_delta_l2"] is not None}
    return {
        "code": code, "outcome": OUTCOMES[code], "fidelity_and_masks_ok": valid,
        "normalization_contractions_defined": contractions_defined,
        "initial_supported_interfaces": [v for k, v in interfaces.items() if k in supported[0]],
        "trained_supported_interfaces": [v for k, v in interfaces.items() if k in supported[1]],
        "earliest_supported_interface": next((v for k, v in interfaces.items() if k in candidates), None),
        "largest_absolute_parameter_change": max(changes, key=lambda k: changes[k]["delta_l2"]),
        "largest_relative_parameter_change_nonzero_initial": max(nonzero, key=lambda k: nonzero[k]["relative_delta_l2"]),
        "amplification": amplification,
        "criteria": "Reuse QKV C/D/E thresholds and strict sample majority. B requires zero initial C/D/E cases and trained majority for all three; C requires initial supported mechanisms retained, >=2x median margin and <=0.5x mean normalization retention; A requires the same nonempty mechanism set without that joint amplification. Otherwise E. Failed replay/mask checks or undefined normalization contractions force E.",
        "limitation": "Operational evidence classification, not causal proof. Parameter delta rankings do not identify a causal parameter. Scale evidence localizes projected memory/KV, not the origin inside Qwen or context_projection. Partial or conflicting evidence remains inconclusive.",
    }


def diagnose_initial_vs_trained(trained: WaypointDecoder, contexts: list[CachedContext], *,
                                config: TinyOverfitConfig, hidden_size: int,
                                recorded_path: Path, device: str) -> dict:
    torch.manual_seed(config.seed)
    initial = WaypointDecoder(hidden_size, config).to(device)
    _, predictions = evaluate(initial, contexts, beta=config.smooth_l1_beta, device=device)
    fidelity = initialization_fidelity(recorded_path, predictions)
    result = {"status": "init_vs_trained_diagnostic_not_a_phase_pass", "sample_count": len(contexts),
              "initialization_fidelity": fidelity, "initialization_seed": config.seed,
              "initial_checkpoint_loaded": False, "backward_calls": 0, "optimizer_steps": 0,
              "measurement_notes": ["CPU construction immediately after torch.manual_seed, then .to(device), exactly as fit_cached_contexts; no trained state is loaded into initial.",
                                    "Both states use the same in-memory cached contexts and eval mode. Prediction fidelity uses existing reload atol=rtol=1e-5; it is not a bitwise reconstruction guarantee.",
                                    "Memory statistics pool valid tokens; logits pool all sample/head/query combinations. Full attention or hidden tensors are not persisted.",
                                    "Zero initial parameter norms have undefined relative deltas (null); absolute changes remain reported."]}
    if not fidelity["passed"]:
        return result | {"comparison_performed": False,
                         "interpretation": {"code": "D", "outcome": OUTCOMES["D"], "reason": fidelity["reason"]}}
    changes = parameter_changes(initial, trained)
    before = state_diagnostic(initial, contexts, device=device)
    after = state_diagnostic(trained, contexts, device=device)
    return result | {"comparison_performed": True, "parameter_changes": changes,
                     "initial": before, "trained": after,
                     "planner_state_unchanged": before["qkv"]["planner_state_unchanged"] and after["qkv"]["planner_state_unchanged"],
                     "interpretation": interpret_comparison(before, after, changes)}
