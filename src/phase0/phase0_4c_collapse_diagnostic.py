from __future__ import annotations

import torch
from torch import nn

from src.phase0.phase0_4c_tiny_overfit import CachedContext
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder, masked_waypoint_loss


def pairwise_statistics(rows: torch.Tensor, *, matrices: bool = True) -> dict:
    rows = rows.detach().cpu().double()
    norms = torch.linalg.vector_norm(rows, dim=-1)
    distances = torch.cdist(rows, rows, compute_mode="donot_use_mm_for_euclid_dist")
    products = norms[:, None] * norms[None, :]
    defined = products > 0
    cosine = torch.full_like(products, float("nan"))
    cosine[defined] = ((rows @ rows.T)[defined] / products[defined]).clamp(-1, 1)
    off_diagonal = ~torch.eye(len(rows), dtype=torch.bool)
    l2 = distances[off_diagonal]
    cos = cosine[off_diagonal & defined]
    result = {
        "shape": list(rows.shape), "row_norms": norms.tolist(),
        "off_diagonal_l2": {"min": float(l2.min()), "mean": float(l2.mean()), "max": float(l2.max())},
        "off_diagonal_cosine": {
            name: float(getattr(cos, name)()) if cos.numel() else None
            for name in ("min", "mean", "max")
        },
        "undefined_off_diagonal_cosine_count": int((off_diagonal & ~defined).sum()),
    }
    if matrices:
        result["pairwise_l2"] = distances.tolist()
        result["pairwise_cosine"] = [
            [float(cosine[i, j]) if defined[i, j] else None for j in range(len(rows))]
            for i in range(len(rows))
        ]
    return result


def stage_statistics(value: torch.Tensor) -> dict:
    result = pairwise_statistics(value[0], matrices=False)
    result["shape"] = list(value.shape)
    return result


def prediction_delta(real: torch.Tensor, intervention: torch.Tensor) -> dict:
    delta = (real - intervention).double()
    return {"mean_absolute_m": float(delta.abs().mean()),
            "max_absolute_m": float(delta.abs().max()),
            "trajectory_l2_m": float(torch.linalg.vector_norm(delta)),
            "mean_waypoint_l2_m": float(torch.linalg.vector_norm(delta, dim=-1).mean())}


def representation_probe(planner: WaypointDecoder, contexts: list[CachedContext], *, device: str) -> dict:
    planner.eval()
    per_sample, hidden_means, memory_means = [], [], []
    modules = {f"decoder_layer_{i + 1}": layer for i, layer in enumerate(planner.decoder.layers)}
    if planner.decoder.norm is not None:
        modules["final_decoder_norm"] = planner.decoder.norm
    modules["waypoint_projection"] = planner.waypoint_projection
    with torch.no_grad():
        for index, context in enumerate(contexts):
            hidden = context.hidden_states.to(device).to(planner.context_projection.weight.dtype)
            mask = context.attention_mask.to(device)
            memory = planner.context_projection(hidden)
            hidden_mean = hidden[mask.bool()].float().mean(dim=0).cpu()
            memory_mean = memory[mask.bool()].float().mean(dim=0).cpu()
            hidden_means.append(hidden_mean)
            memory_means.append(memory_mean)
            stages = {"input_waypoint_queries": stage_statistics(planner.waypoint_queries.weight[None])}

            def capture(name: str):
                def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
                    stages[name] = stage_statistics(output)
                return hook

            handles = [module.register_forward_hook(capture(name)) for name, module in modules.items()]
            try:
                real = planner(hidden, mask)
            finally:
                for handle in handles:
                    handle.remove()
            queries = planner.waypoint_queries.weight.unsqueeze(0)
            zero = planner.waypoint_projection(planner.decoder(
                queries, torch.zeros_like(memory), memory_key_padding_mask=~mask.bool(),
            ))
            other = contexts[(index + 1) % len(contexts)]
            swapped = planner(other.hidden_states.to(device), other.attention_mask.to(device))
            per_sample.append({
                "sample_token": context.sample_token, "stages": stages,
                "context": {"valid_tokens": int(mask.sum()),
                            "masked_mean_hidden_norm": float(hidden_mean.norm()),
                            "masked_mean_projected_memory_norm": float(memory_mean.norm())},
                "real_vs_zero_memory": prediction_delta(real, zero),
                "real_vs_other_memory": prediction_delta(real, swapped),
                "other_memory_sample_token": other.sample_token,
            })
    aggregate = {}
    for stage in per_sample[0]["stages"]:
        values = [row["stages"][stage] for row in per_sample]
        cosines = [v["off_diagonal_cosine"]["mean"] for v in values
                   if v["off_diagonal_cosine"]["mean"] is not None]
        aggregate[stage] = {
            "mean_pairwise_l2": sum(v["off_diagonal_l2"]["mean"] for v in values) / len(values),
            "min_pairwise_l2": min(v["off_diagonal_l2"]["min"] for v in values),
            "mean_pairwise_cosine": sum(cosines) / len(cosines) if cosines else None,
            "samples_with_defined_cosine": len(cosines),
        }
    return {
        "mode": "eval", "additional_decoder_norm": planner.decoder.norm is not None,
        "per_sample": per_sample, "stage_aggregates": aggregate,
        "pooled_contexts": {
            "interpretation": "Masked means are diagnostic summaries, not proof of semantic separability; tokens are not aligned across samples.",
            "sample_tokens": [c.sample_token for c in contexts],
            "hidden_states": pairwise_statistics(torch.stack(hidden_means)),
            "projected_memory": pairwise_statistics(torch.stack(memory_means)),
        },
        "interventions": "Zero memory is applied after context_projection; other memory uses the next sample cyclically with its own attention mask.",
    }


def backward_probe(planner: WaypointDecoder, contexts: list[CachedContext], *, beta: float, device: str) -> dict:
    planner.eval()
    planner.zero_grad(set_to_none=True)
    predictions, losses = [], []
    for context in contexts:
        prediction = planner(context.hidden_states.to(device), context.attention_mask.to(device))
        prediction.retain_grad()
        predictions.append(prediction)
        losses.append(masked_waypoint_loss(prediction, context.target.to(device),
                                           context.valid_mask.to(device), beta=beta))
    objective = torch.stack(losses).mean()
    objective.backward()
    gradients = torch.cat([prediction.grad.detach().cpu() for prediction in predictions])
    groups = {"context_projection": planner.context_projection,
              **{f"decoder_layer_{i + 1}": layer for i, layer in enumerate(planner.decoder.layers)},
              "waypoint_projection": planner.waypoint_projection}
    return {
        "mode": "eval; configured dropout unchanged, stochastic dropout disabled for diagnosis",
        "objective": "mean over eight samples of the existing masked SmoothL1 coordinate mean",
        "loss": float(objective.detach()), "beta": beta, "backward_calls": 1, "optimizer_steps": 0,
        "output_gradient_scaling": "Includes each sample's valid-coordinate mean and the 1/N sample mean; reported mean averages the N retained prediction gradients.",
        "mean_dL_dpredicted_x_by_timestep": gradients.mean(dim=0)[:, 0].tolist(),
        "mean_dL_dpredicted_y_by_timestep": gradients.mean(dim=0)[:, 1].tolist(),
        "per_sample_output_gradients": [
            {"sample_token": c.sample_token, "dL_dprediction": gradient.tolist()}
            for c, gradient in zip(contexts, gradients, strict=True)
        ],
        "query_gradients": pairwise_statistics(planner.waypoint_queries.weight.grad),
        "module_gradient_l2": {
            name: float(torch.stack([p.grad.detach().double().square().sum()
                                     for p in module.parameters()]).sum().sqrt())
            for name, module in groups.items()
        },
    }


def diagnose(planner: WaypointDecoder, contexts: list[CachedContext], *, beta: float, device: str) -> dict:
    before = {name: value.detach().cpu().clone() for name, value in planner.state_dict().items()}
    result = {
        "status": "diagnostic_complete_not_a_phase_pass", "sample_count": len(contexts),
        "cosine_convention": "Zero-norm vectors have undefined cosine (JSON null); summaries exclude undefined pairs. Off-diagonal statistics include both directions.",
        "query_embeddings": pairwise_statistics(planner.waypoint_queries.weight),
        "representations": representation_probe(planner, contexts, device=device),
        "backward": backward_probe(planner, contexts, beta=beta, device=device),
        "interpretation_guide": {
            "query_parameter_collapse": "Nearly identical query embeddings suggest query-parameter collapse.",
            "first_layer": "Distinct embeddings but collapsed layer-1 outputs localize the next audit to layer-1 attention and normalization.",
            "second_layer": "Distinct layer-1 outputs but collapsed layer-2 outputs localize the next audit to layer 2.",
            "output_projection": "Distinct final decoder representations but collapsed waypoints point to the shared projection.",
            "context_ignored": "Small real-vs-zero and real-vs-other deltas suggest functional context insensitivity on these eight samples.",
            "temporal_differentiation": "Material cross-context deltas with collapsed timesteps indicate context use with failed temporal differentiation.",
            "query_jacobian": "Different direct output gradients with near-zero or similar query gradients support suppression/coupling of query-specific supervision by the decoder Jacobian.",
        },
    }
    unchanged = all(torch.equal(before[name], value.detach().cpu())
                    for name, value in planner.state_dict().items())
    if not unchanged:
        raise RuntimeError("diagnostic changed planner state")
    result["planner_state_unchanged"] = unchanged
    return result
