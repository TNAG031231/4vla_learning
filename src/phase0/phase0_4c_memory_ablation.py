from __future__ import annotations

import torch
from torch import nn

from src.phase0.phase0_4c_collapse_diagnostic import pairwise_statistics
from src.phase0.phase0_4c_init_diagnostic import state_diagnostic
from src.phase0.phase0_4c_layer2_diagnostic import diagnose_layer2
from src.phase0.phase0_4c_qkv_diagnostic import distribution
from src.phase0.phase0_4c_tiny_overfit import CachedContext
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder


def trajectory_diversity(predictions: list[dict]) -> dict:
    points = torch.tensor([row["predicted_waypoints"] for row in predictions], dtype=torch.float64)
    samples = []
    for row, trajectory in zip(predictions, points, strict=True):
        delta = trajectory[1:] - trajectory[:-1]
        samples.append({
            "sample_token": row["sample_token"], "waypoints": trajectory.tolist(),
            "pairwise": pairwise_statistics(trajectory, matrices=False),
            "longitudinal_increments_m": delta[:, 0].tolist(),
            "backward_longitudinal_steps": int((delta[:, 0] < 0).sum()),
            "temporal_path_length_m": float(delta.norm(dim=-1).sum()),
            "first_to_last_distance_m": float((trajectory[-1] - trajectory[0]).norm()),
        })
    return {"per_sample": samples,
            "across_samples_by_timestep": [pairwise_statistics(points[:, t], matrices=False)
                                            for t in range(points.shape[1])],
            "across_samples_trajectory_l2": pairwise_statistics(points.flatten(1), matrices=False),
            "scope": "All six predicted timestamps; raw meters. Pairwise metrics exclude diagonal. These are descriptive diversity measures, not extra PASS conditions. GT validity masks remain in predictions JSONL."}


def collect_diagnostics(planner: WaypointDecoder, contexts: list[CachedContext],
                        predictions: list[dict], *, device: str) -> dict:
    raw = []

    def capture_raw(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        context = contexts[len(raw)]
        values = output[0, context.attention_mask[0].bool().to(output.device)].detach().double()
        raw.append({"sample_token": context.sample_token,
                    "valid_token_norms": distribution(values.norm(dim=-1)),
                    "feature_rms": float(values.square().mean().sqrt())})

    handle = planner.context_projection.register_forward_hook(capture_raw)
    try:
        state = state_diagnostic(planner, contexts, device=device)
    finally:
        handle.remove()
    layer2 = diagnose_layer2(planner, contexts, device=device)
    return {
        "status": "memory_normalization_ablation_diagnostics",
        "raw_projected_memory": raw, "decoder_memory_after_layernorm": state["memory"],
        "qkv": state["qkv"], "collapse_progression": state["collapse_progression"],
        "layer2_stages": layer2["stage_aggregates"], "layer2_per_sample": layer2["per_sample"],
        "cross_query_attention": layer2["attention_aggregate"],
        "cross_sample_attention": layer2["cross_sample_attention_summary_deltas"],
        "execution_order": layer2["execution_order"], "norm_first": layer2["norm_first"],
        "trajectory_diversity": trajectory_diversity(predictions),
        "planner_state_unchanged": state["qkv"]["planner_state_unchanged"] and layer2["planner_state_unchanged"],
        "diagnostic_backward_calls": 0, "diagnostic_optimizer_steps": 0,
        "measurement_notes": [
            "Raw context_projection output and normalized decoder memory are reported separately; LayerNorm regulates the latter.",
            "Original eval forward and attention backend are unchanged. Separate attention replays never feed into training; inspect fidelity flags.",
            "Logits aggregate samples, heads and queries. Memory statistics exclude padding. No full hidden or attention tensors are saved.",
            "Mechanistic evidence and the unchanged six-condition tiny-overfit gate must be assessed separately.",
        ],
    }
