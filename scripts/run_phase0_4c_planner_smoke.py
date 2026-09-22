#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.phase0.development_projection import IsolationCounters
from src.phase0.phase0_4b_lora_full import load_config as load_semantic_config
from src.phase0.phase0_4b_lora_smoke import load_temporal_records, select_tiny
from src.phase0.phase0_4b_protocol import adapt_record, serialize_action
from src.phase0.phase0_4c_two_turn_planner import (
    PlannerConfig, WaypointDecoder, contextual_hidden_states, freeze_backbone, load_config,
    masked_waypoint_loss, planning_inputs, planning_messages, predicted_action_inference,
)
from src.phase0.qwen3vl_dataset_adapter import (
    collect_git_provenance, load_config as load_ego_config, resolve_derived_path,
)
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies
from src.phase0.qwen3vl_smoke import resolve_image_path


def run_smoke(*, dataset_root: Path, derived_root: Path, config: PlannerConfig) -> dict:
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(ROOT):
        raise ValueError("smoke artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"smoke output already exists: {output}")
    semantic = load_semantic_config(ROOT / "configs/phase0_4b_lora_full.yaml")
    records = load_temporal_records(ROOT, derived_root, split="train")
    ego_config = load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml")
    samples = [adapt_record(row, ego_config) for row in records]
    selected = select_tiny(
        [s for s in samples if s.target.longitudinal_valid and s.target.lateral_valid],
        config.train_subset_size, config.seed,
    )
    targets = {row["sample_token"]: row for row in records
               if row["sample_token"] in {s.sample_token for s in selected}}
    runtime = default_runtime_dependencies()
    device = runtime.device_selector("cuda:0")
    dtype = runtime.dtype_selector(semantic.precision)
    if dtype != torch.bfloat16:
        raise ValueError("selected Phase 0.4b model requires BF16 support")
    torch.manual_seed(config.seed)
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, semantic.local_files_only)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype,
                                semantic.attention_implementation, semantic.local_files_only)
    adapter = resolve_derived_path(derived_root, config.selected_adapter_relative_path)
    model = runtime.adapter_loader(base, adapter).to(device)
    freeze_backbone(model)
    planner = WaypointDecoder(model.config.text_config.hidden_size, config).to(device)
    print(json.dumps({"event": "selected_adapter_loaded", "adapter": str(adapter),
                      "planner_trainable_parameters": sum(p.numel() for p in planner.parameters())}),
          flush=True)
    training = []
    for sample in selected:
        images = [runtime.image_loader(resolve_image_path(dataset_root, path))
                  for path in sample.observation.image_paths]
        action = serialize_action(sample.target.longitudinal, sample.target.lateral)
        inputs, evidence = planning_inputs(
            processor, planning_messages(sample.observation, images, action), device,
        )
        hidden = contextual_hidden_states(model, inputs)
        planner.train()
        planner.zero_grad(set_to_none=True)
        waypoints = planner(hidden, inputs["attention_mask"])
        row = targets[sample.sample_token]
        target = torch.tensor([row["future_waypoints"]], device=device, dtype=torch.float32)
        mask = torch.tensor([row["trajectory_valid_mask"]], device=device, dtype=torch.bool)
        loss = masked_waypoint_loss(waypoints, target, mask, beta=config.smooth_l1_beta)
        if not torch.isfinite(loss):
            raise ValueError("teacher-forced waypoint loss is not finite")
        loss.backward()
        gradients = {
            name: sum(float(p.grad.detach().float().norm()) for p in module.parameters()
                      if p.grad is not None)
            for name, module in planner.named_children()
        }
        entry = {
            "sample_token": sample.sample_token, "conditioning_type": "gt_action_teacher_forced_train",
            **evidence, "hidden_state_field": "hidden_states[-1]",
            "hidden_state_shape": list(hidden.shape), "hidden_state_dtype": str(hidden.dtype),
            "waypoint_shape": list(waypoints.shape),
            "waypoints_finite": bool(torch.isfinite(waypoints).all()),
            "loss": float(loss.detach()), "backward_succeeded": True,
            "planner_gradient_norms": gradients,
            "planner_gradients_finite": all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                            for p in planner.parameters()),
            "planner_gradients_nonzero": all(value > 0 for value in gradients.values()),
        }
        training.append(entry)
        print(json.dumps({"event": "teacher_forced_backward", "sample_token": sample.sample_token,
                          "loss": entry["loss"], "gradient_norms": gradients}), flush=True)
        del hidden, waypoints, loss
    sample = selected[0]
    images = [runtime.image_loader(resolve_image_path(dataset_root, path))
              for path in sample.observation.image_paths]
    _, predicted = predicted_action_inference(
        model=model, planner=planner, observation=sample.observation, images=images,
        processor=processor, generation_kwargs=semantic.generation_kwargs, device=device,
    )
    predicted["sample_token"] = sample.sample_token
    lora = [p for name, p in model.named_parameters() if "lora_" in name]
    freeze = {
        "qwen_and_lora_frozen": all(not p.requires_grad and p.grad is None for p in model.parameters()),
        "lora_parameter_tensors": len(lora),
        "lora_frozen": bool(lora) and all(not p.requires_grad and p.grad is None for p in lora),
        "planner_trainable": all(p.requires_grad for p in planner.parameters()),
    }
    passed = (
        freeze["qwen_and_lora_frozen"] and freeze["lora_frozen"] and freeze["planner_trainable"]
        and all(e["planner_gradients_nonzero"] and e["planner_gradients_finite"]
                and e["waypoints_finite"] and e["waypoint_shape"] == [1, 6, 2] for e in training)
        and predicted["waypoints_finite"] and predicted["waypoint_shape"] == [1, 6, 2]
        and predicted["action_context_matches"]
    )
    result = {
        "status": "smoke_passed" if passed else "smoke_failed",
        "run_kind": "two_turn_planner_chain_only", "trajectory_performance_evaluated": False,
        "execution_git_commit": collect_git_provenance(ROOT).commit,
        "model_id": FIXED_MODEL_ID, "model_revision": FIXED_REVISION,
        "processor_revision": FIXED_REVISION, "selected_adapter": str(adapter),
        "selected_adapter_loaded": True, "config": asdict(config),
        "transformers_version": runtime.package_version("transformers"),
        "peft_version": runtime.package_version("peft"),
        "hidden_size": model.config.text_config.hidden_size, "freeze_policy": freeze,
        "teacher_forced_training": training, "predicted_action_inference": predicted,
        "train_records_validated": len(records), "train_sample_count": len(selected),
        "validation_sample_count": 0, "predicted_action_sample_count": 1,
        "optimizer_steps": 0, "test_records_read": 0, **asdict(IsolationCounters()),
        "test_evaluation_performed": False,
    }
    output.mkdir(parents=True)
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0.4c-1 two-turn continuous waypoint smoke.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_4c_planner_smoke.yaml")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--dry-run", action="store_true", help="show config without data or model access")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_no_data_or_model_access", "config": asdict(config)}, indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error("set NUSCENES_ROOT and VLA_DERIVED_ROOT or provide root arguments")
    result = run_smoke(dataset_root=args.dataset_root, derived_root=args.derived_root, config=config)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
