#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.phase0.phase0_4b_lora_full import load_config as load_semantic_config
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4c_collapse_diagnostic import diagnose
from src.phase0.phase0_4c_init_diagnostic import diagnose_initial_vs_trained
from src.phase0.phase0_4c_layer2_diagnostic import diagnose_layer2
from src.phase0.phase0_4c_qkv_diagnostic import diagnose_qkv, token_context
from src.phase0.phase0_4c_tiny_overfit import (
    TinyOverfitConfig, cache_contexts, load_config, select_training_samples, write_json,
)
from src.phase0.phase0_4c_two_turn_planner import PlannerConfig, WaypointDecoder, freeze_backbone
from src.phase0.qwen3vl_dataset_adapter import (
    collect_git_provenance, load_config as load_ego_config, resolve_derived_path,
)
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies

OUTPUT_RELATIVE_DIR = "phase_0_4/two_turn_planner_collapse_diagnostic_v0_3"
LAYER2_OUTPUT_RELATIVE_DIR = "phase_0_4/two_turn_planner_layer2_diagnostic_v0_3"
QKV_OUTPUT_RELATIVE_DIR = "phase_0_4/two_turn_planner_qkv_diagnostic_v0_3"
INIT_OUTPUT_RELATIVE_DIR = "phase_0_4/two_turn_planner_init_vs_trained_diagnostic_v0_3"


def run(*, dataset_root: Path, derived_root: Path, split: str = "train", probe: str = "collapse") -> dict:
    if split != "train":
        raise ValueError("collapse diagnostic only permits train")
    directories = {"collapse": OUTPUT_RELATIVE_DIR, "layer2": LAYER2_OUTPUT_RELATIVE_DIR,
                   "qkv": QKV_OUTPUT_RELATIVE_DIR, "init-vs-trained": INIT_OUTPUT_RELATIVE_DIR}
    if probe not in directories:
        raise ValueError("probe must be collapse, layer2, qkv or init-vs-trained")
    relative_dir = directories[probe]
    output = resolve_derived_path(derived_root, relative_dir)
    if output.is_relative_to(ROOT):
        raise ValueError("diagnostic artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    diagnostic_commit = collect_git_provenance(ROOT).commit
    config = load_config(ROOT / "configs/phase0_4c_tiny_overfit.yaml")
    source = resolve_derived_path(derived_root, config.output_relative_dir)
    checkpoint_path = source / "planner_state.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if TinyOverfitConfig(**saved["training_config"]) != config:
        raise ValueError("checkpoint training configuration differs from v0.3 configuration")
    records = load_temporal_records(ROOT, derived_root, split="train")
    samples = select_training_samples(
        records, load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml"), config,
    )
    if [s.sample_token for s in samples] != saved["tiny_sample_tokens"]:
        raise ValueError("reconstructed tiny subset differs from checkpoint sample order")
    targets = {row["sample_token"]: row for row in records
               if row["sample_token"] in saved["tiny_sample_tokens"]}
    semantic = load_semantic_config(ROOT / "configs/phase0_4b_lora_full.yaml")
    runtime = default_runtime_dependencies()
    device = runtime.device_selector("cuda:0")
    dtype = runtime.dtype_selector(semantic.precision)
    if dtype != torch.bfloat16:
        raise ValueError("selected Phase 0.4b model requires BF16 support")
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, semantic.local_files_only)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype,
                                semantic.attention_implementation, semantic.local_files_only)
    adapter = resolve_derived_path(derived_root, config.selected_adapter_relative_path)
    model = runtime.adapter_loader(base, adapter).to(device)
    freeze_backbone(model)
    if model.config.text_config.hidden_size != saved["hidden_size"]:
        raise ValueError("checkpoint and Qwen hidden sizes differ")
    captured_ids = []

    def capture_ids(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        captured_ids.append(kwargs["input_ids"][0].detach().cpu().tolist())

    handle = model.register_forward_pre_hook(capture_ids, with_kwargs=True) if probe == "qkv" else None
    try:
        contexts, evidence = cache_contexts(
            model=model, processor=processor, samples=samples, records=targets,
            dataset_root=dataset_root, device=device, runtime=runtime,
        )
    finally:
        if handle is not None:
            handle.remove()
    tokenizer = processor.tokenizer if probe == "qkv" else None
    token_contexts = None
    if probe == "qkv":
        token_contexts = {
            sample.sample_token: token_context(ids, tokenizer, sample.observation.frame_texts, item["action_token_ids"])
            for sample, ids, item in zip(samples, captured_ids, evidence, strict=True)
        }
    frozen = all(not p.requires_grad and p.grad is None for p in model.parameters())
    del model, base, processor
    torch.cuda.empty_cache()
    planner = WaypointDecoder(saved["hidden_size"], PlannerConfig(**saved["planner_config"])).to(device)
    planner.load_state_dict(saved["planner_state_dict"])
    if probe == "init-vs-trained":
        result = diagnose_initial_vs_trained(planner, contexts, config=config,
            hidden_size=saved["hidden_size"], recorded_path=source / "predictions_before.jsonl", device=device)
    elif probe == "qkv":
        result = diagnose_qkv(planner, contexts, device=device, token_contexts=token_contexts, tokenizer=tokenizer)
    else:
        result = (diagnose(planner, contexts, beta=config.smooth_l1_beta, device=device)
                  if probe == "collapse" else diagnose_layer2(planner, contexts, device=device))
    result["provenance"] = {
        "checkpoint": str(checkpoint_path), "checkpoint_provenance": saved["provenance"],
        "diagnostic_git_commit": diagnostic_commit,
        "model_id": FIXED_MODEL_ID, "model_revision": FIXED_REVISION,
        "selected_adapter": str(adapter), "qwen_and_lora_frozen": frozen,
        "conditioning_type": "gt_action_teacher_forced_train_diagnostic",
        "training_config": saved["training_config"], "contexts": evidence,
        "validation_evaluation_performed": False, "test_evaluation_performed": False,
        "test_records_read": 0, "checkpoint_written": False,
        "torch_version": str(torch.__version__),
        "transformers_version": runtime.package_version("transformers"),
    }
    output.mkdir(parents=True)
    write_json(output / "diagnostic.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only v0.3 checkpoint collapse diagnostic; no optimizer.")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--probe", choices=("collapse", "layer2", "qkv", "init-vs-trained"), default="collapse")
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    args = parser.parse_args(argv)
    if args.dataset_root is None or args.derived_root is None:
        parser.error("set NUSCENES_ROOT and VLA_DERIVED_ROOT or provide root arguments")
    result = run(dataset_root=args.dataset_root, derived_root=args.derived_root, split=args.split, probe=args.probe)
    if args.probe == "init-vs-trained":
        print(json.dumps({"status": result["status"], "interpretation": result["interpretation"],
                          "initialization_fidelity": result["initialization_fidelity"],
                          "comparison_performed": result["comparison_performed"],
                          "output": str(args.derived_root / INIT_OUTPUT_RELATIVE_DIR / "diagnostic.json")},
                         indent=2, allow_nan=False))
        return 0
    if args.probe == "qkv":
        print(json.dumps({
            "status": result["status"], "interpretation": result["interpretation"],
            "mask_accounting": [{"sample_token": s["sample_token"], **s["mask"]} for s in result["per_sample"]],
            "q_projection_weights": result["q_projection_weights"],
            "logits": result["logit_aggregate_over_samples_heads_queries"],
            "planner_state_unchanged": result["planner_state_unchanged"],
            "output": str(args.derived_root / QKV_OUTPUT_RELATIVE_DIR / "diagnostic.json"),
        }, indent=2, allow_nan=False))
        return 0
    if args.probe == "layer2":
        print(json.dumps({
            "status": result["status"], "norm_first": result["norm_first"],
            "planner_state_unchanged": result["planner_state_unchanged"],
            "execution_order": result["execution_order"], "stages": result["stage_aggregates"],
            "attention": result["attention_aggregate"],
            "attention_replay_max_absolute_difference": max(
                s["cross_attention"]["replay_output_max_absolute_difference"] for s in result["per_sample"]),
            "output": str(args.derived_root / LAYER2_OUTPUT_RELATIVE_DIR / "diagnostic.json"),
        }, indent=2, allow_nan=False))
        return 0
    print(json.dumps({
        "status": result["status"], "planner_state_unchanged": result["planner_state_unchanged"],
        "query_pairwise_l2": result["query_embeddings"]["off_diagonal_l2"],
        "stages": result["representations"]["stage_aggregates"],
        "context_deltas": [{"sample_token": row["sample_token"],
                            "zero": row["real_vs_zero_memory"], "other": row["real_vs_other_memory"]}
                           for row in result["representations"]["per_sample"]],
        "mean_dL_dx": result["backward"]["mean_dL_dpredicted_x_by_timestep"],
        "mean_dL_dy": result["backward"]["mean_dL_dpredicted_y_by_timestep"],
        "query_gradient_norms": result["backward"]["query_gradients"]["row_norms"],
        "output": str(args.derived_root / OUTPUT_RELATIVE_DIR / "diagnostic.json"),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
