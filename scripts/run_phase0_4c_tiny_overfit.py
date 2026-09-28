#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.phase0.development_projection import IsolationCounters
from src.phase0.phase0_4b_lora_full import load_config as load_semantic_config
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4c_tiny_overfit import (
    TinyOverfitConfig, cache_contexts, fit_cached_contexts, load_config, select_training_samples, write_json,
)
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder, freeze_backbone
from src.phase0.qwen3vl_dataset_adapter import (
    collect_git_provenance, load_config as load_ego_config, resolve_derived_path,
)
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


def run(*, dataset_root: Path, derived_root: Path, config: TinyOverfitConfig,
        split: str = "train") -> dict:
    if split != "train":
        raise ValueError("Phase 0.4c-2 only permits train")
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(ROOT):
        raise ValueError("tiny-overfit artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"tiny-overfit output already exists: {output}")
    semantic = load_semantic_config(ROOT / "configs/phase0_4b_lora_full.yaml")
    records = load_temporal_records(ROOT, derived_root, split="train")
    samples = select_training_samples(
        records, load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml"), config,
    )
    selected_tokens = {sample.sample_token for sample in samples}
    targets = {row["sample_token"]: row for row in records if row["sample_token"] in selected_tokens}
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
    hidden_size = model.config.text_config.hidden_size
    planner = WaypointDecoder(hidden_size, config)
    counts = {
        "qwen_trainable_parameters": sum(p.numel() for name, p in model.named_parameters()
                                          if "lora_" not in name and p.requires_grad),
        "lora_trainable_parameters": sum(p.numel() for name, p in model.named_parameters()
                                          if "lora_" in name and p.requires_grad),
        "lora_parameter_tensors": sum("lora_" in name for name, _ in model.named_parameters()),
        "planner_trainable_parameters": sum(p.numel() for p in planner.parameters() if p.requires_grad),
    }
    print(json.dumps({"event": "selected_adapter_loaded", **counts}), flush=True)
    if (counts["qwen_trainable_parameters"] or counts["lora_trainable_parameters"]
            or not counts["lora_parameter_tensors"] or counts["planner_trainable_parameters"] != 2_764_546):
        raise ValueError("frozen Qwen/LoRA or Phase 0.4c-1 planner parameter contract mismatch")
    del planner
    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", asdict(config))
    contexts, evidence = cache_contexts(
        model=model, processor=processor, samples=samples, records=targets,
        dataset_root=dataset_root, device=device, runtime=runtime,
    )
    write_json(output / "tiny_subset.json", evidence)
    provenance = {
        "run_kind": "real_frozen_qwen_gt_action_tiny_train_overfit",
        "execution_git_commit": collect_git_provenance(ROOT).commit,
        "model_id": FIXED_MODEL_ID, "model_revision": FIXED_REVISION,
        "processor_revision": FIXED_REVISION, "selected_adapter": str(adapter),
        "selected_adapter_loaded": True, "freeze_policy": counts,
        "qwen_and_lora_frozen": all(not p.requires_grad and p.grad is None for p in model.parameters()),
        "train_records_validated": len(records), "validation_sample_count": 0,
        "validation_evaluation_performed": False, "test_evaluation_performed": False,
        "test_records_read": 0, **asdict(IsolationCounters()),
        "context_cache_policy": "in_process_cpu_bfloat16_for_fixed_context_frozen_backbone_tiny_overfit_only",
        "transformers_version": runtime.package_version("transformers"),
        "peft_version": runtime.package_version("peft"),
    }
    del model, base, processor
    torch.cuda.empty_cache()
    return fit_cached_contexts(contexts=contexts, config=config, hidden_size=hidden_size,
                               device=device, output=output, provenance=provenance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0.4c-2 eight-train-sample planner tiny overfit.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_4c_tiny_overfit.yaml")
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
    result = run(dataset_root=args.dataset_root, derived_root=args.derived_root, config=config, split=args.split)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "tiny_overfit_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
