#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.phase0.drivema_stage2_sft import Stage2Runner, evaluate
from src.phase0.drivema_stage2b import load_config
from src.phase0.phase0_4b_lora_full import load_config as load_action_config
from src.phase0.phase0_4b_protocol import StructuredCollator
from src.phase0.phase0_4c_full_train import prepare_data, write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


def compare_predictions(baseline: list[dict], optimized: list[dict], before: dict, after: dict) -> dict:
    mismatches = []
    fields = ("sample_token", "scene_token", "action_context", "parsed_action", "trajectory_text",
              "input_ids", "invalid_reason", "predicted_waypoints", "target_waypoints", "trajectory_valid_mask")
    for index, (left, right) in enumerate(zip(baseline, optimized, strict=True)):
        changed = [field for field in fields if left[field] != right[field]]
        spans = [[{key: span[key] for key in ("turn", "start", "end", "token_ids")}
                  for span in row["assistant_spans"]] for row in (left, right)]
        if spans[0] != spans[1]:
            changed.append("assistant_spans")
        if changed:
            mismatches.append({"index": index, "sample_token": left["sample_token"], "fields": changed})
    return {"passed": not mismatches and before == after,
            "sample_count": len(baseline), "mismatches": mismatches, "metrics_exact_match": before == after,
            "invalid_count_match": before["trajectory"]["invalid_prediction_count"] ==
                                   after["trajectory"]["invalid_prediction_count"]}


def summarize(performance: list[dict], elapsed: float, metrics: dict, peak: int | None) -> dict:
    latencies = [item["total_seconds"] for item in performance for _ in item["sample_tokens"]]
    return {"total_inference_seconds": elapsed, "samples_per_second": len(latencies) / elapsed,
            "p50_latency_seconds": float(torch.tensor(latencies, dtype=torch.float64).quantile(.50)),
            "p95_latency_seconds": float(torch.tensor(latencies, dtype=torch.float64).quantile(.95)),
            "latency_definition": "per_sample_batch_completion; batch latency repeated for each member; no queue wait",
            "peak_allocated_bytes": peak, "metrics": metrics,
            "phase_seconds": {name: sum(item.get(name, 0.) for item in performance)
                              for name in ("turn1_seconds", "turn2_seconds", "log_prob_seconds")},
            "generated_tokens": {turn: sum(item["generated_tokens"][turn] for item in performance)
                                 for turn in ("1", "2")}, "batches": performance}


def markdown_summary(result: dict) -> str:
    a, b = result["baseline"], result["optimized"]
    lines = ["# Stage 2C GPU Benchmark", "", f"Correctness gate: **{result['consistency']['passed']}**", "",
             "| Metric | Baseline | Optimized |", "| --- | --- | --- |"]
    for label, key in (("Total inference time (s)", "total_inference_seconds"),
                       ("Samples/s", "samples_per_second"), ("P50 latency (s)", "p50_latency_seconds"),
                       ("P95 latency (s)", "p95_latency_seconds"), ("Peak GPU allocated bytes", "peak_allocated_bytes")):
        lines.append(f"| {label} | {a[key]} | {b[key]} |")
    lines.append(f"| Valid trajectory rate | {1 - a['metrics']['trajectory']['invalid_prediction_rate']} | {1 - b['metrics']['trajectory']['invalid_prediction_rate']} |")
    for label, key in (("ADE@3s (m)", "ade_3s_m"),
                       ("FDE@3s (m)", "fde_3s_m"), ("Invalid count", "invalid_prediction_count")):
        lines.append(f"| {label} | {a['metrics']['trajectory'][key]} | {b['metrics']['trajectory'][key]} |")
    for key in ("longitudinal_macro_f1", "lateral_macro_f1", "joint_accuracy"):
        lines.append(f"| {key} | {a['metrics']['action'][key]} | {b['metrics']['action'][key]} |")
    lines += ["", f"Throughput speedup: {result['throughput_speedup']:.4f}x",
              f"End-to-end latency reduction: {result['latency_reduction_fraction']:.2%}",
              "", "Latency is batch completion time per sample, not batch time divided by batch size.",
              "Total time includes image loading, preprocessing, two turns, optional probability forward and evaluation; excludes model/data intake, warmup and output writes.",
              "Full action metrics, per-turn timings, token counts and mismatch locations are in benchmark.json.",
              "Cross-turn KV reuse: not implemented. Each turn uses its own Transformers cache.",
              "A failed correctness gate invalidates the speedup claim; inspect saved predictions before further use."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One fixed-validation paired GPU benchmark; no training or test intake.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="existing Stage 2B checkpoint directory containing adapter/")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/drivema_stage2b_full.yaml")
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--output", type=Path, required=True, help="new directory outside repository")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--warmup-samples", type=int, default=4)
    args = parser.parse_args(argv)
    if args.dataset_root is None or args.derived_root is None:
        parser.error("provide dataset/derived roots or set NUSCENES_ROOT and VLA_DERIVED_ROOT")
    if not 1 <= args.warmup_samples <= args.samples <= 3594:
        parser.error("require 1 <= warmup-samples <= samples <= 3594")
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("benchmark output must be a new directory outside repository")
    adapter = args.checkpoint / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError("checkpoint must contain an existing adapter/adapter_config.json")
    from peft import PeftModel
    config = load_config(args.config)
    semantic = load_action_config(ROOT / "configs/phase0_4b_lora_full.yaml")
    runtime = default_runtime_dependencies()
    device = runtime.device_selector("cuda:0")
    dtype = runtime.dtype_selector("bfloat16")
    if dtype != torch.bfloat16:
        raise ValueError("BF16-capable GPU required")
    _, validation, records, intake = prepare_data(ROOT, args.derived_root)
    selected = sorted(validation, key=lambda sample: sample.sample_token)[:args.samples]
    if len(selected) != args.samples:
        raise ValueError("insufficient fixed validation samples")
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, "sdpa", True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).to(device).eval()
    collator = StructuredCollator(processor, runtime.image_loader, args.dataset_root)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "model_id": FIXED_MODEL_ID,
                "revision": FIXED_REVISION, "git_commit": collect_git_provenance(ROOT).commit,
                "torch": torch.__version__, "transformers": runtime.package_version("transformers"),
                "peft": runtime.package_version("peft"), "python": platform.python_version(),
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                "dtype": str(dtype), "attention": "sdpa", "config": asdict(config),
                "action_generation": semantic.generation_kwargs, "data_intake": intake,
                "sample_tokens": [s.sample_token for s in selected], "batch_size": args.batch_size,
                "warmup_samples": args.warmup_samples, "cross_turn_cache_reuse": False,
                "baseline_generation_use_cache": model.generation_config.use_cache}
    write_json(output / "metadata.json", metadata)
    # Warm filesystem/image decoding for the same fixed population before either timer.
    for sample in selected:
        collator.messages(sample, expected_split="validation")
    tracks, predictions = {}, {}
    for name, optimized in (("baseline", False), ("optimized", True)):
        torch.manual_seed(config.seed)
        runner = Stage2Runner(processor, collator, device, semantic.generation_kwargs,
                              config.trajectory_max_new_tokens, inference_batch_size=args.batch_size,
                              optimized=optimized, compute_log_probs=not optimized)
        evaluate(model, runner, selected[:args.warmup_samples], records)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        performance = []
        started = time.perf_counter()
        rows, metrics = evaluate(model, runner, selected, records, performance=performance, progress_interval=16)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        tracks[name] = summarize(performance, elapsed, metrics, torch.cuda.max_memory_allocated(device))
        predictions[name] = rows
        write_predictions(output / f"{name}_predictions.jsonl", rows)
        write_json(output / f"{name}_metrics.json", tracks[name])
    result = {**tracks, "metadata": metadata,
              "consistency": compare_predictions(predictions["baseline"], predictions["optimized"],
                                                  tracks["baseline"]["metrics"], tracks["optimized"]["metrics"]),
              "throughput_speedup": tracks["optimized"]["samples_per_second"] / tracks["baseline"]["samples_per_second"],
              "latency_reduction_fraction": 1 - tracks["optimized"]["total_inference_seconds"] /
                                                tracks["baseline"]["total_inference_seconds"]}
    write_json(output / "benchmark.json", result)
    (output / "summary.md").write_text(markdown_summary(result))
    print(json.dumps({"output": str(output), "consistency": result["consistency"]}, indent=2))
    return 0 if result["consistency"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
