#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.phase0.drivema_stage2b import fit, load_config, select_subset
from src.phase0.drivema_stage2_sft import Stage2Runner, evaluate
from src.phase0.phase0_4b_lora_full import load_config as load_action_config
from src.phase0.phase0_4b_protocol import StructuredCollator
from src.phase0.phase0_4c_full_train import prepare_data, write_predictions
from src.phase0.phase0_4c_no_action import aligned_metrics, compare
from src.baselines.ego_history_mlp import read_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, resolve_derived_path
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


def compare_baselines(rows: list[dict], records: dict, paths: list[Path]) -> dict:
    primary = 'stage2b'
    converted = [{**r, 'conditioning_type': primary,
                  'target_waypoints': torch.tensor(r['target_waypoints'], dtype=torch.float32).tolist()} for r in rows]
    models = {primary: aligned_metrics(converted, primary, records)}
    for path, track in zip(paths, ('predicted_action', 'no_action_neutral_context', 'ego_history_mlp'), strict=True):
        models[track] = aligned_metrics(read_predictions(path, track), track, records)
    return compare(models, sorted(records), primary_track=primary, primary_name=primary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Stage 2B pilot/full SFT; no test intake.')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/drivema_stage2b_pilot.yaml')
    parser.add_argument('--mode', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--output-relative-dir')
    parser.add_argument('--resume', type=Path, help='completed checkpoint directory; resume writes a new run directory')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--baseline-predictions', nargs=3, type=Path,
                        metavar=('WAYPOINT', 'NO_ACTION', 'MLP'), help='required for full final paired comparison')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.mode == 'pilot' and (config.train_samples != 512 or config.monitor_samples != 128):
        parser.error('pilot requires fixed 512 train / 128 validation')
    if args.mode == 'full':
        if args.epochs is None or args.learning_rate is None or not args.output_relative_dir or not args.baseline_predictions:
            parser.error('full requires Pilot-informed --epochs, --learning-rate, new --output-relative-dir and three --baseline-predictions')
        if args.epochs < 1 or not 0 < args.learning_rate < float('inf'):
            parser.error('epochs and learning rate must be positive and finite')
        config = replace(config, train_samples=None, epochs=args.epochs, learning_rate=args.learning_rate)
    elif args.epochs is not None or args.learning_rate is not None:
        parser.error('Pilot settings are frozen in its config')
    if args.output_relative_dir:
        config = replace(config, output_relative_dir=args.output_relative_dir)
    if args.dry_run:
        print(json.dumps(asdict(config), indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error('provide dataset and derived roots')
    output = resolve_derived_path(args.derived_root, config.output_relative_dir)
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError('output must be a new directory outside the repository')
    from peft import PeftModel
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('BF16 GPU required')
    train, validation, records, data_summary = prepare_data(ROOT, args.derived_root)
    if len(validation) != 3594:
        raise ValueError('full frozen validation must contain 3594 samples')
    train = select_subset(train, config.train_samples, config.seed, 'train')
    monitor = select_subset(validation, config.monitor_samples, config.seed, 'validation')
    torch.manual_seed(config.seed)
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, 'sdpa', True)
    adapter = (args.resume / 'adapter' if args.resume else
               resolve_derived_path(args.derived_root, config.selected_adapter_relative_path))
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    semantic = load_action_config(ROOT / 'configs/phase0_4b_lora_full.yaml')
    runner = Stage2Runner(processor, StructuredCollator(processor, runtime.image_loader, args.dataset_root),
                          device, semantic.generation_kwargs, config.trajectory_max_new_tokens)
    result = fit(model=model, runner=runner, train=train, monitor=monitor, records=records, config=config,
                 output=output, resume=args.resume, provenance={'data': data_summary,
                 'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION,
                 'git_commit': collect_git_provenance(ROOT).commit, 'mode': args.mode})
    if args.mode == 'full':
        selected = Path(result['best']['checkpoint']) / 'adapter'
        restored = PeftModel.from_pretrained(model.unload(), selected).to(device)
        rows, metrics = evaluate(restored, runner, validation, records)
        write_predictions(output / 'final_validation_predictions.jsonl', rows)
        write_json(output / 'final_validation_metrics.json', metrics)
        validation_records = {s.sample_token: records[s.sample_token] for s in validation}
        write_json(output / 'baseline_comparison.json', compare_baselines(rows, validation_records, args.baseline_predictions))
        result['full_validation'] = 'completed_3594_selected_checkpoint'
        write_json(output / 'training_summary.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
