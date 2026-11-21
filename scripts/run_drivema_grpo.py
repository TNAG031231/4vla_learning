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

from src.phase0.drivema_grpo_evaluation import (
    baseline_provenance, behavior_statistics, load_baseline_predictions,
)
from src.phase0.drivema_grpo_train import (
    evaluate_and_save, fit, load_config, prepare_data, run_identity, training_rollout_config, validate_config,
)
from src.phase0.drivema_grpo_replay import (
    load_observation_replay, load_observation_schedule, low512_replay, validate_replay_checkpoint,
    validate_replay_configuration,
)
from src.phase0.drivema_stage2_sft import Stage2Runner
from src.phase0.drivema_stage2b import select_subset
from src.phase0.phase0_4b_lora_full import load_config as load_action_config
from src.phase0.phase0_4b_protocol import StructuredCollator
from src.phase0.phase0_4b_evaluation import factorized_metrics
from src.phase0.phase0_4c_evaluation import aggregate_metrics
from src.phase0.phase0_4c_full_train import write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, resolve_derived_path
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Unified full-train GRPO; cumulative steps; explicit validation only.')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/drivema_grpo.yaml')
    parser.add_argument('--mode', choices=('train', 'evaluate-monitor', 'evaluate-full'), default='train')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--local-model-dir', type=Path, help='local copy of the frozen Stage 2B base model revision')
    parser.add_argument('--resume', type=Path, help='GRPO checkpoint; restores current/reference and training state')
    parser.add_argument('--checkpoint', type=Path, help='GRPO checkpoint for evaluation-only modes')
    parser.add_argument('--baseline-predictions', type=Path,
                        help='saved Stage 2B full validation JSONL; evaluate-full only; skips SFT inference')
    parser.add_argument('--max-steps', type=int, help='cumulative optimizer step budget, not additional steps')
    parser.add_argument('--evaluation-steps', nargs='+', type=int)
    parser.add_argument('--output-relative-dir', help='new directory relative to derived root, including on resume')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--observation-replay-archive', type=Path, help='original Stage 3D export ZIP')
    source.add_argument('--observation-schedule', type=Path, help='frozen experiment observation schedule JSON')
    source.add_argument('--low512-validation', type=Path, help='original repeat256 worker_a_full_validation directory')
    parser.add_argument('--dry-run', action='store_true', help='validate configuration only; no data/model loading')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    overrides = {key: getattr(args, key) for key in ('max_steps', 'evaluation_steps', 'output_relative_dir')
                 if getattr(args, key) is not None}
    config = replace(config, **overrides)
    validate_config(config)
    rollout = training_rollout_config(ROOT, config)
    replay = load_observation_replay(args.observation_replay_archive) if args.observation_replay_archive else None
    if args.observation_schedule:
        replay = load_observation_schedule(args.observation_schedule)
    if (config.objective != 'grpo') != (args.low512_validation is not None):
        parser.error('--low512-validation is required for the matched CE arms and applies only to them')
    if args.low512_validation is not None:
        replay = low512_replay(json.loads((args.low512_validation / 'metadata.json').read_text()))
    if config.consistency_weight != .5 and replay is None:
        parser.error('consistency ablation requires --observation-replay-archive or --observation-schedule')
    if replay is not None:
        validate_replay_configuration(asdict(config), rollout, replay)
    if args.mode == 'train' and args.checkpoint is not None:
        parser.error('training uses --resume, not --checkpoint')
    if args.mode != 'train' and (args.checkpoint is None or args.resume is not None):
        parser.error('evaluation requires --checkpoint and no --resume')
    if args.baseline_predictions is not None and args.mode != 'evaluate-full':
        parser.error('--baseline-predictions requires evaluate-full')
    if args.low512_validation is not None and args.mode == 'evaluate-full' and args.baseline_predictions is None:
        args.baseline_predictions = args.low512_validation / 'sft_full_predictions.jsonl'
    if args.dry_run:
        print(json.dumps({'training': asdict(config), 'rollout': asdict(rollout),
                          'train_selection': ('original step-by-step observation replay' if replay else
                                              'all eligible train records; rollout.train_samples is unused'),
                          'mode': args.mode,
                          'observation_replay': {'steps': len(replay['steps']),
                                                 'unique_observations': len({r['sample_token'] for r in replay['steps']}),
                                                 'source': replay['source_archive']} if replay else None}, indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error('provide dataset/derived roots')
    output = resolve_derived_path(args.derived_root, config.output_relative_dir)
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError('output must be a new directory outside the repository')
    train, validation, records, intake = prepare_data(ROOT, args.derived_root)
    print(json.dumps({'data_intake': intake, 'full_train_selected': len(train)}, indent=2), flush=True)
    checkpoint = args.resume or args.checkpoint
    if checkpoint is not None and replay is not None:
        state = torch.load(checkpoint / 'unified_state.pt', map_location='cpu', weights_only=True)
        validate_replay_checkpoint(state, replay, config.consistency_weight)
        if args.low512_validation is not None and args.mode == 'evaluate-full' and state['step'] != 512:
            parser.error('Low-512 full validation requires checkpoint step 512')
    initial = resolve_derived_path(args.derived_root, rollout.checkpoint_relative_path)
    semantic = load_action_config(ROOT / 'configs/phase0_4b_lora_full.yaml')
    baseline_source = None
    if args.baseline_predictions is not None:
        baseline = load_baseline_predictions(args.baseline_predictions, validation, records)
        baseline_source = baseline_provenance(
            args.baseline_predictions, rollout.checkpoint_relative_path, FIXED_MODEL_ID, FIXED_REVISION,
            {'trajectory_max_new_tokens': rollout.trajectory_max_new_tokens, 'batch_size': 1,
             'optimized': True, 'action_generation': semantic.generation_kwargs})
        baseline_metrics = {'action': factorized_metrics(baseline),
                            'trajectory': aggregate_metrics(baseline, 'predicted_action'),
                            'behavior': behavior_statistics(baseline)}
    from peft import PeftModel
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('BF16-capable GPU required')
    model_source = FIXED_MODEL_ID
    if args.local_model_dir is not None:
        model_source = str(args.local_model_dir.resolve(strict=True))
    processor = runtime.processor_loader(model_source, FIXED_REVISION, True)
    base = runtime.model_loader(model_source, FIXED_REVISION, dtype, 'sdpa', True)
    model = PeftModel.from_pretrained(base, (checkpoint or initial) / 'adapter', is_trainable=True)
    reference = checkpoint / 'adapter/reference' if checkpoint else initial / 'adapter'
    model.load_adapter(reference, adapter_name='reference', is_trainable=False)
    model.to(device)
    runner = Stage2Runner(processor, StructuredCollator(processor, runtime.image_loader, args.dataset_root),
                          device, semantic.generation_kwargs, rollout.trajectory_max_new_tokens,
                          optimized=True, compute_log_probs=False)
    metadata = {'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION, 'model_source': model_source,
                'initial_sft_checkpoint': rollout.checkpoint_relative_path, 'data_intake': intake,
                'git_commit': collect_git_provenance(ROOT).commit, 'dtype': str(dtype), 'attention': 'sdpa',
                'torch': str(torch.__version__), 'transformers': runtime.package_version('transformers'),
                'peft': runtime.package_version('peft')}
    if replay is not None:
        metadata['observation_replay'] = replay
    if args.mode == 'train':
        fit(model=model, runner=runner, train=train, validation=validation, records=records,
            rollout_config=rollout, config=config, output=output, metadata=metadata, resume=args.resume)
    else:
        state = torch.load(checkpoint / 'unified_state.pt', map_location='cpu', weights_only=True)
        monitor = select_subset(validation, config.monitor_samples, rollout.seed, 'validation')
        if state['identity'] != run_identity(config, rollout, train, monitor, runner, metadata):
            raise ValueError('evaluation model/data/monitor identity differs')
        output.mkdir(parents=True, exist_ok=False)
        population = validation if args.mode == 'evaluate-full' else monitor
        if args.baseline_predictions is not None:
            write_predictions(output / 'sft_full_predictions.jsonl', baseline)
            write_json(output / 'sft_full_metrics.json', baseline_metrics)
        elif args.mode == 'evaluate-full':
            baseline, _ = evaluate_and_save(model, runner, population, records, output, 'sft_full',
                                            adapter='reference')
        else:
            baseline = state['baseline']
        evaluate_and_save(model, runner, population, records, output,
                          f'{args.mode}_{state["step"]}', baseline)
        write_json(output / 'metadata.json', {**metadata, 'checkpoint': str(checkpoint),
                                            'step': state['step'], 'mode': args.mode,
                                            'population_tokens': [s.sample_token for s in population],
                                            'identity': state['identity']})
        if baseline_source is not None:
            write_json(output / 'baseline_source.json', baseline_source)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
