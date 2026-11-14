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

from src.phase0.drivema_grpo_train import (
    evaluate_and_save, fit, load_config, prepare_data, run_identity, validate_config,
)
from src.phase0.drivema_stage2_sft import Stage2Runner
from src.phase0.drivema_stage2b import select_subset
from src.phase0.drivema_stage3a import load_config as load_rollout_config
from src.phase0.phase0_4b_lora_full import load_config as load_action_config
from src.phase0.phase0_4b_protocol import StructuredCollator
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
    parser.add_argument('--resume', type=Path, help='GRPO checkpoint; restores current/reference and training state')
    parser.add_argument('--checkpoint', type=Path, help='GRPO checkpoint for evaluation-only modes')
    parser.add_argument('--max-steps', type=int, help='cumulative optimizer step budget, not additional steps')
    parser.add_argument('--evaluation-steps', nargs='+', type=int)
    parser.add_argument('--output-relative-dir', help='new directory relative to derived root, including on resume')
    parser.add_argument('--dry-run', action='store_true', help='validate configuration only; no data/model loading')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    overrides = {key: getattr(args, key) for key in ('max_steps', 'evaluation_steps', 'output_relative_dir')
                 if getattr(args, key) is not None}
    config = replace(config, **overrides)
    validate_config(config)
    rollout = load_rollout_config(ROOT / config.rollout_config)
    if args.mode == 'train' and args.checkpoint is not None:
        parser.error('training uses --resume, not --checkpoint')
    if args.mode != 'train' and (args.checkpoint is None or args.resume is not None):
        parser.error('evaluation requires --checkpoint and no --resume')
    if args.dry_run:
        print(json.dumps({'training': asdict(config), 'rollout': asdict(rollout),
                          'train_selection': 'all eligible train records; rollout.train_samples is unused',
                          'mode': args.mode}, indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error('provide dataset/derived roots')
    output = resolve_derived_path(args.derived_root, config.output_relative_dir)
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError('output must be a new directory outside the repository')
    train, validation, records, intake = prepare_data(ROOT, args.derived_root)
    print(json.dumps({'data_intake': intake, 'full_train_selected': len(train)}, indent=2), flush=True)
    checkpoint = args.resume or args.checkpoint
    initial = resolve_derived_path(args.derived_root, rollout.checkpoint_relative_path)
    from peft import PeftModel
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('BF16-capable GPU required')
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, 'sdpa', True)
    model = PeftModel.from_pretrained(base, (checkpoint or initial) / 'adapter', is_trainable=True)
    reference = checkpoint / 'adapter/reference' if checkpoint else initial / 'adapter'
    model.load_adapter(reference, adapter_name='reference', is_trainable=False)
    model.to(device)
    semantic = load_action_config(ROOT / 'configs/phase0_4b_lora_full.yaml')
    runner = Stage2Runner(processor, StructuredCollator(processor, runtime.image_loader, args.dataset_root),
                          device, semantic.generation_kwargs, rollout.trajectory_max_new_tokens,
                          optimized=True, compute_log_probs=False)
    metadata = {'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION,
                'initial_sft_checkpoint': rollout.checkpoint_relative_path, 'data_intake': intake,
                'git_commit': collect_git_provenance(ROOT).commit, 'dtype': str(dtype), 'attention': 'sdpa',
                'torch': str(torch.__version__), 'transformers': runtime.package_version('transformers'),
                'peft': runtime.package_version('peft')}
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
        if args.mode == 'evaluate-full':
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
