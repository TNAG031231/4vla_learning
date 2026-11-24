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
from peft import PeftModel
from transformers import GenerationConfig
import yaml

from src.phase0.drivema_longitudinal_branches import (
    DIRECTIONS, DRAWS, evaluate_pair, pair_differences, select_pairs, summarize_pairs,
)
from src.phase0.drivema_stage3_rewards import RewardConfig
from src.phase0.drivema_visual_ablation import select_frozen_adapter
from src.phase0.phase0_4b_protocol import StructuredCollator, adapt_record
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, load_config, resolve_derived_path, sha256_file
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies

LOW_TRAIN = 'drivema/parallel_coverage_run01/worker_a_continuation'
OUTPUT = 'drivema/low512_longitudinal_branches_run01'
SEED = 20261124


def prepare(derived: Path, per_stratum: int) -> tuple[dict, dict, dict, dict]:
    meta = json.loads((resolve_derived_path(derived, LOW_TRAIN) / 'metadata.json').read_text())
    identity = meta['identity']
    schedule = identity['observation_sequence']
    if (identity['experiment']['arm'] != 'repeat256' or len(schedule) != 512
            or len({r['sample_token'] for r in schedule}) != 256
            or (meta['model_id'], meta['revision']) != (FIXED_MODEL_ID, FIXED_REVISION)):
        raise ValueError('expected the frozen Low-512 training run')
    temporal = yaml.safe_load((ROOT / 'configs/phase0_4_temporal_dataset.yaml').read_text())
    manifest = resolve_derived_path(derived, temporal['output_relative_dir']) / 'train.jsonl'
    source = identity['data_intake']['manifests']['train']
    if manifest != resolve_derived_path(derived, source['relative_path']) or sha256_file(manifest) != source['sha256']:
        raise ValueError('train manifest differs from checkpoint intake')
    visited = {s['sample_token'] for s in schedule}
    records = {}
    with manifest.open() as stream:
        for row in map(json.loads, stream):
            if row['sample_token'] in visited:
                records[row['sample_token']] = row
    plan = select_pairs(records, schedule, per_stratum, SEED)
    return meta, identity, records, plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare forced longitudinal actions on Low-512 training observations.')
    parser.add_argument('stage', choices=('plan', 'run'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--local-model-dir', type=Path, default=os.environ.get('QWEN_LOCAL_DIR'))
    parser.add_argument('--output-relative-dir', default=OUTPUT)
    parser.add_argument('--per-stratum', type=int, default=8)
    parser.add_argument('--near-tie-g', type=float, default=.02, help='absolute reward difference, inclusive')
    args = parser.parse_args(argv)
    if args.derived_root is None or (args.stage == 'run' and args.dataset_root is None):
        parser.error('provide a derived root for plan and both derived/dataset roots for run')
    if not 0 <= args.near_tie_g <= 1:
        parser.error('near-tie-g must lie within [0,1]')
    meta, identity, records, plan = prepare(args.derived_root, args.per_stratum)
    if args.stage == 'plan':
        print(json.dumps(plan, indent=2), flush=True)
        return 0
    output = resolve_derived_path(args.derived_root, args.output_relative_dir)
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('output must be a new directory outside the repository')
    checkpoint = resolve_derived_path(args.derived_root, LOW_TRAIN) / 'checkpoint_512'
    state = torch.load(checkpoint / 'unified_state.pt', map_location='cpu', weights_only=True)
    if state['step'] != 512 or state['identity'] != identity:
        raise ValueError('loaded checkpoint differs from Low-512 training intake')
    del state
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16 or meta['dtype'] != str(dtype) or meta['attention'] != 'sdpa':
        raise ValueError('Low-512 requires BF16 SDPA')
    source = str(args.local_model_dir.resolve(strict=True)) if args.local_model_dir else FIXED_MODEL_ID
    processor = runtime.processor_loader(source, FIXED_REVISION, True)
    base = runtime.model_loader(source, FIXED_REVISION, dtype, 'sdpa', True)
    model = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=False).to(device)
    select_frozen_adapter(model, 'default')
    collator = StructuredCollator(processor, runtime.image_loader, args.dataset_root)
    rollout = identity['rollout']
    sampling = GenerationConfig(
        **{k: rollout[k] for k in ('do_sample', 'temperature', 'top_k', 'top_p')},
        num_beams=1, max_new_tokens=rollout['trajectory_max_new_tokens'], use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
    greedy_max_tokens = identity['validation']['trajectory_max_new_tokens']
    reward = RewardConfig(**rollout['reward'])
    ego = load_config(ROOT / 'configs/phase0_3_dataset_adapter.yaml')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    write_json(output / 'metadata.json', {
        'mode': 'train_forced_longitudinal_branch_diagnostic', 'checkpoint': str(checkpoint),
        'step': 512, 'source_commit': meta['git_commit'], 'git_commit': collect_git_provenance(ROOT).commit,
        'model_source': source, 'model_revision': FIXED_REVISION, 'dtype': str(dtype), 'attention': 'sdpa',
        'train_manifest': identity['data_intake']['manifests']['train'], 'sampling': sampling.to_dict(),
        'greedy_max_tokens': greedy_max_tokens, 'reward': rollout['reward'],
        'seeds': 'distinct across observations/batches/draws; paired across actions within each draw',
        'positive_difference': 'alternative better than keep; by mean sampled g or by lower greedy ADE',
        'near_tie_g': args.near_tie_g,
    })
    rows = []
    with (output / 'pairs.jsonl').open('x', buffering=1) as stream:
        for i, pair in enumerate(plan['pairs']):
            record = records[pair['sample_token']]
            sample = adapt_record(record, ego, expected_split='train')
            messages = collator.messages(sample, expected_split='train')
            branches = evaluate_pair(model, processor, messages, sample, record, pair['alternative'],
                                     SEED + i * 2 * DRAWS, device, sampling, greedy_max_tokens, reward)
            row = {**pair, 'split': 'train', 'branches': branches,
                   'differences': pair_differences(branches, pair['alternative'])}
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            print(f"Completed longitudinal pair {i + 1}/{plan['observations']}", flush=True)
    result = {'overall': summarize_pairs(rows, args.near_tie_g), 'strata': {}}
    for alternative in DIRECTIONS:
        for target in ('keep', alternative):
            subset = [r for r in rows if r['alternative'] == alternative and r['target_longitudinal'] == target]
            result['strata'][f'keep_vs_{alternative}__target_{target}'] = summarize_pairs(subset, args.near_tie_g)
    write_json(output / 'summary.json', result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
