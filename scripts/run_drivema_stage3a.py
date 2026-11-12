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

from src.phase0.drivema_stage2_sft import Stage2Runner
from src.phase0.drivema_stage3a import (
    generation_configs, group_markdown, load_config, prepare_train, rollout_group,
)
from src.phase0.drivema_stage3_rewards import summarize_groups
from src.phase0.phase0_4b_protocol import StructuredCollator
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, resolve_derived_path
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Stage 3A train-only group rollout; no optimizer updates.')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/drivema_stage3a.yaml')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(asdict(config), indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error('provide dataset/derived roots')
    output = resolve_derived_path(args.derived_root, config.output_relative_dir)
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError('rollout output must be a new directory outside the repository')
    samples, records, intake = prepare_train(ROOT, args.derived_root)
    if len(samples) < config.train_samples:
        raise ValueError('insufficient eligible training observations')
    selected = samples[:config.train_samples]
    checkpoint = resolve_derived_path(args.derived_root, config.checkpoint_relative_path)
    from peft import PeftModel
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('BF16-capable GPU required')
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, 'sdpa', True)
    model = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=False).to(device).eval()
    collator = StructuredCollator(processor, runtime.image_loader, args.dataset_root)
    runner = Stage2Runner(processor, collator, device, {}, config.trajectory_max_new_tokens)
    torch.manual_seed(config.seed)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metadata.json', {
        'config': asdict(config), 'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION,
        'policy_checkpoint': str(checkpoint), 'policy_role': 'frozen_sft_behavior_policy',
        'git_commit': collect_git_provenance(ROOT).commit, 'seed': config.seed,
        'torch': torch.__version__, 'transformers': runtime.package_version('transformers'),
        'peft': runtime.package_version('peft'), 'dtype': str(dtype), 'attention': 'sdpa',
        'generation_config': [g.to_dict() for g in generation_configs(processor, config)],
        'use_model_defaults': False,
        'sample_tokens': [s.sample_token for s in selected], 'data_intake': intake,
        'optimizer_updates': 0,
    })
    groups = []
    with (output / 'groups.jsonl').open('w') as stream:
        for sample in selected:
            group = rollout_group(model=model, processor=processor, sample=sample, images=runner.images(sample),
                                  record=records[sample.sample_token], config=config, device=device)
            stream.write(json.dumps(group, allow_nan=False) + '\n')
            stream.flush()
            groups.append(group)
            print(json.dumps({'sample_token': sample.sample_token, **group['statistics']}), flush=True)
    write_json(output / 'summary.json', summarize_groups(groups))
    (output / 'example.md').write_text(group_markdown(groups[0]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
