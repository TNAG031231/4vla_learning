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
from src.phase0.drivema_stage3a import load_config as load_rollout_config, prepare_train
from src.phase0.drivema_stage3b import fit, load_config
from src.phase0.phase0_4b_protocol import StructuredCollator
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, resolve_derived_path
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Stage 3B online LoRA GRPO GPU smoke; at most two updates.')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/drivema_stage3b_smoke.yaml')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    rollout = load_rollout_config(ROOT / config.rollout_config)
    if args.dry_run:
        print(json.dumps({'training': asdict(config), 'rollout': asdict(rollout)}, indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error('provide dataset/derived roots')
    output = resolve_derived_path(args.derived_root, config.output_relative_dir)
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError('smoke output must be a new directory outside the repository')
    samples, records, intake = prepare_train(ROOT, args.derived_root)
    if len(samples) < rollout.train_samples:
        raise ValueError('insufficient eligible train observations')
    samples = samples[:rollout.train_samples]
    checkpoint = resolve_derived_path(args.derived_root, rollout.checkpoint_relative_path)
    from peft import PeftModel
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('BF16-capable GPU required')
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, 'sdpa', True)
    current = (args.resume or checkpoint) / 'adapter'
    reference = args.resume / 'adapter/reference' if args.resume else checkpoint / 'adapter'
    model = PeftModel.from_pretrained(base, current, is_trainable=True)
    model.load_adapter(reference, adapter_name='reference', is_trainable=False)
    model.to(device)
    collator = StructuredCollator(processor, runtime.image_loader, args.dataset_root)
    runner = Stage2Runner(processor, collator, device, {}, rollout.trajectory_max_new_tokens)
    fit(model=model, runner=runner, samples=samples, records=records, rollout_config=rollout,
        config=config, output=output, resume=args.resume, metadata={
            'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION,
            'initial_sft_checkpoint': str(checkpoint), 'data_intake': intake,
            'git_commit': collect_git_provenance(ROOT).commit, 'dtype': str(dtype), 'attention': 'sdpa',
            'torch': torch.__version__, 'transformers': runtime.package_version('transformers'),
            'peft': runtime.package_version('peft'),
        })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
