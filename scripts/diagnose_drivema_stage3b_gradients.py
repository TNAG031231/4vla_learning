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
from src.phase0.drivema_stage3b import GRPOConfig
from src.phase0.drivema_stage3b_gradient_diagnostic import diagnose_gradients
from src.phase0.phase0_4b_protocol import StructuredCollator
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, resolve_derived_path
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Fixed Step 1 rollout: Old TF vs Generation LoRA gradients; no update.')
    parser.add_argument('--rollout', type=Path, required=True, help='Existing rollout_1.json; sibling metadata.json required')
    parser.add_argument('--output', type=Path, required=True, help='New diagnostic JSON outside the repository')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    args = parser.parse_args(argv)
    if args.dataset_root is None or args.derived_root is None:
        parser.error('provide dataset/derived roots')
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError('diagnostic output must be a new file outside the repository')
    group = json.loads(args.rollout.read_text())
    metadata_path = args.rollout.parent / 'metadata.json'
    source = json.loads(metadata_path.read_text())
    config = GRPOConfig(**source['config'])
    rollout = load_rollout_config(ROOT / config.rollout_config)
    if (source['model_id'] != FIXED_MODEL_ID or source['revision'] != FIXED_REVISION
            or source['resume'] is not None or source['identity']['rollout'] != asdict(rollout)
            or group['optimizer_step'] != 1 or group['split'] != 'train'
            or group['sample_token'] not in source['identity']['sample_tokens']):
        raise ValueError('requires original SFT Step 1 rollout with matching model/rollout metadata')
    samples, _, intake = prepare_train(ROOT, args.derived_root)
    sample = next(s for s in samples if s.sample_token == group['sample_token'])
    checkpoint = resolve_derived_path(args.derived_root, rollout.checkpoint_relative_path)
    from peft import PeftModel
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('BF16-capable GPU required')
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, 'sdpa', True)
    model = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=True)
    model.load_adapter(checkpoint / 'adapter', adapter_name='reference', is_trainable=False)
    model.to(device)
    collator = StructuredCollator(processor, runtime.image_loader, args.dataset_root)
    runner = Stage2Runner(processor, collator, device, {}, rollout.trajectory_max_new_tokens)
    result = diagnose_gradients(model, runner, sample, runner.images(sample), group, config)
    result['metadata'] = {
        'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION,
        'initial_sft_checkpoint': str(checkpoint), 'input_rollout': str(args.rollout.resolve()),
        'input_metadata': str(metadata_path.resolve()), 'source_metadata': source,
        'config': asdict(config), 'data_intake': intake,
        'git_commit': collect_git_provenance(ROOT).commit, 'dtype': str(dtype), 'attention': 'sdpa',
        'torch': torch.__version__, 'transformers': runtime.package_version('transformers'),
        'peft': runtime.package_version('peft'), 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None,
        'denominators': {'A': 'independent no-grad old TF', 'B': 'saved Generation log-prob'},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps({'output': str(output), 'comparison': result['comparison'], 'checks': result['checks']},
                     allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
