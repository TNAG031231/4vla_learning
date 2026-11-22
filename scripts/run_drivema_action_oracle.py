#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
import yaml

from src.phase0.drivema_action_oracle import intervention_action_ids, replay_turn2
from src.phase0.drivema_action_oracle_metrics import summarize
from src.phase0.drivema_grpo_evaluation import load_baseline_predictions
from src.phase0.drivema_visual_ablation import select_frozen_adapter
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4b_protocol import SFTSample, StructuredCollator, parse_action
from src.phase0.phase0_4c_evaluation import prediction_metrics
from src.phase0.phase0_4c_full_train import eligible_samples
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, load_config, resolve_derived_path, sha256_file
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION, validate_processor_inputs
from src.phase0.qwen3vl_lora_smoke import _move_batch, default_runtime_dependencies


def wrong_actions(samples: list[SFTSample], predictions: list[dict]) -> list[tuple[SFTSample, dict]]:
    selected = []
    for sample, row in zip(samples, predictions, strict=True):
        if not (sample.target.longitudinal_valid and sample.target.lateral_valid):
            raise ValueError(f'joint action target unavailable: {sample.sample_token}')
        if not (row['prediction_valid'] and row['conditioning_type'] == 'predicted_action'
                and row['fde_3s_m'] is not None and row['parsed_action'] is not None
                and parse_action(row['action_context']) == row['parsed_action']):
            raise ValueError(f'invalid predicted-action or 3-second control: {sample.sample_token}')
        if row['action_context'] != sample.target.training_text():
            selected.append((sample, row))
    return selected


def evaluate(selected: list[tuple[SFTSample, dict]], collator: StructuredCollator,
             model: torch.nn.Module, device: str, max_tokens: int, output: Path) -> list[dict]:
    processor = collator.processor
    rows = []
    control_types = set()
    started = time.monotonic()
    with (output / 'pairs.jsonl').open('x', buffering=1) as stream:
        for index, (sample, saved) in enumerate(selected, start=1):
            messages = collator.messages(sample, expected_split='validation')
            first = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
            validate_processor_inputs(first, expected_batch_size=1,
                                      expected_image_count=len(sample.observation.image_paths))
            first = _move_batch(first, device)
            oracle_action = {'longitudinal': sample.target.longitudinal, 'lateral': sample.target.lateral}
            changed = tuple(axis for axis in oracle_action if oracle_action[axis] != saved['parsed_action'][axis])
            control = changed not in control_types
            if control:
                action, trajectory = saved['assistant_spans']
                replay = replay_turn2(model, processor, first, saved, action['token_ids'], max_tokens)
                if replay['trajectory_ids'] != trajectory['token_ids']:
                    raise ValueError(f"predicted-action replay differs from archived trajectory: {sample.sample_token}")
                control_types.add(changed)
            action_ids = intervention_action_ids(processor, sample)
            oracle = replay_turn2(model, processor, first, saved, action_ids, max_tokens)
            points = oracle['waypoints']
            metrics = prediction_metrics(
                torch.tensor(points) if points is not None else None,
                torch.tensor(saved['target_waypoints']), torch.tensor(saved['trajectory_valid_mask']))
            row = {
                'sample_token': sample.sample_token, 'scene_token': sample.scene_token, 'split': 'validation',
                'conditioning_type': 'gt_action_diagnostic', 'predicted_action': saved['parsed_action'],
                'oracle_action': oracle_action, 'changed_axes': changed, 'control_replayed': control,
                'baseline_waypoints': saved['predicted_waypoints'], 'oracle_waypoints': points,
                'target_waypoints': saved['target_waypoints'], 'trajectory_valid_mask': saved['trajectory_valid_mask'],
                'oracle_trajectory_text': oracle['trajectory_text'], 'invalid_reason': oracle['invalid_reason'],
                'baseline_ade_3s_m': saved['ade_3s_m'], 'baseline_fde_3s_m': saved['fde_3s_m'],
                'oracle_ade_3s_m': metrics.get('ade_3s_m'), 'oracle_fde_3s_m': metrics.get('fde_3s_m'),
            }
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            if index == 1 or index % 16 == 0 or index == len(selected):
                remaining = (len(selected) - index) * (time.monotonic() - started) / index
                print(json.dumps({'completed': index, 'total': len(selected),
                                  'remaining_minutes': round(remaining / 60, 1)}), flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validation-only second-turn GT-action intervention on saved joint mistakes.')
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--local-model-dir', type=Path, default=os.environ.get('QWEN_LOCAL_DIR'))
    parser.add_argument('--full-validation-relative-dir', required=True)
    parser.add_argument('--output-relative-dir', required=True)
    args = parser.parse_args(argv)
    if args.dataset_root is None or args.derived_root is None:
        parser.error('provide dataset and derived roots')
    full = resolve_derived_path(args.derived_root, args.full_validation_relative_dir)
    output = resolve_derived_path(args.derived_root, args.output_relative_dir)
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('output must be a new directory outside the repository')
    meta = json.loads((full / 'metadata.json').read_text())
    if (meta['mode'], meta['model_id'], meta['revision']) != ('evaluate-full', FIXED_MODEL_ID, FIXED_REVISION):
        raise ValueError('expected full validation of the frozen Qwen base and revision')
    checkpoint = resolve_derived_path(args.derived_root, meta['checkpoint'])
    state = torch.load(checkpoint / 'unified_state.pt', map_location='cpu', weights_only=True)
    if state['step'] != meta['step'] or state['identity'] != meta['identity']:
        raise ValueError('checkpoint and full-validation run differ')
    inference = meta['identity']['validation']
    if not (inference['optimized'] and inference['batch_size'] == 1
            and not inference['action_generation']['do_sample'] and inference['action_generation']['num_beams'] == 1):
        raise ValueError('expected optimized batch-one greedy control predictions')
    temporal = yaml.safe_load((ROOT / 'configs/phase0_4_temporal_dataset.yaml').read_text())
    manifest = resolve_derived_path(args.derived_root, temporal['output_relative_dir']) / 'validation.jsonl'
    source_manifest = meta['identity']['data_intake']['manifests']['validation']
    if (manifest != resolve_derived_path(args.derived_root, source_manifest['relative_path'])
            or sha256_file(manifest) != source_manifest['sha256']):
        raise ValueError('validation manifest differs from checkpoint intake')
    records = load_temporal_records(ROOT, args.derived_root, split='validation')
    samples, _ = eligible_samples(records, load_config(ROOT / 'configs/phase0_3_dataset_adapter.yaml'), 'validation')
    path = full / f"evaluate-full_{state['step']}_predictions.jsonl"
    predictions = load_baseline_predictions(path, samples, {r['sample_token']: r for r in records})
    selected = wrong_actions(samples, predictions)
    del records, predictions, state
    print(json.dumps({'validation': len(samples), 'wrong_actions': len(selected),
                      'checkpoint': str(checkpoint), 'output': str(output)}), flush=True)
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16 or meta['dtype'] != str(dtype) or meta['attention'] != 'sdpa':
        raise ValueError('diagnostic requires the BF16 SDPA validation configuration')
    source = (str(args.local_model_dir.resolve(strict=True)) if args.local_model_dir
              else meta.get('model_source', meta['model_id']))
    processor = runtime.processor_loader(source, FIXED_REVISION, True)
    base = runtime.model_loader(source, FIXED_REVISION, dtype, 'sdpa', True)
    model = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=False).to(device)
    select_frozen_adapter(model, 'default')
    collator = StructuredCollator(processor, runtime.image_loader, args.dataset_root)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metadata.json', {
        'mode': 'validation_gt_action_diagnostic', 'checkpoint': str(checkpoint), 'step': meta['step'],
        'source_predictions': str(path), 'source_git_commit': meta['git_commit'],
        'git_commit': collect_git_provenance(ROOT).commit, 'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION,
        'model_source': source, 'dtype': str(dtype), 'attention': 'sdpa', 'inference': inference,
        'validation_count': len(samples), 'scene_count': len({s.scene_token for s in samples}),
        'wrong_action_count': len(selected), 'delta': 'GT action minus original predicted action',
        'control': 'original full-validation predictions; replay first wrong sample per changed-axis pattern',
    })
    rows = evaluate(selected, collator, model, device, inference['trajectory_max_new_tokens'], output)
    result = summarize(rows, len(samples))
    write_json(output / 'summary.json', result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
