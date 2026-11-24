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
import yaml

from src.phase0.drivema_action_crossover import crossover_plan, decomposition, summarize_plan
from src.phase0.drivema_action_oracle import intervention_action_ids, replay_turn2
from src.phase0.drivema_action_oracle_metrics import TRAJECTORY_ERRORS
from src.phase0.drivema_visual_ablation import select_frozen_adapter
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4b_protocol import StructuredCollator
from src.phase0.phase0_4c_evaluation import prediction_metrics
from src.phase0.phase0_4c_full_train import eligible_samples, write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, load_config, resolve_derived_path, sha256_file
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION, validate_processor_inputs
from src.phase0.qwen3vl_lora_smoke import _move_batch, default_runtime_dependencies


BASELINE = 'drivema/low768_reward_shape_run01/additive_full'
FEEDBACK = 'drivema/low768_action_geometry_run01/full'
CHECKPOINT = 'drivema/low768_reward_shape_run01/additive/checkpoint_768'


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def load_inputs(derived_root: Path, oracle_dir: Path) -> tuple:
    metas, predictions = [], []
    for directory, weight in ((BASELINE, 0.), (FEEDBACK, 1.38)):
        full = resolve_derived_path(derived_root, directory)
        meta = json.loads((full / 'metadata.json').read_text())
        reward = meta['identity']['rollout']['reward']
        if not ((meta['mode'], meta['step'], meta['model_id'], meta['revision']) ==
                ('evaluate-full', 768, FIXED_MODEL_ID, FIXED_REVISION)
                and reward['composition'] == 'additive' and reward.get('action_geometry_weight', 0.) == weight):
            raise ValueError(f'expected matched additive-768 and feedback-768 controls: {directory}')
        rows = read_rows(full / 'evaluate-full_768_predictions.jsonl')
        if set(meta['population_tokens']) != {r['sample_token'] for r in rows}:
            raise ValueError(f'incomplete frozen validation population: {directory}')
        metas.append(meta)
        predictions.append(rows)
    old, new = metas
    inference = old['identity']['validation']
    if not (new['identity']['validation'] == inference and inference['optimized'] and inference['batch_size'] == 1
            and not inference['action_generation']['do_sample'] and inference['action_generation']['num_beams'] == 1
            and old['dtype'] == new['dtype'] == 'torch.bfloat16' and old['attention'] == new['attention'] == 'sdpa'
            and old['data_intake']['manifests']['validation'] == new['data_intake']['manifests']['validation']):
        raise ValueError('incompatible validation conditions')
    oracle = json.loads((oracle_dir / 'metadata.json').read_text())
    if not (oracle['mode'] == 'validation_gt_action_diagnostic' and oracle['inference'] == inference
            and oracle['validation_count'] == len(predictions[0])
            and oracle['checkpoint'] == old['checkpoint'] and oracle['source_git_commit'] == old['git_commit']
            and all(oracle[k] == old[k] for k in ('step', 'model_id', 'revision', 'dtype', 'attention'))):
        raise ValueError('oracle cache was generated under a different baseline')
    plan = crossover_plan(*predictions, read_rows(oracle_dir / 'pairs.jsonl'))
    return metas, predictions, plan


def run_replays(rows: list[dict], saved: dict, feedback: dict, samples: dict, collator: StructuredCollator,
                model: torch.nn.Module, device: str, max_tokens: int) -> None:
    processor = collator.processor
    for row in rows:
        token = row['sample_token']
        if row['source'] == 'baseline':
            continue
        action = feedback[token]['assistant_spans'][0]['token_ids']
        sample = samples[token]
        if row['source'] == 'oracle' and intervention_action_ids(processor, sample) == action:
            continue
        row['source'] = 'replay'
        messages = collator.messages(sample, expected_split='validation')
        first = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              return_dict=True, return_tensors='pt')
        validate_processor_inputs(first, expected_batch_size=1, expected_image_count=len(sample.observation.image_paths))
        result = replay_turn2(model, processor, _move_batch(first, device), saved[token], action, max_tokens)
        row['mixed_waypoints'] = points = result['waypoints']
        row['mixed_invalid_reason'] = result['invalid_reason']
        target = saved[token]
        metrics = prediction_metrics(torch.tensor(points) if points is not None else None,
                                     torch.tensor(target['target_waypoints']), torch.tensor(target['trajectory_valid_mask']))
        row['mixed'] = {k: metrics.get(k) for k in TRAJECTORY_ERRORS}
        print(f"Replayed {token}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Cross #71 predicted actions with the fixed #69 trajectory decoder.')
    parser.add_argument('mode', choices=('plan', 'evaluate'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--oracle-dir', required=True, type=Path, help='extracted #70 cache directory')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--local-model-dir', type=Path, default=os.environ.get('QWEN_LOCAL_DIR'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.derived_root is None or (args.mode == 'evaluate' and args.dataset_root is None):
        parser.error('provide derived root and, for evaluation, dataset root')
    metas, predictions, plan = load_inputs(args.derived_root, args.oracle_dir)
    print(json.dumps(summarize_plan(plan), indent=2), flush=True)
    if args.mode == 'plan':
        return 0
    output = (args.output_dir or args.derived_root / 'drivema/action_crossover_run01').resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('choose a new output directory outside the checkout')
    old, new = metas
    checkpoint = resolve_derived_path(args.derived_root, CHECKPOINT)
    state = torch.load(checkpoint / 'unified_state.pt', map_location='cpu', weights_only=True)
    if state['step'] != old['step'] or state['identity'] != old['identity']:
        raise ValueError('checkpoint differs from full-validation control')
    del state
    manifest = old['data_intake']['manifests']['validation']
    temporal = yaml.safe_load((ROOT / 'configs/phase0_4_temporal_dataset.yaml').read_text())
    path = resolve_derived_path(args.derived_root, temporal['output_relative_dir']) / 'validation.jsonl'
    if path != resolve_derived_path(args.derived_root, manifest['relative_path']) or sha256_file(path) != manifest['sha256']:
        raise ValueError('validation manifest differs from checkpoint intake')
    records = load_temporal_records(ROOT, args.derived_root, split='validation')
    samples, _ = eligible_samples(records, load_config(ROOT / 'configs/phase0_3_dataset_adapter.yaml'), 'validation')
    by_token = {s.sample_token: s for s in samples}
    if by_token.keys() != {r['sample_token'] for r in plan}:
        raise ValueError('sample population differs from full validation')
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('crossover requires the archived bfloat16 decoder')
    source = str(args.local_model_dir.resolve(strict=True)) if args.local_model_dir else old['model_source']
    processor = runtime.processor_loader(source, FIXED_REVISION, True)
    base = runtime.model_loader(source, FIXED_REVISION, dtype, 'sdpa', True)
    model = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=False).to(device)
    select_frozen_adapter(model, 'default')
    collator = StructuredCollator(processor, runtime.image_loader, args.dataset_root)
    baseline, feedback = [{r['sample_token']: r for r in rows} for rows in predictions]
    run_replays(plan, baseline, feedback, by_token, collator, model, device,
                old['identity']['validation']['trajectory_max_new_tokens'])
    result = decomposition(plan)
    output.mkdir(parents=True, exist_ok=False)
    write_predictions(output / 'pairs.jsonl', plan)
    write_json(output / 'metadata.json', {
        'mode': 'validation_predicted_action_crossover', 'baseline_checkpoint': str(checkpoint),
        'feedback_checkpoint': new['checkpoint'], 'oracle_dir': str(args.oracle_dir), 'inference': old['identity']['validation'],
        'model_id': old['model_id'], 'revision': FIXED_REVISION, 'git_commit': collect_git_provenance(ROOT).commit,
        'baseline_full_dir': BASELINE, 'feedback_full_dir': FEEDBACK, 'validation_manifest': manifest})
    write_json(output / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
