#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.benchmark_drivema_stage2c import compare_predictions
from src.phase0.drivema_stage2_sft import Stage2Runner, evaluate
from src.phase0.phase0_4b_protocol import StructuredCollator, parse_action
from src.phase0.phase0_4c_full_train import prepare_data, write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


def first_difference(left: list, right: list) -> int | None:
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)) if len(left) != len(right) else None)


def analyze_saved(left: list[dict], right: list[dict]) -> dict:
    rows = []
    for a, b in zip(left, right, strict=True):
        if a['sample_token'] != b['sample_token']:
            raise ValueError('saved sample order differs')
        spans = [{s['turn']: s for s in row['assistant_spans']} for row in (a, b)]
        actions_equal = spans[0][1]['token_ids'] == spans[1][1]['token_ids']
        row = {'sample_token': a['sample_token'], 'action_tokens_equal': actions_equal,
               'actions': [a['action_context'], b['action_context']]}
        if 2 not in spans[0] or 2 not in spans[1]:
            row['category'] = 'NO_TURN2'
        else:
            prefixes = [item['input_ids'][:span[2]['start']] for item, span in zip((a, b), spans)]
            suffixes = [item['input_ids'][span[1]['end']:span[2]['start']]
                        for item, span in zip((a, b), spans)]
            row.update(prefix_first_difference=first_difference(*prefixes),
                       suffix_equal=suffixes[0] == suffixes[1], suffix_token_ids=suffixes,
                       generation_first_difference=first_difference(spans[0][2]['token_ids'], spans[1][2]['token_ids']))
            row['category'] = ('TURN1_MISMATCH' if not actions_equal else
                               'TURN2_PREFIX_MISMATCH' if prefixes[0] != prefixes[1] else
                               'TURN2_GENERATION_MISMATCH' if row['generation_first_difference'] is not None else 'MATCH')
        rows.append(row)
    return {'counts': dict(Counter(row['category'] for row in rows)), 'samples': rows}


@contextmanager
def capture(model, sample_tokens: list[str], tokenizer):
    """Diagnostic-only CPU copies; never used for throughput measurement."""
    base = model.get_base_model()
    core = base.model
    original = model.generate
    calls = []
    active_tokens = []

    def generate(**kwargs):
        tokens = sample_tokens if not calls else active_tokens
        call = {'tokens': tokens, 'inputs': {k: v.detach().cpu().clone() for k, v in kwargs.items()
                                           if isinstance(v, torch.Tensor)},
                'generation': {**model.generation_config.to_dict(),
                               **{k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}},
                'vision_start_token_id': model.config.vision_start_token_id,
                'steps': []}
        calls.append(call)
        result = original(**kwargs)
        if len(calls) == 1:
            width = kwargs['input_ids'].shape[1]
            for token, sequence in zip(tokens, result, strict=True):
                ids = sequence[width:].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if tokenizer.eos_token_id in ids and parse_action(text) is not None:
                    active_tokens.append(token)
        return result

    def positions(module, args, kwargs):
        calls[-1]['steps'].append({k: kwargs[k].detach().cpu().clone()
                                  for k in ('position_ids', 'cache_position') if kwargs.get(k) is not None})
        calls[-1]['steps'][-1]['rope_deltas'] = core.rope_deltas.detach().cpu().clone()

    def logits(module, args, kwargs, output):
        calls[-1]['steps'][-1]['logits'] = output.logits[:, -1].detach().cpu().clone()
        calls[-1]['steps'][-1]['attention_mask'] = kwargs['attention_mask'].detach().cpu().clone()

    model.generate = generate
    handles = [core.language_model.register_forward_pre_hook(positions, with_kwargs=True),
               base.register_forward_hook(logits, with_kwargs=True)]
    try:
        yield calls
    finally:
        model.generate = original
        for handle in handles:
            handle.remove()


def sample_trace(call: dict, token: str) -> dict:
    index = call['tokens'].index(token)
    inputs = call['inputs']
    mask = inputs['attention_mask'][index].bool()
    # Count image placeholders using the actual per-image grids, in sample order.
    image_counts = (inputs['input_ids'] == call['vision_start_token_id']).sum(dim=1).tolist()
    grid_start = sum(image_counts[:index])
    grid_end = grid_start + image_counts[index]
    grids = inputs['image_grid_thw']
    patch_start = int(grids[:grid_start].prod(dim=1).sum())
    patch_end = int(grids[:grid_end].prod(dim=1).sum())
    return {'input_ids': inputs['input_ids'][index, mask],
            'attention_mask': inputs['attention_mask'][index, mask],
            'padding': int((~mask).sum()),
            'image_grid_thw': grids[grid_start:grid_end],
            'pixel_values': inputs['pixel_values'][patch_start:patch_end],
            'steps': [{'position_ids': step['position_ids'][:, index, mask] if i == 0 else
                                      step['position_ids'][:, index],
                       'attention_mask': step['attention_mask'][index, int((~mask).sum()):],
                       'cache_position': step['cache_position'],
                       'rope_deltas': step['rope_deltas'][index],
                       'logits': step['logits'][index]} for i, step in enumerate(call['steps'])]}


def top2(logits: torch.Tensor) -> dict:
    values, ids = logits.float().topk(2)
    return {'argmax': int(ids[0]), 'top2_ids': ids.tolist(), 'top2_logits': values.tolist(),
            'margin': float(values[0] - values[1])}


def compare_traces(reference: list[dict], candidate: list[dict], rows: list[dict], other: list[dict],
                   output: Path, label: str) -> list[dict]:
    report = []
    for a, b in zip(rows, other, strict=True):
        token = a['sample_token']
        for turn in (1, 2):
            sa = next((s for s in a['assistant_spans'] if s['turn'] == turn), None)
            sb = next((s for s in b['assistant_spans'] if s['turn'] == turn), None)
            if sa is None or sb is None:
                report.append({'sample_token': token, 'turn': turn, 'status': 'missing_turn'})
                continue
            ca = next(c for c in reference if token in c['tokens'] and c['turn'] == turn)
            cb = next(c for c in candidate if token in c['tokens'] and c['turn'] == turn)
            ta, tb = sample_trace(ca, token), sample_trace(cb, token)
            difference = first_difference(sa['token_ids'], sb['token_ids'])
            entry = {'sample_token': token, 'turn': turn, 'first_different_token_zero_based': difference,
                     'inputs_equal': {key: torch.equal(ta[key], tb[key]) for key in
                                      ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')},
                     'generation_kwargs_equal': ca['generation'] == cb['generation'],
                     'padding': [ta['padding'], tb['padding']]}
            # Inspect only the shared generated prefix, including the first divergent prediction.
            steps = min(len(ta['steps']), len(tb['steps']), difference + 1 if difference is not None else len(ta['steps']))
            entry['position_ids_equal_through_comparison'] = all(torch.equal(x['position_ids'], y['position_ids'])
                                                                for x, y in zip(ta['steps'][:steps], tb['steps'][:steps]))
            entry['attention_masks_equal_through_comparison'] = all(
                torch.equal(x['attention_mask'], y['attention_mask'])
                for x, y in zip(ta['steps'][:steps], tb['steps'][:steps]))
            entry['rope_deltas_equal_after_padding'] = all(
                torch.equal(x['rope_deltas'] + ta['padding'], y['rope_deltas'] + tb['padding'])
                for x, y in zip(ta['steps'][:steps], tb['steps'][:steps]))
            entry['cache_positions_equal_after_padding'] = all(
                torch.equal(x['cache_position'][ta['padding'] if i == 0 else 0:] - ta['padding'],
                            y['cache_position'][tb['padding'] if i == 0 else 0:] - tb['padding'])
                for i, (x, y) in enumerate(zip(ta['steps'][:steps], tb['steps'][:steps])))
            entry['position_metadata'] = [{key: [trace['steps'][i][key].tolist() for trace in (ta, tb)]
                                          for key in ('cache_position', 'rope_deltas')}
                                         for i in sorted({0, steps - 1}) if steps]
            if difference is not None and difference < steps:
                x, y = ta['steps'][difference]['logits'], tb['steps'][difference]['logits']
                entry.update(serial=top2(x), candidate=top2(y),
                             max_abs_logit_delta=float((x.float() - y.float()).abs().max()),
                             same_prefix=bool(entry['inputs_equal']['input_ids']))
                torch.save({'serial_logits': x, 'candidate_logits': y,
                            'serial_positions': ta['steps'][difference]['position_ids'],
                            'candidate_positions': tb['steps'][difference]['position_ids']},
                           output / f'{label}_{token}_turn{turn}_first_difference.pt')
            report.append(entry)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Analyze saved Stage 2C outputs, then optionally diagnose one original batch of four.')
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--sample-token', help='focal saved failure; preserve its original batch companions')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    args = parser.parse_args(argv)
    saved = [[json.loads(line) for line in (args.benchmark_dir / f'{name}_predictions.jsonl').read_text().splitlines()]
             for name in ('baseline', 'optimized')]
    analysis = analyze_saved(*saved)
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('diagnostics must be outside repository')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'saved_analysis.json', analysis)
    print(json.dumps({'saved_counts': analysis['counts'], 'analysis': str(output / 'saved_analysis.json')}), flush=True)
    if not args.gpu:
        return 0
    metadata = json.loads((args.benchmark_dir / 'metadata.json').read_text())
    if args.dataset_root is None or args.derived_root is None:
        parser.error('GPU diagnostic requires dataset and derived roots')
    if metadata['batch_size'] != 4:
        raise ValueError('this diagnostic requires the original batch-4 benchmark')
    failures = [r for r in analysis['samples'] if r['category'] not in ('MATCH', 'NO_TURN2')]
    if not failures:
        raise ValueError('no saved failure to diagnose')
    focal = args.sample_token or next((r['sample_token'] for r in failures if r['category'] in
                                      ('TURN1_MISMATCH', 'TURN2_PREFIX_MISMATCH')), failures[0]['sample_token'])
    index = metadata['sample_tokens'].index(focal)
    tokens = metadata['sample_tokens'][index // 4 * 4:index // 4 * 4 + 4]
    if len(tokens) != 4:
        raise ValueError('select a complete original group of four')
    from peft import PeftModel
    runtime = default_runtime_dependencies()
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('BF16 GPU required')
    _, validation, records, _ = prepare_data(ROOT, args.derived_root)
    by_token = {s.sample_token: s for s in validation}
    samples = [by_token[t] for t in tokens]
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, 'sdpa', True)
    model = PeftModel.from_pretrained(base, Path(metadata['checkpoint']) / 'adapter').to('cuda:0').eval()
    write_json(output / 'runtime.json', {'source_metadata': metadata, 'sample_tokens': tokens,
               'torch': torch.__version__, 'transformers': runtime.package_version('transformers'),
               'gpu': torch.cuda.get_device_name(), 'generation_config': model.generation_config.to_dict(),
               'vision_start_token_id': model.config.vision_start_token_id})
    collator = StructuredCollator(processor, runtime.image_loader, args.dataset_root)
    tracks, metrics, traces = {}, {}, {}
    for name, optimized, size in (('serial', False, 1), ('batch1', True, 1), ('batch4', True, 4)):
        runner = Stage2Runner(processor, collator, 'cuda:0', metadata['action_generation'],
                              metadata['config']['trajectory_max_new_tokens'], optimized=optimized,
                              inference_batch_size=size, compute_log_probs=False)
        rows, calls = [], []
        for offset in range(0, 4, size):
            group = samples[offset:offset + size]
            with capture(model, [s.sample_token for s in group], processor.tokenizer) as captured:
                predictions, _ = evaluate(model, runner, group, records)
            for turn, call in enumerate(captured, 1):
                call['turn'] = turn
            calls.extend(captured)
            rows.extend(predictions)
        from src.phase0.phase0_4c_evaluation import aggregate_metrics
        from src.phase0.phase0_4b_evaluation import factorized_metrics
        tracks[name], traces[name] = rows, calls
        metrics[name] = {'trajectory': aggregate_metrics(rows, 'predicted_action'), 'action': factorized_metrics(rows)}
        write_predictions(output / f'{name}_predictions.jsonl', rows)
    saved_by_token = {r['sample_token']: r for r in saved[0]}
    saved_rows = [saved_by_token[t] for t in tokens]
    saved_metrics = {'trajectory': aggregate_metrics(saved_rows, 'predicted_action'),
                     'action': factorized_metrics(saved_rows)}
    result = {'serial_vs_saved': compare_predictions(saved_rows, tracks['serial'], saved_metrics, metrics['serial'])}
    for name in ('batch1', 'batch4'):
        result[name] = {'consistency': compare_predictions(tracks['serial'], tracks[name], metrics['serial'], metrics[name]),
                        'trace_comparison': compare_traces(traces['serial'], traces[name], tracks['serial'],
                                                           tracks[name], output, name)}
    write_json(output / 'diagnosis.json', result)
    print(json.dumps({'output': str(output), 'serial_vs_saved': result['serial_vs_saved']['passed'],
                      **{name: result[name]['consistency']['passed'] for name in ('batch1', 'batch4')}}), flush=True)
    return 0 if (result['serial_vs_saved']['passed'] and
                 all(result[name]['consistency']['passed'] for name in ('batch1', 'batch4'))) else 1


if __name__ == '__main__':
    raise SystemExit(main())
