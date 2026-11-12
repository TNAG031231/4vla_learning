#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.diagnose_drivema_stage2c import top2
from src.phase0.phase0_4c_tiny_overfit import write_json


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def token_agreement(left: list[dict], right: list[dict]) -> dict:
    matched, total, exact = 0, 0, 0
    for a, b in zip(left, right, strict=True):
        if a['sample_token'] != b['sample_token']:
            raise ValueError('sample alignment differs')
        aa = {s['turn']: s['token_ids'] for s in a['assistant_spans']}
        bb = {s['turn']: s['token_ids'] for s in b['assistant_spans']}
        exact += aa == bb
        for turn in aa.keys() | bb.keys():
            x, y = aa.get(turn, []), bb.get(turn, [])
            matched += sum(i == j for i, j in zip(x, y))
            total += max(len(x), len(y))
    return {'exact_sample_count': exact, 'sample_count': len(left), 'exact_sample_rate': exact / len(left),
            'matched_tokens': matched, 'compared_tokens': total, 'token_rate': matched / total,
            'definition': 'position-wise within each turn, includes EOS; missing/truncated suffix tokens count as mismatches'}


def audit_logits(directory: Path, diagnosis: dict) -> list[dict]:
    serial = {r['sample_token']: r for r in read_rows(directory / 'serial_predictions.jsonl')}
    report = []
    for track in ('batch1', 'batch4'):
        candidates = {r['sample_token']: r for r in read_rows(directory / f'{track}_predictions.jsonl')}
        for entry in diagnosis[track]['trace_comparison']:
            index = entry.get('first_different_token_zero_based')
            if index is None or 'serial' not in entry:
                continue
            token, turn = entry['sample_token'], entry['turn']
            values = torch.load(directory / f'{track}_{token}_turn{turn}_first_difference.pt',
                                map_location='cpu', weights_only=True)
            row = {'sample_token': token, 'turn': turn, 'index': index, 'track': track,
                   'same_input_prefix': entry['same_prefix'], 'original_summary': {
                       'serial': entry['serial'], 'candidate': entry['candidate']}}
            for name, predictions in (('serial', serial), ('candidate', candidates)):
                ids = next(s['token_ids'] for s in predictions[token]['assistant_spans'] if s['turn'] == turn)
                stats = top2(values[f'{name}_logits'])
                row[name] = {**stats, 'generated_token': ids[index], 'raw_argmax_matches_token': stats['argmax'] == ids[index]}
            row['max_abs_logit_delta'] = float((values['serial_logits'].float() - values['candidate_logits'].float()).abs().max())
            row['status'] = ('RAW_SELECTION_ALIGNED' if all(row[k]['raw_argmax_matches_token'] for k in ('serial', 'candidate'))
                             else 'PROCESSED_SCORES_OR_ALIGNMENT_REQUIRES_CHECK')
            report.append(row)
    return report


def build_report(benchmark: dict, diagnosis: dict, agreement: dict, audit: list[dict]) -> dict:
    a, b = (benchmark[k]['metrics'] for k in ('baseline', 'optimized'))
    quality = {key: (a['trajectory'][key] is not None and b['trajectory'][key] is not None
                     and b['trajectory'][key] <= a['trajectory'][key])
               for key in ('ade_3s_m', 'fde_3s_m', 'invalid_prediction_rate')}
    structural = diagnosis['batch4']['trace_comparison']
    checks = ('position_ids_equal_through_comparison', 'rope_deltas_equal_after_padding',
              'cache_positions_equal_after_padding', 'attention_masks_equal_through_comparison',
              'generation_kwargs_equal')
    return {'strict_consistency_original': benchmark['consistency'],
            'structure': {'scope': 'diagnosed four samples only; not a proof for all 64',
                          'checks': {key: dict(Counter(str(row.get(key, 'NOT_RECORDED')) for row in structural)) for key in checks},
                          'per_sample': structural},
            'token_consistency': agreement, 'logit_audit': audit,
            'quality': {'baseline': a, 'optimized': b, 'trajectory_non_regression': quality,
                        'trajectory_quality_passed': all(quality.values()),
                        'criteria': 'separate post-hoc descriptive gate: ADE3/FDE3/invalid rate <= baseline; no statistical or cross-dataset claim',
                        'action_quality': 'reported separately; no new action tolerance or aggregate acceptance threshold'},
            'performance': {name: {key: benchmark[name][key] for key in (
                'total_inference_seconds', 'samples_per_second', 'p50_latency_seconds', 'p95_latency_seconds', 'peak_allocated_bytes')}
                            for name in ('baseline', 'optimized')},
            'throughput_speedup': benchmark['throughput_speedup'],
            'closeout': 'review raw-selection audit and scope; strict mismatch remains failed; no automatic Stage 3 gate'}


def markdown(report: dict) -> str:
    lines = ['# Stage 2C 分项验收', '',
             f"严格一致性（保留原结果）：{report['strict_consistency_original']['passed']}",
             f"完整生成序列一致率：{report['token_consistency']['exact_sample_rate']:.2%}",
             f"逐位置 token 一致率（含 EOS，缺失计错）：{report['token_consistency']['token_rate']:.2%}",
             f"轨迹质量独立描述性 gate：{report['quality']['trajectory_quality_passed']}",
             '', '| 项目 | Baseline | Optimized |', '| --- | --- | --- |']
    for key in report['performance']['baseline']:
        lines.append(f"| {key} | {report['performance']['baseline'][key]} | {report['performance']['optimized'][key]} |")
    for section, keys in (('trajectory', ('ade_3s_m', 'fde_3s_m', 'invalid_prediction_count', 'invalid_prediction_rate')),
                          ('action', ('longitudinal_macro_f1', 'lateral_macro_f1', 'joint_accuracy'))):
        for key in keys:
            lines.append(f"| {key} | {report['quality']['baseline'][section][key]} | {report['quality']['optimized'][section][key]} |")
    lines += ['', f"吞吐加速：{report['throughput_speedup']}x", '',
              '结构检查只覆盖诊断样本；各项通过/失败/未记录计数：',
              '```json', json.dumps(report['structure']['checks'], indent=2), '```',
              '完整 Action 指标、逐样本结构检查及修正后的 raw-logits 审计见 acceptance.json。',
              'raw argmax 与实际 token 相同，仅说明已保存分歧位置的最终选择一致，不证明所有 logits processor 均未生效。',
              '若仍有 RAW_SELECTION 未对齐，需检查 processed scores；不能直接归因于 BF16。',
              '本报告不覆盖历史文件，不将严格 mismatch 改为 PASS，也不自动批准 Stage 3。']
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Offline Stage 2C acceptance and saved-logit audit; no GPU/model loading.')
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    benchmark = json.loads((args.benchmark_dir / 'benchmark.json').read_text())
    diagnosis = json.loads((args.diagnostic_dir / 'diagnosis.json').read_text())
    agreement = token_agreement(*(read_rows(args.benchmark_dir / f'{name}_predictions.jsonl')
                                  for name in ('baseline', 'optimized')))
    audit = audit_logits(args.diagnostic_dir, diagnosis)
    report = build_report(benchmark, diagnosis, agreement, audit)
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError('report output must be outside repository')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'acceptance.json', report)
    (args.output / 'summary.md').write_text(markdown(report))
    print(json.dumps({'output': str(args.output), 'raw_selection': dict(Counter(r['status'] for r in audit)),
                      'strict_consistency': benchmark['consistency']['passed']}, indent=2))
    return 0  # report completion, not an acceptance exit code


if __name__ == '__main__':
    raise SystemExit(main())
