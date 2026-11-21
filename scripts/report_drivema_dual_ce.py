from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from scripts.report_drivema_grpo_ablation import read_rows
from src.phase0.drivema_grpo_diagnostics import validation_diagnostics
from src.phase0.drivema_grpo_replay import extend_low512, low512_replay, replay_identity
from src.phase0.drivema_grpo_train import OBJECTIVE_CE_WEIGHTS, TrainingConfig
from src.phase0.phase0_4c_tiny_overfit import write_json


def historical_runs(derived: Path, trajectory: Path | None, meta: Path | None) -> tuple[Path, Path | None]:
    def locate(explicit: Path | None, pattern: str, arms: tuple[str, str]) -> Path | None:
        if explicit is not None:
            return explicit.resolve(strict=True)
        matches = sorted(p for p in (derived / 'drivema').glob(pattern)
                         if any((p / f'{arm}_full/metadata.json').is_file() for arm in arms))
        if len(matches) > 1:
            raise ValueError(f'multiple {pattern} runs; specify --trajectory-run or --meta-run: {matches}')
        return matches[0] if matches else None

    trajectory = locate(trajectory, 'pr66_ce_*', ('grpo_ce', 'ce_only'))
    if trajectory is None:
        raise ValueError('provide --trajectory-run pointing to the completed PR #66 experiment')
    for arm in ('grpo_ce', 'ce_only'):
        full = trajectory / f'{arm}_full'
        if not all((full / name).is_file() for name in ('metadata.json', 'evaluate-full_512_predictions.jsonl')):
            raise ValueError(f'missing completed PR #66 full validation: {full}')
    return trajectory, locate(meta, 'meta_ce_*', ('grpo_meta_ce', 'meta_ce_only'))


def report(run: Path, low: Path, step: int, config: TrainingConfig, trajectory: Path | None,
           meta: Path | None, *, repetitions: int = 10000) -> dict:
    control = json.loads((low / 'metadata.json').read_text())
    replay = low512_replay(control)
    old = control['identity']
    tokens = control['population_tokens']
    if len(tokens) != 3594 or len(set(tokens)) != 3594:
        raise ValueError('expected 3594 unique Low-512 validation observations')
    extended = replay_identity(replay['baseline_identity'], extend_low512(replay))
    dual = {name: {**extended, 'dual_supervision': replace(config, dual_ce_first_step=first).dual_supervision()}
            for name, first in (('A', 1), ('B', 513))}
    # Each entry names the full-validation directory, expected checkpoint step and objective identity.
    sources = {'low512': (low, 512, old), 'A': (run / f'A{step}_full', step, dual['A'])}
    missing = []
    if step == 512:
        sources['sft'] = (low, 0, old)
        for name in ('grpo_ce', 'ce_only', 'grpo_meta_ce', 'meta_ce_only'):
            key, parent = ('trajectory_supervision', trajectory) if name in ('grpo_ce', 'ce_only') else (
                'meta_action_supervision', meta)
            if parent is None or not (parent / f'{name}_full/metadata.json').is_file():
                missing.append(name)
                continue
            identity = {**old, key: {'objective': name, 'weight': OBJECTIVE_CE_WEIGHTS[name]}}
            sources[name] = (parent / f'{name}_full', 512, identity)
        high = low.parent / 'worker_b_full_validation'
        if (high / 'metadata.json').is_file():
            schedule = json.loads((high / 'metadata.json').read_text())['observation_replay']
            if (schedule['experiment']['arm'] != 'unique512' or schedule['experiment']['kind'] != 'coverage'
                    or len(schedule['steps']) != 512 or len({r['sample_token'] for r in schedule['steps']}) != 512
                    or schedule['experiment']['consistency_weight'] != .5):
                raise ValueError('expected the original High-512 coverage arm')
            sources['high512'] = (high, 512, replay_identity(replay['baseline_identity'], schedule))
        else:
            missing.append('high512')
        pairs = [(name, 'A') for name in sources if name != 'A']
    elif step == 768:
        sources['B'] = (run / 'B768_full', 768, dual['B'])
        pairs = [('low512', 'A'), ('low512', 'B'), ('B', 'A')]
    else:
        raise ValueError('expected full validation at 512 or 768')

    def rows_for(name: str) -> list[dict]:
        directory, steps, expected = sources[name]
        metadata = json.loads((directory / 'metadata.json').read_text())
        if (metadata['identity'] != expected or metadata['step'] != (steps or 512)
                or metadata['mode'] != 'evaluate-full' or metadata['population_tokens'] != tokens):
            raise ValueError(f'{name}: full-validation checkpoint, objective or population mismatch')
        label = f'evaluate-full_{steps}' if steps else 'sft_full'
        rows = read_rows(directory / f'{label}_predictions.jsonl')
        if ([r['sample_token'] for r in rows] != tokens or len({r['scene_token'] for r in rows}) != 140
                or any(r['split'] != 'validation' for r in rows)):
            raise ValueError(f'{name}: full-validation predictions do not match the 3594 observations / 140 scenes')
        return rows

    primary, paired = {}, {}
    a_rows = rows_for('A')
    for reference, current in pairs:
        diagnostics = validation_diagnostics(rows_for(reference), a_rows if current == 'A' else rows_for(current),
                                             old['rollout']['reward'], repetitions, 20261116,
                                             reference_label=reference, current_label=current)
        for label, role in ((reference, 'baseline'), (current, 'current')):
            metrics = diagnostics['comparison'][role]
            primary[label] = {key: metrics['trajectory'][key] for key in ('ade_3s_m', 'fde_3s_m')}
            primary[label]['joint_accuracy'] = metrics['action']['joint_accuracy']
        paired[f'{current}_minus_{reference}'] = {
            **{key: diagnostics['overall'][key] for key in ('ade_3s_m', 'fde_3s_m')},
            'joint_accuracy_delta': primary[current]['joint_accuracy'] - primary[reference]['joint_accuracy'],
            'joint_accuracy_bootstrap': diagnostics['action_behavior_cluster_bootstrap']['intervals']['joint_accuracy']}
    result = {'step': step, 'observations': len(tokens), 'scenes': 140, 'primary': primary, 'paired': paired,
              'bootstrap': diagnostics['bootstrap'], 'unavailable_controls': missing,
              'full_validation_dirs': {name: str(entry[0]) for name, entry in sources.items()}}
    write_json(run / f'comparison_{step}.json', result)
    print(json.dumps({'step': step, 'primary': primary, 'unavailable_controls': missing}, indent=2), flush=True)
    return result
