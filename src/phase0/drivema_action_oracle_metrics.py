from __future__ import annotations

from src.phase0.drivema_grpo_diagnostics import paired_statistics


TRAJECTORY_ERRORS = ('ade_3s_m', 'fde_3s_m')


def _measurements(row: dict, arm: str) -> dict:
    errors = {key: row[f'{arm}_{key}'] for key in TRAJECTORY_ERRORS}
    return {
        **errors, 'scene_token': row['scene_token'], 'weighted_error_m': None,
        'prediction_valid': all(error is not None for error in errors.values()),
    }


def summarize(rows: list[dict], population: int) -> dict:
    """Summarize oracle minus predicted context on the original action errors."""
    pairs = [(_measurements(row, 'baseline'), _measurements(row, 'oracle')) for row in rows]
    repetitions, seed = 10_000, 20261116
    statistics = paired_statistics(pairs, repetitions=repetitions, seed=seed,
                                   reference_label='predicted', current_label='oracle')
    paired = {key: statistics[key] for key in TRAJECTORY_ERRORS}
    common_count = paired['ade_3s_m']['sample_count']
    result = {
        'population_count': population, 'wrong_action_count': len(rows),
        'wrong_action_scene_count': len({row['scene_token'] for row in rows}),
        'paired_valid_count': common_count,
        'oracle_invalid_count': sum(row['invalid_reason'] is not None for row in rows),
        'paired': paired,
        'bootstrap': {'unit': 'scene', 'repetitions': repetitions, 'seed': seed,
                      'estimand': 'sample-weighted mean paired delta on wrong-action rows'},
    }
    if common_count == len(rows):
        result['full_population_mean_delta_m'] = {
            key: sum(oracle[key] - baseline[key] for baseline, oracle in pairs) / population
            for key in TRAJECTORY_ERRORS
        }
    return result
