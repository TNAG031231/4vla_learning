from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_action_geometry import action_contributions, main
from scripts.run_drivema_action_geometry import collect_results
from src.phase0.drivema_grpo_replay import replay_identity, reward_shape_replay
from src.phase0.drivema_grpo_train import group_metrics
from src.phase0.drivema_stage3_rewards import assign_advantages
from src.phase0.phase0_4c_full_train import write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from test_drivema_grpo import arguments, case, config, low512_control, processor, records, runner, samples
from test_drivema_grpo_diagnostics import group_and_history, prediction, reward
from test_drivema_reward_shape_report import evaluations


def test_action_contributions_separate_changed_actions_and_use_common_valid_ade_denominator():
    reference = [prediction(str(i), 'scene', 1.) for i in range(5)]
    current = [prediction(str(i), 'scene', error) for i, error in enumerate((0., 3., 2., None, 5.))]
    wrong = {'longitudinal': 'keep', 'lateral': 'left'}
    reference[0]['parsed_action'] = wrong
    current[1]['parsed_action'] = wrong
    reference[2]['parsed_action'] = wrong
    current[2]['parsed_action'] = {'longitudinal': 'decelerate', 'lateral': 'left'}
    current[3]['parsed_action'] = None
    for row in (reference[4], current[4]):
        row['target']['lateral_valid'] = False
    result = action_contributions(reference, current[::-1])
    assert result['paired_ade_count'] == 4 and result['excluded_ade_count'] == 1
    transitions = result['joint_correctness']
    assert transitions['wrong_to_correct']['contribution_to_overall_delta_m'] == -.25
    assert transitions['correct_to_wrong'] == {
        'observations': 2, 'paired_ade_count': 1, 'mean_delta_m': 2., 'contribution_to_overall_delta_m': .5}
    assert transitions['wrong_to_wrong']['contribution_to_overall_delta_m'] == .25
    assert transitions['joint_target_unavailable']['contribution_to_overall_delta_m'] == 1.
    changes = result['predicted_action_change']
    assert changes['changed']['observations'] == 3 and changes['unparsed']['paired_ade_count'] == 0
    for groups in (changes, transitions):
        assert sum(g['contribution_to_overall_delta_m'] for g in groups.values()) == 1.5


def test_report_compares_feedback_with_archived_additive_and_rejects_mismatched_identity(evaluations, reward):
    low_dir, control = evaluations
    low = json.loads((low_dir / 'metadata.json').read_text())
    control_training = control / 'additive/metadata.json'
    control_meta = json.loads(control_training.read_text())
    control_full = json.loads((control / 'additive_full/metadata.json').read_text())
    control_meta['config'] = {'output_relative_dir': str(Path(control.name) / 'additive')}
    write_json(control_training, control_meta)

    run = low_dir.parent / 'action_geometry'
    (run / 'train').mkdir(parents=True)
    (run / 'full').mkdir()
    weight = yaml.safe_load((ROOT / 'configs/drivema_low768_action_geometry.yaml').read_text())['action_geometry_weight']
    replay = reward_shape_replay(low, 'additive', action_geometry_weight=weight)
    identity = replay_identity(replay['baseline_identity'], replay)
    write_json(run / 'train/metadata.json', {**control_meta, 'identity': identity,
               'config': {'output_relative_dir': 'action_geometry/train'}})
    group, _ = group_and_history(reward)
    group['statistics'] = assign_advantages(group['candidates'], reward.advantage_eps, action_geometry_weight=weight)
    metrics = group_metrics(group, reward)
    write_predictions(run / 'train/training_history.jsonl', [
        {**metrics, 'step': s['optimizer_step'], 'sample_token': s['sample_token']} for s in replay['steps'][512:]])
    full = {**control_full, 'identity': identity, 'observation_replay': replay,
            'checkpoint': '/training/action_geometry/train/checkpoint_768'}
    write_json(run / 'full/metadata.json', full)
    template = prediction('v', 's', .5)
    rows = [{**template, 'sample_token': token, 'scene_token': f's{i % 140}'}
            for i, token in enumerate(low['population_tokens'])]
    write_predictions(run / 'full/evaluate-full_768_predictions.jsonl', rows[::-1])
    for path in (run / 'train/summary.json', run / 'full/baseline_source.json',
                 run / 'full/evaluate-full_768_metrics.json'):
        write_json(path, {})
    (run / 'train/checkpoint_768').mkdir()
    (run / 'train/checkpoint_768/training_state.pt').touch()
    cli = ['--low512-validation', str(low_dir), '--control-root', str(control),
           '--run-root', str(run), '--bootstrap-repetitions', '20']
    archive = low_dir.parent / 'results.tar.gz'
    collect_results(low_dir.parent, low_dir, control, run, archive)
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert sum(name.endswith('evaluate-full_768_predictions.jsonl') for name in names) == 2
        assert sum(name.endswith('training_history.jsonl') for name in names) == 2
        assert not any('checkpoint_768' in name for name in names)
        result = json.load(bundle.extractfile('action_geometry/comparison.json'))
    assert result['paired']['ade_3s_m']['cluster_bootstrap_95ci']['mean_delta_m'] == [-.25, -.25]
    assert result['trajectory_change']['mean_distance_m'] == .25
    assert result['training_signal'] == {'training_groups': 256, 'nonzero_feedback_groups': 256,
                                         'nonzero_feedback_with_tied_label_rewards': 0}
    write_json(run / 'full/metadata.json', {**full, 'identity': control_full['identity']})
    with pytest.raises(ValueError, match='full-validation identity'):
        main(cli)
