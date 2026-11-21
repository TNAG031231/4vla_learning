from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prepare_drivema_parallel as prepare
from scripts import run_drivema_parallel as launch
from src.phase0.drivema_grpo_replay import load_observation_replay, validate_replay_checkpoint
from src.phase0.drivema_grpo_schedules import coverage_schedules, reward_selection
from src.phase0.drivema_parallel import run_concurrent
from test_drivema_grpo_diagnostics import archive, reward
from test_drivema_grpo_ablation import completed_ablation
from test_drivema_stage2_sft import case, records, samples


def test_two_processes_overlap_with_separate_state_and_same_gpu(tmp_path):
    code = ('import os,time,pathlib; p=pathlib.Path.cwd(); '
            'p.joinpath("state").write_text(os.environ["CUDA_VISIBLE_DEVICES"]); '
            'p.parent.joinpath(p.name+".ready").touch(); '
            'other=p.parent/("b.ready" if p.name=="a" else "a.ready"); '
            'end=time.monotonic()+5; '
            '\nwhile not other.exists() and time.monotonic()<end: time.sleep(.01)\n'
            'assert other.exists(), "workers did not overlap"')
    trees = {w: tmp_path / w for w in ('a', 'b')}
    for tree in trees.values():
        tree.mkdir()
    result = run_concurrent({w: [sys.executable, '-c', code] for w in trees}, trees,
                            {**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}, tmp_path)
    assert result['exit_codes'] == {'a': 0, 'b': 0}
    assert len(set(result['pids'].values())) == 2
    assert all((t / 'state').read_text() == '0' for t in trees.values())


def test_failure_stops_sibling_and_preserves_status(tmp_path):
    commands = {'a': [sys.executable, '-c', 'import sys; sys.exit(7)'],
                'b': [sys.executable, '-c', 'import time; time.sleep(30)']}
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='worker failed'):
        run_concurrent(commands, {w: tmp_path for w in commands}, os.environ.copy(), tmp_path)
    assert time.monotonic() - start < 10
    status = json.loads((tmp_path / 'process_status.json').read_text())
    assert status['a']['exit_code'] == 7 and status['b']['exit_code'] is not None


@pytest.fixture
def suite(archive, tmp_path):
    save, _, _ = archive
    output = tmp_path / 'suite'
    prepare.main(['--round', 'reward', '--baseline-archive', str(save('original.zip')),
                  '--output', str(output)])
    return output / 'suite.json'


def test_reward_suite_dry_run_isolates_workers_and_resume(suite, tmp_path, capsys):
    capsys.readouterr()
    args = ['--suite', str(suite), '--worktree-a', str(tmp_path / 'a'), '--worktree-b', str(tmp_path / 'b'),
            '--dataset-root', str(tmp_path), '--derived-root', str(tmp_path), '--run-relative-dir', 'run', '--dry-run']
    for stage in ('pilot', 'continue', 'evaluate'):
        extra = ['--pilot-approved'] if stage == 'continue' else []
        if stage == 'evaluate':
            extra = ['--baseline-predictions', str(tmp_path / 'sft.jsonl')]
        assert launch.main(args + ['--stage', stage] + extra) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['concurrent_processes'] == 2 and not result['gpu_execution']
        a, b = result['commands'].values()
        assert a != b
        for worker, command in result['commands'].items():
            assert command[command.index('--max-steps') + 1] == ('32' if stage == 'pilot' else '256')
            assert worker in command[command.index('--output-relative-dir') + 1]
            if stage == 'pilot':
                assert '--resume' not in command and '--checkpoint' not in command
            elif stage == 'continue':
                assert f'{worker}_pilot/checkpoint_32' in command[command.index('--resume') + 1]
    with pytest.raises(SystemExit):
        launch.main(args + ['--stage', 'continue'])
    (tmp_path / 'run/worker_a_pilot').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        launch.main(args + ['--stage', 'pilot'])


def test_reward_suite_rejects_changed_observation(suite, tmp_path):
    path = suite.parent / 'worker_a.json'
    schedule = json.loads(path.read_text())
    schedule['steps'][0]['scene_token'] = 'wrong'
    path.write_text(json.dumps(schedule))
    with pytest.raises(ValueError, match='step alignment'):
        launch.main(['--suite', str(suite), '--stage', 'pilot', '--worktree-a', str(tmp_path / 'a'),
            '--worktree-b', str(tmp_path / 'b'), '--dataset-root', str(tmp_path), '--derived-root', str(tmp_path),
            '--run-relative-dir', 'run', '--dry-run'])


def test_coverage_schedule_matches_gt_and_budget(archive, samples):
    save, _, _ = archive
    replay = load_observation_replay(save('original.zip'))
    original = samples['train'][0]
    train = [replace(original, sample_token=f't{i}', scene_token=f's{i % 16}') for i in range(512)]
    replay['baseline_identity']['train_tokens'] = [s.sample_token for s in train]
    replay['steps'] = [{'optimizer_step': i + 1, 'sample_token': s.sample_token, 'scene_token': s.scene_token}
                       for i, s in enumerate(train[:256])]
    pair = coverage_schedules(replay, train, .25, {'rationale': 'fixture'}, 123)
    assert pair == coverage_schedules(replay, train, .25, {'rationale': 'fixture'}, 123)
    for arm, unique, repetitions in (('repeat256', 256, 2), ('unique512', 512, 1)):
        schedule = pair[arm]
        counts = Counter(r['sample_token'] for r in schedule['steps'])
        assert len(counts) == unique and set(counts.values()) == {repetitions}
        assert len(schedule['steps']) == 512
        assert schedule['coverage']['matching_tiers_for_new_observations'] == {'scene_and_gt_action': 256}
        assert all(r['total_visits'] == repetitions for r in schedule['steps'])
        state = {'identity': {'observation_sequence': schedule['steps'], 'experiment': schedule['experiment'],
                             'rollout': {'reward': {'consistency_weight': .25}}}}
        validate_replay_checkpoint(state, schedule, .25)
        other = pair['unique512' if arm == 'repeat256' else 'repeat256']
        with pytest.raises(ValueError):
            validate_replay_checkpoint(state, other, .25)
    assert pair['repeat256']['coverage']['gt_action_exposures'] == pair['unique512']['coverage']['gt_action_exposures']


def test_coverage_requires_results_and_explicit_selection(archive, tmp_path, capsys):
    save, _, _ = archive
    path = tmp_path / 'coverage'
    prepare.main(['--round', 'coverage', '--baseline-archive', str(save('baseline.zip')),
                  '--output', str(path), '--dry-run'])
    assert 'blocked_pending' in capsys.readouterr().out and not path.exists()
    worker = {'verified_steps': 256, 'validation': {'ablation_vs_sft': {'comparison': {'population_count': 3594}}}}
    report = {'round': 'reward', 'status': 'full_validation_complete', 'weights': {'0.0': 'a', '0.25': 'b', '0.5': 'original'},
              'workers': {'worker_a': worker, 'worker_b': deepcopy(worker)}}
    assert reward_selection(report, .25, 'paired results reviewed')['consistency_weight'] == .25
    report['workers']['worker_b']['verified_steps'] = 32
    with pytest.raises(ValueError, match='evidence'):
        reward_selection(report, .25, 'reviewed')


def test_action_bootstrap_keeps_invalids_in_denominator():
    from test_drivema_grpo_diagnostics import prediction
    from src.phase0.drivema_grpo_bootstrap import action_behavior_bootstrap
    baseline = [prediction(f'v{i}', f'scene{i}', 1.) for i in range(4)]
    current = deepcopy(baseline)
    for row in current:
        row['parsed_action'] = None
    result = action_behavior_bootstrap(baseline, current, 100, 17)
    assert result['intervals']['joint_accuracy']['delta_95ci'] == [-1., -1.]
    assert result['intervals']['longitudinal_macro_f1']['delta_95ci'] == [-.25, -.25]
    assert result['intervals']['consistency_mean']['undefined_replicates'] == 100


def test_parallel_report_producer_to_coverage_gate(completed_ablation, tmp_path):
    from scripts.report_drivema_parallel import main as report
    from src.phase0.drivema_grpo_replay import replay_identity
    from src.phase0.drivema_stage3a import load_config
    from test_drivema_grpo_diagnostics import group_and_history
    import shutil
    archive_path, pilot, continuation, evaluation = completed_ablation
    suite_dir = tmp_path / 'report-suite'
    prepare.main(['--round', 'reward', '--baseline-archive', str(archive_path), '--output', str(suite_dir)])
    root = tmp_path / 'runs'
    root.mkdir()
    for worker, weight in (('worker_a', 0.), ('worker_b', .25)):
        schedule = json.loads((suite_dir / f'{worker}.json').read_text())
        identity = deepcopy(schedule['baseline_identity'])
        identity['rollout']['reward']['consistency_weight'] = weight
        identity = replay_identity(identity, schedule)
        reward_config = replace(load_config(ROOT / 'configs/drivema_stage3a.yaml').reward, consistency_weight=weight)
        for source, suffix in ((pilot, 'pilot'), (continuation, 'continuation'), (evaluation, 'full_validation')):
            target = root / f'{worker}_{suffix}'
            shutil.copytree(source, target)
            metadata = json.loads((target / 'metadata.json').read_text())
            metadata['identity'] = identity
            if suffix == 'full_validation':
                metadata['checkpoint'] = str(root / f'{worker}_continuation/checkpoint_256')
            else:
                metadata['config']['consistency_weight'] = weight
                metadata['resume'] = None if suffix == 'pilot' else str(root / f'{worker}_pilot/checkpoint_4')
                rows = []
                for step in (range(1, 5) if suffix == 'pilot' else range(5, 257)):
                    group, history = group_and_history(reward_config)
                    group['optimizer_step'] = history['step'] = step
                    history.update(loss=0., policy_loss=0.)
                    history['effective_advantage_groups'] = [step * int(n) for n in history['effective_advantage_group']]
                    (target / f'rollout_{step}.json').write_text(json.dumps(group))
                    rows.append(history)
                (target / 'training_history.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (target / 'metadata.json').write_text(json.dumps(metadata))
    output = tmp_path / 'comparison'
    report(['--suite', str(suite_dir / 'suite.json'), '--run-root', str(root), '--output', str(output),
            '--bootstrap-repetitions', '20'])
    result = json.loads((output / 'comparison.json').read_text())
    assert result['worker_b_vs_worker_a']['comparison']['population_count'] == 3594
    assert set(result['weights']) == {'0.0', '0.25', '0.5'}
    assert reward_selection(result, .25, 'synthetic integration only')['consistency_weight'] == .25


@pytest.mark.parametrize('resource', ['gpu', 'disk'])
def test_resource_preflight_stops_before_process_start(suite, tmp_path, monkeypatch, resource):
    from types import SimpleNamespace
    monkeypatch.setattr(launch.subprocess, 'check_output',
                        lambda command, **kwargs: '' if 'status' in command else 'same-commit')
    monkeypatch.setattr(launch, 'gpu_snapshot', lambda _: {'free_mib': 0 if resource == 'gpu' else 32768})
    monkeypatch.setattr(launch.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0 if resource == 'disk' else 100 * 2**30))
    def no_processes(*args, **kwargs):
        pytest.fail('resource admission failure must precede process launch')
    monkeypatch.setattr(launch, 'run_concurrent', no_processes)
    with pytest.raises(ValueError, match='headroom'):
        launch.main(['--suite', str(suite), '--stage', 'pilot', '--worktree-a', str(tmp_path / 'a'),
            '--worktree-b', str(tmp_path / 'b'), '--dataset-root', str(tmp_path), '--derived-root', str(tmp_path),
            '--run-relative-dir', 'run'])
    assert not (tmp_path / 'run').exists()


def test_reward_shape_launcher_runs_two_matched_workers_before_evaluation(tmp_path, monkeypatch):
    from scripts import run_drivema_reward_shape as launcher

    low = tmp_path / 'worker_a_full_validation'
    low.mkdir()
    for name in ('metadata.json', 'sft_full_predictions.jsonl', 'evaluate-full_512_predictions.jsonl'):
        (low / name).touch()
    parent = tmp_path / 'worker_a_continuation/checkpoint_512'
    (parent / 'adapter/reference').mkdir(parents=True)
    for directory in (parent / 'adapter', parent / 'adapter/reference'):
        (directory / 'adapter_config.json').touch()
        (directory / 'adapter_model.safetensors').touch()
    for name in ('unified_state.pt', 'training_state.pt'):
        (parent / name).touch()
    stages = []

    def launch(commands, worktrees, environment, directory):
        stages.append(directory.name)
        assert environment['HF_HUB_OFFLINE'] == environment['TRANSFORMERS_OFFLINE'] == '1'
        assert set(commands) == {'additive', 'multiplicative'}
        for arm, command in commands.items():
            def option(flag):
                return command[command.index(flag) + 1]

            assert option('--reward-shape') == arm
            assert option('--low512-validation') == str(low)
            assert option('--local-model-dir') == str(tmp_path)
            assert option('--mode') == directory.name
            if directory.name == 'train':
                assert option('--resume') == str(parent)
                assert option('--output-relative-dir') == f'run/{arm}'
            else:
                assert option('--checkpoint') == str(tmp_path / 'run' / arm / 'checkpoint_768')
                assert option('--baseline-predictions') == str(low / 'sft_full_predictions.jsonl')
                assert option('--output-relative-dir') == f'run/{arm}_full'

    reports = []
    monkeypatch.setattr(launcher, 'run_concurrent', launch)
    monkeypatch.setattr(launcher, 'report', lambda args: reports.append(args) or 0)
    args = ['--dataset-root', str(tmp_path), '--derived-root', str(tmp_path), '--local-model-dir', str(tmp_path),
            '--low512-validation', str(low), '--run-relative-dir', 'run']
    assert launcher.main(args) == 0
    assert stages == ['train', 'evaluate-full']
    assert reports == [['--low512-validation', str(low), '--run-root', str(tmp_path / 'run')]]
    with pytest.raises(FileExistsError):
        launcher.main(args)
