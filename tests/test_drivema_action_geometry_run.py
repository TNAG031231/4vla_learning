from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_drivema_action_geometry


@pytest.mark.parametrize('stage, checkpoint_option, checkpoint_name, output', [
    ('train', '--resume', 'worker_a_continuation/checkpoint_512', 'train'),
    ('evaluate-full', '--checkpoint', 'low768_action_geometry_run01/train/checkpoint_768', 'full'),
])
def test_launcher_starts_only_the_feedback_worker(monkeypatch, tmp_path, stage, checkpoint_option,
                                                 checkpoint_name, output):
    calls = []
    monkeypatch.setattr(run_drivema_action_geometry.subprocess, 'run',
                        lambda command, **options: calls.append((command, options)))
    run_drivema_action_geometry.main([stage, '--derived-root', str(tmp_path)])
    (command, options), = calls
    assert command[command.index('--mode') + 1] == stage
    assert command[command.index('--action-geometry-weight') + 1] == '1.38'
    assert command[command.index('--reward-shape') + 1] == 'additive'
    assert command[command.index(checkpoint_option) + 1].endswith('/' + checkpoint_name)
    assert command[command.index('--output-relative-dir') + 1] == f'drivema/low768_action_geometry_run01/{output}'
    assert options['env']['HF_HUB_OFFLINE'] == '1'
    if stage == 'evaluate-full':
        assert command[command.index('--baseline-predictions') + 1].endswith('/sft_full_predictions.jsonl')


@pytest.mark.parametrize('stage, output', [('train', 'train'), ('evaluate-full', 'full')])
def test_early_launcher_starts_at_sft_and_evaluates_its_own_step_512(monkeypatch, tmp_path, stage, output):
    calls = []
    monkeypatch.setattr(run_drivema_action_geometry.subprocess, 'run',
                        lambda command, **options: calls.append(command))
    run_drivema_action_geometry.main([stage, '--derived-root', str(tmp_path), '--feedback-start-step', '1'])
    command, = calls
    assert command[command.index('--action-geometry-weight') + 1] == '1.38'
    assert command[command.index('--feedback-start-step') + 1] == '1'
    assert command[command.index('--max-steps') + 1] == '512'
    assert command[command.index('--reward-shape') + 1] == 'additive'
    start = command.index('--evaluation-steps') + 1
    assert command[start:start + 6] == ['0', '32', '64', '128', '256', '512']
    assert command[command.index('--output-relative-dir') + 1] == f'drivema/early_action_geometry_run01/{output}'
    assert '--resume' not in command
    if stage == 'train':
        assert '--checkpoint' not in command
    else:
        assert command[command.index('--checkpoint') + 1].endswith('/early_action_geometry_run01/train/checkpoint_512')
        assert command[command.index('--baseline-predictions') + 1].endswith('/sft_full_predictions.jsonl')
