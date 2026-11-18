from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_drivema_grpo as cli
from scripts import report_drivema_grpo_ablation as report_cli
from src.phase0 import drivema_grpo_train as training
from src.phase0.drivema_grpo_replay import (
    load_observation_replay, replay_identity, validate_replay_checkpoint, validate_replay_configuration,
)
from src.phase0.drivema_stage3a import load_config as stage3a_config
from src.phase0.drivema_stage3_rewards import candidate_rewards, assign_advantages
from src.phase0.drivema_stage2b import select_subset
from test_drivema_grpo import (
    arguments, config, validation_predict, assert_nested_equal,
    case, records, samples, processor, runner, tiny_qwen, lora_model, dual_model, controlled_rollout,
)
from test_drivema_grpo_diagnostics import archive, reward, group_and_history


def test_grpo_only_override_and_frozen_stage3a(tmp_path):
    cfg = training.load_config(ROOT / 'configs/drivema_grpo_no_consistency.yaml')
    actual = training.training_rollout_config(ROOT, cfg)
    frozen = stage3a_config(ROOT / cfg.rollout_config)
    assert actual.reward.consistency_weight == 0 and frozen.reward.consistency_weight == .5
    assert replace(actual, reward=replace(actual.reward, consistency_weight=.5)) == frozen
    path = tmp_path / 'bad.yaml'
    values = yaml.safe_load((ROOT / cfg.rollout_config).read_text())
    for key, value in [('consistency_weight', 0), ('ade_weight', .6), ('advantage_eps', .01)]:
        changed = deepcopy(values)
        changed['reward'][key] = value
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError, match='reward definition'):
            stage3a_config(path)
    for value in (-1, .1, float('nan'), True):
        with pytest.raises(ValueError, match='consistency weight'):
            training.validate_config(replace(cfg, consistency_weight=value))


def test_zero_weight_changes_only_turn2_rewards_and_advantages(reward):
    group, _ = group_and_history(reward)
    zero = deepcopy(group['candidates'])
    from src.phase0.phase0_4b_protocol import ActionTarget
    for c, before in zip(zero, group['candidates']):
        c['rewards'] = candidate_rewards(c, ActionTarget('stop', 'straight', True, True),
                                        [[0., 0.]] * 6, [True] * 6, replace(reward, consistency_weight=0))
        assert c['rewards']['turn_rewards'][1] == c['rewards']['trajectory']
        assert {k: v for k, v in c['rewards'].items() if k != 'turn_rewards'} == {
            k: v for k, v in before['rewards'].items() if k != 'turn_rewards'}
    statistics = assign_advantages(zero, reward.advantage_eps)
    assert statistics['zero_variance'] == [False, False]
    assert [c['turn_advantages'][0] for c in zero] == [c['turn_advantages'][0] for c in group['candidates']]
    assert zero[0]['turn_advantages'][1] > zero[1]['turn_advantages'][1]
    assert group['candidates'][0]['turn_advantages'][1] < group['candidates'][1]['turn_advantages'][1]


def test_replay_archive_dry_run_and_config_isolation(archive, capsys):
    save, _, _ = archive
    path = save('baseline.zip')
    replay = load_observation_replay(path)
    assert [s['optimizer_step'] for s in replay['steps']] == list(range(1, 257))
    cfg = training.load_config(ROOT / 'configs/drivema_grpo_no_consistency.yaml')
    rollout = training.training_rollout_config(ROOT, cfg)
    validate_replay_configuration(asdict(cfg), rollout, replay)
    for changed in (replace(cfg, learning_rate=2e-6), replace(cfg, beta=.02),
                    replace(cfg, epsilon=.1), replace(cfg, max_steps=257)):
        with pytest.raises(ValueError):
            validate_replay_configuration(asdict(changed), rollout, replay)
    args = ['--config', str(ROOT / 'configs/drivema_grpo_no_consistency.yaml'), '--dry-run']
    with pytest.raises(SystemExit):
        cli.main(args)
    assert cli.main(args + ['--observation-replay-archive', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['rollout']['reward']['consistency_weight'] == 0
    assert result['observation_replay']['steps'] == 256


@pytest.mark.parametrize('field', ['sample_token', 'optimizer_step'])
def test_external_replay_mismatch(archive, field):
    save, contents, _ = archive
    key = 'grpo_pilot_run01/rollout_1.json'
    group = json.loads(contents[key])
    group[field] = 'mismatch'
    with pytest.raises(ValueError, match='mismatch'):
        load_observation_replay(save('bad.zip', {key: json.dumps(group)}))


@pytest.fixture
def matched_arguments(arguments, config):
    arguments['runner'].compute_log_probs = False
    monitor = select_subset(arguments['validation'], config.monitor_samples, arguments['rollout_config'].seed, 'validation')
    identity = training.run_identity(config, arguments['rollout_config'], arguments['train'], monitor,
                                     arguments['runner'], arguments['metadata'])
    selected = [arguments['train'][(i + 1) % len(arguments['train'])] for i in range(256)]
    replay = {'source_archive': 'producer-fixture.zip', 'baseline_identity': identity,
              'steps': [{'optimizer_step': i + 1, 'sample_token': s.sample_token, 'scene_token': s.scene_token}
                        for i, s in enumerate(selected)]}
    return {**arguments, 'rollout_config': replace(arguments['rollout_config'],
                reward=replace(arguments['rollout_config'].reward, consistency_weight=0)),
            'metadata': {**arguments['metadata'], 'observation_replay': replay}}


@pytest.mark.parametrize('weight', [0., .25])
def test_real_cpu_lora_replay_resume_and_unchanged_reference(dual_model, controlled_rollout, matched_arguments,
                                                           config, validation_predict, monkeypatch, tmp_path, weight):
    cfg = replace(config, consistency_weight=weight, max_steps=2, evaluation_steps=[0, 1, 2])
    matched_arguments['rollout_config'] = replace(matched_arguments['rollout_config'], reward=replace(
        matched_arguments['rollout_config'].reward, consistency_weight=weight))
    def rollout_with_extra_rng(**kwargs):
        torch.rand(37)
        return controlled_rollout[0](**kwargs)
    monkeypatch.setattr(training, 'rollout_group', rollout_with_extra_rng)
    initial = deepcopy(dual_model)
    reference = {n: p.detach().clone() for n, p in initial.named_parameters() if '.reference.' in n}
    training.fit(model=dual_model, config=cfg, output=tmp_path / 'continuous', **matched_arguments)
    training.fit(model=initial, config=replace(cfg, max_steps=1), output=tmp_path / 'pilot', **matched_arguments)
    checkpoint = tmp_path / 'pilot/checkpoint_1'
    # The live policy equals the saved step-1 current/reference pair; optimizer/RNG restore is real.
    training.fit(model=initial, config=cfg, output=tmp_path / 'resumed', resume=checkpoint, **matched_arguments)
    assert_nested_equal(dual_model.state_dict(), initial.state_dict())
    for n, p in initial.named_parameters():
        if n in reference:
            assert torch.equal(p, reference[n]) and not p.requires_grad
    expected = matched_arguments['metadata']['observation_replay']['steps']
    for step in (1, 2):
        g = json.loads((tmp_path / f'continuous/rollout_{step}.json').read_text())
        assert all(g[k] == expected[step - 1][k] for k in expected[step - 1])
        assert all(c['rewards']['turn_rewards'][1] == c['rewards']['trajectory'] + weight * c['rewards']['consistency']
                   for c in g['candidates'])
        update = json.loads((tmp_path / f'continuous/update_{step}.json').read_text())
        assert update['lora_parameter_delta_l2'] > 0 and update['gradient_norm'] > 0
    assert json.loads((tmp_path / 'continuous/rollout_2.json').read_text()) == json.loads(
        (tmp_path / 'resumed/rollout_2.json').read_text())
    state = torch.load(checkpoint / 'unified_state.pt', weights_only=True)
    replay = matched_arguments['metadata']['observation_replay']
    validate_replay_checkpoint(state, replay, weight)
    with pytest.raises(ValueError, match='old GRPO forbidden'):
        validate_replay_checkpoint({'identity': replay['baseline_identity']}, replay, 0)
    changed = deepcopy(replay)
    changed['steps'][0]['sample_token'] = 'different'
    with pytest.raises(ValueError, match='old GRPO forbidden'):
        validate_replay_checkpoint(state, changed, 0)
    with pytest.raises(FileExistsError):
        training.fit(model=initial, config=cfg, output=tmp_path / 'pilot', **matched_arguments)
    with pytest.raises(ValueError, match='identity'):
        training.fit(model=initial, config=replace(cfg, max_steps=3, consistency_weight=.5),
                     rollout_config=replace(matched_arguments['rollout_config'], reward=replace(
                         matched_arguments['rollout_config'].reward, consistency_weight=.5)),
                     output=tmp_path / 'wrong-reward', resume=checkpoint,
                     **{k: v for k, v in matched_arguments.items() if k != 'rollout_config'})
    report = report_cli.audit_ablation(replay, [tmp_path / 'pilot', tmp_path / 'resumed'], 2, weight)
    assert report['verified_steps'] == 2
    if weight == 0:
        assert report['training']['ranking']['error_m']['combined']['pair_counts']['reversed'] == 0


def test_scene_mismatch_fails_before_update(dual_model, matched_arguments, config, tmp_path):
    matched_arguments['metadata']['observation_replay']['steps'][0]['scene_token'] = 'wrong'
    with pytest.raises(ValueError, match='scene differs'):
        training.fit(model=dual_model, config=replace(config, consistency_weight=0),
                     output=tmp_path / 'unused', **matched_arguments)
    assert not (tmp_path / 'unused').exists()


def test_cli_fresh_sft_initialization_resume_and_saved_sft_evaluation(
        dual_model, controlled_rollout, matched_arguments, config, validation_predict,
        processor, runner, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
    replay = matched_arguments['metadata']['observation_replay']
    replay['baseline_identity'].update(model=FIXED_MODEL_ID, revision=FIXED_REVISION,
        initial_sft_checkpoint=matched_arguments['rollout_config'].checkpoint_relative_path)
    monkeypatch.setattr(cli, 'load_observation_replay', lambda _: replay)
    initial = tmp_path / matched_arguments['rollout_config'].checkpoint_relative_path / 'adapter'
    dual_model.save_pretrained(initial, selected_adapters=['default'])
    unloaded = dual_model.unload()
    def load_base(*args):
        model = Qwen3VLForConditionalGeneration(deepcopy(unloaded.config))
        model.load_state_dict(unloaded.state_dict())
        return model
    runtime = SimpleNamespace(device_selector=lambda _: 'cpu', dtype_selector=lambda _: torch.bfloat16,
                              processor_loader=lambda *a: processor, model_loader=load_base,
                              image_loader=lambda _: Image.new('RGB', (64, 64)), package_version=lambda _: 'test')
    monkeypatch.setattr(cli, 'default_runtime_dependencies', lambda: runtime)
    monkeypatch.setattr(cli, 'collect_git_provenance', lambda _: SimpleNamespace(commit='synthetic-test'))
    monkeypatch.setattr(cli, 'Stage2Runner', lambda *a, **kw: runner)
    monkeypatch.setattr(cli, 'prepare_data', lambda *a: (
        matched_arguments['train'], matched_arguments['validation'], matched_arguments['records'],
        matched_arguments['metadata']['data_intake']))
    monkeypatch.setattr(training, 'rollout_group', controlled_rollout[0])
    loaded = []
    original = PeftModel.from_pretrained
    def load_adapter(model, path, **kwargs):
        loaded.append(path)
        return original(model, path, **kwargs)
    monkeypatch.setattr(PeftModel, 'from_pretrained', load_adapter)
    cfg = replace(config, consistency_weight=0, max_steps=1, evaluation_steps=[0, 1, 2])
    path = tmp_path / 'zero.yaml'
    path.write_text(yaml.safe_dump(asdict(cfg)))
    common = ['--config', str(path), '--dataset-root', str(tmp_path), '--derived-root', str(tmp_path),
              '--observation-replay-archive', 'producer-backed.zip']
    assert cli.main(common) == 0
    assert loaded == [initial]
    checkpoint = tmp_path / cfg.output_relative_dir / 'checkpoint_1'
    assert cli.main(common + ['--resume', str(checkpoint), '--max-steps', '2',
                              '--output-relative-dir', 'ablation-resumed']) == 0
    assert loaded[-1] == checkpoint / 'adapter'
    bad = tmp_path / 'old-grpo'
    bad.mkdir()
    torch.save({'identity': replay['baseline_identity']}, bad / 'unified_state.pt')
    with pytest.raises(ValueError, match='old GRPO forbidden'):
        cli.main(common + ['--resume', str(bad), '--max-steps', '2', '--output-relative-dir', 'bad-resume'])
    assert len(loaded) == 2
    with pytest.raises(ValueError, match='new directory'):
        cli.main(common)
    assert cli.main(common + ['--mode', 'evaluate-full', '--checkpoint', str(checkpoint),
                              '--output-relative-dir', 'first-eval']) == 0
    saved = tmp_path / 'first-eval/sft_full_predictions.jsonl'
    evaluations = []
    evaluate = cli.evaluate_and_save
    def current_only(*args, **kwargs):
        evaluations.append(kwargs.get('adapter', 'default'))
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(cli, 'evaluate_and_save', current_only)
    assert cli.main(common + ['--mode', 'evaluate-full', '--checkpoint', str(checkpoint),
                              '--baseline-predictions', str(saved), '--output-relative-dir', 'reuse-eval']) == 0
    assert evaluations == ['default']


@pytest.fixture
def completed_ablation(archive, reward, tmp_path):
    from test_drivema_grpo_diagnostics import prediction
    save, contents, _ = archive
    path = save('original.zip')
    replay = load_observation_replay(path)
    identity = deepcopy(replay['baseline_identity'])
    identity['rollout']['reward']['consistency_weight'] = 0.
    identity = replay_identity(identity, replay)
    pilot, continuation, evaluation = [tmp_path / d for d in ('pilot', 'continuation', 'evaluation')]
    for directory in (pilot, continuation, evaluation):
        directory.mkdir()
    for directory, start, end in ((pilot, 1, 4), (continuation, 5, 256)):
        metadata = {'identity': identity, 'config': {'consistency_weight': 0.},
                    'resume': None if start == 1 else str(pilot / 'checkpoint_4')}
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        history = []
        for step in range(start, end + 1):
            group, update = group_and_history(replace(reward, consistency_weight=0))
            group['optimizer_step'] = update['step'] = step
            update['effective_advantage_groups'] = [step * int(n) for n in update['effective_advantage_group']]
            update.update(loss=0., policy_loss=0.)
            (directory / f'rollout_{step}.json').write_text(json.dumps(group))
            history.append(update)
        (directory / 'training_history.jsonl').write_text('\n'.join(json.dumps(r) for r in history))
    baseline = contents['grpo_full_validation_reuse_sft_run01/sft_full_predictions.jsonl']
    (evaluation / 'sft_full_predictions.jsonl').write_text(baseline)
    rows = [prediction(f'v{i}', f'scene-{i % 2}', .25) for i in range(3594)]
    (evaluation / 'evaluate-full_256_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (evaluation / 'metadata.json').write_text(json.dumps({
        'identity': identity, 'step': 256, 'mode': 'evaluate-full',
        'checkpoint': str(continuation / 'checkpoint_256'), 'population_tokens': [r['sample_token'] for r in rows]}))
    return path, pilot, continuation, evaluation


def test_full_three_way_reporting_and_pilot_audit(completed_ablation, tmp_path):
    path, pilot, continuation, evaluation = completed_ablation
    assert report_cli.main(['--baseline-archive', str(path), '--run-dirs', str(pilot), '--steps', '4',
                            '--output', str(tmp_path / 'pilot-audit.json')]) == 0
    output = tmp_path / 'full-comparison.json'
    assert report_cli.main(['--baseline-archive', str(path), '--run-dirs', str(pilot), str(continuation),
                            '--steps', '256', '--evaluation-dir', str(evaluation), '--output', str(output),
                            '--bootstrap-repetitions', '100']) == 0
    result = json.loads(output.read_text())
    original = result['validation']['ablation_vs_original_grpo']['overall']['ade_3s_m']
    sft = result['validation']['ablation_vs_sft']['overall']['ade_3s_m']
    assert original['original_grpo_mean_m'] == .5 and original['ablation_mean_m'] == .25
    assert original['mean_delta_m'] == -.25 and sft['mean_delta_m'] == -.75
    assert original['cluster_bootstrap_95ci']['mean_delta_m'] == [-.25, -.25]
    assert result['training']['ranking']['error_m']['combined']['pair_counts']['reversed'] == 0
    with pytest.raises(SystemExit):
        report_cli.main(['--baseline-archive', str(path), '--run-dirs', str(pilot), '--steps', '4',
                         '--output', str(output)])


@pytest.mark.parametrize('failure', ['sequence', 'reward', 'zero_update', 'missing_step'])
def test_audit_rejects_failed_pilot(completed_ablation, failure):
    path, pilot, _, _ = completed_ablation
    if failure in ('sequence', 'reward'):
        file = pilot / 'rollout_1.json'
        group = json.loads(file.read_text())
        if failure == 'sequence':
            group['sample_token'] = 'wrong-observation'
        else:
            group['candidates'][0]['rewards']['turn_rewards'][1] += .5
        file.write_text(json.dumps(group))
    else:
        file = pilot / 'training_history.jsonl'
        rows = report_cli.read_rows(file)
        if failure == 'zero_update':
            for row in rows:
                row['lora_parameter_delta_l2'] = 0.
        else:
            rows.pop()
        file.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        report_cli.audit_ablation(load_observation_replay(path), [pilot], 4)
