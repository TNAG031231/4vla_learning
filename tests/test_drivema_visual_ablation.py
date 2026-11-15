from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_drivema_visual_ablation as cli
from src.phase0 import drivema_visual_ablation as ablation
from src.phase0.drivema_stage2_sft import evaluate
from src.phase0.drivema_stage2_protocol import serialize_trajectory
from src.phase0.drivema_visual_interventions import CONDITIONS, VisualAblationRunner, donor_mapping, intervene
from src.phase0.drivema_visual_metrics import (
    align_predictions, condition_report, difference_in_differences, paired_sensitivity,
)
from src.phase0.phase0_4c_evaluation import prediction_metrics
from src.phase0.phase0_4c_full_train import write_predictions
from test_drivema_stage2_sft import (
    case, records, samples, processor, runner, tiny_qwen, lora_model, scripted_generate,
)


@pytest.fixture
def config():
    return ablation.load_config(ROOT / 'configs/drivema_visual_ablation.yaml')


@pytest.fixture
def population(samples):
    return [replace(sample, scene_token=f'validation-scene-{i % 2}')
            for i, sample in enumerate(samples['validation'])]


@pytest.fixture
def clean(population, records, processor, runner, lora_model, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate', scripted_generate(
        processor, ['longitudinal=keep; lateral=straight', serialize_trajectory([[1., 0.]] * 6)], []))
    runner.optimized = True
    runner.compute_log_probs = False
    rows, _ = evaluate(lora_model, runner, population, {r['sample_token']: r for r in records['validation']})
    return rows


@pytest.mark.parametrize('condition', CONDITIONS)
def test_intervention_pixels_dimensions_and_slots(condition):
    images = [Image.new('RGB', size, (i + 1, 2, 3)) for i, size in enumerate(((32, 48), (64, 32), (48, 64)))]
    before = [image.tobytes() for image in images]
    calls = []
    def loader(path):
        calls.append(path)
        return Image.new('RGB', (16, 24), (11, 12, 13))
    result, changed = intervene(images, condition, (128, 128, 128),
                                [{'donor_path': f'donor-{i}'} for i in range(3)], loader)
    expected = [0, 1] if condition == 'history_gray' else [2] if condition == 'current_gray' else [0, 1, 2]
    assert changed == expected
    assert [image.size for image in result] == [image.size for image in images]
    assert len(result) == len(images)
    for i, image in enumerate(result):
        if i in changed:
            expected_color = (11, 12, 13) if condition == 'scene_shuffle' else (128,) * 3
            assert image.getpixel((0, 0)) == expected_color
        else:
            assert image.tobytes() == before[i]
    assert [image.tobytes() for image in images] == before
    assert len(calls) == (3 if condition == 'scene_shuffle' else 0)


def test_single_frame_history_not_applicable():
    image = Image.new('RGB', (64, 64), (1, 2, 3))
    result, indices = intervene([image], 'history_gray', (128,) * 3, [], lambda path: None)
    assert result == [image] and indices == []


def test_shuffle_seed_validation_scene_and_missing_slots(population):
    mapping = donor_mapping(population, population, 42)
    assert mapping == donor_mapping(population, list(reversed(population)), 42)
    assert mapping != donor_mapping(population, population, 43)
    for sample in population:
        rows = mapping[sample.sample_token]
        assert len(rows) == len(sample.observation.image_paths)
        assert [r['source_history_slot'] for r in rows] == [i for i, v in enumerate(sample.observation.history_valid_mask) if v]
        assert all(r['donor_split'] == 'validation' and r['donor_scene_token'] != sample.scene_token for r in rows)
    with pytest.raises(ValueError, match='different validation scene'):
        donor_mapping(population[:1], population[:1], 42)
    with pytest.raises(ValueError, match='validation donors'):
        donor_mapping(population, [replace(population[0], split='test')], 42)


@pytest.mark.parametrize('condition', CONDITIONS)
def test_real_processor_preserves_input_and_targets(condition, population, records, clean, processor, runner, tmp_path):
    original = deepcopy((population, records))
    wrapped = VisualAblationRunner(processor, runner.collator, 'cpu', runner.action_generation, 192,
                                   optimized=True, compute_log_probs=False)
    wrapped.configure(condition, donor_mapping(population, population, 42), (128,) * 3, clean,
                      tmp_path / 'evidence.jsonl')
    for sample in population:
        images = wrapped.images(sample)
        evidence = wrapped.evidence[sample.sample_token]
        assert all(evidence['structure_matches'].values())
        assert len(images) == sum(sample.observation.history_valid_mask)
        slots = evidence['intervened_history_slots']
        if condition == 'history_gray':
            assert all(i < len(sample.observation.history_valid_mask) - 1 for i in slots)
            assert evidence['applicable'] == (len(images) > 1)
        if condition == 'current_gray':
            assert slots == [len(sample.observation.history_valid_mask) - 1]
    assert (population, records) == original


def test_structure_failure_persisted_before_generation(population, clean, processor, runner, tmp_path):
    altered = deepcopy(clean)
    altered[0]['input_ids'][0] += 1
    wrapped = VisualAblationRunner(processor, runner.collator, 'cpu', runner.action_generation, 192)
    evidence = tmp_path / 'evidence.jsonl'
    wrapped.configure('gray', donor_mapping(population, population, 42), (128,) * 3, altered, evidence)
    with pytest.raises(ValueError, match='saved clean prefix differs'):
        wrapped.images(population[0])
    assert not json.loads(evidence.read_text())['structure_matches']['saved_clean_prefix']


def test_pairing_metrics_denominators_invalid_and_history(clean):
    first = deepcopy(clean)
    for row in first:
        row.update(prediction_metrics(torch.zeros(6, 2), torch.zeros(6, 2), torch.ones(6, dtype=torch.bool)))
        row['target_waypoints'] = [[0., 0.]] * 6
        row['trajectory_valid_mask'] = [True] * 6
    changed = deepcopy(first)
    for row in changed:
        row.update(prediction_metrics(torch.tensor([[3., 4.]] * 6), torch.zeros(6, 2), torch.ones(6, dtype=torch.bool)))
        row['visual_intervention'] = {'applicable': True}
    changed[0]['parsed_action'] = {'longitudinal': 'stop', 'lateral': 'left'}
    changed[1].update(prediction_valid=False, predicted_waypoints=None, invalid_reason='invalid_trajectory_text')
    changed[1]['parsed_action'] = None
    changed[2]['visual_intervention']['applicable'] = False
    first[3].update(prediction_valid=False, predicted_waypoints=None, invalid_reason='invalid_trajectory_text')
    report = paired_sensitivity(first, list(reversed(changed)))
    assert report['paired_errors']['ade_3s_m'] == {'sample_count': 2, 'clean': 0., 'perturbed': 5., 'delta': 5.}
    assert report['trajectory_change'] == {'sample_count': 2, 'mean_distance_m': 5.}
    assert report['action_change']['joint']['rate'] == pytest.approx(1 / 3)
    assert report['validity_transitions']['trajectory'] == {
        'valid_to_valid': 2, 'valid_to_invalid': 1, 'invalid_to_valid': 1, 'invalid_to_invalid': 0}
    grouped = condition_report(first, changed)
    assert grouped['not_applicable_count'] == 1
    assert grouped['intervened_only']['common_valid_count'] == 1
    with pytest.raises(ValueError, match='incomplete run'):
        paired_sensitivity(first, changed[:-1])
    with pytest.raises(ValueError, match='unique population'):
        paired_sensitivity(first, changed + changed[:1])
    for key in ('scene_token', 'split', 'target', 'target_waypoints', 'trajectory_valid_mask'):
        bad = deepcopy(changed)
        bad[0][key] = 'mismatch'
        with pytest.raises(ValueError, match='contract mismatch'):
            align_predictions(first, bad)
    grpo_changed = [{**r, 'visual_intervention': {'applicable': True}} for r in first]
    did = difference_in_differences(first, changed, changed, grpo_changed)
    assert did['paired_errors']['overall']['ade_3s_m']['difference_in_differences'] == -10
    assert did['paired_errors']['overall']['ade_3s_m']['sample_count'] == 2
    empty = condition_report(first, [{**r, 'visual_intervention': {'applicable': False}} for r in changed])
    assert empty['intervened_only']['population_count'] == 0
    assert empty['intervened_only']['clean'] is None


def test_gt_horizon_denominator(clean):
    first = deepcopy(clean)
    changed = deepcopy(clean)
    for rows in (first, changed):
        rows[0]['trajectory_valid_mask'][-1] = False
        rows[0]['fde_3s_m'] = None
    report = paired_sensitivity(first, changed)
    assert report['paired_errors']['ade_3s_m']['sample_count'] == 4
    assert report['paired_errors']['fde_3s_m']['sample_count'] == 3


def test_end_to_end_shared_adapters_outputs(config, population, records, clean, processor, runner,
                                           lora_model, tmp_path, monkeypatch):
    lora_model.add_adapter('reference', deepcopy(lora_model.peft_config['default']))
    parameters = {n: p.detach().clone() for n, p in lora_model.named_parameters()}
    selected = []
    original = lora_model.set_adapter
    def select(name):
        selected.append(name)
        original(name)
    monkeypatch.setattr(lora_model, 'set_adapter', select)
    wrapped = VisualAblationRunner(processor, runner.collator, 'cpu', runner.action_generation, 192,
                                   optimized=True, compute_log_probs=False)
    report = ablation.evaluate_conditions(
        lora_model, wrapped, population, {r['sample_token']: r for r in records['validation']},
        {'sft': clean, 'grpo': clean}, donor_mapping(population, population, config.seed), config, tmp_path)
    assert selected == ['reference', 'default']
    assert all(not p.requires_grad and torch.equal(p, parameters[n]) for n, p in lora_model.named_parameters())
    assert report == ablation.summarize(tmp_path)
    assert len(list(tmp_path.glob('*_predictions.jsonl'))) == 10
    assert len(list(tmp_path.glob('*_input_evidence.jsonl'))) == 8
    for model in ('sft', 'grpo'):
        for condition in CONDITIONS:
            overall = report['models'][model][condition]['overall']
            assert overall['population_count'] == 4
            rows = [json.loads(line) for line in (tmp_path / f'{model}_{condition}_predictions.jsonl').read_text().splitlines()]
            assert all(s['token_log_probs'] is None for r in rows for s in r['assistant_spans'])
    metadata = {'config': asdict(config), 'sample_count': 4, 'derived_root': str(tmp_path.resolve())}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    ablation.validate_sanity(tmp_path, config, tmp_path)
    with pytest.raises(ValueError, match='same checkpoint'):
        ablation.validate_sanity(tmp_path, replace(config, seed=3), tmp_path)


def test_cli_config_and_dry_run(config, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run must not load data or a model')
    monkeypatch.setattr(cli, 'run', forbidden)
    assert cli.main(['--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['conditions'] == list(CONDITIONS)
    with pytest.raises(SystemExit):
        cli.main(['--samples', '2', '--dry-run'])
    with pytest.raises(ValueError, match='escapes'):
        cli.main(['--dry-run', '--output-relative-dir', '../escape'])
    bad = tmp_path / 'config.yaml'
    import yaml
    bad.write_text(yaml.safe_dump({**asdict(config), 'gray_rgb': [128, 1, 128]}))
    with pytest.raises(ValueError, match='neutral gray'):
        ablation.load_config(bad)
    with pytest.raises(ValueError, match='new directory'):
        ablation.run(ROOT, tmp_path, tmp_path, replace(config, output_relative_dir='.'), 4, None)
    with pytest.raises(ValueError, match='sanity-output'):
        ablation.run(ROOT, tmp_path, tmp_path, config, 128, None)


def test_real_tiny_qwen_frozen_inference(population, records, processor, runner, lora_model, tmp_path):
    selected = population[:1]
    targets = {r['sample_token']: r for r in records['validation']}
    runner.optimized = True
    runner.compute_log_probs = False
    runner.action_generation = {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 2}
    clean_rows, _ = evaluate(lora_model, runner, selected, targets)
    wrapped = VisualAblationRunner(processor, runner.collator, 'cpu', runner.action_generation, 2,
                                   optimized=True, compute_log_probs=False)
    wrapped.configure('gray', donor_mapping(selected, population, 42), (128,) * 3, clean_rows,
                      tmp_path / 'real_evidence.jsonl')
    ablation.select_frozen_adapter(lora_model, 'default')
    rows, _ = evaluate(lora_model, wrapped, selected, targets)
    assert len(rows) == 1
    assert rows[0]['log_prob_definition'] == 'not_computed'
    assert all(not p.requires_grad and p.grad is None for p in lora_model.parameters())


@pytest.fixture
def intake_artifacts(config, clean, samples, records, runner, tmp_path, monkeypatch):
    from src.phase0.drivema_grpo_train import load_config as training_config, run_identity
    from src.phase0.drivema_stage3a import load_config as rollout_config
    from src.phase0.phase0_4b_protocol import adapt_record
    from src.phase0.qwen3vl_dataset_adapter import load_config as ego_config, sha256_file
    from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
    temporal = []
    baseline = []
    for i in range(128):
        row = deepcopy(records['validation'][i % 4])
        row['sample_token'] = f'monitor-{i}'
        row['historical_sample_tokens'][-1] = row['sample_token']
        row['scene_token'] = f'validation-scene-{i % 2}'
        temporal.append(row)
        prediction = deepcopy(clean[i % 4])
        prediction.update(sample_token=row['sample_token'], scene_token=row['scene_token'])
        baseline.append(prediction)
    ego = ego_config(ROOT / 'configs/phase0_3_dataset_adapter.yaml')
    monitor = [adapt_record(row, ego, expected_split='validation') for row in temporal]
    import yaml
    directory = yaml.safe_load((ROOT / 'configs/phase0_4_temporal_dataset.yaml').read_text())['output_relative_dir']
    path = tmp_path / directory / 'validation.jsonl'
    path.parent.mkdir(parents=True)
    write_predictions(path, temporal)
    metadata = {'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION,
                'initial_sft_checkpoint': 'synthetic-sft', 'data_intake': {'manifests': {'validation': {
                    'relative_path': str(path.relative_to(tmp_path)), 'sha256': sha256_file(path)}}}}
    identity = run_identity(training_config(ROOT / 'configs/drivema_grpo.yaml'),
                            rollout_config(ROOT / 'configs/drivema_stage3a.yaml'),
                            samples['train'], monitor, runner, metadata)
    checkpoint = tmp_path / config.checkpoint_relative_path
    checkpoint.mkdir(parents=True)
    torch.save({'identity': identity, 'step': 256, 'baseline': baseline}, checkpoint / 'unified_state.pt')
    for relative in (config.sft_clean_relative_path, config.grpo_clean_relative_path):
        write_predictions(tmp_path / relative, list(reversed(baseline)))
    loaded_splits = []
    def load(repository, derived_root, *, split):
        loaded_splits.append(split)
        return temporal
    monkeypatch.setattr(ablation, 'load_temporal_records', load)
    return loaded_splits, baseline


def test_checkpoint_intake_alignment_and_baseline_contract(config, intake_artifacts, tmp_path):
    splits, baseline = intake_artifacts
    selected, donors, records, clean_rows, inference, intake = ablation.prepare_inputs(ROOT, tmp_path, config)
    assert splits == ['validation']
    assert [s.sample_token for s in selected] == [r['sample_token'] for r in baseline]
    assert clean_rows['sft'] == baseline and clean_rows['grpo'] == baseline
    assert len(donors) == 128 and not inference['compute_log_probs']
    bad = deepcopy(baseline)
    bad[0]['target_waypoints'][0][0] += 1
    write_predictions(tmp_path / config.grpo_clean_relative_path, bad)
    with pytest.raises(ValueError, match='target_waypoints mismatch'):
        ablation.prepare_inputs(ROOT, tmp_path, config)
    write_predictions(tmp_path / config.grpo_clean_relative_path, baseline)
    bad = deepcopy(baseline)
    bad[0]['raw_output'] = 'different'
    write_predictions(tmp_path / config.sft_clean_relative_path, bad)
    with pytest.raises(ValueError, match='frozen baseline'):
        ablation.prepare_inputs(ROOT, tmp_path, config)


def test_cli_run_loads_one_base_and_two_frozen_adapters(config, intake_artifacts, lora_model, processor,
                                                       runner, tmp_path, monkeypatch, capsys):
    from peft import PeftModel
    counts = {'base': 0, 'adapter': []}
    def load_base(*args):
        counts['base'] += 1
        return lora_model.get_base_model()
    def load_current(base, path, *, is_trainable):
        assert not is_trainable
        counts['adapter'].append(str(path))
        return lora_model
    def load_reference(path, *, adapter_name, is_trainable):
        assert adapter_name == 'reference' and not is_trainable
        counts['adapter'].append(str(path))
        lora_model.add_adapter('reference', deepcopy(lora_model.peft_config['default']))
    monkeypatch.setattr(PeftModel, 'from_pretrained', load_current)
    monkeypatch.setattr(lora_model, 'load_adapter', load_reference)
    runtime = SimpleNamespace(device_selector=lambda d: 'cpu', dtype_selector=lambda d: torch.bfloat16,
                              processor_loader=lambda *a: processor, model_loader=load_base,
                              image_loader=runner.collator.image_loader, package_version=lambda name: 'fixture')
    monkeypatch.setattr(ablation, 'default_runtime_dependencies', lambda: runtime)
    monkeypatch.setattr(ablation, 'collect_git_provenance', lambda root: SimpleNamespace(commit='fixture'))
    args = ['--derived-root', str(tmp_path), '--dataset-root', str(tmp_path), '--samples', '4']
    assert cli.main(args) == 0
    assert counts['base'] == 1 and len(counts['adapter']) == 2
    output = tmp_path / config.output_relative_dir
    assert (output / 'summary.json').exists()
    assert cli.main(['--mode', 'summarize', '--input-dir', str(output)]) == 0
    with pytest.raises(ValueError, match='new directory'):
        cli.main(args)
