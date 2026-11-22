from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_drivema_action_oracle as oracle
from scripts import run_drivema_grpo as grpo
from src.phase0 import drivema_grpo_train as training
from src.phase0 import drivema_stage3a
from src.phase0.drivema_stage2_protocol import serialize_trajectory
from src.phase0.drivema_stage2b import select_subset
from src.phase0.phase0_4c_full_train import write_predictions
from test_drivema_stage2_sft import case, lora_model, processor, records, runner, scripted_generate, tiny_qwen
from test_drivema_stage3b import dual_model


def test_full_validation_producer_to_oracle_cli(dual_model, processor, records, runner, tmp_path, monkeypatch):
    for index, row in enumerate(records['validation']):
        row.update(longitudinal_action='keep' if index == 0 else 'accelerate', lateral_action='straight')
    temporal = yaml.safe_load((ROOT / 'configs/phase0_4_temporal_dataset.yaml').read_text())
    manifests = tmp_path / temporal['output_relative_dir']
    manifests.mkdir(parents=True)
    for split, rows in records.items():
        write_predictions(manifests / f'{split}.jsonl', rows)
    reads = []

    def read_records(repository, derived, *, split):
        reads.append(split)
        return records[split]

    for module in (training, drivema_stage3a, oracle):
        monkeypatch.setattr(module, 'load_temporal_records', read_records)
    train, validation, _, intake = training.prepare_data(ROOT, tmp_path)
    config = replace(training.load_config(ROOT / 'configs/drivema_grpo.yaml'), monitor_samples=2)
    config_path = tmp_path / 'training.yaml'
    config_path.write_text(yaml.safe_dump(asdict(config)))
    rollout = training.training_rollout_config(ROOT, config)
    runner.optimized, runner.compute_log_probs = True, False
    identity = training.run_identity(config, rollout, train, select_subset(validation, 2, rollout.seed, 'validation'),
                                     runner, {'model_id': oracle.FIXED_MODEL_ID, 'revision': oracle.FIXED_REVISION,
                                              'initial_sft_checkpoint': rollout.checkpoint_relative_path,
                                              'data_intake': intake})
    checkpoint = tmp_path / 'checkpoint_768'
    dual_model.save_pretrained(checkpoint / 'adapter')
    torch.save({'step': 768, 'identity': identity}, checkpoint / 'unified_state.pt')
    base = dual_model.unload()
    calls = []
    responses = ['longitudinal=keep; lateral=straight', serialize_trajectory([[1., 0.]] * 6)]
    monkeypatch.setattr(type(base), 'generate', scripted_generate(processor, responses, calls))
    runtime = SimpleNamespace(device_selector=lambda _: 'cpu', dtype_selector=lambda _: torch.bfloat16,
                              processor_loader=lambda *args: processor, model_loader=lambda *args: deepcopy(base),
                              image_loader=runner.collator.image_loader, package_version=lambda _: 'test')
    for module in (grpo, oracle):
        monkeypatch.setattr(module, 'default_runtime_dependencies', lambda: runtime)
        monkeypatch.setattr(module, 'collect_git_provenance', lambda _: SimpleNamespace(commit='test-fixture'))
    common = ['--dataset-root', str(tmp_path), '--derived-root', str(tmp_path)]
    assert grpo.main(common + ['--config', str(config_path), '--mode', 'evaluate-full', '--checkpoint', str(checkpoint),
                               '--output-relative-dir', 'full']) == 0
    reads.clear()
    calls.clear()
    args = common + ['--full-validation-relative-dir', 'full', '--local-model-dir', str(tmp_path),
                     '--output-relative-dir', 'diagnostic']
    assert oracle.main(args) == 0
    assert reads == ['validation']
    result = tmp_path / 'diagnostic'
    metadata = json.loads((result / 'metadata.json').read_text())
    assert metadata['step'] == 768 and Path(metadata['checkpoint']) == checkpoint
    assert metadata['inference'] == identity['validation']
    pairs = [json.loads(line) for line in (result / 'pairs.jsonl').read_text().splitlines()]
    assert [pair['sample_token'] for pair in pairs] == [sample.sample_token for sample in validation[1:]]
    assert all(pair['changed_axes'] == ['longitudinal'] and pair['conditioning_type'] == 'gt_action_diagnostic'
               for pair in pairs)
    assert sum(pair['control_replayed'] for pair in pairs) == 1
    assert len(calls) == len(pairs) + 1
    summary = json.loads((result / 'summary.json').read_text())
    assert (summary['population_count'], summary['wrong_action_count'], summary['paired_valid_count']) == (4, 3, 3)
    assert summary['full_population_mean_delta_m']['ade_3s_m'] == 0

    torch.save({'step': 767, 'identity': identity}, checkpoint / 'unified_state.pt')
    with pytest.raises(ValueError, match='checkpoint and full-validation run differ'):
        oracle.main(args[:-1] + ['mismatched'])
    assert not (tmp_path / 'mismatched').exists()
