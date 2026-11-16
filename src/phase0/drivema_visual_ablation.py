from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
import yaml

from src.phase0.drivema_grpo_evaluation import evaluate_population, load_baseline_predictions
from src.phase0.drivema_visual_interventions import CONDITIONS, VisualAblationRunner, donor_mapping
from src.phase0.drivema_visual_metrics import condition_report, difference_in_differences
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4b_protocol import SFTSample, StructuredCollator
from src.phase0.phase0_4c_full_train import eligible_samples, write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import (
    collect_git_provenance, load_config as load_ego_config, resolve_derived_path, sha256_file,
)
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies


@dataclass(frozen=True)
class AblationConfig:
    seed: int
    gray_rgb: list[int]
    checkpoint_relative_path: str
    sft_clean_relative_path: str
    grpo_clean_relative_path: str
    output_relative_dir: str


def load_config(path: Path) -> AblationConfig:
    config = AblationConfig(**yaml.safe_load(path.read_text()))
    if type(config.seed) is not int or len(config.gray_rgb) != 3 or any(
            type(value) is not int or not 0 <= value <= 255 for value in config.gray_rgb):
        raise ValueError('seed must be integer and gray_rgb must be three RGB bytes')
    if len(set(config.gray_rgb)) != 1:
        raise ValueError('gray_rgb must be neutral gray')
    for name in ('checkpoint_relative_path', 'sft_clean_relative_path', 'grpo_clean_relative_path',
                 'output_relative_dir'):
        resolve_derived_path(Path('/derived'), getattr(config, name))
    return config


def select_frozen_adapter(model: nn.Module, adapter: str) -> None:
    model.set_adapter(adapter)
    model.requires_grad_(False)
    model.eval()


def prepare_inputs(repository: Path, derived_root: Path, config: AblationConfig
                   ) -> tuple[list[SFTSample], list[SFTSample], dict, dict, dict, dict]:
    checkpoint = resolve_derived_path(derived_root, config.checkpoint_relative_path)
    state = torch.load(checkpoint / 'unified_state.pt', map_location='cpu', weights_only=True)
    if state['step'] != 256:
        raise ValueError('P3-A requires GRPO checkpoint step 256')
    identity = state['identity']
    if identity['model'] != FIXED_MODEL_ID or identity['revision'] != FIXED_REVISION:
        raise ValueError('checkpoint model/revision differs')
    inference = identity['validation']
    if (inference['batch_size'] != 1 or not inference['optimized'] or inference['compute_log_probs']
            or inference['action_generation']['do_sample']
            or inference['action_generation']['num_beams'] != 1):
        raise ValueError('saved baseline must use optimized batch-one greedy inference without log probs')
    manifest = identity['data_intake']['manifests']['validation']
    manifest_path = resolve_derived_path(derived_root, manifest['relative_path'])
    if sha256_file(manifest_path) != manifest['sha256']:
        raise ValueError('validation manifest differs from checkpoint intake')
    temporal = yaml.safe_load((repository / 'configs/phase0_4_temporal_dataset.yaml').read_text())
    expected = resolve_derived_path(derived_root, temporal['output_relative_dir']) / 'validation.jsonl'
    if manifest_path != expected:
        raise ValueError('checkpoint validation manifest path differs from temporal intake')
    records = load_temporal_records(repository, derived_root, split='validation')
    ego = load_ego_config(repository / 'configs/phase0_3_dataset_adapter.yaml')
    donors, intake = eligible_samples(records, ego, 'validation')
    by_token = {sample.sample_token: sample for sample in donors}
    tokens = identity['monitor_tokens']
    if len(tokens) != 128 or len(set(tokens)) != 128:
        raise ValueError('P3-A requires the fixed 128-sample monitor population')
    samples = [by_token[token] for token in tokens]
    records = {row['sample_token']: row for row in records}
    clean = {name: load_baseline_predictions(resolve_derived_path(derived_root, path), samples, records)
             for name, path in (('sft', config.sft_clean_relative_path), ('grpo', config.grpo_clean_relative_path))}
    if clean['sft'] != state['baseline']:
        raise ValueError('saved SFT clean predictions differ from checkpoint frozen baseline')
    return samples, donors, records, clean, inference, intake


def summarize(output: Path) -> dict:
    def read(name: str) -> list[dict]:
        with (output / name).open() as stream:
            return [json.loads(line) for line in stream]
    clean = {model: read(f'{model}_clean_predictions.jsonl') for model in ('sft', 'grpo')}
    predictions = {model: {condition: read(f'{model}_{condition}_predictions.jsonl') for condition in CONDITIONS}
                   for model in clean}
    comparisons = {model: {condition: condition_report(clean[model], rows) for condition, rows in tracks.items()}
                   for model, tracks in predictions.items()}
    return {'models': comparisons, 'difference_in_differences': {
        condition: difference_in_differences(clean['sft'], predictions['sft'][condition],
                                             clean['grpo'], predictions['grpo'][condition])
        for condition in CONDITIONS},
        'interpretation': 'test-time visual dependency; not retrained ego-only ablation; '
                          'unchanged predictions do not prove vision is unused; '
                          'lateral left/right do not identify turns or lane changes'}


def evaluate_conditions(model: nn.Module, runner: VisualAblationRunner, samples: list[SFTSample], records: dict,
                        clean: dict, mapping: dict, config: AblationConfig, output: Path) -> dict:
    write_json(output / 'donor_mapping.json', {'seed': config.seed, 'mapping': mapping,
                                             'resize': 'PIL bicubic to each source slot resolution'})
    for name, adapter in (('sft', 'reference'), ('grpo', 'default')):
        select_frozen_adapter(model, adapter)
        write_predictions(output / f'{name}_clean_predictions.jsonl', clean[name])
        for condition in CONDITIONS:
            runner.configure(condition, mapping, tuple(config.gray_rgb), clean[name],
                             output / f'{name}_{condition}_input_evidence.jsonl')
            rows, metrics = evaluate_population(model, runner, samples, records)
            for row in rows:
                row['visual_intervention'] = runner.evidence[row['sample_token']]
            write_predictions(output / f'{name}_{condition}_predictions.jsonl', rows)
            write_json(output / f'{name}_{condition}_metrics.json', metrics)
            write_json(output / f'{name}_{condition}_comparison.json', condition_report(clean[name], rows))
    summary = summarize(output)
    write_json(output / 'summary.json', summary)
    return summary


def validate_sanity(path: Path, config: AblationConfig, derived_root: Path) -> None:
    metadata = json.loads((path / 'metadata.json').read_text())
    expected = asdict(config)
    actual = metadata['config'].copy()
    actual['output_relative_dir'] = expected['output_relative_dir']
    if (actual != expected or not 4 <= metadata['sample_count'] <= 8
            or metadata['derived_root'] != str(derived_root.resolve())):
        raise ValueError('sanity run must use the same checkpoint, clean inputs and intervention config')
    summarize(path)
    for model in ('sft', 'grpo'):
        for condition in CONDITIONS:
            with (path / f'{model}_{condition}_input_evidence.jsonl').open() as stream:
                evidence = [json.loads(line) for line in stream]
            if len(evidence) != metadata['sample_count'] or any(
                    not all(row['structure_matches'].values()) for row in evidence):
                raise ValueError('sanity visual structure check failed')


def run(repository: Path, dataset_root: Path, derived_root: Path, config: AblationConfig,
        sample_count: int, sanity_output: Path | None) -> dict:
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.exists() or output.is_relative_to(repository):
        raise ValueError('output must be a new directory outside the repository')
    if sample_count == 128:
        if sanity_output is None:
            raise ValueError('full P3-A requires --sanity-output from a completed 4–8 sample run')
        validate_sanity(sanity_output, config, derived_root)
    provenance = collect_git_provenance(repository)
    samples, donors, records, clean, inference, intake = prepare_inputs(repository, derived_root, config)
    # Build on the full monitor order so sanity and full use identical donor assignments.
    mapping = donor_mapping(samples, donors, config.seed)
    for replacements in mapping.values():
        for replacement in replacements:
            for role in ('source', 'donor'):
                record = records[replacement[f'{role}_sample_token']]
                replacement[f'{role}_frame_sample_token'] = record['historical_sample_tokens'][
                    replacement[f'{role}_history_slot']]
    samples = samples[:sample_count]
    clean = {name: rows[:sample_count] for name, rows in clean.items()}
    mapping = {sample.sample_token: mapping[sample.sample_token] for sample in samples}
    from peft import PeftModel
    runtime = default_runtime_dependencies()
    device = runtime.device_selector('cuda:0')
    dtype = runtime.dtype_selector('bfloat16')
    if dtype != torch.bfloat16:
        raise ValueError('BF16-capable GPU required')
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, 'sdpa', True)
    checkpoint = resolve_derived_path(derived_root, config.checkpoint_relative_path)
    model = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=False)
    model.load_adapter(checkpoint / 'adapter/reference', adapter_name='reference', is_trainable=False)
    model.to(device)
    runner = VisualAblationRunner(
        processor, StructuredCollator(processor, runtime.image_loader, dataset_root), device,
        inference['action_generation'], inference['trajectory_max_new_tokens'],
        inference_batch_size=1, optimized=True, compute_log_probs=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metadata.json', {
        'config': asdict(config), 'inference': inference, 'sample_count': sample_count,
        'population_tokens': [s.sample_token for s in samples], 'data_intake': intake,
        'model_id': FIXED_MODEL_ID, 'revision': FIXED_REVISION, 'adapter_roles': {'sft': 'reference', 'grpo': 'default'},
        'derived_root': str(derived_root.resolve()), 'git_commit': provenance.commit,
        'torch': str(torch.__version__), 'transformers': runtime.package_version('transformers'),
        'peft': runtime.package_version('peft'), 'dtype': str(dtype), 'attention': 'sdpa',
        'test_records_read': 0, 'clean_inference': 'reused; no generation',
        'grpo_clean_provenance': 'user-specified step-256 monitor file; tokens/targets/prefix validated; '
                                 'checkpoint does not embed its own current-policy predictions',
    })
    return evaluate_conditions(model, runner, samples, records, clean, mapping, config, output)
