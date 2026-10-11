from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase0_4c_ego_fusion as cli
from src.baselines import ego_history_mlp as ego
from src.phase0 import phase0_4c_ego_fusion as fusion
from src.phase0 import phase0_4c_no_action as neutral
from src.phase0.phase0_4c_two_turn_planner import masked_waypoint_loss
from src.phase0.qwen3vl_dataset_adapter import GitProvenance
from test_ego_history_mlp import case, records, artifacts
from test_phase0_4b_protocol import processor
from test_phase0_4c_two_turn_planner import tiny_qwen
from test_phase0_4c_no_action import config, runtime, references, small


@pytest.fixture
def fusion_config():
    return fusion.load_config(ROOT / "configs/phase0_4c_ego_fusion.yaml", ROOT)


@pytest.fixture
def fusion_inputs(references, small, fusion_config, monkeypatch):
    git = GitProvenance(commit="a" * 40, branch="synthetic", detached_head=False, worktree_clean=True)
    neutral.run(repository=ROOT, dataset_root=references, derived_root=references,
                config=small, git_provenance=git)
    monkeypatch.setattr(neutral, "load_config", lambda *a: small)
    config = replace(fusion_config, planner_dimension=small.planner_dimension,
                     num_decoder_layers=small.num_decoder_layers,
                     gradient_accumulation_steps=small.gradient_accumulation_steps,
                     expected_validation_count=small.expected_validation_count,
                     expected_motion_unavailable_count=small.expected_motion_unavailable_count)
    return dict(repository=ROOT, dataset_root=references, derived_root=references,
                config=config, git_provenance=git)


def test_real_context_ego_condition_gradients_and_no_target_input(
        fusion_config, records, artifacts, processor, tiny_qwen, monkeypatch):
    assert sum(p.numel() for p in fusion.EgoWaypointDecoder(2560, fusion_config).parameters()) == 2_800_898
    config = replace(fusion_config, planner_dimension=16, num_decoder_layers=1, dropout=0.)
    planner = fusion.EgoWaypointDecoder(32, config)
    planner.normalization = ego.normalization_stats(records[0], config.normalization_epsilon)
    train, _, raw, _ = fusion.direct.prepare_data(ROOT, artifacts[0])
    s = train[-1]
    sample = fusion.FusionSample(s.sample_token, s.scene_token, s.split, s.observation,
                                 raw[s.sample_token]["ego_motion_history"])
    monkeypatch.setattr(fusion.direct, "resolve_image_path", lambda root, path: root / path)
    runner = fusion.FusionRunner(tiny_qwen, processor, SimpleNamespace(
        image_loader=lambda p: Image.new("RGB", (64, 64))), Path("unused"), "cpu")
    before, evidence = runner.predict(planner, sample)
    assert before.shape == (1, 6, 2) and evidence["ego_feature_shape"] == [1, 21]
    assert runner.context_evidence["conditioning_type"] == fusion.TRACK
    assert runner.context_evidence["rendered_context"].count(neutral.NEUTRAL_CONTEXT) == 1
    assert not hasattr(sample, "target")
    raw[s.sample_token].update(future_waypoints=[[987654., -987654.]] * 6,
                               longitudinal_action="stop", lateral_action="right")
    after, _ = runner.predict(planner, sample)
    assert torch.equal(before, after)
    masked_waypoint_loss(before, torch.ones_like(before), torch.ones(1, 6, dtype=torch.bool), beta=1.).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in planner.ego_encoder.parameters())
    assert any(p.grad.abs().sum() > 0 for p in planner.ego_encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in tiny_qwen.parameters())
    optimizer = torch.optim.AdamW(planner.parameters(), lr=config.learning_rate)
    weight = planner.ego_encoder[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(weight, planner.ego_encoder[0].weight)
    hidden, mask = torch.zeros(1, 3, 32), torch.ones(1, 3, dtype=torch.long)
    assert torch.equal(planner(hidden, mask), planner(hidden, mask, torch.zeros(1, 16)))
    assert not torch.allclose(planner(hidden, mask, planner.ego_encoder(torch.zeros(1, 21))),
                              planner(hidden, mask, planner.ego_encoder(torch.ones(1, 21))))


def test_two_epoch_producer_intake_training_reload_and_comparisons(fusion_inputs, runtime, records):
    kwargs = fusion_inputs
    output = kwargs["derived_root"] / kwargs["config"].output_relative_dir
    model_count = len(runtime[1])
    dry = fusion.run(**kwargs, dry_run=True)
    assert dry["ego_normalization_fit_split"] == "train"
    assert len(runtime[1]) == model_count and not output.exists()
    result = fusion.run(**kwargs)
    assert set(result["models"]) == {fusion.TRACK, *fusion.REFERENCE_FILES}
    assert result["delta_definition"] == "fusion_minus_reference; negative_is_better"
    assert result["pairwise"]["fusion_minus_no_action_neutral_context"]["ade_3s_m"]["paired_sample_count"] == 4
    summary = json.loads((output / "training_summary.json").read_text())
    assert summary["selected_epoch"] == 2 and summary["optimizer_steps"] == 4
    assert summary["reload_consistency"]["reload_consistency"]
    assert summary["test_isolation"]["test_records_read"] == 0
    assert summary["test_isolation"]["test_evaluation_performed"] is False
    stats = json.loads((output / "normalization_stats.json").read_text())
    assert stats == ego.normalization_stats(records[0], kwargs["config"].normalization_epsilon)
    reloaded = fusion.reload_planner(output / "epoch_2.pt", "cpu")
    assert reloaded.normalization == stats
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["freeze_policy"]["qwen_trainable_parameters"] == 0
    assert metadata["freeze_policy"]["lora_trainable_parameters"] == 0
    assert metadata["conditioning_protocol"] == {"train": fusion.TRACK, "validation": fusion.TRACK}
    subset = json.loads((output / "motion_unavailable_comparison.json").read_text())
    assert subset["sample_count"] == 1
    assert not (output / "epoch_3.pt").exists()
    with pytest.raises(FileExistsError):
        fusion.run(**kwargs)


@pytest.mark.parametrize("failure", ["split", "target", "metadata"])
def test_external_intake_rejects_before_model_access(fusion_inputs, monkeypatch, failure):
    kwargs = fusion_inputs
    if failure == "split":
        kwargs["validation_split"] = "test"
    else:
        directory = kwargs["derived_root"] / "phase_0_4/no_action_ablation_v0_1"
        if failure == "target":
            path = directory / "epoch_2_predictions.jsonl"
            rows = ego.read_predictions(path, neutral.TRACK)
            rows[0]["target_waypoints"][0][0] += 1
            neutral.full.write_predictions(path, rows)
        else:
            path = directory / "run_metadata.json"
            metadata = json.loads(path.read_text())
            metadata["config"]["seed"] += 1
            neutral.write_json(path, metadata)
    monkeypatch.setattr(fusion.convergence, "prepare_run", lambda *a, **k: pytest.fail("model accessed"))
    with pytest.raises(ValueError):
        fusion.run(**kwargs, dry_run=True)
    assert not (kwargs["derived_root"] / kwargs["config"].output_relative_dir).exists()


def test_cli_test_split_rejected_before_run(monkeypatch):
    monkeypatch.setattr(cli, "run", lambda **k: pytest.fail("run accessed"))
    with pytest.raises(SystemExit):
        cli.main(["--validation-split", "test"])
