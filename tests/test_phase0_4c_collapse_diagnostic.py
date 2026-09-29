from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import diagnose_phase0_4c_collapse as cli
from src.phase0 import phase0_4c_collapse_diagnostic as diagnostic
from src.phase0.phase0_4c_tiny_overfit import CachedContext, load_config
from src.phase0.phase0_4c_two_turn_planner import PlannerConfig, WaypointDecoder, masked_waypoint_loss


@pytest.fixture
def setup():
    config = replace(load_config(ROOT / "configs/phase0_4c_tiny_overfit.yaml"),
                     planner_dimension=8, num_heads=2)
    torch.manual_seed(42)
    planner = WaypointDecoder(16, config)
    contexts = [CachedContext(
        f"synthetic-{i}", torch.randn(1, 3 + i, 16).bfloat16(),
        torch.tensor([[1] * (2 + i) + [0]]),
        torch.tensor([[[float(t + 1), i * 0.1] for t in range(6)]]),
        torch.tensor([[True, True, False, True, True, True]]),
    ) for i in range(8)]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield planner, contexts, config
    finally:
        torch.set_num_threads(threads)


def test_pairwise_statistics_and_zero_norm_cosine():
    result = diagnostic.pairwise_statistics(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
    torch.testing.assert_close(torch.tensor(result["pairwise_l2"]),
                               torch.tensor([[0., 2 ** .5, 2.], [2 ** .5, 0., 2 ** .5], [2., 2 ** .5, 0.]]))
    assert result["pairwise_cosine"] == [[1., 0., -1.], [0., 1., 0.], [-1., 0., 1.]]
    assert result["row_norms"] == [1., 1., 1.]
    assert result["off_diagonal_cosine"]["mean"] == pytest.approx(-1 / 3)
    zero = diagnostic.pairwise_statistics(torch.zeros(6, 8))
    assert zero["off_diagonal_l2"] == {"min": 0., "mean": 0., "max": 0.}
    assert zero["off_diagonal_cosine"] == {"min": None, "mean": None, "max": None}
    assert zero["undefined_off_diagonal_cosine_count"] == 30
    assert zero["pairwise_cosine"] == [[None] * 6 for _ in range(6)]
    json.dumps(zero, allow_nan=False)


@pytest.mark.parametrize("additional_norm", [False, True])
def test_representations_match_normal_forward_and_context_interventions(setup, additional_norm):
    planner, contexts, _ = setup
    if additional_norm:
        planner.decoder.norm = nn.LayerNorm(8)
    planner.eval()
    report = diagnostic.representation_probe(planner, contexts, device="cpu")
    assert report["additional_decoder_norm"] == additional_norm
    assert len(report["per_sample"]) == 8
    with torch.no_grad():
        for i, row in enumerate(report["per_sample"]):
            c, other = contexts[i], contexts[(i + 1) % 8]
            real = planner(c.hidden_states, c.attention_mask)
            swapped = planner(other.hidden_states, other.attention_mask)
            stages = row["stages"]
            assert stages["waypoint_projection"] == diagnostic.stage_statistics(real)
            assert stages["decoder_layer_1"]["shape"] == [1, 6, 8]
            assert ("final_decoder_norm" in stages) == additional_norm
            assert row["real_vs_other_memory"] == diagnostic.prediction_delta(real, swapped)
            hidden = c.hidden_states.float()
            memory = planner.context_projection(hidden)
            queries = planner.waypoint_queries.weight[None]
            zero = planner.waypoint_projection(planner.decoder(
                queries, torch.zeros_like(memory), memory_key_padding_mask=~c.attention_mask.bool()))
            assert row["real_vs_zero_memory"] == diagnostic.prediction_delta(real, zero)
            assert row["context"]["valid_tokens"] == 2 + i
            expected_mean = hidden[:, :-1].mean(dim=1)[0]
            assert row["context"]["masked_mean_hidden_norm"] == pytest.approx(float(expected_mean.norm()))
    assert report["pooled_contexts"]["hidden_states"]["shape"] == [8, 16]
    assert all(not module._forward_hooks for module in planner.modules())
    json.dumps(report, allow_nan=False)


def test_backward_once_matches_full_mean_and_never_updates_parameters(setup, monkeypatch):
    planner, contexts, config = setup
    before = {k: v.clone() for k, v in planner.state_dict().items()}
    reference = WaypointDecoder(16, config).eval()
    reference.load_state_dict(before)
    losses = [masked_waypoint_loss(reference(c.hidden_states, c.attention_mask), c.target,
                                   c.valid_mask, beta=config.smooth_l1_beta) for c in contexts]
    objective = torch.stack(losses).mean()
    expected = torch.autograd.grad(objective, tuple(reference.parameters()))
    calls = []
    original_backward = torch.Tensor.backward

    def backward(tensor, *args, **kwargs):
        calls.append(tensor)
        return original_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", backward)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("optimizer constructed"))
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *a, **kw: pytest.fail("optimizer constructed"))
    monkeypatch.setattr(torch, "save", lambda *a, **kw: pytest.fail("checkpoint saved"))
    report = diagnostic.diagnose(planner, contexts, beta=config.smooth_l1_beta, device="cpu")
    assert len(calls) == 1 and report["planner_state_unchanged"]
    assert report["backward"]["loss"] == pytest.approx(float(objective.detach()))
    for parameter, gradient in zip(planner.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)
    for name, value in planner.state_dict().items():
        assert torch.equal(value, before[name])
    for context, row in zip(contexts, report["backward"]["per_sample_output_gradients"], strict=True):
        with torch.no_grad():
            pred = planner(context.hidden_states, context.attention_mask)
        derivative = ((pred - context.target) / config.smooth_l1_beta).clamp(-1, 1)
        derivative[~context.valid_mask] = 0
        derivative /= (2 * context.valid_mask.sum() * len(contexts))
        torch.testing.assert_close(torch.tensor(row["dL_dprediction"]), derivative[0])
        assert not context.hidden_states.requires_grad and context.hidden_states.grad is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mismatch", [None, "configuration", "sample order"])
@pytest.mark.parametrize("probe", ["collapse", "layer2", "qkv"])
def test_cli_checkpoint_intake_and_artifact_preservation(setup, tmp_path, monkeypatch, mismatch, probe):
    planner, contexts, config = setup
    source = tmp_path / config.output_relative_dir
    source.mkdir(parents=True)
    # Same fields as fit_cached_contexts() checkpoint producer, with actual module state_dict.
    saved = {"planner_state_dict": planner.state_dict(),
             "planner_config": {name: getattr(config, name) for name in PlannerConfig.__dataclass_fields__},
             "training_config": asdict(config), "hidden_size": 16,
             "tiny_sample_tokens": [c.sample_token for c in contexts], "provenance": {"run_kind": "synthetic"}}
    if mismatch == "configuration":
        saved["training_config"]["smooth_l1_beta"] = 2.
    elif mismatch == "sample order":
        saved["tiny_sample_tokens"].reverse()
    checkpoint = source / "planner_state.pt"
    torch.save(saved, checkpoint)
    paths = [checkpoint]
    for version in ("v0_1", "v0_2", "v0_3"):
        path = tmp_path / "phase_0_4" / f"two_turn_planner_tiny_overfit_{version}" / "sentinel.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preserve")
        paths.append(path)
    if probe != "collapse":
        old_diagnostic = tmp_path / cli.OUTPUT_RELATIVE_DIR / "diagnostic.json"
        old_diagnostic.parent.mkdir(parents=True)
        old_diagnostic.write_text("preserve previous collapse diagnostic")
        paths.append(old_diagnostic)
    if probe == "qkv":
        old_layer2 = tmp_path / cli.LAYER2_OUTPUT_RELATIVE_DIR / "diagnostic.json"
        old_layer2.parent.mkdir(parents=True)
        old_layer2.write_text("preserve previous layer2 diagnostic")
        paths.append(old_layer2)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "collect_git_provenance", lambda path: SimpleNamespace(commit="test-only"))
    monkeypatch.setattr(cli, "load_temporal_records", lambda *a, **kw: [{"sample_token": c.sample_token} for c in contexts])
    monkeypatch.setattr(cli, "select_training_samples", lambda *a: [
        SimpleNamespace(sample_token=c.sample_token, observation=SimpleNamespace(frame_texts=())) for c in contexts])
    if mismatch:
        monkeypatch.setattr(cli, "default_runtime_dependencies", lambda: pytest.fail("model accessed"))
        with pytest.raises(ValueError, match=mismatch):
            cli.run(dataset_root=tmp_path, derived_root=tmp_path, probe=probe)
        return
    model = nn.Linear(16, 16)
    model.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))
    tokenizer = SimpleNamespace(convert_ids_to_tokens=str, decode=lambda ids: str(ids), all_special_ids=[])
    runtime = SimpleNamespace(
        device_selector=lambda value: "cpu", dtype_selector=lambda value: torch.bfloat16,
        processor_loader=lambda *a: SimpleNamespace(tokenizer=tokenizer), model_loader=lambda *a: model,
        adapter_loader=lambda base, path: base, package_version=lambda name: "synthetic",
    )
    monkeypatch.setattr(cli, "default_runtime_dependencies", lambda: runtime)

    def cache(**kwargs):
        assert all(not p.requires_grad and p.grad is None for p in kwargs["model"].parameters())
        if probe == "qkv":
            monkeypatch.setattr(model, "forward", lambda **kwargs: None)
            for context in contexts:
                model(input_ids=torch.arange(context.hidden_states.shape[1])[None])
            return contexts, [{"action_token_ids": [0]} for _ in contexts]
        return contexts, []

    monkeypatch.setattr(cli, "cache_contexts", cache)
    monkeypatch.setattr(torch, "save", lambda *a, **kw: pytest.fail("checkpoint saved"))
    result = cli.run(dataset_root=tmp_path, derived_root=tmp_path, probe=probe)
    output = tmp_path / {"collapse": cli.OUTPUT_RELATIVE_DIR, "layer2": cli.LAYER2_OUTPUT_RELATIVE_DIR,
                         "qkv": cli.QKV_OUTPUT_RELATIVE_DIR}[probe]
    assert json.loads((output / "diagnostic.json").read_text()) == result
    assert list(output.iterdir()) == [output / "diagnostic.json"]
    assert result["provenance"]["qwen_and_lora_frozen"]
    assert not model._forward_pre_hooks
    if probe == "qkv":
        assert all(token["token_id"] == token["absolute_index"] for s in result["per_sample"]
                   for h in s["heads"] for q in h["queries"] for token in q["top_tokens"])
    for path, (content, modified) in before.items():
        assert path.read_bytes() == content and path.stat().st_mtime_ns == modified
    with pytest.raises(FileExistsError):
        cli.run(dataset_root=tmp_path, derived_root=tmp_path, probe=probe)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_reject_split_before_any_access(split, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("checkpoint accessed"))
    monkeypatch.setattr(cli, "load_temporal_records", lambda *a, **kw: pytest.fail("dataset accessed"))
    with pytest.raises(ValueError, match="only permits train"):
        cli.run(dataset_root=tmp_path, derived_root=tmp_path, split=split)
    with pytest.raises(SystemExit) as error:
        cli.main(["--split", split])
    assert error.value.code == 2


def test_projection_collapse_is_visible_separately_from_decoder_diversity(setup):
    planner, contexts, _ = setup
    with torch.no_grad():
        planner.waypoint_projection.weight.zero_()
        planner.waypoint_projection.bias.copy_(torch.tensor([9.804, -0.085]))
    result = diagnostic.representation_probe(planner, contexts, device="cpu")
    for row in result["per_sample"]:
        assert row["stages"]["decoder_layer_2"]["off_diagonal_l2"]["mean"] > 0
        assert row["stages"]["waypoint_projection"]["off_diagonal_l2"]["max"] == 0
        assert row["real_vs_zero_memory"]["max_absolute_m"] == 0
        assert row["real_vs_other_memory"]["max_absolute_m"] == 0
