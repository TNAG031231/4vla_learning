from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phase0_4b_lora_smoke import main
from src.phase0 import phase0_4b_lora_smoke as smoke
from src.phase0.phase0_4b_protocol import ActionTarget
from src.phase0.qwen3vl_dataset_adapter import GitProvenance
from test_phase0_4b_protocol import case, processor, samples


def test_selection_keeps_single_direction_supervision(samples):
    samples[0] = replace(samples[0], target=ActionTarget(None, "left", False, True))
    samples[1] = replace(samples[1], target=ActionTarget("stop", None, True, False))
    samples[2] = replace(samples[2], target=ActionTarget(None, None, False, False))
    chosen = smoke.select_tiny(samples, 3, 7)
    assert samples[0] in chosen and samples[1] in chosen and samples[2] not in chosen
    assert smoke.select_tiny(list(reversed(samples)), 3, 7) == chosen
    with pytest.raises(ValueError, match="only permits train"):
        smoke.select_tiny([replace(samples[0], split="test")], 1, 7)


def test_invalid_predictions_count_as_wrong_and_masks_are_independent():
    predictions = [
        {"target": {"longitudinal": "keep", "lateral": "left", "longitudinal_valid": True,
                    "lateral_valid": True}, "parsed_action": None},
        {"target": {"longitudinal": None, "lateral": "left", "longitudinal_valid": False,
                    "lateral_valid": True}, "parsed_action": {"longitudinal": "stop", "lateral": "left"}},
    ]
    result = smoke.metrics(predictions)
    assert result["joint_accuracy"] == result["longitudinal_accuracy"] == 0
    assert result["lateral_accuracy"] == result["parser_success_rate"] == 0.5
    assert result["longitudinal_count"] == 1 and result["lateral_count"] == 2


def test_cli_rejects_test_before_environment_access():
    with pytest.raises(SystemExit) as error:
        main(["--split", "test"])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="only permits train"):
        smoke.load_samples(Path("missing"), Path("missing"), split="test")


def test_config_prevents_full_training(tmp_path):
    import yaml
    config_path = ROOT / "configs/phase0_4b_lora_smoke.yaml"
    config = smoke.load_config(config_path)
    assert config.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    settings = yaml.safe_load(config_path.read_text())
    settings["train_subset_size"] = 14253
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="tiny subset"):
        smoke.load_config(bad)


def test_synthetic_runtime_save_fresh_reload_and_artifact(processor, samples, tmp_path, monkeypatch):
    selected = [replace(samples[0], target=ActionTarget("keep", "straight", True, True))]
    loaded = []
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(use_cache=True)

        def to(self, device):
            return self

        def forward(self, **batch):
            assert (batch["labels"] != -100).any()
            weight = self.lora_B if hasattr(self, "lora_B") else self.weight
            return SimpleNamespace(loss=(weight - 1).square().mean())

        def generate(self, **batch):
            text = "longitudinal=keep; lateral=straight" if hasattr(self, "lora_B") else "longitudinal=stop; lateral=left"
            ids = processor.tokenizer.encode(text, add_special_tokens=False)
            return torch.cat((batch["input_ids"], torch.tensor([ids])), dim=1)

        def gradient_checkpointing_enable(self):
            pass

        def enable_input_require_grads(self):
            pass

        def save_pretrained(self, path):
            path.mkdir()
            torch.save(self.lora_B.detach(), path / "synthetic_adapter.pt")

    def model_loader(*args):
        model = TinyModel()
        loaded.append(model)
        return model

    def inject(model, config):
        model.lora_B = torch.nn.Parameter(torch.zeros(1))
        return model

    def reload_adapter(model, path):
        assert model is loaded[1] and model is not loaded[0]
        model.lora_B = torch.nn.Parameter(torch.load(path / "synthetic_adapter.pt", weights_only=True))
        return model

    runtime = SimpleNamespace(
        model_loader=model_loader, processor_loader=lambda *args: processor,
        lora_config_factory=lambda **kwargs: kwargs, lora_injector=inject,
        adapter_loader=reload_adapter, optimizer_factory=lambda params, lr: torch.optim.SGD(params, lr=lr),
        image_loader=lambda path: Image.new("RGB", (64, 64)), dtype_selector=lambda _: torch.float32,
        device_selector=lambda _: "cpu", inference_context=nullcontext, package_version=lambda _: "synthetic",
    )
    monkeypatch.setattr(smoke, "default_runtime_dependencies", lambda: runtime)
    monkeypatch.setattr(smoke, "load_samples", lambda *args: selected)
    config = replace(smoke.load_config(ROOT / "configs/phase0_4b_lora_smoke.yaml"),
                     train_subset_size=1, max_steps=2, gradient_accumulation_steps=1, learning_rate=0.1)
    result = smoke.run_smoke(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                             config=config, git_provenance=GitProvenance("a" * 40, "synthetic", False, True))
    assert result["status"] == "smoke_passed"
    assert result["final_loss"] < result["initial_loss"]
    assert result["before_metrics"]["joint_accuracy"] == 0
    assert result["after_metrics"]["joint_accuracy"] == 1
    assert result["lora_parameters_updated"] is True
    assert result["lora_B_before"]["nonzero_element_count"] == 0
    assert result["lora_B_after"]["nonzero_element_count"] == 1
    assert result["test_records_read"] == result["test_images_opened"] == result["test_labels_read"] == 0
    output = tmp_path / config.output_relative_dir
    assert json.loads((output / "smoke_result.json").read_text())["reload_status"] == "success"
    with pytest.raises(FileExistsError):
        smoke.run_smoke(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path,
                        config=config, git_provenance=GitProvenance("a" * 40, "synthetic", False, True))
    assert len(loaded) == 2
