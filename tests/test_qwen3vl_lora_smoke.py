from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_qwen3vl_lora_smoke as cli_module
from src.actions.schema import ACTION_SCHEMA
from src.phase0.qwen3vl_dataset_adapter import (
    GitProvenance,
    adapter_record,
    load_config as load_adapter_config,
)
import src.phase0.qwen3vl_lora_smoke as lora_module
from src.phase0.qwen3vl_lora_smoke import (
    IGNORE_INDEX,
    LORA_TARGET_MODULES,
    Qwen3VLSupervisedCollator,
    RuntimeDependencies,
    TrainingArtifact,
    build_training_messages,
    inject_lora,
    load_adapter_samples,
    load_config,
    run_lora_smoke,
    run_training_steps,
    select_train_subset,
    select_validation_subset,
    trainable_parameter_report,
)
from src.phase0.qwen3vl_zero_shot import AdapterSample


CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_lora_smoke.yaml"
ADAPTER_CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_dataset_adapter.yaml"
ACTION_TOKEN_IDS = {
    action: 200 + index for index, action in enumerate(ACTION_SCHEMA)
}


def _sample(
    index: int,
    *,
    split: str = "train",
    action: str = "keep",
    ego_state_text: str = "Current ego state:\nspeed_mps=1.000000",
) -> AdapterSample:
    return AdapterSample(
        sample_token=f"sample-{index:04d}",
        scene_token=f"scene-{index:04d}",
        split=split,
        variant="image_ego_state",
        cam_front_path=f"samples/CAM_FRONT/{split}-{index:04d}.jpg",
        target_action=action,
        ego_state_text=ego_state_text,
        adapter_record_sha256=f"{index + 1:064x}",
    )


def _class_covered_samples(repeats: int = 2) -> tuple[AdapterSample, ...]:
    return tuple(
        _sample(index, action=action)
        for index, action in enumerate(ACTION_SCHEMA * repeats)
    )


def _git() -> GitProvenance:
    return GitProvenance(
        commit="a" * 40,
        branch="task_p0_3d1_lora_smoke",
        detached_head=False,
        worktree_clean=True,
    )


class FakeProcessor:
    def __init__(self, generation_output: str = "keep") -> None:
        self.generation_output = generation_output
        self.full_conversations = []

    @staticmethod
    def _prefix_ids(conversation) -> list[int]:
        text = conversation[0]["content"][1]["text"]
        extra = [104] if "extra_token" in text else []
        return [900, 901, 101, 102, *extra, 103]

    def apply_chat_template(self, messages, **kwargs):
        conversations = messages if isinstance(messages[0], list) else [messages]
        rows = []
        for conversation in conversations:
            row = self._prefix_ids(conversation)
            if conversation[-1]["role"] == "assistant":
                action = conversation[-1]["content"][0]["text"]
                row = [*row, ACTION_TOKEN_IDS[action], 999]
                self.full_conversations.append(conversation)
            rows.append(row)
        width = max(len(row) for row in rows)
        input_ids = torch.tensor(
            [row + [0] * (width - len(row)) for row in rows],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in rows],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.ones((len(rows) * 2, 3)),
            "image_grid_thw": torch.ones((len(rows), 3), dtype=torch.long),
            "mm_token_type_ids": attention_mask.clone(),
        }

    def batch_decode(self, values, **kwargs):
        rows = values.tolist() if hasattr(values, "tolist") else values
        decoded = []
        for row in rows:
            tokens = [int(token) for token in row if int(token) != 999]
            if 777 in tokens:
                decoded.append("driving")
                continue
            action = next(
                (
                    name
                    for name, token_id in ACTION_TOKEN_IDS.items()
                    if token_id in tokens
                ),
                self.generation_output,
            )
            decoded.append(action)
        return decoded


class FakeModel(torch.nn.Module):
    def __init__(self, generation_output: str = "keep") -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.zeros(4))
        self.config = SimpleNamespace(use_cache=True)
        self.generation_output = generation_output
        self.forward_input_ids = []
        self.operation_history = []
        self.gradient_checkpointing_enabled = False
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True

    def forward(self, input_ids, **kwargs):
        self.operation_history.append("forward")
        self.forward_input_ids.append(input_ids.detach().clone())
        loss = (self.lora_A - 1.0).pow(2).mean()
        return SimpleNamespace(loss=loss)

    def generate(self, input_ids, **kwargs):
        self.operation_history.append("generate")
        token = 777 if self.generation_output == "driving" else ACTION_TOKEN_IDS[
            self.generation_output
        ]
        generated = torch.full(
            (input_ids.shape[0], 1), token, dtype=input_ids.dtype
        )
        return torch.cat((input_ids, generated), dim=1)

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(b"fake-lora")


def _runtime(
    generation_output: str = "keep",
    *,
    reloaded_generation_output: str | None = None,
):
    state = {
        "models": [],
        "lora_kwargs": None,
        "adapter_path": None,
        "opened_images": [],
    }
    processor = FakeProcessor(generation_output)

    def model_loader(*args):
        model = FakeModel(generation_output)
        state["models"].append(model)
        return model

    def lora_config_factory(**kwargs):
        state["lora_kwargs"] = kwargs
        return kwargs

    def add_lora(model, config):
        model.register_parameter(
            "lora_A", torch.nn.Parameter(torch.zeros(2))
        )
        model.register_parameter(
            "lora_B", torch.nn.Parameter(torch.zeros(2))
        )
        return model

    def adapter_loader(model, path):
        state["adapter_path"] = path
        if reloaded_generation_output is not None:
            model.generation_output = reloaded_generation_output
        return add_lora(model, {})

    def image_loader(path):
        state["opened_images"].append(path)
        return object()

    dependencies = RuntimeDependencies(
        processor_loader=lambda *args: processor,
        model_loader=model_loader,
        lora_config_factory=lora_config_factory,
        lora_injector=add_lora,
        adapter_loader=adapter_loader,
        optimizer_factory=lambda parameters, learning_rate: torch.optim.SGD(
            parameters, lr=learning_rate
        ),
        image_loader=image_loader,
        dtype_selector=lambda preference: torch.float32,
        device_selector=lambda device: "cpu",
        inference_context=nullcontext,
        package_version=lambda name: f"fake-{name}",
    )
    return dependencies, processor, state


def test_config_freezes_real_peft_lora_contract() -> None:
    config = load_config(CONFIG_PATH)
    assert config.model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert config.model_revision == lora_module.FIXED_REVISION
    assert config.input_variant == "image_ego_state"
    assert config.optimization_split == "train"
    assert config.evaluation_split == "validation"
    assert config.lora_r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert config.lora_bias == "none"
    assert config.lora_task_type == "CAUSAL_LM"
    assert config.lora_target_modules == LORA_TARGET_MODULES


def test_optional_runtime_packages_are_lazy_imported(monkeypatch) -> None:
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"transformers", "peft", "accelerate"}:
            raise AssertionError(f"unexpected eager import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    importlib.reload(lora_module)


def test_train_selection_is_deterministic_class_covered_and_train_only() -> None:
    samples = _class_covered_samples()
    first = select_train_subset(samples, size=10, seed=17)
    second = select_train_subset(tuple(reversed(samples)), size=10, seed=17)
    assert [sample.sample_token for sample in first] == [
        sample.sample_token for sample in second
    ]
    assert {sample.target_action for sample in first} == set(ACTION_SCHEMA)
    assert {sample.split for sample in first} == {"train"}
    with pytest.raises(ValueError, match="train only"):
        select_train_subset(
            (*samples, _sample(99, split="validation")), size=10, seed=17
        )


def test_validation_selection_never_enters_optimization() -> None:
    validation = tuple(
        _sample(index, split="validation", action=ACTION_SCHEMA[index % 6])
        for index in range(12)
    )
    selected = select_validation_subset(validation, size=4, seed=9)
    assert len(selected) == 4
    assert {sample.split for sample in selected} == {"validation"}
    with pytest.raises(ValueError, match="validation only"):
        select_validation_subset(
            (*validation, _sample(99, split="train")), size=4, seed=9
        )


def test_existing_adapter_record_is_consumed_without_schema_duplication(
    tmp_path: Path,
) -> None:
    adapter_config = load_adapter_config(ADAPTER_CONFIG_PATH)
    motion = {
        "speed_mps": 2.0,
        "longitudinal_acceleration_mps2": 0.25,
        "yaw_rate_radps": -0.125,
        "source": "ego_pose_past_difference",
        "timestamp_source": "CAM_FRONT_sample_data",
        "availability": "full",
        "history_interval_sec": 0.5,
        "acceleration_interval_sec": 1.0,
        "unavailable_reason": None,
    }
    source = {
        "sample_token": "sample-0001",
        "scene_token": "scene-0001",
        "timestamp": 1,
        "split": "train",
        "cam_front_path": "samples/CAM_FRONT/0001.jpg",
        "current_ego_motion": motion,
        "target_action": "keep",
        "projection_record_sha256": "b" * 64,
    }
    record = adapter_record(
        source,
        variant="image_ego_state",
        config=adapter_config,
        source_file_sha256=adapter_config.source_files["train"].sha256,
    )
    path = tmp_path / "train_image_ego_state.jsonl"
    path.write_text(
        lora_module.canonical_json_bytes(record).decode("utf-8") + "\n",
        encoding="utf-8",
    )
    artifact = TrainingArtifact(
        receipt_sha256="c" * 64,
        adapter_config=adapter_config,
        train_path=path,
        validation_path=tmp_path / "unused.jsonl",
        train_sha256="d" * 64,
        validation_sha256="e" * 64,
    )
    samples = load_adapter_samples(path=path, split="train", artifact=artifact)
    assert len(samples) == 1
    assert samples[0].ego_state_text is not None
    assert samples[0].target_action == "keep"


def test_collator_masks_prompt_image_ego_and_padding_tokens() -> None:
    config = load_config(CONFIG_PATH)
    processor = FakeProcessor()
    opened = []
    collator = Qwen3VLSupervisedCollator(
        processor=processor,
        image_loader=lambda path: opened.append(path) or object(),
        nuscenes_root=Path("/dataset"),
        config=config,
    )
    samples = (
        _sample(0, action="keep"),
        _sample(1, action="stop", ego_state_text="extra_token"),
    )
    batch = collator(samples, expected_split="train")
    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
        "labels",
    }
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["image_grid_thw"].shape[0] == len(samples)
    assert len(opened) == len(samples)
    for index, sample in enumerate(samples):
        labels = batch["labels"][index]
        mask = batch["attention_mask"][index]
        supervised = labels[labels != IGNORE_INDEX].tolist()
        assert processor.batch_decode([supervised])[0] == sample.target_action
        supervised_positions = torch.nonzero(
            labels != IGNORE_INDEX, as_tuple=False
        ).flatten()
        assert len(supervised_positions) == 2
        assert torch.all(
            labels[: supervised_positions[0]] == IGNORE_INDEX
        )
        assert torch.all(labels[mask == 0] == IGNORE_INDEX)
        assert labels[0].item() == IGNORE_INDEX
        assert labels[1].item() == IGNORE_INDEX
        assert labels[2].item() == IGNORE_INDEX
    assert torch.equal(batch["mm_token_type_ids"], batch["attention_mask"])
    assert batch["pixel_values"].shape[0] == 2 * len(samples)


def test_collator_rejects_test_before_image_access() -> None:
    opened = []
    collator = Qwen3VLSupervisedCollator(
        processor=FakeProcessor(),
        image_loader=lambda path: opened.append(path),
        nuscenes_root=Path("/dataset"),
        config=load_config(CONFIG_PATH),
    )
    with pytest.raises(ValueError, match="only permits train and validation"):
        collator([_sample(0, split="test")], expected_split="test")
    assert opened == []


def test_training_conversation_assistant_contains_only_exact_action() -> None:
    messages = build_training_messages(
        _sample(0, action="left_lateral"), object(), load_config(CONFIG_PATH)
    )
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "left_lateral"}],
    }
    assert "left_lateral" not in messages[0]["content"][1]["text"].splitlines()[-1]


def test_lora_injection_freezes_base_and_reports_trainable_parameters() -> None:
    dependencies, _, state = _runtime()
    model = inject_lora(
        FakeModel(), config=load_config(CONFIG_PATH), dependencies=dependencies
    )
    named = dict(model.named_parameters())
    assert named["base_weight"].requires_grad is False
    assert named["lora_A"].requires_grad is True
    assert named["lora_B"].requires_grad is True
    assert state["lora_kwargs"] == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    report = trainable_parameter_report(model, LORA_TARGET_MODULES)
    assert report["total_parameter_count"] == 8
    assert report["trainable_parameter_count"] == 4
    assert report["trainable_percentage"] == 50.0
    assert report["lora_target_module_names"] == list(LORA_TARGET_MODULES)


def test_fake_finite_loss_training_step_uses_train_samples_only() -> None:
    config = replace(
        load_config(CONFIG_PATH),
        max_steps=2,
        gradient_accumulation_steps=2,
    )
    dependencies, processor, _ = _runtime()
    model = inject_lora(
        FakeModel(), config=config, dependencies=dependencies
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    collator = Qwen3VLSupervisedCollator(
        processor=processor,
        image_loader=lambda path: object(),
        nuscenes_root=Path("/dataset"),
        config=config,
    )
    history = run_training_steps(
        model=model,
        samples=_class_covered_samples(1),
        collator=collator,
        optimizer=optimizer,
        config=config,
        device="cpu",
    )
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(history)))
    assert history[-1] < history[0]
    assert len(model.forward_input_ids) == 4


def test_test_split_fails_before_artifact_model_or_image_access(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        lora_module,
        "load_training_artifact",
        lambda **kwargs: pytest.fail("artifact must not be accessed"),
    )
    with pytest.raises(ValueError, match="only permits train"):
        run_lora_smoke(
            config=load_config(CONFIG_PATH),
            config_relative_path="configs/phase0_3_lora_smoke.yaml",
            repository_root=tmp_path,
            nuscenes_root=tmp_path,
            derived_root=tmp_path,
            optimization_split="test",
            evaluation_split="validation",
            git_provenance=_git(),
            dependencies=None,
        )


def test_cli_rejects_test_before_environment_or_runtime_access(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NUSCENES_ROOT", raising=False)
    monkeypatch.delenv("VLA_DERIVED_ROOT", raising=False)
    assert cli_module.main(["--optimization-split", "test"]) == 2


@pytest.mark.parametrize(
    ("max_steps", "expected_status"),
    [
        (2, "smoke_passed"),
        (1, "smoke_completed_without_loss_decrease"),
    ],
)
def test_before_after_learning_evidence_and_smoke_status(
    tmp_path: Path, monkeypatch, max_steps: int, expected_status: str
) -> None:
    train_samples = (*_class_covered_samples(1), _sample(99, action="keep"))
    validation_samples = tuple(
        _sample(index + 20, split="validation", action=ACTION_SCHEMA[index])
        for index in range(2)
    )
    artifact = SimpleNamespace(
        train_path=Path("train.jsonl"),
        validation_path=Path("validation.jsonl"),
        receipt_sha256="f" * 64,
    )
    monkeypatch.setattr(
        lora_module, "load_training_artifact", lambda **kwargs: artifact
    )
    monkeypatch.setattr(
        lora_module,
        "load_adapter_samples",
        lambda path, split, artifact: (
            train_samples if split == "train" else validation_samples
        ),
    )
    config = replace(
        load_config(CONFIG_PATH),
        train_subset_size=7,
        validation_subset_size=2,
        max_steps=max_steps,
        gradient_accumulation_steps=1,
        learning_rate=0.1,
    )
    dependencies, _, state = _runtime(
        "stop", reloaded_generation_output="keep"
    )
    repository_root = tmp_path / "repository"
    derived_root = tmp_path / "derived"
    nuscenes_root = tmp_path / "nuscenes"
    repository_root.mkdir()
    derived_root.mkdir()
    nuscenes_root.mkdir()
    result = run_lora_smoke(
        config=config,
        config_relative_path="configs/phase0_3_lora_smoke.yaml",
        repository_root=repository_root,
        nuscenes_root=nuscenes_root,
        derived_root=derived_root,
        optimization_split="train",
        evaluation_split="validation",
        git_provenance=_git(),
        dependencies=dependencies,
    )
    checkpoint = Path(result["checkpoint"]["adapter_path"])
    assert checkpoint == state["adapter_path"]
    assert (checkpoint / "adapter_model.safetensors").is_file()
    assert result["checkpoint"]["full_model_saved"] is False
    assert result["checkpoint_reload_result"]["completed"] is True
    assert result["training_sample_count"] == 7
    assert result["validation_smoke_sample_count"] == 2
    assert set(result["selected_train_action_distribution"]) == set(ACTION_SCHEMA)
    assert result["test_records_read"] == 0
    assert result["test_images_opened"] == 0
    assert result["test_labels_read"] == 0
    assert result["test_evaluation_performed"] is False
    assert result["validation_smoke_metrics"]["parser_success_count"] == 2
    assert (checkpoint.parent / "smoke_result.json").is_file()
    pretrain_tokens = [
        prediction["sample_token"]
        for prediction in result["pretrain_tiny_predictions"]
    ]
    posttrain_tokens = [
        prediction["sample_token"]
        for prediction in result["tiny_overfit_predictions"]
    ]
    assert pretrain_tokens == posttrain_tokens
    before_metrics = result["pretrain_tiny_metrics"]
    after_metrics = result["tiny_overfit_metrics"]
    summary = result["learning_summary"]
    assert summary["train_accuracy_before"] == before_metrics["accuracy"]
    assert summary["train_accuracy_after"] == after_metrics["accuracy"]
    assert summary["train_accuracy_delta"] == pytest.approx(
        after_metrics["accuracy"] - before_metrics["accuracy"]
    )
    assert summary["train_macro_f1_before"] == before_metrics["macro_f1"]
    assert summary["train_macro_f1_after"] == after_metrics["macro_f1"]
    assert summary["train_macro_f1_delta"] == pytest.approx(
        after_metrics["macro_f1"] - before_metrics["macro_f1"]
    )
    assert summary["checkpoint_reload_completed"] is True
    assert summary["loss_decreased"] is (max_steps > 1)
    assert result["status"] == expected_status
    assert all("/train-" in str(path) for path in state["opened_images"][:7])
    trained_model = state["models"][0]
    assert trained_model.operation_history[:7] == ["generate"] * 7
    assert trained_model.operation_history[7] == "forward"
    assert trained_model.gradient_checkpointing_enabled is True
    assert trained_model.input_grads_enabled is True
    assert trained_model.config.use_cache is False


def test_reload_inference_invalid_output_stays_invalid() -> None:
    dependencies, processor, _ = _runtime("driving")
    model = FakeModel("driving")
    prediction = lora_module._generate_prediction(
        model=model,
        processor=processor,
        sample=_sample(0, split="validation"),
        image=object(),
        config=load_config(CONFIG_PATH),
        device="cpu",
        inference_context=dependencies.inference_context,
    )
    assert prediction["raw_output"] == "driving"
    assert prediction["parsed_action"] is None
    assert prediction["parser_success"] is False
    assert prediction["invalid_reason"] == "output_not_exact_allowed_action"
