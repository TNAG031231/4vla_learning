from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.actions.schema import ACTION_SCHEMA
from src.phase0.qwen3vl_dataset_adapter import (
    GitProvenance,
    adapter_record,
    canonical_json_bytes,
    load_config as load_adapter_config,
    sha256_file,
)
import src.phase0.qwen3vl_zero_shot as zero_shot_module
from src.phase0.qwen3vl_zero_shot import (
    AdapterArtifact,
    RuntimeDependencies,
    build_inference_messages,
    build_metrics,
    load_adapter_artifact,
    load_config,
    run_zero_shot,
    select_adapter_samples,
)
from scripts import run_qwen3vl_zero_shot as cli_module


CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_zero_shot.yaml"
ADAPTER_CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_dataset_adapter.yaml"


def _motion() -> dict[str, object]:
    return {
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


def _source(index: int, action: str) -> dict[str, object]:
    return {
        "sample_token": f"sample-{index:04d}",
        "scene_token": f"scene-{index // 2:04d}",
        "timestamp": index,
        "split": "validation",
        "cam_front_path": f"samples/CAM_FRONT/{index:04d}.jpg",
        "current_ego_motion": _motion(),
        "target_action": action,
        "projection_record_sha256": hashlib.sha256(
            f"projection-{index}".encode()
        ).hexdigest(),
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )


def _adapter_artifact(tmp_path: Path, count: int = 12) -> AdapterArtifact:
    adapter_config = load_adapter_config(ADAPTER_CONFIG_PATH)
    paths = {}
    hashes = {}
    actions = ACTION_SCHEMA * ((count // len(ACTION_SCHEMA)) + 1)
    for variant in zero_shot_module.VARIANTS:
        records = [
            adapter_record(
                _source(index, actions[index]),
                variant=variant,
                config=adapter_config,
                source_file_sha256=(
                    adapter_config.source_files["validation"].sha256
                ),
            )
            for index in range(count)
        ]
        path = tmp_path / f"validation_{variant}.jsonl"
        _write_jsonl(path, records)
        paths[variant] = path
        hashes[variant] = sha256_file(path)
    return AdapterArtifact(
        receipt_sha256="d" * 64,
        validation_paths=paths,
        validation_sha256=hashes,
        adapter_config=adapter_config,
    )


def _receipt(
    config,
    adapter_config,
    validation_hashes: dict[str, str],
) -> dict[str, object]:
    outputs = {
        "train_image_only": {
            "relative_path": "train_image_only.jsonl",
            "sha256": "1" * 64,
            "record_count": adapter_config.source_files["train"].record_count,
        },
        "train_image_ego_state": {
            "relative_path": "train_image_ego_state.jsonl",
            "sha256": "2" * 64,
            "record_count": adapter_config.source_files["train"].record_count,
        },
    }
    for variant in zero_shot_module.VARIANTS:
        outputs[f"validation_{variant}"] = {
            "relative_path": config.adapter_validation_files[variant],
            "sha256": validation_hashes[variant],
            "record_count": config.adapter_validation_record_count,
        }
    return {
        "adapter_version": config.adapter_version,
        "adapter_schema_version": config.adapter_schema_version,
        "generated_at_utc": "2026-08-10T00:00:00Z",
        "git": {
            "commit": "a" * 40,
            "branch": "adapter-branch",
            "detached_head": False,
            "worktree_clean": True,
        },
        "config": {
            "relative_path": config.adapter_config_relative_path,
            "sha256": config.adapter_config_sha256,
        },
        "source_projection": {
            "relative_dir": adapter_config.source_projection_relative_dir,
            "receipt_relative_path": adapter_config.source_projection_receipt,
            "receipt_sha256": adapter_config.expected_source_receipt_sha256,
            "projection_version": (
                adapter_config.expected_source_projection_version
            ),
            "projection_schema_version": (
                adapter_config.expected_source_projection_schema_version
            ),
            "git_commit": adapter_config.expected_source_projection_git_commit,
            "train_relative_path": (
                adapter_config.source_files["train"].relative_path
            ),
            "train_sha256": adapter_config.source_files["train"].sha256,
            "train_record_count": (
                adapter_config.source_files["train"].record_count
            ),
            "validation_relative_path": (
                adapter_config.source_files["validation"].relative_path
            ),
            "validation_sha256": (
                adapter_config.source_files["validation"].sha256
            ),
            "validation_record_count": (
                adapter_config.source_files["validation"].record_count
            ),
        },
        "prompt": {
            "task_prompt": adapter_config.task_prompt,
            "allowed_actions": list(adapter_config.allowed_actions),
            "ego_state_field_order": list(adapter_config.ego_field_order),
            "float_precision": adapter_config.float_precision,
            "unavailable_token": adapter_config.unavailable_token,
        },
        "outputs": outputs,
        "source_records_parsed": {
            "train": adapter_config.source_files["train"].record_count,
            "validation": config.adapter_validation_record_count,
        },
        "adapter_records_written": {
            "train_image_only": adapter_config.source_files["train"].record_count,
            "train_image_ego_state": (
                adapter_config.source_files["train"].record_count
            ),
            "validation_image_only": config.adapter_validation_record_count,
            "validation_image_ego_state": config.adapter_validation_record_count,
        },
        "absolute_path_leak_count": 0,
        "invalid_cam_front_path_count": 0,
        "invalid_source_record_hash_count": 0,
        "duplicate_sample_token_count": 0,
        "forbidden_split_records_seen": 0,
        "combined_manifest_accessed": False,
        "test_files_opened": 0,
        "test_records_read": 0,
        "test_images_opened": 0,
        "test_labels_read": 0,
        "test_evaluation_performed": False,
        "image_files_opened": 0,
        "model_load_performed": False,
        "processor_load_performed": False,
        "tokenization_performed": False,
        "training_performed": False,
    }


def _frozen_adapter_directory(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    adapter_config = load_adapter_config(ADAPTER_CONFIG_PATH)
    adapter_dir = tmp_path / config.adapter_relative_dir
    adapter_dir.mkdir(parents=True)
    validation_hashes = {}
    for variant in zero_shot_module.VARIANTS:
        path = adapter_dir / config.adapter_validation_files[variant]
        path.write_bytes(b"{}\n" * config.adapter_validation_record_count)
        validation_hashes[variant] = sha256_file(path)
    receipt_path = adapter_dir / config.adapter_receipt_relative_path
    receipt_path.write_bytes(
        canonical_json_bytes(
            _receipt(config, adapter_config, validation_hashes)
        )
        + b"\n"
    )
    return config, adapter_dir, receipt_path


def _rewrite_config(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "zero_shot.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


class FakeTensor:
    def __init__(self, shape=(1, 4), dtype="torch.int64") -> None:
        self.shape = shape
        self.dtype = dtype


class FakeBatch(dict):
    def __init__(self) -> None:
        super().__init__(
            {
                "input_ids": FakeTensor(),
                "attention_mask": FakeTensor(),
                "pixel_values": FakeTensor((8, 1536), "torch.float32"),
                "image_grid_thw": FakeTensor((1, 3)),
            }
        )
        self.device = None

    def to(self, device: str):
        self.device = device
        return self


class FakeGenerated:
    shape = (1, 1)


class FakeOutput:
    def __getitem__(self, key):
        return FakeGenerated()


class FakeProcessor:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return FakeBatch()

    def batch_decode(self, generated, **kwargs):
        return [self.outputs.pop(0)]


class FakeModel:
    def __init__(self, dtype: object) -> None:
        self.dtype = dtype
        self.config = SimpleNamespace(
            _commit_hash=zero_shot_module.FIXED_REVISION,
            _attn_implementation="sdpa",
        )
        self.device = None
        self.generate_calls = []

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        return None

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return FakeOutput()


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def is_bf16_supported(self) -> bool:
        return True


class FakeTorch:
    bfloat16 = "torch.bfloat16"
    float16 = "torch.float16"
    cuda = FakeCuda()

    def inference_mode(self):
        return nullcontext()


def _runtime(outputs: list[str]):
    processor = FakeProcessor(outputs)
    model_holder = {}

    def load_model(model_id, revision, dtype, attention, local_only):
        model = FakeModel(dtype)
        model_holder["model"] = model
        return model

    runtime = RuntimeDependencies(
        torch=FakeTorch(),
        processor_loader=lambda *args: processor,
        model_loader=load_model,
        image_loader=lambda path: object(),
        package_version=lambda name: "4.test",
    )
    return runtime, processor, model_holder


def _git() -> GitProvenance:
    return GitProvenance(
        commit="e" * 40,
        branch="zero-shot-test",
        detached_head=False,
        worktree_clean=True,
    )


def _run(
    tmp_path: Path,
    monkeypatch,
    *,
    variant: str = "image_only",
    outputs: list[str] | None = None,
    count: int = 2,
):
    config = load_config(CONFIG_PATH)
    artifact = _adapter_artifact(tmp_path / "adapter", count=max(count, 12))
    monkeypatch.setattr(
        zero_shot_module, "load_adapter_artifact", lambda **kwargs: artifact
    )
    repository_root = tmp_path / "repository"
    derived_root = tmp_path / "derived"
    nuscenes_root = tmp_path / "nuscenes"
    repository_root.mkdir()
    derived_root.mkdir()
    for index in range(count):
        image_path = nuscenes_root / f"samples/CAM_FRONT/{index:04d}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
    runtime, processor, model_holder = _runtime(
        outputs or ["keep"] * count
    )
    result = run_zero_shot(
        config=config,
        config_relative_path="configs/phase0_3_zero_shot.yaml",
        repository_root=repository_root,
        nuscenes_root=nuscenes_root,
        derived_root=derived_root,
        input_variant=variant,
        split="validation",
        max_samples=count,
        git_provenance=_git(),
        dependencies=runtime,
        now_utc=lambda: "2026-08-10T00:00:00Z",
    )
    return result, processor, model_holder, artifact, config


def test_frozen_configuration_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config.model_revision == zero_shot_module.FIXED_REVISION
    assert config.processor_revision == zero_shot_module.FIXED_REVISION
    assert config.prompt_version == "phase0.3c-zero-shot-v0.1"
    assert config.generation_kwargs == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 16,
    }
    assert config.variants == ("image_only", "image_ego_state")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.__setitem__("model_revision", "f" * 40), "model_revision"),
        (lambda raw: raw.__setitem__("prompt_version", "changed"), "prompt_version"),
        (
            lambda raw: raw["generation_kwargs"].__setitem__("do_sample", True),
            "generation_kwargs",
        ),
        (lambda raw: raw.__setitem__("variants", ["image_only"]), "variants"),
    ],
)
def test_configuration_rejects_contract_changes(tmp_path, change, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_rewrite_config(tmp_path, change))


def test_validation_adapter_artifact_is_accepted(tmp_path: Path) -> None:
    config, _, _ = _frozen_adapter_directory(tmp_path)
    artifact = load_adapter_artifact(
        config=config,
        repository_root=REPOSITORY_ROOT,
        derived_root=tmp_path,
    )
    assert artifact.receipt_sha256
    assert set(artifact.validation_sha256) == set(zero_shot_module.VARIANTS)


def test_wrong_adapter_validation_hash_is_rejected(tmp_path: Path) -> None:
    config, adapter_dir, _ = _frozen_adapter_directory(tmp_path)
    with (adapter_dir / config.adapter_validation_files["image_only"]).open(
        "ab"
    ) as output:
        output.write(b"tamper\n")
    with pytest.raises(ValueError, match="validation SHA-256 mismatch"):
        load_adapter_artifact(
            config=config,
            repository_root=REPOSITORY_ROOT,
            derived_root=tmp_path,
        )


def test_wrong_adapter_schema_is_rejected(tmp_path: Path) -> None:
    config, _, receipt_path = _frozen_adapter_directory(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["adapter_schema_version"] = "wrong"
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(ValueError, match="adapter_schema_version mismatch"):
        load_adapter_artifact(
            config=config,
            repository_root=REPOSITORY_ROOT,
            derived_root=tmp_path,
        )


def test_test_split_rejected_before_adapter_model_or_image_access(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        zero_shot_module,
        "load_adapter_artifact",
        lambda **kwargs: pytest.fail("adapter must not be accessed"),
    )
    with pytest.raises(ValueError, match="only permits validation"):
        run_zero_shot(
            config=load_config(CONFIG_PATH),
            config_relative_path="configs/phase0_3_zero_shot.yaml",
            repository_root=tmp_path,
            nuscenes_root=tmp_path,
            derived_root=tmp_path,
            input_variant="image_only",
            split="test",
            max_samples=10,
            git_provenance=_git(),
            dependencies=None,
        )


def test_cli_rejects_test_before_environment_access(monkeypatch) -> None:
    monkeypatch.delenv("NUSCENES_ROOT", raising=False)
    monkeypatch.delenv("VLA_DERIVED_ROOT", raising=False)
    assert (
        cli_module.main(
            ["--input-variant", "image_only", "--split", "test"]
        )
        == 2
    )


def test_variants_select_identical_first_ten_tokens(tmp_path: Path) -> None:
    artifact = _adapter_artifact(tmp_path)
    image_only = select_adapter_samples(
        artifact=artifact, input_variant="image_only", max_samples=10
    )
    image_ego = select_adapter_samples(
        artifact=artifact, input_variant="image_ego_state", max_samples=10
    )
    assert [sample.sample_token for sample in image_only] == [
        sample.sample_token for sample in image_ego
    ] == [f"sample-{index:04d}" for index in range(10)]
    assert all(sample.ego_state_text is None for sample in image_only)
    assert all(sample.ego_state_text is not None for sample in image_ego)


def test_tampered_adapter_record_hash_is_rejected(tmp_path: Path) -> None:
    artifact = _adapter_artifact(tmp_path)
    path = artifact.validation_paths["image_only"]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["adapter_record_sha256"] = "f" * 64
    _write_jsonl(path, records)
    with pytest.raises(ValueError, match="adapter record SHA-256 mismatch"):
        select_adapter_samples(
            artifact=artifact, input_variant="image_only", max_samples=1
        )


def test_image_only_prompt_contains_no_adapter_ego_text(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    sample = select_adapter_samples(
        artifact=_adapter_artifact(tmp_path),
        input_variant="image_only",
        max_samples=1,
    )[0]
    messages = build_inference_messages(sample, object(), config)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][1]["text"] == config.task_prompt
    assert "Current ego state:" not in messages[0]["content"][1]["text"]


def test_image_ego_state_uses_exact_adapter_serialization(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    sample = select_adapter_samples(
        artifact=_adapter_artifact(tmp_path),
        input_variant="image_ego_state",
        max_samples=1,
    )[0]
    messages = build_inference_messages(sample, object(), config)
    assert messages[0]["content"][1]["text"] == (
        f"{sample.ego_state_text}\n\n{config.task_prompt}"
    )


def test_inference_message_excludes_target_and_forbidden_fields(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    sample = select_adapter_samples(
        artifact=_adapter_artifact(tmp_path),
        input_variant="image_ego_state",
        max_samples=1,
    )[0]
    messages = build_inference_messages(sample, object(), config)
    assert len(messages) == 1
    assert set(messages[0]) == {"role", "content"}
    assert all(item.get("type") in {"image", "text"} for item in messages[0]["content"])
    serialized = repr(messages)
    for forbidden in (
        "target_action",
        "future_ego_trajectory",
        "nearby_agents",
        "gt_boxes",
        "occupancy",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize("action", ACTION_SCHEMA)
def test_strict_legacy_parser_accepts_exact_actions(action: str) -> None:
    assert zero_shot_module.parse_action_output(action)["predicted_action"] == action


@pytest.mark.parametrize(
    "raw_output",
    ("The vehicle should keep.", "driving", "KEEP because...", "keep."),
)
def test_strict_legacy_parser_rejects_non_exact_outputs(raw_output: str) -> None:
    parsed = zero_shot_module.parse_action_output(raw_output)
    assert parsed["parser_success"] is False
    assert parsed["predicted_action"] is None


def test_metrics_perfect_predictions() -> None:
    records = [
        {"target_action": action, "parsed_action": action}
        for action in ACTION_SCHEMA
    ]
    metrics = build_metrics(records)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["parser_success_count"] == 6
    assert metrics["invalid_output_count"] == 0


def test_metrics_missing_classes_and_mixed_predictions() -> None:
    metrics = build_metrics(
        [
            {"target_action": "keep", "parsed_action": "keep"},
            {"target_action": "stop", "parsed_action": "keep"},
        ]
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(1 / 9)
    assert metrics["target_class_distribution"]["accelerate"] == 0
    assert metrics["prediction_class_distribution"]["keep"] == 2


def test_invalid_prediction_is_incorrect_and_ground_truth_false_negative() -> None:
    metrics = build_metrics(
        [{"target_action": "decelerate", "parsed_action": None}]
    )
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["invalid_output_count"] == 1
    assert metrics["per_class_recall"]["decelerate"] == 0.0
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 0


def test_run_writes_predictions_metrics_and_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    result, processor, model_holder, artifact, config = _run(
        tmp_path,
        monkeypatch,
        outputs=["keep", "driving"],
    )
    assert result["status"] == "created"
    output_dir = result["output_dir"]
    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 2
    assert set(predictions[0]) == {
        "sample_token",
        "scene_token",
        "split",
        "input_variant",
        "target_action",
        "raw_output",
        "parsed_action",
        "parser_success",
        "invalid_reason",
        "is_correct",
        "model_id",
        "model_revision",
        "processor_revision",
        "prompt_version",
        "parser_version",
        "generation_config_version",
        "adapter_schema_version",
        "adapter_record_sha256",
    }
    assert predictions[1]["parsed_action"] is None
    assert predictions[1]["invalid_reason"] == "output_not_exact_allowed_action"
    assert len(processor.messages) == 2
    assert len(model_holder["model"].generate_calls) == 2
    assert all(
        call["do_sample"] is False
        and call["num_beams"] == 1
        and call["max_new_tokens"] == 16
        for call in model_holder["model"].generate_calls
    )
    receipt = json.loads((output_dir / "run_receipt.json").read_text())
    assert receipt["adapter"]["receipt_sha256"] == artifact.receipt_sha256
    assert receipt["adapter"]["validation_jsonl_sha256"] == dict(
        artifact.validation_sha256
    )
    assert receipt["config"]["sha256"] == config.config_sha256
    assert receipt["artifacts"]["predictions_sha256"] == sha256_file(
        output_dir / "predictions.jsonl"
    )
    assert receipt["test_records_read"] == 0
    assert receipt["test_images_opened"] == 0
    assert receipt["test_labels_read"] == 0
    assert receipt["test_evaluation_performed"] is False
    assert receipt["validation_label_used_as_model_input"] is False


def test_invalid_output_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    result, _, model_holder, _, _ = _run(
        tmp_path, monkeypatch, outputs=["driving", "moving"]
    )
    assert result["receipt"]["generation"]["retry_count"] == 0
    assert len(model_holder["model"].generate_calls) == 2


def test_matching_existing_artifact_is_revalidated_without_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    result, _, _, artifact, config = _run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        zero_shot_module, "load_adapter_artifact", lambda **kwargs: artifact
    )
    blocked_runtime = RuntimeDependencies(
        torch=FakeTorch(),
        processor_loader=lambda *args: pytest.fail("processor must not load"),
        model_loader=lambda *args: pytest.fail("model must not load"),
        image_loader=lambda path: pytest.fail("image must not open"),
        package_version=lambda name: "unused",
    )
    second = run_zero_shot(
        config=config,
        config_relative_path="configs/phase0_3_zero_shot.yaml",
        repository_root=tmp_path / "repository",
        nuscenes_root=tmp_path / "nuscenes",
        derived_root=tmp_path / "derived",
        input_variant="image_only",
        split="validation",
        max_samples=2,
        git_provenance=_git(),
        dependencies=blocked_runtime,
    )
    assert result["output_dir"] == second["output_dir"]
    assert second["status"] == "already_exists"


def test_existing_tampered_artifact_is_not_overwritten(
    tmp_path: Path, monkeypatch
) -> None:
    result, _, _, artifact, config = _run(tmp_path, monkeypatch)
    predictions_path = result["output_dir"] / "predictions.jsonl"
    predictions_path.write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(
        zero_shot_module, "load_adapter_artifact", lambda **kwargs: artifact
    )
    with pytest.raises(ValueError, match="predictions SHA-256 mismatch"):
        run_zero_shot(
            config=config,
            config_relative_path="configs/phase0_3_zero_shot.yaml",
            repository_root=tmp_path / "repository",
            nuscenes_root=tmp_path / "nuscenes",
            derived_root=tmp_path / "derived",
            input_variant="image_only",
            split="validation",
            max_samples=2,
            git_provenance=_git(),
            dependencies=None,
        )
    assert predictions_path.read_text(encoding="utf-8") == "tampered\n"
