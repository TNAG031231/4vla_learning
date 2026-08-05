from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.schema import ACTION_SCHEMA
from src.phase0.qwen3vl_smoke import (
    FIXED_GENERATION_KWARGS,
    FIXED_REVISION,
    RuntimeDependencies,
    load_smoke_config,
    parse_action_output,
    run_smoke,
    select_first_sample,
    smoke_exit_code,
    write_smoke_artifact,
)
from src.phase0.qwen_preflight import sha256_file


REPOSITORY_ROOT = PROJECT_ROOT
CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_qwen_smoke.yaml"


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    OutOfMemoryError = FakeOutOfMemoryError

    def __init__(self, *, available: bool = True, bf16_supported: bool = True):
        self.available = available
        self.bf16_supported = bf16_supported
        self.reset_called = False

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16_supported

    def reset_peak_memory_stats(self) -> None:
        self.reset_called = True

    def memory_allocated(self) -> int:
        return 128 if self.reset_called else 0

    def max_memory_allocated(self) -> int:
        return 512

    def max_memory_reserved(self) -> int:
        return 768

    def get_device_properties(self, index: int) -> object:
        assert index == 0
        return SimpleNamespace(name="Fake GPU", total_memory=32 * 1024**3)


class FakeTorch:
    __version__ = "2.5.1+cu124"
    bfloat16 = "torch.bfloat16"
    float16 = "torch.float16"
    version = SimpleNamespace(cuda="12.4")

    def __init__(self, *, available: bool = True, bf16_supported: bool = True):
        self.cuda = FakeCuda(
            available=available,
            bf16_supported=bf16_supported,
        )

    def inference_mode(self):
        return nullcontext()


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str):
        self.shape = shape
        self.dtype = dtype
        self.device = "cpu"

    def to(self, device: str):
        self.device = device
        return self


class FakeGeneratedIds:
    shape = (1, 1)


class FakeOutputIds:
    def __init__(self):
        self.trim_slice = None

    def __getitem__(self, item):
        self.trim_slice = item
        return FakeGeneratedIds()


class FakeBatch(dict):
    def to(self, device: str):
        for value in self.values():
            if hasattr(value, "to"):
                value.to(device)
        return self


class FakeProcessor:
    def __init__(self, raw_output: object = "keep"):
        self.raw_output = raw_output
        self.messages = None
        self.template_kwargs = None
        self.decode_kwargs = None
        self.decoded_ids = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return FakeBatch(
            {
                "input_ids": FakeTensor((1, 4), "torch.int64"),
                "attention_mask": FakeTensor((1, 4), "torch.int64"),
                "pixel_values": FakeTensor((8, 3, 14, 14), "torch.float32"),
                "image_grid_thw": FakeTensor((1, 3), "torch.int64"),
            }
        )

    def batch_decode(self, generated_ids, **kwargs):
        self.decoded_ids = generated_ids
        self.decode_kwargs = kwargs
        return [self.raw_output]


class FakeModel:
    def __init__(self, *, revision: str = FIXED_REVISION, generation_error=None):
        self.config = SimpleNamespace(
            _commit_hash=revision,
            _attn_implementation="sdpa",
        )
        self.dtype = "torch.bfloat16"
        self.device = "cpu"
        self.generation_error = generation_error
        self.output_ids = FakeOutputIds()
        self.generate_kwargs = None
        self.eval_called = False

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if self.generation_error is not None:
            raise self.generation_error
        return self.output_ids


def _git_runner(root: Path, arguments: tuple[str, ...]) -> str:
    del root
    return {
        ("rev-parse", "HEAD"): "a" * 40,
        ("branch", "--show-current"): "phase-0.3a2-qwen3vl-smoke",
        ("status", "--porcelain"): "",
    }[arguments]


@pytest.fixture
def config():
    return load_smoke_config(CONFIG_PATH)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    token: str,
    *,
    split: str = "train",
    path: str | None = None,
    label: str = "stop",
) -> dict[str, object]:
    return {
        "sample_token": token,
        "scene_token": f"scene-{token}",
        "split": split,
        "cam_front_path": path or f"samples/CAM_FRONT/{token}.jpg",
        "meta_action": label,
        "future_ego_trajectory": [[999.0, 999.0]],
    }


def _runtime(
    *,
    raw_output: object = "keep",
    available: bool = True,
    bf16_supported: bool = True,
    model_revision: str = FIXED_REVISION,
    model_error: Exception | None = None,
    generation_error: Exception | None = None,
    image_error: Exception | None = None,
):
    torch = FakeTorch(
        available=available,
        bf16_supported=bf16_supported,
    )
    processor = FakeProcessor(raw_output)
    model = FakeModel(
        revision=model_revision,
        generation_error=generation_error,
    )
    calls: dict[str, object] = {"processor": [], "model": []}

    def processor_loader(*args):
        calls["processor"].append(args)
        return processor

    def model_loader(*args):
        calls["model"].append(args)
        if model_error is not None:
            raise model_error
        model.dtype = args[2]
        return model

    def image_loader(path: Path):
        if image_error is not None:
            raise image_error
        with Image.open(path) as image:
            return image.convert("RGB")

    dependencies = RuntimeDependencies(
        torch=torch,
        processor_loader=processor_loader,
        model_loader=model_loader,
        image_loader=image_loader,
        package_version=lambda package: "4.57.6",
    )
    return dependencies, processor, model, calls


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config,
    *,
    rows: list[dict[str, object]] | None = None,
    create_image: bool = True,
    runtime=None,
):
    derived_root = tmp_path / "derived"
    nuscenes_root = tmp_path / "nuscenes"
    cache_root = tmp_path / "hf-cache"
    derived_root.mkdir()
    nuscenes_root.mkdir()
    cache_root.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_root))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    manifest_path = derived_root / config.manifest_relative_path
    actual_rows = rows or [_row("sample-1")]
    _write_manifest(manifest_path, actual_rows)
    selected_path = actual_rows[0]["cam_front_path"]
    if create_image and isinstance(selected_path, str):
        image_path = nuscenes_root / selected_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "white").save(image_path)
    pinned_config = replace(
        config,
        expected_manifest_sha256=sha256_file(manifest_path),
    )
    dependencies = runtime or _runtime()[0]
    return run_smoke(
        config=pinned_config,
        repository_root=REPOSITORY_ROOT,
        nuscenes_root=nuscenes_root,
        derived_root=derived_root,
        dependencies=dependencies,
        git_runner=_git_runner,
    )


def test_config_is_valid(config):
    assert config.model_revision == FIXED_REVISION
    assert config.allowed_actions == ACTION_SCHEMA
    assert dict(config.generation_kwargs) == FIXED_GENERATION_KWARGS


def test_config_rejects_missing_model_revision(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw.pop("model_revision")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="model_revision"):
        load_smoke_config(path)


def test_config_rejects_main_revision(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model_revision"] = "main"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be main"):
        load_smoke_config(path)


def test_config_rejects_test_split(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["allowed_split"] = "test"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="train or validation"):
        load_smoke_config(path)


def test_runtime_rejects_test_split_before_manifest_access(
    tmp_path,
    config,
):
    with pytest.raises(ValueError, match="runtime allowed_split"):
        run_smoke(
            config=replace(config, allowed_split="test"),
            repository_root=REPOSITORY_ROOT,
            nuscenes_root=tmp_path,
            derived_root=tmp_path,
        )


def test_manifest_sha_matches(tmp_path, monkeypatch, config):
    artifact = _run(tmp_path, monkeypatch, config)
    assert artifact["manifest"]["sha256_matches"] is True


def test_manifest_sha_mismatch_blocks_before_parsing(tmp_path, monkeypatch, config):
    derived_root = tmp_path / "derived"
    nuscenes_root = tmp_path / "nuscenes"
    cache_root = tmp_path / "cache"
    for path in (derived_root, nuscenes_root, cache_root):
        path.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_root))
    _write_manifest(
        derived_root / config.manifest_relative_path,
        [_row("sample-1")],
    )
    artifact = run_smoke(
        config=config,
        repository_root=REPOSITORY_ROOT,
        nuscenes_root=nuscenes_root,
        derived_root=derived_root,
        dependencies=_runtime()[0],
        git_runner=_git_runner,
    )
    assert artifact["status"] == "failed"
    assert artifact["manifest_records_parsed"] == 0
    assert artifact["failures"][0]["code"] == "manifest_sha256_mismatch"


def test_selects_first_legal_train_sample(tmp_path):
    path = tmp_path / "manifest.jsonl"
    _write_manifest(
        path,
        [
            _row("validation", split="validation"),
            _row("first"),
            _row("second"),
        ],
    )
    sample, count = select_first_sample(path, "train")
    assert sample.sample_token == "first"
    assert sample.manifest_line_number == 2
    assert count == 2


def test_selection_does_not_depend_on_label(tmp_path):
    path = tmp_path / "manifest.jsonl"
    _write_manifest(
        path,
        [_row("first", label="stop"), _row("second", label="keep")],
    )
    sample, _ = select_first_sample(path, "train")
    assert sample.sample_token == "first"


def test_cam_front_traversal_is_rejected(tmp_path):
    path = tmp_path / "manifest.jsonl"
    _write_manifest(path, [_row("bad", path="samples/CAM_FRONT/../secret.jpg")])
    with pytest.raises(ValueError, match="CAM_FRONT"):
        select_first_sample(path, "train")


def test_missing_image_is_blocked(tmp_path, monkeypatch, config):
    artifact = _run(tmp_path, monkeypatch, config, create_image=False)
    assert artifact["status"] == "blocked"
    assert artifact["failures"][0]["code"] == "image_missing"


def test_undecodable_image_is_failed(tmp_path, monkeypatch, config):
    dependencies, _, _, _ = _runtime(image_error=OSError("bad image"))
    artifact = _run(tmp_path, monkeypatch, config, runtime=dependencies)
    assert artifact["status"] == "failed"
    assert artifact["failures"][0]["code"] == "image_decode_failed"


def test_processor_receives_only_image_and_fixed_prompt(tmp_path, monkeypatch, config):
    dependencies, processor, _, _ = _runtime()
    artifact = _run(tmp_path, monkeypatch, config, runtime=dependencies)
    assert artifact["status"] == "passed"
    content = processor.messages[0]["content"]
    assert [item["type"] for item in content] == ["image", "text"]
    assert content[1]["text"] == config.task_prompt
    assert set(content[0]) == {"type", "image"}


def test_processor_records_actual_shapes_and_device(tmp_path, monkeypatch, config):
    artifact = _run(tmp_path, monkeypatch, config)
    processor = artifact["processor"]
    assert processor["input_ids_shape"] == [1, 4]
    assert processor["pixel_values_shape"] == [8, 3, 14, 14]
    assert processor["image_grid_thw_shape"] == [1, 3]
    assert processor["pixel_values_device"] == "cuda:0"
    assert processor["image_width_pixels"] == 32
    assert processor["image_height_pixels"] == 24


def test_generation_is_deterministic_and_decode_is_strict(
    tmp_path,
    monkeypatch,
    config,
):
    dependencies, processor, model, _ = _runtime()
    _run(tmp_path, monkeypatch, config, runtime=dependencies)
    for key, value in FIXED_GENERATION_KWARGS.items():
        assert model.generate_kwargs[key] == value
    assert "temperature" not in model.generate_kwargs
    assert processor.decode_kwargs == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def test_generation_trims_input_tokens(tmp_path, monkeypatch, config):
    dependencies, _, model, _ = _runtime()
    _run(tmp_path, monkeypatch, config, runtime=dependencies)
    row_slice, token_slice = model.output_ids.trim_slice
    assert row_slice == slice(None)
    assert token_slice == slice(4, None)


@pytest.mark.parametrize("action", ACTION_SCHEMA)
def test_strict_parser_accepts_each_action(action):
    result = parse_action_output(action)
    assert result["parser_success"] is True
    assert result["predicted_action"] == action


def test_strict_parser_normalizes_case_and_outer_whitespace():
    result = parse_action_output("  LEFT_LATERAL\n")
    assert result["parser_success"] is True
    assert result["normalized_output"] == "left_lateral"


@pytest.mark.parametrize(
    "raw_output",
    [
        "keep because the road is clear",
        '{"action":"keep"}',
        "keep stop",
        "",
        "turn left",
        None,
    ],
)
def test_strict_parser_rejects_non_exact_outputs(raw_output):
    result = parse_action_output(raw_output)
    assert result["parser_success"] is False
    assert result["predicted_action"] is None


def test_invalid_output_is_not_retried(tmp_path, monkeypatch, config):
    dependencies, _, model, _ = _runtime(raw_output="Action: keep")
    artifact = _run(tmp_path, monkeypatch, config, runtime=dependencies)
    assert artifact["status"] == "completed_with_invalid_output"
    assert artifact["generation"]["retry_count"] == 0
    assert model.generate_kwargs is not None


def test_bf16_is_selected_when_supported(tmp_path, monkeypatch, config):
    dependencies, _, _, calls = _runtime(bf16_supported=True)
    _run(tmp_path, monkeypatch, config, runtime=dependencies)
    assert calls["model"][0][2] == "torch.bfloat16"


def test_fp16_is_controlled_fallback(tmp_path, monkeypatch, config):
    dependencies, _, _, calls = _runtime(bf16_supported=False)
    _run(tmp_path, monkeypatch, config, runtime=dependencies)
    assert calls["model"][0][2] == "torch.float16"


def test_cuda_unavailable_is_blocked(tmp_path, monkeypatch, config):
    artifact = _run(
        tmp_path,
        monkeypatch,
        config,
        runtime=_runtime(available=False)[0],
    )
    assert artifact["status"] == "blocked"
    assert artifact["failures"][0]["code"] == "cuda_unavailable"


def test_model_load_failure_is_structured(tmp_path, monkeypatch, config):
    artifact = _run(
        tmp_path,
        monkeypatch,
        config,
        runtime=_runtime(model_error=RuntimeError("load failed"))[0],
    )
    assert artifact["status"] == "failed"
    assert artifact["failures"][0]["code"] == "model_load_failed"


def test_model_load_oom_is_structured(tmp_path, monkeypatch, config):
    artifact = _run(
        tmp_path,
        monkeypatch,
        config,
        runtime=_runtime(model_error=FakeOutOfMemoryError("oom"))[0],
    )
    assert artifact["status"] == "failed"
    assert artifact["failures"][0]["code"] == "model_load_oom"


def test_resolved_revision_mismatch_is_failed(tmp_path, monkeypatch, config):
    artifact = _run(
        tmp_path,
        monkeypatch,
        config,
        runtime=_runtime(model_revision="b" * 40)[0],
    )
    assert artifact["status"] == "failed"
    assert artifact["failures"][0]["code"] == "resolved_revision_mismatch"


def test_atomic_artifact_write(tmp_path):
    output_path = tmp_path / "nested/smoke_result.json"
    write_smoke_artifact({"status": "passed"}, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"status": "passed"}
    assert list(output_path.parent.glob(".*.tmp")) == []


def test_test_isolation_fields_are_fixed(tmp_path, monkeypatch, config):
    artifact = _run(tmp_path, monkeypatch, config)
    assert artifact["test_records_read"] == 0
    assert artifact["test_images_opened"] == 0
    assert artifact["test_labels_read"] == 0
    assert artifact["test_evaluation_performed"] is False


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("passed", 0),
        ("completed_with_invalid_output", 0),
        ("blocked", 2),
        ("failed", 1),
    ],
)
def test_exit_code_contract(status, expected):
    assert smoke_exit_code(status) == expected


def test_cli_success_and_invalid_output_are_successful(monkeypatch, tmp_path, config):
    import scripts.run_qwen3vl_smoke as cli

    monkeypatch.setenv("NUSCENES_ROOT", str(tmp_path))
    monkeypatch.setenv("VLA_DERIVED_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "load_smoke_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "smoke_output_path",
        lambda *args: tmp_path / "result.json",
    )
    monkeypatch.setattr(cli, "write_smoke_artifact", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "run_smoke",
        lambda **kwargs: {"status": "completed_with_invalid_output"},
    )
    assert cli.main(["--config", "config.yaml"]) == 0


@pytest.mark.parametrize(("status", "expected"), [("blocked", 2), ("failed", 1)])
def test_cli_blocked_and_failed_are_nonzero(
    monkeypatch,
    tmp_path,
    config,
    status,
    expected,
):
    import scripts.run_qwen3vl_smoke as cli

    monkeypatch.setenv("NUSCENES_ROOT", str(tmp_path))
    monkeypatch.setenv("VLA_DERIVED_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "load_smoke_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "smoke_output_path",
        lambda *args: tmp_path / "result.json",
    )
    monkeypatch.setattr(cli, "write_smoke_artifact", lambda *args: None)
    monkeypatch.setattr(cli, "run_smoke", lambda **kwargs: {"status": status})
    assert cli.main(["--config", "config.yaml"]) == expected


def test_cli_rejects_unknown_argument():
    import scripts.run_qwen3vl_smoke as cli

    with pytest.raises(SystemExit) as error:
        cli.main(["--config", "config.yaml", "--model-id", "other"])
    assert error.value.code == 2
