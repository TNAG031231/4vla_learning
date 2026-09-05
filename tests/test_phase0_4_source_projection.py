from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_trainval_manifest as producer
from src.phase0 import development_projection as development
from src.phase0 import phase0_4_source_projection as projection
from src.phase0.manifest import json_record
from src.phase0.qwen_preflight import sha256_file
from test_phase0_3_development_projection import _mapping_payload, _write_inputs
from test_trainval_manifest import build_scene, evaluate, rules


CONFIG = ROOT / "configs/phase0_4_source_projection.yaml"


@pytest.fixture
def config():
    return projection.load_config(CONFIG, ROOT)


@pytest.fixture
def source(tmp_path):
    reader = build_scene(tmp_path, tuple(index * 500_000 for index in range(7)))
    # Sample time and CAM_FRONT time deliberately differ.
    reader.tables["sample_data"]["camera-0"]["timestamp"] = 10_000
    decision = evaluate(reader, tmp_path)
    assert decision.record is not None
    return json_record(decision.record)


@pytest.fixture
def selection(config):
    return development.select_development_scenes(
        _mapping_payload(), config.source_contract,
    )


def project(source, config, selection, split="train"):
    return projection.project_record(
        source, split=split, config=config,
        split_mapping_sha256=source["split_mapping_sha256"],
        selection=selection,
    )


@pytest.mark.parametrize(
    "split,scene", [("train", "scene-000"), ("validation", "scene-560")],
)
def test_real_producer_shape_preserved(source, config, selection, split, scene):
    source.update(split=split, scene_token=scene)
    result = project(source, config, selection, split)
    assert len(result["future_ego_trajectory"]) == 7
    assert result["future_ego_trajectory"] == source["future_ego_trajectory"]
    assert result["timestamp"] == 0
    assert result["current_ego_pose"]["timestamp_us"] == 10_000
    assert result["current_ego_pose"] == source["current_ego_pose"]
    assert result["current_ego_motion"] == source["current_ego_motion"]
    assert result["source_legacy_meta_action"] == source["meta_action"]
    assert "target_action" not in result
    assert "nearby_agents" not in result
    assert result["source_audit_record"] is None
    assert result["source_manifest_schema_version"] == source["manifest_schema_version"]


@pytest.mark.parametrize(
    "split,scene",
    [("test", "scene-700"), ("train", "scene-700"), ("train", "scene-560")],
)
def test_split_and_scene_rejection(source, config, selection, split, scene):
    source.update(split=split, scene_token=scene)
    with pytest.raises(ValueError, match="only permits|outside"):
        project(source, config, selection, split)


@pytest.mark.parametrize("change,message", [
    (lambda p: p.pop(), "7 points"),
    (lambda p: p.append(p[-1].copy()), "7 points"),
    (lambda p: p[2].update(t_sec=p[1]["t_sec"]), "strictly increasing"),
    (lambda p: p[2].update(future_sample_token=p[1]["future_sample_token"]), "unique"),
    (lambda p: p[1].update(future_sample_token=""), "nonempty"),
    (lambda p: p[0].update(future_sample_token="wrong"), "anchor"),
    (lambda p: p[0].update(t_sec=0.01), "anchor"),
    (lambda p: p[0].update(x_m=0.01), "origin"),
    (lambda p: p[0].update(y_m=0.01), "origin"),
    (lambda p: p[0].update(heading_delta_rad=0.01), "origin"),
    (lambda p: p[6].update(t_sec=3.076), "tolerance"),
])
def test_trajectory_contract_rejections(source, config, selection, change, message):
    source["scene_token"] = "scene-000"
    change(source["future_ego_trajectory"])
    with pytest.raises(ValueError, match=message):
        project(source, config, selection)


@pytest.mark.parametrize("field", ["t_sec", "x_m", "y_m", "heading_delta_rad"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_trajectory_rejected(source, config, selection, field, value):
    source["scene_token"] = "scene-000"
    source["future_ego_trajectory"][1][field] = value
    with pytest.raises(ValueError, match="finite"):
        project(source, config, selection)


@pytest.mark.parametrize(
    "field", ["source_frame", "target_frame", "x_axis", "y_axis", "unit", "transform"],
)
def test_coordinate_contract_rejected(source, config, selection, field):
    source["scene_token"] = "scene-000"
    source["coordinate_metadata"]["future_ego_trajectory"][field] = "wrong"
    with pytest.raises(ValueError, match="coordinate"):
        project(source, config, selection)


def test_existing_tolerance_is_preserved(source, config, selection):
    source["scene_token"] = "scene-000"
    source["future_ego_trajectory"][1]["t_sec"] = 0.575
    result = project(source, config, selection)
    assert result["future_ego_trajectory"][1]["t_sec"] == 0.575


def test_pose_timestamp_policy_rejected(source, config, selection):
    source["scene_token"] = "scene-000"
    source["current_ego_pose"]["timestamp_source"] = "sample"
    with pytest.raises(ValueError, match="CAM_FRONT"):
        project(source, config, selection)


def setup_pipeline(tmp_path, config):
    dataroot = tmp_path / "nuscenes"
    train = build_scene(dataroot, tuple(i * 500_000 for i in range(7)), "scene-000")
    validation = build_scene(
        tmp_path / "validation", tuple(i * 500_000 for i in range(7)), "scene-560",
    )
    # Give the second synthetic scene globally unique sample/camera/pose tokens.
    for table, records in validation.tables.items():
        if table == "scene":
            records["scene-560"]["first_sample_token"] = "val-sample-0"
            records["scene-560"]["last_sample_token"] = "val-sample-6"
            train.tables[table].update(records)
            continue
        for token, record in records.items():
            record = dict(record)
            record["token"] = f"val-{token}"
            for field in ("prev", "next", "sample_token", "ego_pose_token"):
                if record.get(field):
                    record[field] = f"val-{record[field]}"
            if "data" in record:
                record["data"] = {"CAM_FRONT": f"val-{record['data']['CAM_FRONT']}"}
            train.tables[table][f"val-{token}"] = record
    source_config = replace(
        config.source_contract, expected_sample_counts={"train": 1, "validation": 1},
    )
    source_config = _write_inputs(tmp_path / "derived", source_config)
    config = replace(config, source_contract=source_config)
    calls = []

    def selected_producer(**kwargs):
        calls.append(kwargs)
        tokens = kwargs["scene_tokens"]
        assert not set(tokens) & {f"scene-{i:03d}" for i in range(700, 850)}
        assert kwargs["audit_index"] == {}
        # Run the actual frozen producer on the populated scene in each split.
        return producer.build_records(**{**kwargs, "scene_tokens": (tokens[0],)})

    kwargs = dict(
        config=config, repository_root=ROOT,
        derived_root=tmp_path / "derived", nuscenes_root=dataroot,
        reader_factory=lambda: train, producer=selected_producer,
        producer_inputs=development.ProducerInputs(rules(), 3.0, 0.5, 0.075, 50.0),
        git_provenance=development.GitProvenance("e" * 40, "synthetic", False, True),
    )
    return kwargs, train, calls


def test_producer_to_projection_pipeline_and_byte_only_integrity(
    tmp_path, config, monkeypatch,
):
    kwargs, reader, calls = setup_pipeline(tmp_path, config)
    import src.phase0.protocol as protocol

    def forbidden(*args, **kwargs):
        pytest.fail("combined manifest semantic parsing or audit access is forbidden")

    monkeypatch.setattr(protocol, "iter_manifest_rows", forbidden)
    monkeypatch.setattr(protocol, "validate_manifest", forbidden)
    monkeypatch.setattr(producer, "load_audit_index", forbidden)
    monkeypatch.setattr(producer, "build_full_scene_splits", forbidden)
    receipt = projection.build_source_projection(**kwargs)
    assert len(calls) == 2
    for field in asdict(development.IsolationCounters()):
        assert receipt[field] == 0
    output = kwargs["derived_root"] / config.output_relative_dir
    receipt_path = output / "source_projection_receipt.json"
    assert json.loads(receipt_path.read_text()) == receipt
    for split in ("train", "validation"):
        path = output / f"{split}.jsonl"
        record = json.loads(path.read_text())
        assert len(record["future_ego_trajectory"]) == 7
        assert record["split"] == split
        assert receipt["outputs"][split]["sha256"] == sha256_file(path)
        assert receipt["outputs"][split]["record_count"] == 1
    assert not (output / "test.jsonl").exists()
    with pytest.raises(FileExistsError):
        projection.build_source_projection(**kwargs)
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["manifest", "mapping", "rule", "timing"])
def test_intake_failure_before_reader_or_output(tmp_path, config, failure):
    kwargs, _, calls = setup_pipeline(tmp_path, config)
    source_config = kwargs["config"].source_contract
    if failure == "manifest":
        path = kwargs["derived_root"] / source_config.combined_manifest_relative_path
        path.write_bytes(b"changed")
    elif failure in ("mapping", "rule"):
        relative = (
            source_config.scene_mapping_relative_path
            if failure == "mapping" else source_config.selected_rule_relative_path
        )
        path = kwargs["derived_root"] / relative
        payload = json.loads(path.read_text())
        key = (
            "scene_split_mapping_sha256"
            if failure == "mapping" else "split_mapping_sha256"
        )
        payload[key] = "f" * 64
        path.write_text(json.dumps(payload))
    else:
        kwargs["producer_inputs"] = replace(
            kwargs["producer_inputs"], time_tolerance_sec=0.1,
        )
    kwargs["reader_factory"] = lambda: pytest.fail("reader initialized before intake")
    with pytest.raises(ValueError):
        projection.build_source_projection(**kwargs)
    assert calls == []
    assert not (kwargs["derived_root"] / config.output_relative_dir).exists()


@pytest.mark.parametrize(
    "table,token", [("scene", "scene-700"), ("sample", "test-sample")],
)
def test_test_traversal_hard_fails(tmp_path, config, table, token):
    kwargs, reader, _ = setup_pipeline(tmp_path, config)
    reader.tables["sample"]["test-sample"] = {"scene_token": "scene-700"}

    def forbidden_producer(**inputs):
        inputs["nuscenes"].get(table, token)
        pytest.fail("test access should have raised")

    kwargs["producer"] = forbidden_producer
    with pytest.raises(ValueError, match="project test"):
        projection.build_source_projection(**kwargs)
    assert not (kwargs["derived_root"] / config.output_relative_dir).exists()


def test_cross_scene_future_excluded_by_frozen_producer(tmp_path, config):
    kwargs, reader, _ = setup_pipeline(tmp_path, config)
    reader.tables["sample"]["sample-3"]["scene_token"] = "scene-560"
    with pytest.raises(ValueError, match="record count"):
        projection.build_source_projection(**kwargs)
    assert not (kwargs["derived_root"] / config.output_relative_dir).exists()


def test_config_timing_cannot_change(tmp_path):
    payload = yaml.safe_load(CONFIG.read_text())
    payload["trajectory"]["time_tolerance_sec"] = 0.1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="frozen contract"):
        projection.load_config(path, ROOT)


def test_write_failure_does_not_publish_partial_artifact(tmp_path, config, monkeypatch):
    kwargs, _, _ = setup_pipeline(tmp_path, config)

    def fail_receipt(*args, **kwargs):
        raise OSError("synthetic receipt write failure")

    monkeypatch.setattr(projection, "write_canonical_json", fail_receipt)
    with pytest.raises(OSError, match="receipt write failure"):
        projection.build_source_projection(**kwargs)
    output = kwargs["derived_root"] / config.output_relative_dir
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_output_is_deterministic_for_same_producer_inputs(tmp_path, config):
    receipts = []
    for name in ("first", "second"):
        kwargs, _, _ = setup_pipeline(tmp_path / name, config)
        receipts.append(projection.build_source_projection(**kwargs))
    assert receipts[0] == receipts[1]
