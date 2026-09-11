from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from scripts import build_phase0_4_temporal_dataset as builder
from scripts.derive_phase0_4_factorized_targets import derive_record
from src.phase0 import development_projection as development
from src.phase0 import phase0_4_source_projection as projection
from src.phase0.manifest import json_record
from src.phase0.phase0_4_temporal_dataset import (
    FACTORIZED_FIELDS, INPUT_FIELDS, TARGET_FIELDS, build_record, collect_history,
    validate_record,
)
from test_phase0_3_development_projection import _mapping_payload
from test_trainval_manifest import build_scene, evaluate


@pytest.fixture
def case(tmp_path):
    reader = build_scene(tmp_path, tuple(i * 500_000 for i in range(12)))
    for camera in reader.tables["sample_data"].values():
        camera["timestamp"] += 10_000
    config = projection.load_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT)
    selection = development.select_development_scenes(_mapping_payload(), config.source_contract)
    records = []
    for index in range(4):
        source = json_record(evaluate(reader, tmp_path, f"sample-{index}").record)
        source["scene_token"] = "scene-000"
        source = projection.project_record(source, split="train", config=config,
                                           split_mapping_sha256=source["split_mapping_sha256"], selection=selection)
        records.append(derive_record(source, "train", config))
    for sample in reader.tables["sample"].values():
        sample["scene_token"] = "scene-000"
    return reader, records


def test_full_history_alignment_and_waypoints(case):
    reader, records = case
    source = records[3]
    result = build_record(source, collect_history(reader, source, 3), 3)
    assert result["historical_sample_tokens"] == ["sample-1", "sample-2", "sample-3"]
    assert result["historical_sensor_timestamps"] == [510_000, 1_010_000, 1_510_000]
    assert result["historical_relative_times_sec"] == [-1, -0.5, 0]
    assert result["historical_cam_front_paths"] == [f"samples/CAM_FRONT/sample-{i}.jpg" for i in (1, 2, 3)]
    assert result["history_valid_mask"] == [True] * 3
    assert [motion["availability"] for motion in result["ego_motion_history"]] == ["partial", "full", "full"]
    assert result["ego_motion_history"][-1] == source["current_ego_motion"]
    assert result["future_waypoints"] == [[p["x_m"], p["y_m"]] for p in source["future_ego_trajectory"][1:7]]
    assert result["future_waypoints"][0] != [0, 0]
    assert len(result["future_waypoints"]) == 6
    assert all(len(point) == 2 for point in result["future_waypoints"])
    assert result["trajectory_valid_mask"] == [True] * 6
    assert {key: result[key] for key in FACTORIZED_FIELDS} == {key: source[key] for key in FACTORIZED_FIELDS}
    assert not set(INPUT_FIELDS) & set(TARGET_FIELDS)
    assert "future_ego_trajectory" not in result
    assert "source_legacy_meta_action" not in result
    assert result["coordinate_metadata"]["future_ego_trajectory"]["y_axis"] == "left"


@pytest.mark.parametrize("index,mask", [(0, [False, False, True]), (1, [False, True, True])])
def test_scene_start_padding_is_not_an_observation(case, index, mask):
    reader, records = case
    result = build_record(records[index], collect_history(reader, records[index], 3), 3)
    assert result["history_valid_mask"] == mask
    for key in INPUT_FIELDS[:-1]:
        assert len(result[key]) == 3
        assert all(value is None for value, valid in zip(result[key], mask) if not valid)
    tokens = [token for token in result["historical_sample_tokens"] if token is not None]
    assert len(set(tokens)) == len(tokens)
    assert result["ego_motion_history"][-1]["availability"] == ("unavailable" if index == 0 else "partial")


@pytest.mark.parametrize("failure,message", [
    ("future", "strictly past"), ("cross_scene", "scene boundary"),
    ("cycle", "unique"), ("camera", "CAM_FRONT sample mismatch"),
    ("anchor", "anchor timestamp"), ("motion", "anchor motion"),
    ("motion_past", "strictly past"),
])
def test_history_rejects_actual_alignment_errors(case, failure, message):
    reader, records = case
    record = records[3]
    if failure == "future":
        reader.tables["sample"]["sample-3"]["prev"] = "sample-4"
    elif failure == "cross_scene":
        reader.tables["sample"]["sample-2"]["scene_token"] = "scene-560"
    elif failure == "cycle":
        reader.tables["sample"]["sample-2"]["prev"] = "sample-3"
    elif failure == "camera":
        reader.tables["sample_data"]["camera-2"]["sample_token"] = "sample-3"
    elif failure == "anchor":
        record["current_ego_pose"]["timestamp_us"] += 1
    elif failure == "motion":
        record["current_ego_motion"]["speed_mps"] = 99
    else:
        reader.tables["sample_data"]["camera-0"]["timestamp"] = 1_000_000
    with pytest.raises(ValueError, match=message):
        collect_history(reader, record, 3)


def test_test_split_rejected_before_reader(case):
    _, records = case
    records[0]["split"] = "test"
    class ForbiddenReader:
        def get(self, *args):
            pytest.fail("reader accessed")
    with pytest.raises(ValueError, match="train and validation"):
        collect_history(ForbiddenReader(), records[0], 3)
    with pytest.raises(ValueError, match="train and validation"):
        builder.intake(Path("missing"), None, ["test"])


def test_guarded_test_scene_rejection(case):
    reader, records = case
    reader.tables["sample"]["sample-2"]["scene_token"] = "scene-test"
    counters = development.IsolationCounters()
    guarded = development.GuardedNuScenesReader(reader, frozenset({"scene-000"}), frozenset({"scene-test"}), counters)
    with pytest.raises(ValueError, match="project test"):
        collect_history(guarded, records[3], 3)
    assert counters.test_sample_records_read == 1
    assert counters.test_images_opened == 0


def test_pipeline_summary_output_and_readonly_availability(case, tmp_path, monkeypatch):
    reader, records = case
    settings = yaml.safe_load((ROOT / "configs/phase0_4_temporal_dataset.yaml").read_text())
    selection = development.select_development_scenes(
        _mapping_payload(), projection.load_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT).source_contract,
    )
    monkeypatch.setattr(builder, "intake", lambda *args: ({"train": records}, selection))
    summary = builder.build_dataset(tmp_path / "derived", tmp_path, settings, ["train"], lambda: reader, availability_only=True)
    assert not (tmp_path / "derived").exists()
    assert [entry["full_count"] for entry in summary["history_availability"]["train"].values()] == [4, 3, 2, 1]
    summary = builder.build_dataset(tmp_path / "derived", tmp_path, settings, ["train"], lambda: reader)
    assert summary["splits"]["train"]["history_full"] == 2
    assert summary["splits"]["train"]["history_padded"] == 2
    assert summary["splits"]["train"]["trajectory_all_valid"] == 4
    assert all(value == 0 for value in summary["access_evidence"].values())
    output = tmp_path / "derived" / settings["output_relative_dir"]
    restored = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(restored) == 4
    assert (output / "review.html").is_file()
    assert "REVIEWER-ONLY" in (output / "review.html").read_text()
    with pytest.raises(FileExistsError):
        builder.build_dataset(tmp_path / "derived", tmp_path, settings, ["train"], lambda: reader)


def test_frozen_artifact_intake_preserves_targets(case, tmp_path, monkeypatch):
    import hashlib

    reader, records = case
    config = projection.load_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT)
    config = replace(config, source_contract=replace(config.source_contract, expected_sample_counts={"train": 4}))
    mapping = _mapping_payload()
    monkeypatch.setattr(builder, "read_scene_mapping", lambda path: mapping)
    target_dir = tmp_path / builder.FACTORIZED_DIR
    source_dir = tmp_path / config.output_relative_dir
    target_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    for row in records:
        row["split_mapping_sha256"] = mapping["scene_split_mapping_sha256"]
    sources = [{key: value for key, value in row.items()
                if key not in (*FACTORIZED_FIELDS, "motion_features", "source_future_trajectory_version")}
               for row in records]
    payload = ''.join(json.dumps(row) + '\n' for row in sources).encode()
    (source_dir / "train.jsonl").write_bytes(payload)
    monkeypatch.setitem(builder.AUDITED_SOURCE_SHA256, "train", hashlib.sha256(payload).hexdigest())
    target_file = target_dir / "train.jsonl"
    target_file.write_text(''.join(json.dumps(row) + '\n' for row in records))
    restored, _ = builder.intake(tmp_path, config, ["train"])
    assert restored["train"] == records
    records[0]["cam_front_path"] = "samples/wrong.jpg"
    target_file.write_text(''.join(json.dumps(row) + '\n' for row in records))
    with pytest.raises(ValueError, match="changed frozen source"):
        builder.intake(tmp_path, config, ["train"])


@pytest.mark.parametrize("field,value,message", [
    ("history_valid_mask", [True, False, True], "history mask"),
    ("historical_relative_times_sec", [-2, -0.5, 0], "timestamp alignment"),
    ("future_waypoints", [[1, 2]], "shape"),
    ("trajectory_valid_mask", [True] * 5, "six valid"),
    ("split", "test", "split"),
])
def test_serialized_schema_validation(case, field, value, message):
    reader, records = case
    result = build_record(records[3], collect_history(reader, records[3], 3), 3)
    validate_record(result)
    result[field] = value
    with pytest.raises(ValueError, match=message):
        validate_record(result)
