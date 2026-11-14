from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from pyquaternion import Quaternion
import pytest
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from src.phase0 import development_projection as development
from src.phase0 import phase0_4_source_projection as projection
from src.phase0.manifest import json_record
from src.phase0.phase0_5_geometry import (
    Grid, box_cells, box_corners, build_geometry, category_id, load_geometry, save_geometry,
)
from src.phase0.phase0_5_visualization import render_geometry
from test_phase0_3_development_projection import _mapping_payload
from test_trainval_manifest import build_scene, evaluate


@pytest.fixture
def scene(tmp_path):
    reader = build_scene(tmp_path, tuple(i * 500_000 for i in range(8)))
    rotation = Quaternion(axis=[0, 0, 1], angle=math.pi / 2)
    reader.tables["ego_pose"]["pose-0"].update(translation=[10, 20, 0], rotation=list(rotation))
    source = json_record(evaluate(reader, tmp_path).record)
    source["scene_token"] = "scene-000"
    config = projection.load_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT)
    selection = development.select_development_scenes(_mapping_payload(), config.source_contract)
    source = projection.project_record(source, split="train", config=config,
                                       split_mapping_sha256=source["split_mapping_sha256"], selection=selection)
    reader.tables["sample_annotation"] = {}
    for sample in reader.tables["sample"].values():
        sample["scene_token"] = "scene-000"
    for t in range(1, 7):
        for instance, category, xy, size in (
            ("car", "vehicle.car", [4 + t, 1, 0], [2, 4, 1.5]),
            ("bike", "vehicle.bicycle", [2, -3, 0], [0.6, 1.5, 1]),
        ):
            if instance == "bike" and t == 3:
                continue
            token = f"{instance}-{t}"
            reader.tables["sample"][f"sample-{t}"]["anns"].append(token)
            reader.tables["sample_annotation"][token] = {
                "token": token, "instance_token": instance, "sample_token": f"sample-{t}",
                "category_name": category,
                "translation": (rotation.rotate(np.array(xy)) + [10, 20, 0]).tolist(),
                "size": size, "rotation": list(rotation),
            }
    return reader, source


def build(reader, source, grid=None, max_agents=4):
    return build_geometry(reader, source, grid or Grid((-10, 50), (-25, 25), 0.5),
                          max_agents=max_agents, annotation_camera_tolerance_sec=0.075)


def test_current_frame_identity_sizes_and_missing_annotation(scene):
    reader, source = scene
    arrays, metadata = build(reader, source)
    assert metadata["instance_tokens"] == ["bike", "car", None, None]
    np.testing.assert_allclose(arrays["agent_numeric_state"][:, 1, :4],
                               [[5 + t, 1, 4, 2] for t in range(6)], atol=1e-6)
    np.testing.assert_allclose(arrays["agent_numeric_state"][:, 1, 4], 0, atol=1e-6)
    assert metadata["annotation_tokens"][0][1] == "car-1"
    assert metadata["annotation_tokens"][5][1] == "car-6"
    assert metadata["source_sample_tokens"] == [f"sample-{i}" for i in range(1, 7)]
    assert arrays["agent_category_id"][0].tolist() == [2, 1, 0, 0]
    assert not arrays["agent_valid_mask"][2, 0]
    assert arrays["agent_motion_source"][2, 0] == 0
    assert metadata["annotation_tokens"][2][0] is None
    assert arrays["temporal_occupancy"].shape == (6, 2, 120, 100)
    assert not arrays["temporal_occupancy"][2, 1].any()


@pytest.mark.parametrize("state", [
    [0, 0, 4, 2, 0], [1.1, -0.3, 3.7, 1.6, 0.71],
    [3.9, 1, 2, 1, 0.4], [20, 0, 2, 1, 0], [-4, -4, 0.4, 0.5, 1.2],
])
def test_raster_matches_independent_polygon_intersections(state):
    grid = Grid((-4, 4), (-4, 4), 0.5)
    state = np.array(state)
    x, y, status = box_cells(state, grid)
    actual = np.zeros(grid.shape, dtype=bool)
    actual[x, y] = True
    polygon = Polygon(box_corners(state))
    expected = np.zeros_like(actual)
    for i, j in np.ndindex(grid.shape):
        x0, y0 = -4 + i * 0.5, -4 + j * 0.5
        expected[i, j] = polygon.intersection(box(x0, y0, x0 + 0.5, y0 + 0.5)).area > 1e-10
    np.testing.assert_array_equal(actual, expected)
    assert (status == "outside") == (not expected.any())


def test_empty_scene_and_object_raster_union(scene):
    reader, source = scene
    arrays, _ = build(reader, source)
    grid = Grid((-10, 50), (-25, 25), 0.5)
    for t in range(6):
        for i in np.flatnonzero(arrays["agent_valid_mask"][t]):
            x, y, _ = box_cells(arrays["agent_numeric_state"][t, i], grid)
            channel = arrays["agent_category_id"][t, i] - 1
            assert arrays["temporal_occupancy"][t, channel, x, y].all()
    for sample in reader.tables["sample"].values():
        sample["anns"] = []
    empty, meta = build(reader, source)
    assert not empty["agent_valid_mask"].any()
    assert not empty["temporal_occupancy"].any()
    assert empty["temporal_occupancy_valid_mask"].all()
    assert meta["instance_count"] == 0


def test_artifact_roundtrip_and_render(scene, tmp_path):
    arrays, metadata = build(*scene)
    path = tmp_path / "geometry.npz"
    save_geometry(path, arrays, metadata)
    restored, sidecar = load_geometry(path)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], restored[key])
        assert arrays[key].dtype == restored[key].dtype
    assert sidecar == json.loads(json.dumps(metadata))
    render_geometry(restored, sidecar, tmp_path / "review.png")
    assert (tmp_path / "review.png").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        save_geometry(path, arrays, metadata)


@pytest.mark.parametrize("failure,match", [
    ("test", "train/validation"), ("scene", "scene boundary"),
    ("timestamp", "sample time"), ("camera", "annotation/camera"),
    ("annotation", "different sample"), ("capacity", "no objects dropped"),
])
def test_rejects_boundary_contract_errors(scene, failure, match):
    reader, source = scene
    if failure == "test":
        source["split"] = "test"
    elif failure == "scene":
        reader.tables["sample"]["sample-2"]["scene_token"] = "other-scene"
    elif failure == "timestamp":
        reader.tables["sample"]["sample-2"]["timestamp"] += 100_000
    elif failure == "camera":
        reader.tables["sample_data"]["camera-2"]["timestamp"] += 100_000
    elif failure == "annotation":
        reader.tables["sample_annotation"]["car-2"]["sample_token"] = "sample-3"
    with pytest.raises(ValueError, match=match):
        build(reader, source, max_agents=1 if failure == "capacity" else 4)


def test_category_channels_are_exclusive():
    assert category_id("vehicle.car") == 1
    assert category_id("vehicle.bicycle") == 2
    assert category_id("vehicle.motorcycle") == 2
    assert category_id("human.pedestrian.adult") == 2
    assert category_id("movable_object.trafficcone") == 0


def test_grid_rejects_invalid_extent():
    with pytest.raises(ValueError, match="positive integer"):
        Grid((0, 2.1), (0, 2), 0.5)
