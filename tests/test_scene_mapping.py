from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.schema import ACTION_SCHEMA, LABEL_RULE_VERSION
from src.phase0.scene_mapping import (
    build_scene_mapping_payload,
    ensure_scene_mapping,
    hash_scene_histograms,
    read_scene_mapping,
)
from src.phase0.stratified_split import SPLIT_STRATEGY_VERSION


def scene_inputs() -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, int]],
]:
    train_tokens = tuple(f"train-{index:03d}" for index in range(700))
    val_tokens = tuple(f"val-{index:03d}" for index in range(150))
    official_splits = {
        **{token: "train" for token in train_tokens},
        **{token: "val" for token in val_tokens},
    }
    project_splits = {
        **{
            token: ("train" if index < 560 else "validation")
            for index, token in enumerate(train_tokens)
        },
        **{token: "test" for token in val_tokens},
    }
    histograms = {
        token: {
            action: int(action == ACTION_SCHEMA[index % len(ACTION_SCHEMA)])
            for action in ACTION_SCHEMA
        }
        for index, token in enumerate(train_tokens)
    }
    return official_splits, project_splits, histograms


def mapping_payload(
    official_splits: dict[str, str],
    project_splits: dict[str, str],
    histograms: dict[str, dict[str, int]],
) -> dict[str, object]:
    return build_scene_mapping_payload(
        nuscenes_version="v1.0-trainval",
        official_splits=official_splits,
        project_splits=project_splits,
        split_seed=20260710,
        split_strategy_version=SPLIT_STRATEGY_VERSION,
        label_rule_version=LABEL_RULE_VERSION,
        scene_histogram_sha256=hash_scene_histograms(histograms),
    )


def test_scene_histogram_and_mapping_hashes_ignore_input_order() -> None:
    official, project, histograms = scene_inputs()
    reversed_histograms = dict(reversed(tuple(histograms.items())))
    reversed_official = dict(reversed(tuple(official.items())))
    reversed_project = dict(reversed(tuple(project.items())))

    first = mapping_payload(official, project, histograms)
    second = mapping_payload(
        reversed_official,
        reversed_project,
        reversed_histograms,
    )

    assert hash_scene_histograms(histograms) == hash_scene_histograms(
        reversed_histograms
    )
    assert first == second


def test_mapping_hash_changes_when_one_assignment_changes() -> None:
    official, project, histograms = scene_inputs()
    changed_project = dict(project)
    changed_project["train-000"] = "validation"
    changed_project["train-560"] = "train"

    first = mapping_payload(official, project, histograms)
    changed = mapping_payload(official, changed_project, histograms)

    assert first["scene_split_mapping_sha256"] != changed[
        "scene_split_mapping_sha256"
    ]


def test_existing_mapping_mismatch_fails_without_overwrite(tmp_path: Path) -> None:
    official, project, histograms = scene_inputs()
    expected = mapping_payload(official, project, histograms)
    path = tmp_path / "scene_mapping.json"
    ensure_scene_mapping(path, expected, allow_create=True)
    original_bytes = path.read_bytes()
    changed_project = dict(project)
    changed_project["train-000"] = "validation"
    changed_project["train-560"] = "train"
    changed = mapping_payload(official, changed_project, histograms)

    with pytest.raises(ValueError, match="does not match recomputed"):
        ensure_scene_mapping(path, changed, allow_create=True)

    assert path.read_bytes() == original_bytes
    assert tuple(tmp_path.glob(".scene_mapping.json.*.tmp")) == ()


def test_pilot_reuses_complete_mapping_without_shrinking_it(tmp_path: Path) -> None:
    official, project, histograms = scene_inputs()
    expected = mapping_payload(official, project, histograms)
    path = tmp_path / "scene_mapping.json"

    first_hash = ensure_scene_mapping(path, expected, allow_create=True)
    original_bytes = path.read_bytes()
    second_hash = ensure_scene_mapping(path, expected, allow_create=True)
    persisted = read_scene_mapping(path)

    assert first_hash == second_hash
    assert path.read_bytes() == original_bytes
    assert len(persisted["scenes"]) == 850


def test_full_mode_requires_existing_mapping(tmp_path: Path) -> None:
    official, project, histograms = scene_inputs()
    expected = mapping_payload(official, project, histograms)

    with pytest.raises(ValueError, match="full mode requires"):
        ensure_scene_mapping(
            tmp_path / "missing.json",
            expected,
            allow_create=False,
        )
