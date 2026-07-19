from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from src.actions.schema import ACTION_SCHEMA, LABEL_RULE_VERSION
from src.phase0.manifest import write_canonical_json
from src.phase0.protocol import (
    OFFICIAL_TRAIN_SCENE_COUNT,
    OFFICIAL_VAL_SCENE_COUNT,
    PHASE0_SPLIT_SEED,
    PROJECT_TRAIN_SCENE_COUNT,
    PROJECT_VALIDATION_SCENE_COUNT,
    validate_sha256,
)
from src.phase0.stratified_split import (
    HARD_CONSTRAINT_PENALTY,
    MAX_SWAP_REFINEMENTS,
    SPLIT_STRATEGY_VERSION,
)


MAPPING_SCHEMA_VERSION = "phase0_trainval_scene_split_mapping_v1"
OBJECTIVE_CONFIGURATION = {
    "distribution_distance": "total_variation",
    "train_distribution_weight": 1.0,
    "validation_distribution_weight": 1.0,
    "validation_scene_support_distance_weight": 1.0,
    "hard_constraint_penalty": HARD_CONSTRAINT_PENALTY,
    "max_swap_refinements": MAX_SWAP_REFINEMENTS,
    "validation_scene_fraction": 0.2,
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_scene_histograms(
    scene_histograms: Mapping[str, Mapping[str, int]],
) -> str:
    payload = [
        {
            "scene_token": scene_token,
            "histogram": {
                action: scene_histograms[scene_token].get(action, 0)
                for action in ACTION_SCHEMA
            },
        }
        for scene_token in sorted(scene_histograms)
    ]
    return canonical_sha256(payload)


def build_scene_mapping_payload(
    *,
    nuscenes_version: str,
    official_splits: Mapping[str, str],
    project_splits: Mapping[str, str],
    split_seed: int,
    split_strategy_version: str,
    label_rule_version: str,
    scene_histogram_sha256: str,
) -> dict[str, object]:
    if set(official_splits) != set(project_splits):
        raise ValueError(
            "official and project scene mappings must cover the same scenes"
        )
    validated_histogram_hash = validate_sha256(
        scene_histogram_sha256,
        "scene_histogram_sha256",
    )
    scenes = [
        {
            "scene_token": scene_token,
            "official_split": official_splits[scene_token],
            "project_split": project_splits[scene_token],
        }
        for scene_token in sorted(project_splits)
    ]
    material: dict[str, object] = {
        "mapping_schema_version": MAPPING_SCHEMA_VERSION,
        "nuScenes_version": nuscenes_version,
        "official_train_scene_count": sum(
            split == "train" for split in official_splits.values()
        ),
        "official_val_scene_count": sum(
            split == "val" for split in official_splits.values()
        ),
        "split_seed": split_seed,
        "split_strategy_version": split_strategy_version,
        "label_rule_version": label_rule_version,
        "action_schema": list(ACTION_SCHEMA),
        "scene_histogram_sha256": validated_histogram_hash,
        "objective": dict(OBJECTIVE_CONFIGURATION),
        "scenes": scenes,
    }
    return {
        **material,
        "scene_split_mapping_sha256": canonical_sha256(material),
    }


def validate_scene_mapping_payload(payload: Mapping[str, object]) -> str:
    mapping_hash = validate_sha256(
        payload.get("scene_split_mapping_sha256"),
        "scene_split_mapping_sha256",
    )
    validate_sha256(
        payload.get("scene_histogram_sha256"),
        "scene_histogram_sha256",
    )
    if payload.get("mapping_schema_version") != MAPPING_SCHEMA_VERSION:
        raise ValueError("unsupported scene mapping schema version")
    if payload.get("nuScenes_version") != "v1.0-trainval":
        raise ValueError("scene mapping nuScenes_version mismatch")
    if payload.get("split_seed") != PHASE0_SPLIT_SEED:
        raise ValueError("scene mapping split_seed mismatch")
    if payload.get("split_strategy_version") != SPLIT_STRATEGY_VERSION:
        raise ValueError("scene mapping split_strategy_version mismatch")
    if payload.get("label_rule_version") != LABEL_RULE_VERSION:
        raise ValueError("scene mapping label_rule_version mismatch")
    if payload.get("action_schema") != list(ACTION_SCHEMA):
        raise ValueError("scene mapping action schema mismatch")
    if payload.get("objective") != OBJECTIVE_CONFIGURATION:
        raise ValueError("scene mapping objective configuration mismatch")
    material = {
        key: value
        for key, value in payload.items()
        if key != "scene_split_mapping_sha256"
    }
    if canonical_sha256(material) != mapping_hash:
        raise ValueError("scene split mapping hash mismatch")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 850:
        raise ValueError("scene mapping must contain 850 scenes")
    scene_tokens = []
    official_counts = {"train": 0, "val": 0}
    project_counts = {"train": 0, "validation": 0, "test": 0}
    for scene in scenes:
        if not isinstance(scene, Mapping):
            raise ValueError("scene mapping entry must be an object")
        scene_token = scene.get("scene_token")
        official_split = scene.get("official_split")
        project_split = scene.get("project_split")
        if not isinstance(scene_token, str) or not scene_token:
            raise ValueError("scene mapping entry is missing scene_token")
        if official_split not in {"train", "val"}:
            raise ValueError("scene mapping has an invalid official split")
        if project_split not in {"train", "validation", "test"}:
            raise ValueError("scene mapping has an invalid project split")
        if official_split == "val" and project_split != "test":
            raise ValueError("official val scene must map to project test")
        if official_split == "train" and project_split == "test":
            raise ValueError("official train scene cannot map to project test")
        official_counts[official_split] += 1
        project_counts[project_split] += 1
        scene_tokens.append(scene_token)
    if scene_tokens != sorted(scene_tokens) or len(scene_tokens) != len(
        set(scene_tokens)
    ):
        raise ValueError("scene mapping tokens must be sorted and unique")
    if official_counts != {
        "train": OFFICIAL_TRAIN_SCENE_COUNT,
        "val": OFFICIAL_VAL_SCENE_COUNT,
    }:
        raise ValueError("scene mapping official split counts mismatch")
    if project_counts != {
        "train": PROJECT_TRAIN_SCENE_COUNT,
        "validation": PROJECT_VALIDATION_SCENE_COUNT,
        "test": OFFICIAL_VAL_SCENE_COUNT,
    }:
        raise ValueError("scene mapping project split counts mismatch")
    if payload.get("official_train_scene_count") != official_counts["train"]:
        raise ValueError("scene mapping official train count metadata mismatch")
    if payload.get("official_val_scene_count") != official_counts["val"]:
        raise ValueError("scene mapping official val count metadata mismatch")
    return mapping_hash


def read_scene_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scene mapping root must be an object")
    validate_scene_mapping_payload(payload)
    return payload


def ensure_scene_mapping(
    path: Path,
    expected_payload: Mapping[str, object],
    *,
    allow_create: bool,
) -> str:
    expected_hash = validate_scene_mapping_payload(expected_payload)
    if path.exists():
        existing_payload = read_scene_mapping(path)
        if canonical_json_bytes(existing_payload) != canonical_json_bytes(
            expected_payload
        ):
            raise ValueError("existing scene mapping does not match recomputed mapping")
        return expected_hash
    if not allow_create:
        raise ValueError("full mode requires an existing scene mapping sidecar")

    def validate_written_mapping(temporary_path: Path) -> None:
        written_payload = read_scene_mapping(temporary_path)
        if canonical_json_bytes(written_payload) != canonical_json_bytes(
            expected_payload
        ):
            raise ValueError("written scene mapping does not match expected mapping")

    write_canonical_json(
        expected_payload,
        path,
        validator=validate_written_mapping,
    )
    return expected_hash
