import csv
from collections import Counter
from pathlib import Path

import pytest

from test_meta_action import load_derivation_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_AUDIT_CSV = PROJECT_ROOT / "data" / "phase_1_7_manual_audit.csv"
SUPPLEMENT_AUDIT_CSV = (
    PROJECT_ROOT / "data" / "phase_1_7_lateral_supplement_audit.csv"
)
NUSCENES_ROOT = PROJECT_ROOT / "data" / "nuscenes"
ACTION_SCHEMA = (
    "keep",
    "accelerate",
    "decelerate",
    "stop",
    "left_lateral",
    "right_lateral",
)
EXPECTED_CORRECTIONS = {
    "348c8122f47349429a6cd694dcac86e6": "decelerate",
    "73eb876167f4419a9a6ec1a601abdcaf": "decelerate",
    "7ae00681137b40f5bd7bef3823a82ee2": "decelerate",
    "8b6d496ed9d84469b75836ca1c56959f": "stop",
    "e6b0b282aa174a978272dc2d0a89d560": "accelerate",
}


def _read_audit_rows() -> tuple[dict[str, str], ...]:
    rows = []
    for path in (BASE_AUDIT_CSV, SUPPLEMENT_AUDIT_CSV):
        with path.open(newline="", encoding="utf-8") as input_file:
            rows.extend(csv.DictReader(input_file))
    return tuple(rows)


def _load_nuscenes_or_skip():
    if not (NUSCENES_ROOT / "v1.0-mini").is_dir():
        pytest.skip("nuScenes mini is unavailable under data/nuscenes")

    pytest.importorskip("nuscenes")
    from nuscenes.nuscenes import NuScenes

    return NuScenes(
        version="v1.0-mini",
        dataroot=str(NUSCENES_ROOT),
        verbose=False,
    )


def _print_transition_matrix(transitions: Counter[tuple[str, str]]) -> None:
    print("old_action -> new_action transition matrix:")
    print("old_action,new_action,count")
    for old_action in ACTION_SCHEMA:
        for new_action in ACTION_SCHEMA:
            count = transitions[(old_action, new_action)]
            if count:
                print(f"{old_action},{new_action},{count}")


def test_phase_1_8_108_sample_audit_regression() -> None:
    rows = _read_audit_rows()
    sample_tokens = tuple(row["sample_token"] for row in rows)
    assert len(rows) == 108
    assert len(sample_tokens) == len(set(sample_tokens))

    derivation = load_derivation_module()
    rules = derivation.load_meta_action_rules(
        PROJECT_ROOT / "configs" / "action_rules.yaml"
    )
    data_config = derivation._load_data_config(
        PROJECT_ROOT / "configs" / "data.yaml",
        NUSCENES_ROOT,
    )
    nuscenes = _load_nuscenes_or_skip()
    new_actions = {
        record.sample_token: record.derived_action
        for record in (
            derivation.derive_sample_record(
                nuscenes=nuscenes,
                sample_token=sample_token,
                camera=derivation.CAMERA_CHANNEL,
                rules=rules,
                time_tolerance_sec=data_config.trajectory_time_tolerance_sec,
            )
            for sample_token in sample_tokens
        )
    }

    new_distribution = Counter(new_actions.values())
    transitions = Counter(
        (row["derived_action"], new_actions[row["sample_token"]])
        for row in rows
    )
    new_correct_count = sum(
        new_actions[row["sample_token"]] == row["reviewed_action"]
        for row in rows
    )
    print(f"total samples: {len(rows)}")
    print(f"new derived_action distribution: {dict(sorted(new_distribution.items()))}")
    print(f"new_correct_count: {new_correct_count}")
    _print_transition_matrix(transitions)

    assert set(new_distribution) == set(ACTION_SCHEMA)
    assert new_distribution["left_lateral"] == 5
    assert new_distribution["right_lateral"] == 5
    assert new_distribution["accelerate"] >= 5
    assert new_distribution["decelerate"] >= 13
    assert new_distribution["stop"] >= 20
    assert 45 <= new_distribution["keep"] <= 65
    assert all(
        new_actions[sample_token] == action
        for sample_token, action in EXPECTED_CORRECTIONS.items()
    )
    assert new_correct_count >= 103
    assert not any(
        old_action in {"left_lateral", "right_lateral"}
        and new_action != old_action
        for old_action, new_action in transitions
    )
    assert not any(
        row["derived_action"] == row["reviewed_action"]
        and row["derived_action"] in {"accelerate", "decelerate", "stop"}
        and new_actions[row["sample_token"]] == "keep"
        for row in rows
    )
