import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
SELECT_SCRIPT = DATA_DIRECTORY / "select_manual_review_samples.py"
SUMMARY_SCRIPT = DATA_DIRECTORY / "summarize_manual_review.py"


def load_module(path: Path, name: str) -> ModuleType:
    assert path.is_file(), f"{path} is missing"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(DATA_DIRECTORY))
    try:
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(str(DATA_DIRECTORY))
    return module


def write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_selector_defaults_to_phase_1_7_mini_inputs() -> None:
    selector = load_module(SELECT_SCRIPT, "select_manual_review_samples")

    assert selector.DEFAULT_DERIVED_LABELS == Path(
        "data/outputs/phase_1_7_meta_action_mini/derived_meta_action.jsonl"
    )
    assert selector.DEFAULT_REVIEW_MANIFEST == Path(
        "data/outputs/phase_1_7_manual_audit_mini/review_manifest.jsonl"
    )


def test_selection_warns_when_target_exceeds_available_records(
    tmp_path: Path,
) -> None:
    selector = load_module(SELECT_SCRIPT, "select_manual_review_samples")
    derived_path = write_jsonl(
        tmp_path / "derived.jsonl",
        (
            {
                "sample_token": "keep-1",
                "timestamp_us": 1,
                "cam_front_path": "samples/CAM_FRONT/keep.jpg",
                "derived_action": "keep",
                "label_rule_version": "phase-1.6-meta-action-v0",
                "boundary_flags": [],
                "rule_features": {"delta_x_m": 8.0, "delta_y_m": 0.1},
            },
            {
                "sample_token": "stop-1",
                "timestamp_us": 2,
                "cam_front_path": "samples/CAM_FRONT/stop.jpg",
                "derived_action": "stop",
                "label_rule_version": "phase-1.6-meta-action-v0",
                "boundary_flags": ["all_zero_trajectory"],
                "rule_features": {"delta_x_m": 0.0, "delta_y_m": 0.0},
            },
        ),
    )

    candidates = selector.load_candidates(
        derived_path=derived_path,
        review_manifest_path=None,
        visualization_dir=tmp_path / "visualizations",
    )
    result = selector.select_manual_review_samples(
        candidates=candidates,
        target_count=100,
    )

    assert len(result.records) == 2
    assert "requested 100 samples but only 2 candidates are available" in (
        result.warnings
    )
    assert any(
        warning.startswith("missing derived_action coverage:")
        for warning in result.warnings
    )


def test_selection_tops_up_rare_action_classes(tmp_path: Path) -> None:
    selector = load_module(SELECT_SCRIPT, "select_manual_review_samples")
    action_counts = {
        "keep": 10,
        "accelerate": 7,
        "decelerate": 6,
        "stop": 5,
        "left_lateral": 3,
        "right_lateral": 1,
    }
    rows = []
    for action, count in action_counts.items():
        for index in range(count):
            rows.append(
                {
                    "sample_token": f"{action}-{index:02d}",
                    "timestamp_us": index,
                    "cam_front_path": f"samples/CAM_FRONT/{action}.jpg",
                    "derived_action": action,
                    "label_rule_version": "phase-1.6-meta-action-v0",
                    "boundary_flags": [],
                    "rule_features": {
                        "delta_x_m": float(index + 1),
                        "delta_y_m": 0.0,
                    },
                }
            )
    derived_path = write_jsonl(tmp_path / "derived.jsonl", tuple(rows))

    candidates = selector.load_candidates(
        derived_path=derived_path,
        review_manifest_path=None,
        visualization_dir=tmp_path / "visualizations",
    )
    result = selector.select_manual_review_samples(
        candidates=candidates,
        target_count=30,
    )
    selected_counts: dict[str, int] = {}
    for record in result.records:
        selected_counts[record.derived_action] = (
            selected_counts.get(record.derived_action, 0) + 1
        )

    assert result.candidate_action_counts == action_counts
    assert selected_counts["keep"] >= 5
    assert selected_counts["accelerate"] >= 5
    assert selected_counts["decelerate"] >= 5
    assert selected_counts["stop"] >= 5
    assert selected_counts["left_lateral"] == 3
    assert selected_counts["right_lateral"] == 1
    assert "candidate_count < 5 for derived_action left_lateral: 3" in (
        result.warnings
    )
    assert "candidate_count < 5 for derived_action right_lateral: 1" in (
        result.warnings
    )


def test_selection_covers_actions_vru_safety_and_boundary_samples(
    tmp_path: Path,
) -> None:
    selector = load_module(SELECT_SCRIPT, "select_manual_review_samples")
    derived_rows = tuple(
        {
            "sample_token": f"{action}-{index}",
            "timestamp_us": index,
            "cam_front_path": f"samples/CAM_FRONT/{action}.jpg",
            "derived_action": action,
            "label_rule_version": "phase-1.6-meta-action-v0",
            "safety_rule_version": "not_available",
            "has_vru": index % 2 == 0,
            "safety_status": "unsafe" if index in {2, 5} else "safe",
            "boundary_flags": (
                ["lateral_threshold_boundary"] if index == 4 else []
            ),
            "rule_features": {"delta_x_m": float(index), "delta_y_m": 0.0},
        }
        for index, action in enumerate(selector.ACTION_SCHEMA, start=1)
    )
    derived_path = write_jsonl(tmp_path / "derived.jsonl", derived_rows)

    candidates = selector.load_candidates(
        derived_path=derived_path,
        review_manifest_path=None,
        visualization_dir=tmp_path / "visualizations",
    )
    result = selector.select_manual_review_samples(
        candidates=candidates,
        target_count=6,
    )

    assert [record.derived_action for record in result.records] == list(
        selector.ACTION_SCHEMA
    )
    assert {record.has_vru for record in result.records} == {"yes", "no"}
    assert {record.safety_status for record in result.records} == {
        "safe",
        "unsafe",
    }
    assert any(
        record.selection_reason == "rule_boundary_sample"
        for record in result.records
    )


def test_selection_joins_manifest_and_writes_relative_visualization_path(
    tmp_path: Path,
) -> None:
    selector = load_module(SELECT_SCRIPT, "select_manual_review_samples")
    derived_path = write_jsonl(
        tmp_path / "derived.jsonl",
        (
            {
                "sample_token": "sample-a",
                "timestamp_us": 10,
                "cam_front_path": "samples/CAM_FRONT/a.jpg",
                "derived_action": "keep",
                "label_rule_version": "phase-1.6-meta-action-v0",
                "rule_features": {"delta_x_m": 1.0, "delta_y_m": 0.0},
            },
        ),
    )
    manifest_path = write_jsonl(
        tmp_path / "review_manifest.jsonl",
        (
            {
                "sample_token": "sample-a",
                "scene_token": "scene-a",
                "timestamp": 11,
                "visualization_path": "visualizations/sample-a.png",
                "nearby_agent_count": 3,
                "split": "mini",
            },
        ),
    )
    output_path = tmp_path / "audit.csv"

    candidates = selector.load_candidates(
        derived_path=derived_path,
        review_manifest_path=manifest_path,
        visualization_dir=tmp_path / "visualizations",
    )
    result = selector.select_manual_review_samples(
        candidates=candidates,
        target_count=1,
    )
    selector.write_review_csv(result.records, output_path)

    rows = read_csv(output_path)
    assert rows[0]["scene_token"] == "scene-a"
    assert rows[0]["timestamp"] == "11"
    assert rows[0]["visualization_path"] == (
        "visualizations/sample-a.png"
    )
    assert rows[0]["nearby_agent_count"] == "3"
    assert rows[0]["split"] == "mini"


def test_missing_optional_safety_fields_do_not_crash(tmp_path: Path) -> None:
    selector = load_module(SELECT_SCRIPT, "select_manual_review_samples")
    derived_path = write_jsonl(
        tmp_path / "derived.jsonl",
        (
            {
                "sample_token": "sample-a",
                "timestamp_us": 10,
                "cam_front_path": "samples/CAM_FRONT/a.jpg",
                "derived_action": "keep",
                "label_rule_version": "phase-1.6-meta-action-v0",
                "rule_features": {"delta_x_m": 1.0, "delta_y_m": 0.0},
            },
        ),
    )

    candidates = selector.load_candidates(
        derived_path=derived_path,
        review_manifest_path=None,
        visualization_dir=tmp_path / "visualizations",
    )
    result = selector.select_manual_review_samples(
        candidates=candidates,
        target_count=1,
    )

    assert result.records[0].safety_score_reasonable == "not_available"
    assert result.records[0].safety_status == "not_available"
    assert result.records[0].split == "phase-1.7-mini-audit"


def test_summary_does_not_count_uncertain_as_correct_and_groups_errors(
    tmp_path: Path,
) -> None:
    summary_module = load_module(
        SUMMARY_SCRIPT,
        "summarize_manual_review",
    )
    csv_path = tmp_path / "audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=summary_module.REQUIRED_REVIEW_FIELDS,
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "sample_token": "ok",
                    "derived_action": "keep",
                    "label_correct": "yes",
                    "trajectory_alignment_correct": "yes",
                    "agent_alignment_correct": "yes",
                    "safety_score_reasonable": "yes",
                    "error_type": "",
                },
                {
                    "sample_token": "bad",
                    "derived_action": "keep",
                    "label_correct": "no",
                    "trajectory_alignment_correct": "yes",
                    "agent_alignment_correct": "yes",
                    "safety_score_reasonable": "not_available",
                    "error_type": "wrong_action",
                },
                {
                    "sample_token": "unsure",
                    "derived_action": "stop",
                    "label_correct": "uncertain",
                    "trajectory_alignment_correct": "uncertain",
                    "agent_alignment_correct": "yes",
                    "safety_score_reasonable": "uncertain",
                    "error_type": "boundary_case",
                },
            )
        )

    rows = summary_module.read_review_rows(csv_path)
    summary = summary_module.summarize_review_rows(rows)

    assert summary.total_samples == 3
    assert summary.label_correct_counts["yes"] == 1
    assert summary.label_correct_counts["no"] == 1
    assert summary.label_correct_counts["uncertain"] == 1
    assert summary.correct_label_count == 1
    assert summary.error_type_counts["wrong_action"] == 1
    assert summary.error_type_counts["boundary_case"] == 1
    assert summary.derived_action_error_rates["keep"] == 0.5
    assert "unsure" in summary.uncertain_sample_tokens
