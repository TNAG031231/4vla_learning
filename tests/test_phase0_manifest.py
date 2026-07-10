from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import build_phase0_manifest as manifest_builder


def test_manifest_record_preserves_phase_minus_one_source_and_required_fields() -> None:
    audit_row = SimpleNamespace(
        source_audit="base",
        sample_token="sample-a",
        scene_token="scene-a",
        timestamp="10",
        cam_front_path="samples/CAM_FRONT/a.jpg",
        historical_derived_action="keep",
        reviewed_action="keep",
        label_correct="yes",
        trajectory_alignment_correct="yes",
        agent_alignment_correct="yes",
        label_rule_version="phase-1.6-meta-action-v0.1",
    )
    derived = SimpleNamespace(
        sample_token="sample-a",
        scene_token="scene-a",
        timestamp_us=10,
        cam_front_path="samples/CAM_FRONT/a.jpg",
        derived_action="keep",
        label_rule_version="phase-1.6-meta-action-v0.2",
    )
    trajectory = SimpleNamespace(points=(SimpleNamespace(t_sec=0.0),))
    agents = SimpleNamespace(agents=(SimpleNamespace(instance_token="agent-a"),))

    record = manifest_builder.build_manifest_record(
        audit_row=audit_row,
        derived_record=derived,
        current_ego_state={"frame": "global"},
        trajectory=trajectory,
        nearby_agents=agents,
        split="train",
    )

    payload = manifest_builder.to_json_record(record)

    assert set(payload) >= {
        "sample_token",
        "scene_token",
        "timestamp",
        "cam_front_path",
        "current_ego_state",
        "future_ego_trajectory",
        "nearby_agents",
        "meta_action",
        "label_rule_version",
        "safety_rule_version",
        "split",
        "source_audit_record",
        "coordinate_metadata",
    }
    assert payload["meta_action"] == "keep"
    assert payload["source_audit_record"]["source_audit"] == "base"
    assert payload["source_audit_record"]["sample_token"] == "sample-a"
    assert payload["future_ego_trajectory"] == [{"t_sec": 0.0}]
    assert payload["nearby_agents"] == [{"instance_token": "agent-a"}]
    assert payload["coordinate_metadata"]["future_ego_trajectory"] == {
        "source_frame": "nuScenes_global",
        "target_frame": "current_ego",
        "x_axis": "forward",
        "y_axis": "left",
        "z_axis": "up",
        "unit": "meter",
        "transform": "subtract_current_ego_translation_then_apply_inverse_current_ego_rotation",
    }


def test_manifest_reader_rejects_unknown_action(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"sample_token": "sample", "scene_token": "scene", "meta_action": "turn_left", "split": "train", "label_rule_version": "v0"}\n',
        encoding="utf-8",
    )

    try:
        manifest_builder.read_manifest_samples(manifest_path)
    except ValueError as error:
        assert "Unsupported action" in str(error)
    else:
        raise AssertionError("unknown meta_action must be rejected")
