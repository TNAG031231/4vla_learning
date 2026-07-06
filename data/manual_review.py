from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path


REVIEW_RECORD_FIELDS = (
    "sample_token",
    "scene_token",
    "timestamp",
    "cam_front_path",
    "visualization_path",
    "derived_action",
    "reviewed_action",
    "label_correct",
    "trajectory_alignment_correct",
    "agent_alignment_correct",
    "visualization_sufficient",
    "safety_score_reasonable",
    "error_type",
    "review_note",
    "reviewer",
    "review_time",
    "label_rule_version",
    "safety_rule_version",
    "selection_reason",
    "forward_displacement_m",
    "lateral_displacement_m",
    "total_displacement_m",
    "nearby_agent_count",
)
LABEL_REVIEW_VALUES = frozenset(
    {"yes", "no", "uncertain", "not_applicable"}
)
ALIGNMENT_REVIEW_VALUES = frozenset({"yes", "no", "uncertain"})
SAFETY_REVIEW_VALUES = frozenset(
    {"yes", "no", "uncertain", "not_available"}
)


@dataclass(frozen=True)
class ReviewRecord:
    sample_token: str
    scene_token: str
    timestamp: int
    cam_front_path: str
    visualization_path: str
    derived_action: str
    reviewed_action: str
    label_correct: str
    trajectory_alignment_correct: str
    agent_alignment_correct: str
    visualization_sufficient: str
    safety_score_reasonable: str
    error_type: str
    review_note: str
    reviewer: str
    review_time: str
    label_rule_version: str
    safety_rule_version: str
    selection_reason: str
    forward_displacement_m: float
    lateral_displacement_m: float
    total_displacement_m: float
    nearby_agent_count: int


@dataclass(frozen=True)
class ReviewSummary:
    total_records: int
    label_correct: int
    trajectory_alignment_correct: int
    agent_alignment_correct: int
    visualization_sufficient: int
    safety_score_reasonable: int


@dataclass(frozen=True)
class ReviewCandidate:
    sample_token: str
    scene_token: str
    timestamp: int
    cam_front_path: str
    forward_displacement_m: float
    lateral_displacement_m: float
    total_displacement_m: float
    nearby_agent_count: int


@dataclass(frozen=True)
class ReviewSelection:
    candidate: ReviewCandidate
    reason: str


def validate_review_record(record: ReviewRecord) -> None:
    enum_fields = (
        ("label_correct", record.label_correct, LABEL_REVIEW_VALUES),
        (
            "trajectory_alignment_correct",
            record.trajectory_alignment_correct,
            ALIGNMENT_REVIEW_VALUES,
        ),
        (
            "agent_alignment_correct",
            record.agent_alignment_correct,
            ALIGNMENT_REVIEW_VALUES,
        ),
        (
            "visualization_sufficient",
            record.visualization_sufficient,
            ALIGNMENT_REVIEW_VALUES,
        ),
        (
            "safety_score_reasonable",
            record.safety_score_reasonable,
            SAFETY_REVIEW_VALUES,
        ),
    )
    for field_name, value, allowed_values in enum_fields:
        if value not in allowed_values:
            raise ValueError(
                f"{field_name} must be one of {sorted(allowed_values)}"
            )


def create_review_record(
    sample_token: str,
    scene_token: str,
    timestamp: int,
    cam_front_path: str,
    visualization_path: str,
    selection_reason: str,
    label_rule_version: str,
    safety_rule_version: str,
    forward_displacement_m: float = 0.0,
    lateral_displacement_m: float = 0.0,
    total_displacement_m: float = 0.0,
    nearby_agent_count: int = 0,
) -> ReviewRecord:
    record = ReviewRecord(
        sample_token=sample_token,
        scene_token=scene_token,
        timestamp=timestamp,
        cam_front_path=cam_front_path,
        visualization_path=visualization_path,
        derived_action="not_available",
        reviewed_action="",
        label_correct="uncertain",
        trajectory_alignment_correct="uncertain",
        agent_alignment_correct="uncertain",
        visualization_sufficient="uncertain",
        safety_score_reasonable="not_available",
        error_type="",
        review_note="",
        reviewer="",
        review_time="",
        label_rule_version=label_rule_version,
        safety_rule_version=safety_rule_version,
        selection_reason=selection_reason,
        forward_displacement_m=forward_displacement_m,
        lateral_displacement_m=lateral_displacement_m,
        total_displacement_m=total_displacement_m,
        nearby_agent_count=nearby_agent_count,
    )
    validate_review_record(record)
    return record


def summarize_review_records(
    records: tuple[ReviewRecord, ...],
) -> ReviewSummary:
    return ReviewSummary(
        total_records=len(records),
        label_correct=sum(record.label_correct == "yes" for record in records),
        trajectory_alignment_correct=sum(
            record.trajectory_alignment_correct == "yes"
            for record in records
        ),
        agent_alignment_correct=sum(
            record.agent_alignment_correct == "yes" for record in records
        ),
        visualization_sufficient=sum(
            record.visualization_sufficient == "yes" for record in records
        ),
        safety_score_reasonable=sum(
            record.safety_score_reasonable == "yes" for record in records
        ),
    )


def _choose_candidate(
    ranked_candidates: tuple[ReviewCandidate, ...],
    selected_tokens: set[str],
    selected_scenes: set[str],
) -> ReviewCandidate | None:
    available = tuple(
        candidate
        for candidate in ranked_candidates
        if candidate.sample_token not in selected_tokens
    )
    if not available:
        return None
    return next(
        (
            candidate
            for candidate in available
            if candidate.scene_token not in selected_scenes
        ),
        available[0],
    )


def select_review_candidates(
    candidates: tuple[ReviewCandidate, ...],
    sample_count: int,
) -> tuple[ReviewSelection, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    criteria = (
        (
            "high_forward_displacement_candidate",
            tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        -candidate.forward_displacement_m,
                        candidate.sample_token,
                    ),
                )
            ),
        ),
        (
            "low_displacement_candidate",
            tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        candidate.total_displacement_m,
                        candidate.sample_token,
                    ),
                )
            ),
        ),
        (
            "has_nearby_agents",
            tuple(
                sorted(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.nearby_agent_count > 0
                    ),
                    key=lambda candidate: (
                        -candidate.nearby_agent_count,
                        candidate.sample_token,
                    ),
                )
            ),
        ),
        (
            "no_nearby_agents",
            tuple(
                sorted(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.nearby_agent_count == 0
                    ),
                    key=lambda candidate: candidate.sample_token,
                )
            ),
        ),
        (
            "lateral_displacement_candidate",
            tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        -abs(candidate.lateral_displacement_m),
                        candidate.sample_token,
                    ),
                )
            ),
        ),
        (
            "ordinary_motion_candidate",
            tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        abs(candidate.lateral_displacement_m),
                        -candidate.total_displacement_m,
                        candidate.sample_token,
                    ),
                )
            ),
        ),
    )
    selected_tokens: set[str] = set()
    selected_scenes: set[str] = set()
    selections = []
    for reason, ranked_candidates in criteria:
        if len(selections) >= sample_count:
            break
        candidate = _choose_candidate(
            ranked_candidates,
            selected_tokens,
            selected_scenes,
        )
        if candidate is None:
            continue
        selections.append(ReviewSelection(candidate=candidate, reason=reason))
        selected_tokens.add(candidate.sample_token)
        selected_scenes.add(candidate.scene_token)

    remaining = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if candidate.sample_token not in selected_tokens
            ),
            key=lambda candidate: (
                candidate.scene_token in selected_scenes,
                candidate.scene_token,
                candidate.timestamp,
                candidate.sample_token,
            ),
        )
    )
    for candidate in remaining:
        if len(selections) >= sample_count:
            break
        selections.append(
            ReviewSelection(
                candidate=candidate,
                reason="scene_diversity_fill_candidate",
            )
        )
        selected_scenes.add(candidate.scene_token)

    return tuple(selections)


def create_review_records(
    selections: tuple[ReviewSelection, ...],
    label_rule_version: str,
    safety_rule_version: str,
) -> tuple[ReviewRecord, ...]:
    return tuple(
        create_review_record(
            sample_token=selection.candidate.sample_token,
            scene_token=selection.candidate.scene_token,
            timestamp=selection.candidate.timestamp,
            cam_front_path=selection.candidate.cam_front_path,
            visualization_path=(
                "visualizations/"
                f"{selection.candidate.sample_token}_one_page.png"
            ),
            selection_reason=selection.reason,
            label_rule_version=label_rule_version,
            safety_rule_version=safety_rule_version,
            forward_displacement_m=(
                selection.candidate.forward_displacement_m
            ),
            lateral_displacement_m=(
                selection.candidate.lateral_displacement_m
            ),
            total_displacement_m=selection.candidate.total_displacement_m,
            nearby_agent_count=selection.candidate.nearby_agent_count,
        )
        for selection in selections
    )


def write_review_outputs(
    records: tuple[ReviewRecord, ...],
    manifest_path: Path,
    template_path: Path,
) -> None:
    for record in records:
        validate_review_record(record)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(
                json.dumps(asdict(record), ensure_ascii=False) + "\n"
            )

    template_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as template_file:
        writer = csv.DictWriter(
            template_file,
            fieldnames=REVIEW_RECORD_FIELDS,
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
