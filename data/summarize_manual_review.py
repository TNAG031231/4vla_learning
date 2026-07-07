#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path


REQUIRED_REVIEW_FIELDS = (
    "sample_token",
    "derived_action",
    "label_correct",
    "trajectory_alignment_correct",
    "agent_alignment_correct",
    "safety_score_reasonable",
    "error_type",
)
EXPECTED_REVIEW_VALUES = ("yes", "no", "uncertain")


@dataclass(frozen=True)
class ManualReviewSummary:
    total_samples: int
    derived_action_counts: dict[str, int]
    label_correct_counts: dict[str, int]
    trajectory_alignment_counts: dict[str, int]
    agent_alignment_counts: dict[str, int]
    safety_score_reasonable_counts: dict[str, int]
    derived_action_error_rates: dict[str, float]
    error_type_counts: dict[str, int]
    uncertain_sample_tokens: tuple[str, ...]
    correct_label_count: int


def read_review_rows(path: Path) -> tuple[Mapping[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_fields = [
            field
            for field in REQUIRED_REVIEW_FIELDS
            if field not in (reader.fieldnames or [])
        ]
        if missing_fields:
            raise ValueError(
                "review CSV missing required fields: "
                f"{', '.join(missing_fields)}"
            )
        return tuple(dict(row) for row in reader)


def _count_values(
    rows: tuple[Mapping[str, str], ...],
    field_name: str,
) -> dict[str, int]:
    counter = Counter(row.get(field_name, "").strip() for row in rows)
    return dict(sorted(counter.items()))


def _error_types(rows: tuple[Mapping[str, str], ...]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        raw_value = row.get("error_type", "")
        for value in raw_value.replace(";", ",").split(","):
            normalized = value.strip()
            if normalized:
                counter[normalized] += 1
    return dict(sorted(counter.items()))


def _derived_action_error_rates(
    rows: tuple[Mapping[str, str], ...],
) -> dict[str, float]:
    action_rows: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        action_rows.setdefault(row.get("derived_action", ""), []).append(row)

    rates: dict[str, float] = {}
    for action, grouped_rows in sorted(action_rows.items()):
        reviewed_rows = [
            row
            for row in grouped_rows
            if row.get("label_correct", "").strip() in {"yes", "no"}
        ]
        if not reviewed_rows:
            rates[action] = 0.0
            continue
        incorrect = sum(
            row.get("label_correct", "").strip() == "no"
            for row in reviewed_rows
        )
        rates[action] = incorrect / len(reviewed_rows)
    return rates


def _uncertain_tokens(rows: tuple[Mapping[str, str], ...]) -> tuple[str, ...]:
    uncertain_fields = (
        "label_correct",
        "trajectory_alignment_correct",
        "agent_alignment_correct",
        "safety_score_reasonable",
    )
    return tuple(
        row.get("sample_token", "")
        for row in rows
        if any(row.get(field, "").strip() == "uncertain" for field in uncertain_fields)
    )


def summarize_review_rows(
    rows: tuple[Mapping[str, str], ...],
) -> ManualReviewSummary:
    return ManualReviewSummary(
        total_samples=len(rows),
        derived_action_counts=_count_values(rows, "derived_action"),
        label_correct_counts=_count_values(rows, "label_correct"),
        trajectory_alignment_counts=_count_values(
            rows,
            "trajectory_alignment_correct",
        ),
        agent_alignment_counts=_count_values(rows, "agent_alignment_correct"),
        safety_score_reasonable_counts=_count_values(
            rows,
            "safety_score_reasonable",
        ),
        derived_action_error_rates=_derived_action_error_rates(rows),
        error_type_counts=_error_types(rows),
        uncertain_sample_tokens=_uncertain_tokens(rows),
        correct_label_count=sum(
            row.get("label_correct", "").strip() == "yes" for row in rows
        ),
    )


def print_summary(summary: ManualReviewSummary) -> None:
    print(f"total samples: {summary.total_samples}")
    print(f"derived_action counts: {summary.derived_action_counts}")
    print(f"label_correct counts: {summary.label_correct_counts}")
    print(
        "trajectory_alignment_correct counts: "
        f"{summary.trajectory_alignment_counts}"
    )
    print(
        "agent_alignment_correct counts: "
        f"{summary.agent_alignment_counts}"
    )
    print(
        "safety_score_reasonable counts: "
        f"{summary.safety_score_reasonable_counts}"
    )
    print(
        "derived_action error rates: "
        f"{summary.derived_action_error_rates}"
    )
    print(f"error_type counts: {summary.error_type_counts}")
    print(
        "uncertain sample_tokens: "
        f"{list(summary.uncertain_sample_tokens)}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a filled Phase -1.7 manual review CSV."
    )
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    summary = summarize_review_rows(read_review_rows(arguments.review_csv))
    print_summary(summary)
    if arguments.summary_json is not None:
        arguments.summary_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary_json.write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
