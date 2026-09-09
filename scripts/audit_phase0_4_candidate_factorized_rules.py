#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "data"))

from data.inspect_nuscenes_sample import TrajectoryPoint  # noqa: E402
from scripts import analyze_phase0_4_factorized_features as calibration  # noqa: E402
from src.phase0.phase0_4_factorized_targets import (  # noqa: E402
    LATERAL_CANDIDATE_ACTIONS,
    LONGITUDINAL_CANDIDATE_ACTIONS,
    CandidateDirectionResult,
    CandidateEvaluation,
    CandidateRuleParameters,
    evaluate_candidate_rules,
    extract_motion_features,
    load_candidate_parameters,
    stop_motion_extent,
)


PROVENANCE_NOTICE = (
    "legacy provenance agreement is diagnostic only, not factorized ground truth."
)
BOUNDARY_CATEGORIES = {
    "speed_boundary": "longitudinal_speed_boundary",
    "stop_boundary": "longitudinal_stop_boundary",
    "low_forward_displacement": "longitudinal_low_forward_displacement",
    "lateral_displacement_boundary": "lateral_displacement_boundary",
    "lateral_heading_sign_conflict": "lateral_heading_sign_conflict",
    "lateral_excursion_without_final_displacement": (
        "lateral_excursion_without_final_displacement"
    ),
    "heading_without_lateral_displacement": "heading_without_lateral_displacement",
}


def summarize_direction(
    results: list[CandidateDirectionResult], actions: tuple[str, ...],
) -> dict[str, object]:
    valid_count = sum(result.candidate_valid for result in results)
    count = len(results)
    distribution = Counter(result.candidate_action for result in results)
    return {
        "sample_count": count,
        "candidate_distribution": {action: distribution[action] for action in actions},
        "valid_count": valid_count,
        "valid_rate": valid_count / count if count else None,
        "invalid_count": count - valid_count,
        "invalid_rate": (count - valid_count) / count if count else None,
        "reason_counts": dict(sorted(Counter(
            result.candidate_reason for result in results
        ).items())),
    }


def provenance_diagnostic(
    records: list[dict[str, object]],
    candidates: list[CandidateEvaluation],
    *,
    audited_only: bool,
) -> dict[str, object]:
    groups: dict[str, list[CandidateDirectionResult]] = defaultdict(list)
    for record, candidate in zip(records, candidates):
        if audited_only and record["source_audit_record"] is None:
            continue
        legacy = record["source_legacy_meta_action"]
        direction = (
            candidate.candidate_longitudinal
            if legacy in LONGITUDINAL_CANDIDATE_ACTIONS
            else candidate.candidate_lateral
        )
        groups[legacy].append(direction)

    diagnostics = {}
    for legacy, results in sorted(groups.items()):
        longitudinal = legacy in LONGITUDINAL_CANDIDATE_ACTIONS
        # This reference is used only after feature-based candidate evaluation.
        reference = legacy if longitudinal else {
            "left_lateral": "left", "right_lateral": "right",
        }[legacy]
        summary = summarize_direction(
            results,
            LONGITUDINAL_CANDIDATE_ACTIONS
            if longitudinal else LATERAL_CANDIDATE_ACTIONS,
        )
        valid = sum(result.candidate_valid for result in results)
        agreement = sum(
            result.candidate_valid and result.candidate_action == reference
            for result in results
        )
        diagnostics[legacy] = {
            **summary,
            "compared_candidate_direction": (
                "longitudinal" if longitudinal else "lateral"
            ),
            "agreement_count": agreement,
            "valid_disagreement_count": valid - agreement,
            "agreement_rate_among_valid": agreement / valid if valid else None,
            "agreement_rate_all_samples": agreement / len(results),
        }
    return {
        "sample_count": sum(len(group) for group in groups.values()),
        "by_legacy_provenance": diagnostics,
    }


def boundary_distance(
    reason: str, features: dict[str, float], parameters: CandidateRuleParameters,
) -> tuple[float, str]:
    speed_edges = parameters.bounds(parameters.speed_change_center_mps)
    stop_edges = parameters.bounds(parameters.stop_distance_center_m)
    lateral_edges = parameters.bounds(parameters.lateral_displacement_center_m)
    _, heading_high = parameters.bounds(parameters.heading_support_rad)
    distances = {
        "speed_boundary": (
            min(
                abs(abs(features["delta_speed_proxy_mps"]) - edge)
                for edge in speed_edges
            ),
            "m/s",
        ),
        "stop_boundary": (
            min(abs(stop_motion_extent(features) - edge) for edge in stop_edges), "m",
        ),
        "low_forward_displacement": (
            abs(
                features["forward_displacement_m"]
                - parameters.minimum_forward_displacement_m
            ),
            "m",
        ),
        "lateral_displacement_boundary": (
            min(
                abs(abs(features["final_lateral_displacement_m"]) - edge)
                for edge in lateral_edges
            ),
            "m",
        ),
        "lateral_heading_sign_conflict": (
            abs(abs(features["final_heading_delta_rad"]) - heading_high), "rad",
        ),
        "lateral_excursion_without_final_displacement": (
            abs(features["max_abs_lateral_displacement_m"] - lateral_edges[1]), "m",
        ),
        "heading_without_lateral_displacement": (
            abs(abs(features["final_heading_delta_rad"]) - heading_high), "rad",
        ),
    }
    return distances[reason]


def audit_split(
    records: list[dict[str, object]], parameters: CandidateRuleParameters,
) -> dict[str, object]:
    candidates = []
    boundaries = {category: [] for category in BOUNDARY_CATEGORIES.values()}
    independent = {
        "longitudinal_valid_lateral_invalid": 0,
        "longitudinal_invalid_lateral_valid": 0,
        "both_valid": 0,
        "both_invalid": 0,
    }
    joint = {
        f"{longitudinal} + {lateral}": 0
        for longitudinal in LONGITUDINAL_CANDIDATE_ACTIONS
        for lateral in LATERAL_CANDIDATE_ACTIONS
    }
    for record in records:
        features = extract_motion_features([
            TrajectoryPoint(**point) for point in record["future_ego_trajectory"]
        ])
        candidate = evaluate_candidate_rules(features, parameters)
        candidates.append(candidate)
        longitudinal = candidate.candidate_longitudinal
        lateral = candidate.candidate_lateral
        if candidate.candidate_joint_valid:
            independent["both_valid"] += 1
            joint[f"{longitudinal.candidate_action} + {lateral.candidate_action}"] += 1
        elif longitudinal.candidate_valid:
            independent["longitudinal_valid_lateral_invalid"] += 1
        elif lateral.candidate_valid:
            independent["longitudinal_invalid_lateral_valid"] += 1
        else:
            independent["both_invalid"] += 1
        for direction in (longitudinal, lateral):
            reason = direction.candidate_reason
            if reason not in BOUNDARY_CATEGORIES:
                continue
            distance, unit = boundary_distance(reason, features, parameters)
            boundaries[BOUNDARY_CATEGORIES[reason]].append({
                **{key: record[key] for key in (
                    "sample_token", "scene_token", "split", "cam_front_path",
                    "source_legacy_meta_action",
                )},
                "source_audit_record_present": (
                    record["source_audit_record"] is not None
                ),
                "motion_features": features,
                "candidate_result": asdict(candidate),
                "distance_to_candidate_boundary": distance,
                "distance_unit": unit,
            })
    return {
        "sample_count": len(records),
        "candidate_longitudinal": summarize_direction(
            [result.candidate_longitudinal for result in candidates],
            LONGITUDINAL_CANDIDATE_ACTIONS,
        ),
        "candidate_lateral": summarize_direction(
            [result.candidate_lateral for result in candidates],
            LATERAL_CANDIDATE_ACTIONS,
        ),
        "independent_candidate_validity": independent,
        "candidate_joint_valid_combinations": joint,
        "provenance_only_diagnostic": {
            "all_source": provenance_diagnostic(
                records, candidates, audited_only=False,
            ),
            "audited_subset": provenance_diagnostic(
                records, candidates, audited_only=True,
            ),
        },
        "boundary_samples": {
            category: sorted(rows, key=lambda row: (
                row["distance_to_candidate_boundary"], row["sample_token"],
            ))[:10]
            for category, rows in boundaries.items()
        },
    }


def audit_sources(
    derived_root: Path, splits: list[str], parameters: CandidateRuleParameters,
) -> dict[str, object]:
    sources = calibration.read_audited_sources(derived_root, splits)
    return {
        "status": "provisional",
        "candidate_parameters": asdict(parameters),
        "provenance_notice": PROVENANCE_NOTICE,
        "boundary_selection": (
            "Invalid candidates only, at most 10 per category per split; nearest "
            "decision edge first, then sample_token. Distances retain stated units."
        ),
        "splits": {
            split: {
                "source_sha256": calibration.AUDITED_SOURCE_SHA256[split],
                **audit_split(records, parameters),
            }
            for split, records in sources.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print provisional factorized candidate audit JSON."
    )
    parser.add_argument(
        "--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"),
    )
    parser.add_argument(
        "--split", nargs="+", choices=("train", "validation"),
        default=["train", "validation"],
    )
    args = parser.parse_args(argv)
    if args.derived_root is None:
        parser.error("set VLA_DERIVED_ROOT or provide --derived-root")
    parameters = load_candidate_parameters(
        REPOSITORY_ROOT / "configs/phase0_4_candidate_rules.yaml",
    )
    print(json.dumps(
        audit_sources(args.derived_root, args.split, parameters),
        ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
