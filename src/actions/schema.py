from typing import Final


KEEP: Final = "keep"
ACCELERATE: Final = "accelerate"
DECELERATE: Final = "decelerate"
STOP: Final = "stop"
LEFT_LATERAL: Final = "left_lateral"
RIGHT_LATERAL: Final = "right_lateral"
LABEL_RULE_VERSION: Final = "phase-1.6-meta-action-v0.2"

ACTION_SCHEMA: Final = (
    KEEP,
    ACCELERATE,
    DECELERATE,
    STOP,
    LEFT_LATERAL,
    RIGHT_LATERAL,
)
ACTION_SET: Final = frozenset(ACTION_SCHEMA)


def is_valid_action(action: str) -> bool:
    return action in ACTION_SET


def normalize_action(action: str) -> str:
    normalized = action.strip().lower()
    if not is_valid_action(normalized):
        raise ValueError(f"Unsupported action: {action!r}")
    return normalized
