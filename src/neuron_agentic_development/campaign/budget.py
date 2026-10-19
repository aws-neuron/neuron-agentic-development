"""Workload deadline accounting; execution remains owned by the native parent."""

import math
from datetime import datetime, timezone

from .contracts import ContractError, PolicyError


def seconds(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0:
        raise ContractError(f"{name} must be a finite non-negative number")
    return value


def deadline(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractError("budget.deadline_at must be an ISO timestamp with timezone") from error
    if parsed.utcoffset() is None:
        raise ContractError("budget.deadline_at requires a timezone")
    return parsed


def validate_budget(limits: dict) -> None:
    if "deadline_at" in limits:
        deadline(limits["deadline_at"])
        if "closeout_reserve_seconds" not in limits:
            raise ContractError("a deadline requires closeout_reserve_seconds")
    if "closeout_reserve_seconds" in limits:
        seconds(limits["closeout_reserve_seconds"], "closeout_reserve_seconds")
        if "deadline_at" not in limits:
            raise ContractError("closeout_reserve_seconds requires deadline_at")


def remaining_seconds(limits: dict, now=None):
    if "deadline_at" not in limits:
        return None
    return max(0, (deadline(limits["deadline_at"]) -
                   (now or datetime.now(timezone.utc))).total_seconds()
               - limits["closeout_reserve_seconds"])


def validate_chain(limits: dict, inputs: dict, now=None) -> None:
    estimate = inputs.get("estimated_chain_seconds")
    remaining = remaining_seconds(limits, now)
    if estimate is None:
        if remaining is not None:
            raise ContractError(
                "estimated_chain_seconds must include planning, compilation, benchmark, "
                "required correctness and submission; closeout is reserved separately"
            )
        return
    seconds(estimate, "estimated_chain_seconds")
    if estimate <= 0:
        raise ContractError("estimated_chain_seconds must be positive")
    if remaining is not None and estimate > remaining:
        raise PolicyError(
            f"estimated chain {estimate}s exceeds {remaining:.1f}s before closeout reserve; "
            "choose a cheaper experiment or pause for a resumable handoff"
        )
