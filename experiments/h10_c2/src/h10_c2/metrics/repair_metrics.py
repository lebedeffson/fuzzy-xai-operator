from __future__ import annotations


def score_repair(execution: dict, predicted_cost: float, optimal_cost: float | None) -> dict:
    return {
        "full_recertification_success": bool(execution["full_recertification_success"]),
        "partial_recovery": bool(execution["partial_recovery"]),
        "new_critical_issues": int(execution["new_critical_issues"]),
        "human_actions": int(execution["human_actions"]),
        "plan_cost_regret": predicted_cost - optimal_cost if optimal_cost is not None else float("nan"),
        "recertification_ms": float(execution["runtime_ms"]),
    }

