from __future__ import annotations

from decimal import Decimal

from .models import Case, Gold, MethodResult


def execute_plan(case: Case, result: MethodResult) -> dict[str, object]:
    by_id = {item.atom_id: item for item in case.candidates}
    completed: set[str] = set()
    covered: set[str] = set()
    failed = 0
    new_critical = 0
    for atom_id in result.plan:
        candidate = by_id.get(atom_id)
        if candidate is None:
            failed += 1
            continue
        if (
            not candidate.executable
            or any(dependency not in completed for dependency in candidate.dependencies)
            or any(conflict in completed for conflict in candidate.conflicts)
        ):
            failed += 1
            continue
        if candidate.provider_status == "unsafe":
            new_critical += 1
            failed += 1
            continue
        if candidate.provider_status != "healthy":
            failed += 1
            continue
        completed.add(atom_id)
        covered.update(candidate.covers)
    full = set(case.obligations).issubset(covered) and new_critical == 0
    return {
        "full_recertification_success": full,
        "partial_success": bool(covered) and not full,
        "failed_steps": failed,
        "new_critical_violations": new_critical,
        "plan_cost": sum(
            by_id[item].action_cost
            + by_id[item].human_approval_cost
            + by_id[item].fixed_cost
            for item in completed
        ),
        "completed_steps": len(completed),
    }


def score(case: Case, gold: Gold, result: MethodResult) -> dict[str, object]:
    cut = tuple(sorted(result.cut))
    candidates = {item.atom_id: item for item in case.candidates}
    selected = [candidates[item] for item in cut if item in candidates]
    covered = {obligation for item in selected for obligation in item.covers}
    feasible = all(
        item.repairable and item.executable and item.provider_status == "healthy"
        for item in selected
    )
    membership = gold.status.startswith("CERTIFIED") and cut in gold.optimal_cuts
    if gold.optimal_cost is None:
        raw_regret = None
        regret = 0.0
    elif not feasible or not set(case.obligations).issubset(covered):
        raw_regret = None
        regret = 1.0 + len(set(case.obligations) - covered) / max(1, len(case.obligations))
    else:
        predicted = Decimal(str(result.predicted_cost))
        optimal = Decimal(str(gold.optimal_cost))
        raw = predicted - optimal
        raw_regret = float(raw)
        regret = float(raw / max(abs(optimal), Decimal("1e-24")))
    execution = execute_plan(case, result)
    return {
        "optimal_set_membership": membership,
        "raw_cost_regret": raw_regret,
        "normalized_cost_regret": regret,
        "obligation_coverage": len(covered) / max(1, len(case.obligations)),
        "extra_elements": max(0, len(cut) - min((len(item) for item in gold.optimal_cuts), default=0)),
        "false_certification": result.status == "diagnosed"
        and (not feasible or not set(case.obligations).issubset(covered)),
        **execution,
    }
