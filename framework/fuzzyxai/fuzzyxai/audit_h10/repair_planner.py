from __future__ import annotations

from .models import DiagnosticCutResult, FaultPrediction, RepairAction
from .route_validity import ValidityResult
from .structural_inference import infer_specs


class RepairSetPlanner:
    def plan(
        self,
        cut: DiagnosticCutResult,
        fault: FaultPrediction | None = None,
        validity: ValidityResult | None = None,
    ) -> tuple[RepairAction, ...]:
        active_fields = tuple(dict.fromkeys((validity.missing_fields + validity.mismatched_fields) if validity else cut.cut_nodes))
        specs = infer_specs(active_fields, fault)
        actions: list[RepairAction] = []
        covered: set[str] = set()
        for spec in specs:
            fields = tuple(field for field in active_fields if field in spec.fields and field not in covered)
            if not fields:
                continue
            for target in spec.source_nodes:
                actions.append(
                    RepairAction(
                        target=target,
                        action=spec.repair_action,
                        expected_effect=f"restore_contract_family:{spec.leaf}",
                        preconditions=(f"registered_component_available:{target}",),
                        affected_fields=fields,
                    )
                )
            covered.update(fields)
        for field in active_fields:
            if field not in covered:
                actions.append(
                    RepairAction(
                        target=field,
                        action=f"restore_registered_value:{field}",
                        expected_effect=f"restore_contract:{field}",
                        preconditions=(f"registered_value_available:{field}",),
                        affected_fields=(field,),
                    )
                )
        return tuple(actions)
