from __future__ import annotations

from fuzzyxai.adapters.contracts_v2 import ExplanationPlanDecision, ModelCapabilities


class ExplanationPlanner:
    """Select only available channels whose provenance and quality are disclosed."""

    FIDELITY_MIN_CLASSIFICATION = 0.90
    FIDELITY_MIN_REGRESSION = 0.80

    def plan(
        self,
        capabilities: ModelCapabilities,
        *,
        requested: tuple[str, ...] | str = "auto",
        budget: str = "standard",
        regression: bool = False,
    ) -> ExplanationPlanDecision:
        available = {item.name: item for item in capabilities.channels if item.available}
        candidates = list(available) if requested == "auto" else list(requested)
        selected: list[str] = []
        skipped: dict[str, str] = {}
        limitations: list[str] = []
        fidelity_threshold = self.FIDELITY_MIN_REGRESSION if regression else self.FIDELITY_MIN_CLASSIFICATION
        for channel in candidates:
            descriptor = available.get(channel)
            if descriptor is None:
                skipped[channel] = "channel unavailable"
                continue
            if descriptor.origin == "surrogate":
                if descriptor.fidelity_status != "measured" or descriptor.fidelity is None:
                    skipped[channel] = "surrogate fidelity not measured"
                    continue
                if descriptor.fidelity < fidelity_threshold:
                    skipped[channel] = f"surrogate fidelity {descriptor.fidelity:.3f} below {fidelity_threshold:.2f}"
                    continue
            selected.append(channel)
            limitations.extend(descriptor.limitations)
        if budget == "minimal":
            priority = ("prediction", "predict_proba", "decision_function", "local_contributions", "native_rules")
            selected = [item for item in priority if item in selected][:3]
        elif budget not in {"standard", "full"}:
            raise ValueError("budget must be minimal, standard, or full")
        checks = ["trace_completeness", "stability"]
        if any(available[item].origin == "surrogate" for item in selected):
            checks.append("surrogate_fidelity")
        if "local_contributions" in selected:
            checks.append("contribution_reconstruction")
        return ExplanationPlanDecision(
            selected_channels=tuple(selected),
            skipped_channels=skipped,
            required_quality_checks=tuple(dict.fromkeys(checks)),
            limitations=tuple(dict.fromkeys(limitations)),
        )
