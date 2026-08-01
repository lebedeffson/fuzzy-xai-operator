from __future__ import annotations

from dataclasses import asdict

from fuzzyxai.pipelines.practical import MUTATION_FAMILIES, CrossPipelineService
from fuzzyxai.pipelines.registry import PIPELINE_REGISTRY
from nicegui import ui

SERVICE = CrossPipelineService()


def ui_projection(result: dict[str, object]) -> dict[str, object]:
    """Project API-owned fields without deriving a second UI diagnosis."""
    return {
        "pipeline_status": result["pipeline_status"],
        "stage": result["stage"],
        "contract_id": result["contract_id"],
        "root_cause": result["root_cause"],
        "dependent_violations": result["dependent_violations"],
        "evidence_refs": result["evidence_refs"],
        "action": result["action"],
        "repair_plan": result["repair_plan"],
        "repair_executed": result["repair_executed"],
        "recertified": result["recertified"],
        "new_critical_violations": result["new_critical_violations"],
    }


@ui.page("/")
def main_page() -> None:
    ui.label("FuzzyXAI Cross-Pipeline Control").classes("text-h4 font-bold")
    ui.label("Registered controlled pipelines, measured evidence, repair and full recertification.")
    pipeline = ui.select(tuple(PIPELINE_REGISTRY), value=next(iter(PIPELINE_REGISTRY)), label="Pipeline").classes("w-full")
    mutation = ui.select(tuple(MUTATION_FAMILIES), value="FEATURE_SCHEMA_CASCADE", label="Registered mutation").classes("w-full")
    level = ui.select(("L0", "L1", "L2", "L3", "L4"), value="L1", label="Level").classes("w-full")
    output = ui.json_editor({"content": {"json": {"status": "NOT_RUN"}}}).classes("w-full")

    def run() -> None:
        result = SERVICE.mutate(str(pipeline.value), str(mutation.value), str(level.value))
        output.properties["content"] = {"json": ui_projection(asdict(result))}
        output.update()

    ui.button("Run controlled route", on_click=run).props("color=primary")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="FuzzyXAI Cross-Pipeline", reload=False)
