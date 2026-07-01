from __future__ import annotations


def diagnose_route(risk: dict) -> list[dict]:
    if int(risk.get("chi_crit", 0)) == 1:
        return [
            {
                "diagnostic_id": "D_quality_source_conflict",
                "diagnostic_type": "quality_source_conflict",
                "source": "image_and_segmentation_quality",
                "criticality": "high",
                "message_ru": "конфликт качества источника и модельного сигнала",
                "recommended_action": "block",
            }
        ]
    return []
