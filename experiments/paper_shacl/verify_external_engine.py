"""Cross-check manuscript RDF constraints with the independent pySHACL engine.

Data are eight *published aggregate cost rows*, not real model executions.
Tests inject two known schema faults per row, exactly as in manuscript's
limited rdflib validator. pySHACL is third-party but datasets and fault
injections are author-selected: NOT independent evidence of natural errors.
"""

from __future__ import annotations

import hashlib
import json
import platform
from decimal import Decimal
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF
from rdflib.namespace import SH, XSD

EX = Namespace("https://example.org/fuzzyxai/review-registry#")
PROFILE = Path(__file__).with_name("profile.ttl")

ARCHIVED_AGGREGATES = (
    ("births", "logreg", "0.1338", "0.1338"),
    ("births", "tree", "0.2575", "0.1550"),
    ("births", "forest", "0.1138", "0.1138"),
    ("births", "sgd", "0.1600", "0.1263"),
    ("elnino", "logreg", "0.1192", "0.1238"),
    ("elnino", "tree", "0.3131", "0.3131"),
    ("elnino", "forest", "0.2430", "0.2430"),
    ("elnino", "sgd", "0.4276", "0.4276"),
)


def construct(index: int, scenario: str) -> Graph:
    dataset, estimator, baseline, candidate = ARCHIVED_AGGREGATES[index]
    graph = Graph()
    record = EX[f"record{index}"]
    graph.add((record, RDF.type, EX.PolicyComparison))
    graph.add((record, EX.dataset, EX[f"dataset_{dataset}"]))
    graph.add((record, EX.estimator, EX[f"model_{estimator}"]))
    if scenario != "missing_source":
        graph.add((record, EX.sourceRecord, EX[f"archived_table_1_row_{index + 1}"]))
    graph.add((record, EX.baselineCost, Literal(Decimal(baseline), datatype=XSD.decimal)))
    graph.add((
        record, EX.candidateCost,
        Literal(candidate, datatype=XSD.string)
        if scenario == "invalid_cost_type"
        else Literal(Decimal(candidate), datatype=XSD.decimal),
    ))
    return graph


def run() -> dict:
    profile_bytes = PROFILE.read_bytes()
    shapes = Graph().parse(data=profile_bytes.decode("utf-8"), format="turtle")
    rows = []
    for scenario in ("valid", "missing_source", "invalid_cost_type"):
        for index in range(len(ARCHIVED_AGGREGATES)):
            conforms, report, _ = validate(
                construct(index, scenario),
                shacl_graph=shapes,
                inference="none",
                advanced=False,
                abort_on_first=False,
                allow_infos=False,
                allow_warnings=False,
                meta_shacl=False,
            )
            components = sorted(
                {
                    str(component).rsplit("#", 1)[-1]
                    for result in report.subjects(RDF.type, SH.ValidationResult)
                    for component in report.objects(result, SH.sourceConstraintComponent)
                }
            )
            expected_components = {
                "valid": [],
                "missing_source": ["MinCountConstraintComponent"],
                "invalid_cost_type": ["DatatypeConstraintComponent"],
            }[scenario]
            if components != expected_components:
                raise AssertionError(
                    f"Case {scenario} source row {index + 1}: {components} != {expected_components}"
                )
            expected_conforms = scenario == "valid"
            if bool(conforms) != expected_conforms:
                raise AssertionError(
                    f"Incorrect pySHACL conformity: {scenario} row={index + 1}"
                )
            rows.append({
                "source_row": index + 1,
                "scenario": scenario,
                "expected_problem": not expected_conforms,
                "detected_problem": not bool(conforms),
                "violations": components,
            })

    return {
        "schema": "manuscript_archive_independent_pyshacl_crosscheck_v1",
        "shape_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "source": "8 archived policy-cost aggregate rows; no live ML evaluation",
        "validator": "pyshacl",
        "n_source_rows": len(ARCHIVED_AGGREGATES),
        "n_cases": len(rows),
        "valid_accepted": sum(not r["detected_problem"] for r in rows),
        "known_faults_detected": sum(r["detected_problem"] for r in rows),
        "missed_known_faults": sum(r["expected_problem"] and not r["detected_problem"] for r in rows),
        "environment": {"python": platform.python_version()},
        "limits": [
            "Independent third-party SHACL engine, but the test set and expected results are developer chosen",
            "Only two intentionally introduced structural schema faults, not naturally occurring anomalies",
            "No authentication of RDF sourceRecord URI or primary experimental measurements",
            "No claim about regression costs, model quality, correct policy choices or ontology completeness",
        ],
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True, indent=2))
