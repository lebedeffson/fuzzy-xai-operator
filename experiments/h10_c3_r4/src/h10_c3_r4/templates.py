from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .models import (
    ContractTemplate,
    EdgeTemplate,
    NodeTemplate,
    RepairCandidateTemplate,
    RouteTemplate,
)

PIPELINE_SCHEMAS = {
    "tabular-credit": {
        "modality": "tabular",
        "nodes": (
            ("input", "credit_record", ("schema", "provenance")),
            ("encoder", "categorical_encoder", ("version", "dictionary")),
            ("scaler", "robust_scaler", ("version", "feature_order")),
            ("model", "credit_classifier", ("checkpoint", "schema")),
            ("calibration", "probability_calibrator", ("version", "population")),
            ("explainer", "tabular_local_explainer", ("checkpoint", "feature_map")),
            ("dictionary", "feature_dictionary", ("version", "canonical_hash")),
            ("output", "credit_explanation", ("schema", "provenance")),
        ),
        "relations": (
            ("input", "encoder", "transforms"),
            ("encoder", "scaler", "transforms"),
            ("scaler", "model", "consumes"),
            ("model", "calibration", "calibrates"),
            ("model", "explainer", "explains"),
            ("dictionary", "explainer", "validates"),
            ("explainer", "output", "produces"),
        ),
    },
    "tabular-energy": {
        "modality": "tabular",
        "nodes": (
            ("input", "meter_readings", ("schema", "time_range")),
            ("aggregation", "temporal_aggregator", ("window", "timezone")),
            ("imputation", "missing_value_handler", ("policy", "version")),
            ("model", "energy_regressor", ("checkpoint", "feature_order")),
            ("interval", "interval_calibrator", ("coverage", "population")),
            ("explainer", "regression_explainer", ("checkpoint", "feature_map")),
            ("output", "energy_explanation", ("schema", "provenance")),
        ),
        "relations": (
            ("input", "aggregation", "aggregates"),
            ("aggregation", "imputation", "transforms"),
            ("imputation", "model", "consumes"),
            ("model", "interval", "calibrates"),
            ("model", "explainer", "explains"),
            ("interval", "output", "certifies"),
            ("explainer", "output", "produces"),
        ),
    },
    "text-news": {
        "modality": "text",
        "nodes": (
            ("input", "news_document", ("encoding", "language")),
            ("tokenizer", "subword_tokenizer", ("version", "normalization")),
            ("dictionary", "token_dictionary", ("version", "canonical_hash")),
            ("model", "news_classifier", ("checkpoint", "tokenizer_version")),
            ("calibration", "text_calibrator", ("version", "population")),
            ("explainer", "token_attribution", ("checkpoint", "tokenizer_version")),
            ("reduction", "token_to_feature_reducer", ("loss", "dictionary_version")),
            ("output", "news_explanation", ("schema", "provenance")),
        ),
        "relations": (
            ("input", "tokenizer", "transforms"),
            ("dictionary", "tokenizer", "validates"),
            ("tokenizer", "model", "consumes"),
            ("model", "calibration", "calibrates"),
            ("model", "explainer", "explains"),
            ("explainer", "reduction", "reduces"),
            ("dictionary", "reduction", "validates"),
            ("reduction", "output", "produces"),
        ),
    },
    "text-reviews": {
        "modality": "text",
        "nodes": (
            ("input", "review_document", ("encoding", "language")),
            ("segmenter", "sentence_segmenter", ("version", "policy")),
            ("normalizer", "review_normalizer", ("version", "locale")),
            ("dictionary", "sentiment_dictionary", ("version", "canonical_hash")),
            ("model", "review_classifier", ("checkpoint", "segment_policy")),
            ("explainer", "fragment_attribution", ("checkpoint", "aggregation")),
            ("aggregation", "fragment_aggregator", ("policy", "weights")),
            ("output", "review_explanation", ("schema", "provenance")),
        ),
        "relations": (
            ("input", "segmenter", "transforms"),
            ("segmenter", "normalizer", "transforms"),
            ("normalizer", "model", "consumes"),
            ("dictionary", "model", "validates"),
            ("model", "explainer", "explains"),
            ("explainer", "aggregation", "aggregates"),
            ("aggregation", "output", "produces"),
        ),
    },
    "image-quality": {
        "modality": "image",
        "nodes": (
            ("input", "encoded_image", ("codec", "checksum")),
            ("decoder", "image_decoder", ("version", "color_space")),
            ("resize", "geometric_resize", ("shape", "interpolation")),
            ("normalizer", "pixel_normalizer", ("mean", "std")),
            ("model", "vision_classifier", ("checkpoint", "input_shape")),
            ("explainer", "spatial_attribution", ("checkpoint", "feature_layer")),
            ("coordinates", "coordinate_transform", ("source_shape", "target_shape")),
            ("output", "image_explanation", ("schema", "geometry")),
        ),
        "relations": (
            ("input", "decoder", "transforms"),
            ("decoder", "resize", "transforms"),
            ("resize", "normalizer", "transforms"),
            ("normalizer", "model", "consumes"),
            ("model", "explainer", "explains"),
            ("resize", "coordinates", "validates"),
            ("explainer", "coordinates", "transforms"),
            ("coordinates", "output", "produces"),
        ),
    },
    "timeseries-sensors": {
        "modality": "time_series",
        "nodes": (
            ("input", "sensor_stream", ("sampling_rate", "clock")),
            ("resampler", "time_resampler", ("rate", "interpolation")),
            ("window", "window_builder", ("length", "stride")),
            ("normalizer", "series_normalizer", ("mean", "std")),
            ("model", "sequence_classifier", ("checkpoint", "window_length")),
            ("explainer", "temporal_attribution", ("checkpoint", "time_axis")),
            ("timeline", "timeline_restorer", ("rate", "origin")),
            ("output", "timeseries_explanation", ("schema", "time_axis")),
        ),
        "relations": (
            ("input", "resampler", "transforms"),
            ("resampler", "window", "aggregates"),
            ("window", "normalizer", "transforms"),
            ("normalizer", "model", "consumes"),
            ("model", "explainer", "explains"),
            ("explainer", "timeline", "transforms"),
            ("resampler", "timeline", "validates"),
            ("timeline", "output", "produces"),
        ),
    },
}

STRATUM_COUNTS = {"S2": 20, "S3": 30, "S4": 30, "S5": 30}
BANK_SEEDS = {"development": 410_000, "protocol_validation": 520_000, "sealed": 630_000}


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _wl_labels(
    nodes: tuple[NodeTemplate, ...],
    edges: tuple[EdgeTemplate, ...],
    contracts: tuple[ContractTemplate, ...],
) -> dict[str, str]:
    contract_colors: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for contract in contracts:
        contract_colors[contract.subject_role].append(
            (contract.kind, contract.field, contract.category)
        )
    labels = {
        node.role: _digest(
            (
                node.node_type,
                tuple(sorted(node.attributes)),
                tuple(sorted(contract_colors[node.role])),
            )
        )
        for node in nodes
    }
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source_role].append((edge.relation, edge.target_role))
        incoming[edge.target_role].append((edge.relation, edge.source_role))
    for _ in range(max(1, len(nodes))):
        updated = {
            role: _digest(
                (
                    labels[role],
                    tuple(
                        sorted(
                            (relation, labels[target])
                            for relation, target in outgoing[role]
                        )
                    ),
                    tuple(
                        sorted(
                            (relation, labels[source])
                            for relation, source in incoming[role]
                        )
                    ),
                )
            )
            for role in labels
        }
        if Counter(updated.values()) == Counter(labels.values()):
            labels = updated
            break
        labels = updated
    return labels


def _wl_graph_hash(
    nodes: tuple[NodeTemplate, ...],
    edges: tuple[EdgeTemplate, ...],
    contracts: tuple[ContractTemplate, ...],
) -> str:
    labels = _wl_labels(nodes, edges, contracts)
    edge_colors = tuple(
        sorted(
            (
                labels[edge.source_role],
                edge.relation,
                labels[edge.target_role],
                edge.mandatory,
            )
            for edge in edges
        )
    )
    return _digest((tuple(sorted(labels.values())), edge_colors))


def canonicalize_template(template: RouteTemplate) -> RouteTemplate:
    labels = _wl_labels(
        template.node_schema,
        template.edge_schema,
        template.contract_schema,
    )
    graph_hash = _wl_graph_hash(
        template.node_schema,
        template.edge_schema,
        template.contract_schema,
    )
    edges = {
        f"edge::{edge.source_role}::{edge.target_role}": _digest(
            (
                labels[edge.source_role],
                edge.relation,
                labels[edge.target_role],
                edge.mandatory,
            )
        )
        for edge in template.edge_schema
    }

    def subject_label(role: str) -> str:
        return edges[role] if role.startswith("edge::") else labels[role]

    obligation_signatures = {
        contract.contract_id: _digest(
            (
                subject_label(contract.subject_role),
                contract.kind,
                contract.field,
                contract.category,
                contract.repairable,
                tuple(
                    sorted(labels[role] for role in contract.source_roles)
                ),
            )
        )
        for contract in template.contract_schema
    }
    coverage_hash = _digest(
        sorted(
            (
                labels[candidate.source_role],
                tuple(
                    sorted(
                        obligation_signatures[obligation]
                        for obligation in candidate.covers
                    )
                ),
            )
            for candidate in template.candidates
        )
    )
    mutation_hash = _digest(
        (
            "replace_registered_observation",
            tuple(
                sorted(
                    (
                        obligation_signatures[contract.contract_id],
                        subject_label(contract.subject_role),
                    )
                    for contract in template.contract_schema
                )
            ),
        )
    )
    repair_dependency_hash = _digest(
        (
            "restore_registered_observation",
            sorted(
                (
                    labels[candidate.source_role],
                    tuple(
                        sorted(
                            labels[
                                next(
                                    item.source_role
                                    for item in template.candidates
                                    if item.candidate_id == dependency
                                    or item.source_role == dependency
                                )
                            ]
                            for dependency in candidate.dependencies
                        )
                    ),
                )
                for candidate in template.candidates
            ),
        )
    )
    cost_hash = _digest(
        sorted(
            (
                labels[candidate.source_role],
                candidate.cost,
                candidate.executable,
            )
            for candidate in template.candidates
        )
    )
    canonical_hash = _digest(
        (
            graph_hash,
            coverage_hash,
            mutation_hash,
            repair_dependency_hash,
            cost_hash,
        )
    )
    return replace(
        template,
        graph_hash=graph_hash,
        coverage_hash=coverage_hash,
        mutation_hash=mutation_hash,
        repair_dependency_hash=repair_dependency_hash,
        cost_hash=cost_hash,
        canonical_hash=canonical_hash,
    )


def _candidate_pattern(
    stratum: str,
    obligations: tuple[str, ...],
    variant: int,
) -> tuple[RepairCandidateTemplate, ...]:
    jitter = (variant % 7) * 0.013
    if stratum == "S2":
        return tuple(
            RepairCandidateTemplate(
                f"source-{index}",
                f"repair-{index}",
                (obligation,),
                1.0 + jitter + index * 0.07,
            )
            for index, obligation in enumerate(obligations)
        )
    base = (
        RepairCandidateTemplate(
            "greedy-023",
            "repair-greedy",
            (obligations[0], obligations[2], obligations[3]),
            2.5 + jitter,
        ),
        RepairCandidateTemplate(
            "single-1",
            "repair-single",
            (obligations[1],),
            2.0 + jitter,
        ),
        RepairCandidateTemplate(
            "pair-01",
            "repair-pair-a",
            (obligations[0], obligations[1]),
            2.0 + jitter,
            dependencies=("repair-pair-b",),
        ),
        RepairCandidateTemplate(
            "pair-23",
            "repair-pair-b",
            (obligations[2], obligations[3]),
            2.0 + jitter,
        ),
    )
    if stratum == "S3":
        return base
    if stratum == "S4":
        return (
            *base,
            RepairCandidateTemplate(
                "alternative-02",
                "repair-alt-a",
                (obligations[0], obligations[2]),
                2.0 + jitter,
                dependencies=("repair-alt-b",),
            ),
            RepairCandidateTemplate(
                "alternative-13",
                "repair-alt-b",
                (obligations[1], obligations[3]),
                2.0 + jitter,
            ),
        )
    return (
        *base,
        RepairCandidateTemplate(
            "fallback-last",
            "repair-fallback",
            (obligations[-1],),
            1.8 + jitter,
        ),
        RepairCandidateTemplate(
            "unknown-probe",
            "repair-unknown",
            (obligations[-1],),
            0.4 + jitter,
            executable=False,
        ),
    )


def _build_template(
    split: str,
    pipeline: str,
    stratum: str,
    index: int,
    rng: random.Random,
) -> RouteTemplate:
    schema = PIPELINE_SCHEMAS[pipeline]
    variant = index + {
        "development": 0,
        "protocol_validation": 10_000,
        "sealed": 20_000,
    }[split]
    base_nodes = tuple(
        NodeTemplate(role, node_type, tuple(attributes))
        for role, node_type, attributes in schema["nodes"]
    )
    candidates = _candidate_pattern(
        stratum,
        tuple(f"obligation-{number}" for number in range(4 if stratum != "S5" else 5)),
        variant,
    )
    split_cost_offset = {
        "development": 0.0,
        "protocol_validation": 0.0031,
        "sealed": 0.0062,
    }[split]
    pipeline_cost_offset = tuple(PIPELINE_SCHEMAS).index(pipeline) * 0.0001
    candidates = tuple(
        replace(
            candidate,
            cost=candidate.cost + split_cost_offset + pipeline_cost_offset,
        )
        for candidate in candidates
    )
    repair_nodes = tuple(
        NodeTemplate(
            candidate.source_role,
            f"{pipeline}-repair-provider-{variant % 37}-{position}",
            ("provider_version", "availability"),
        )
        for position, candidate in enumerate(candidates)
    )
    auxiliary_count = 2 + (variant % 7)
    auxiliary = tuple(
        NodeTemplate(
            f"aux-{number}",
            f"{pipeline}-audit-stage-{(variant * 7 + number) % 97}",
            ("trace_schema", f"slot_{(variant + number) % 43}"),
        )
        for number in range(auxiliary_count)
    )
    nodes = (*base_nodes, *repair_nodes, *auxiliary)
    edges = [
        EdgeTemplate(source, target, relation)
        for source, target, relation in schema["relations"]
    ]
    roles = [node.role for node in base_nodes]
    for number, node in enumerate(auxiliary):
        source = "input"
        target = "output"
        edges.append(
            EdgeTemplate(
                source,
                node.role,
                ("validates", "derived_from", "certifies")[(variant + number) % 3],
            )
        )
        edges.append(
            EdgeTemplate(
                node.role,
                target,
                ("produces", "transforms", "aggregates")[(variant * 2 + number) % 3],
            )
        )
    for number, candidate in enumerate(candidates):
        target = roles[(variant + number * 2) % len(roles)]
        edges.append(
            EdgeTemplate(
                candidate.source_role,
                target,
                "validates",
            )
        )
    obligation_count = 4 if stratum != "S5" else 5
    contracts = []
    for number in range(obligation_count):
        target = roles[(variant * 3 + number * 2) % len(roles)]
        if stratum == "S5" and number == obligation_count - 1:
            edge_source, edge_target, _ = schema["relations"][
                variant % len(schema["relations"])
            ]
            target = f"edge::{edge_source}::{edge_target}"
        source_roles = tuple(
            candidate.source_role
            for candidate in candidates
            if f"obligation-{number}" in candidate.covers
            and candidate.executable
        )
        contracts.append(
            ContractTemplate(
                contract_id=f"obligation-{number}",
                subject_role=target,
                kind="equals",
                field=f"registered_state_{number}",
                expected=f"valid-{pipeline}-{stratum}-{index}-{number}",
                category=(
                    "provenance",
                    "preprocessing",
                    "model",
                    "representation",
                    "calibration",
                )[number % 5],
                source_roles=source_roles,
                repairable=not (stratum == "S5" and index % 17 == 0 and number == 4),
            )
        )
    raw = RouteTemplate(
        template_id=f"{split}:{pipeline}:{stratum}:{index:03d}",
        split=split,
        pipeline_family=pipeline,
        modality=str(schema["modality"]),
        stratum=stratum,
        node_schema=nodes,
        edge_schema=tuple(edges),
        contract_schema=tuple(contracts),
        candidates=candidates,
        mutation_grammar_id=(
            f"{split}:mutation:"
            f"{rng.randrange(10_000_000):07d}:{stratum}:{index % 19}"
        ),
        repair_grammar_id=(
            f"{split}:repair:"
            f"{rng.randrange(10_000_000):07d}:{stratum}:{index % 23}"
        ),
        graph_hash="",
        coverage_hash="",
        mutation_hash="",
        repair_dependency_hash="",
        cost_hash="",
        canonical_hash="",
    )
    return canonicalize_template(raw)


def build_template_bank(split: str) -> tuple[RouteTemplate, ...]:
    if split not in BANK_SEEDS:
        raise ValueError(f"unknown R4 template bank: {split}")
    rng = random.Random(BANK_SEEDS[split])
    templates = []
    seen: set[str] = set()
    for pipeline in PIPELINE_SCHEMAS:
        for stratum, count in STRATUM_COUNTS.items():
            for local_index in range(count):
                attempt = local_index
                while True:
                    template = _build_template(
                        split,
                        pipeline,
                        stratum,
                        attempt,
                        rng,
                    )
                    if template.canonical_hash not in seen:
                        break
                    attempt += sum(STRATUM_COUNTS.values())
                templates.append(template)
                seen.add(template.canonical_hash)
    return tuple(templates)


def audit_banks(
    banks: dict[str, tuple[RouteTemplate, ...]],
) -> dict[str, object]:
    fingerprints = {
        split: {
            "canonical_hashes": {item.canonical_hash for item in templates},
            "graph_hashes": {item.graph_hash for item in templates},
            "mutation_hashes": {item.mutation_hash for item in templates},
            "coverage_cost_hashes": {
                _digest((item.coverage_hash, item.cost_hash)) for item in templates
            },
            "repair_dependency_hashes": {
                item.repair_dependency_hash for item in templates
            },
        }
        for split, templates in banks.items()
    }
    intersections = {}
    split_names = sorted(banks)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            for field in fingerprints[left]:
                overlap = fingerprints[left][field] & fingerprints[right][field]
                intersections[f"{left}:{right}:{field}"] = sorted(overlap)
    pipeline_hashes = {
        pipeline: {
            item.graph_hash
            for templates in banks.values()
            for item in templates
            if item.pipeline_family == pipeline
        }
        for pipeline in PIPELINE_SCHEMAS
    }
    pipeline_pairs_distinct = all(
        pipeline_hashes[left].isdisjoint(pipeline_hashes[right])
        for left_index, left in enumerate(pipeline_hashes)
        for right in tuple(pipeline_hashes)[left_index + 1 :]
    )
    return {
        "bank_sizes": {split: len(templates) for split, templates in banks.items()},
        "unique_templates": {
            split: len(fingerprints[split]["canonical_hashes"])
            for split in banks
        },
        "intersections": intersections,
        "pipeline_graph_schemas_distinct": pipeline_pairs_distinct,
        "status": "PASS"
        if all(not overlap for overlap in intersections.values())
        and pipeline_pairs_distinct
        else "FAIL",
    }


def write_template_bank(path: Path, templates: tuple[RouteTemplate, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(template.to_dict(), sort_keys=True) + "\n"
            for template in templates
        ),
        encoding="utf-8",
    )
