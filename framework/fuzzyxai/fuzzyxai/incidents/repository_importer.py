from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

FORBIDDEN_GOLD_FIELDS = frozenset(
    {
        "patch",
        "fix_commit",
        "changed_files",
        "maintainer_post_fix_explanation",
    }
)


@dataclass(frozen=True)
class IncidentInput:
    repository_id: str
    incident_id: str
    buggy_commit: str
    issue_description: str
    failing_tests: tuple[str, ...]
    environment_setup_commit: str
    version: str

    @classmethod
    def from_public_mapping(cls, row: dict[str, object]) -> IncidentInput:
        leaked = FORBIDDEN_GOLD_FIELDS.intersection(row)
        if leaked:
            raise ValueError(f"gold fields are forbidden in method input: {sorted(leaked)}")
        return cls(
            repository_id=str(row["repo"]),
            incident_id=str(row["instance_id"]),
            buggy_commit=str(row["base_commit"]),
            issue_description=str(row["problem_statement"]),
            failing_tests=tuple(str(item) for item in row.get("FAIL_TO_PASS", ())),
            environment_setup_commit=str(row.get("environment_setup_commit", "")),
            version=str(row.get("version", "")),
        )


@dataclass(frozen=True)
class RouteComponent:
    component_id: str
    component_type: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class IncidentRoute:
    repository_id: str
    incident_id: str
    buggy_commit: str
    components: tuple[RouteComponent, ...]
    relations: tuple[tuple[str, str, str], ...]
    observable_text: str
    environment_hash: str
    failing_test_hash: str
    route_graph_hash: str
    importer_version: str = "h10-c5-importer-v1"


class RepositoryImporter:
    """Build a formal route only from pre-fix observable evidence."""

    component_types = (
        "repository",
        "environment",
        "dependencies",
        "data_schema",
        "preprocessing",
        "model",
        "explainer",
        "artifact",
        "test",
        "runtime",
        "configuration",
    )

    def import_incident(self, incident: IncidentInput) -> IncidentRoute:
        components = tuple(
            RouteComponent(
                component_id=f"{incident.incident_id}:{kind}",
                component_type=kind,
                evidence_refs=(
                    f"issue:{incident.incident_id}",
                    f"buggy_commit:{incident.buggy_commit}",
                ),
            )
            for kind in self.component_types
        )
        relations = (
            ("repository", "environment", "produces"),
            ("environment", "dependencies", "loads"),
            ("dependencies", "preprocessing", "depends_on"),
            ("data_schema", "preprocessing", "transforms"),
            ("preprocessing", "model", "produces"),
            ("model", "explainer", "explains"),
            ("explainer", "artifact", "serializes"),
            ("artifact", "test", "verifies"),
            ("configuration", "runtime", "loads"),
            ("runtime", "test", "produces"),
        )
        observable = "\n".join(
            (
                incident.issue_description,
                *incident.failing_tests,
                incident.version,
            )
        )
        environment_hash = hashlib.sha256(
            f"{incident.repository_id}:{incident.environment_setup_commit}:{incident.version}".encode()
        ).hexdigest()
        failing_test_hash = hashlib.sha256(
            json.dumps(incident.failing_tests, sort_keys=True).encode()
        ).hexdigest()
        canonical = {
            "repository_id": incident.repository_id,
            "incident_id": incident.incident_id,
            "buggy_commit": incident.buggy_commit,
            "components": [item.component_type for item in components],
            "relations": relations,
            "environment_hash": environment_hash,
            "failing_test_hash": failing_test_hash,
        }
        route_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return IncidentRoute(
            repository_id=incident.repository_id,
            incident_id=incident.incident_id,
            buggy_commit=incident.buggy_commit,
            components=components,
            relations=relations,
            observable_text=observable,
            environment_hash=environment_hash,
            failing_test_hash=failing_test_hash,
            route_graph_hash=route_hash,
        )
