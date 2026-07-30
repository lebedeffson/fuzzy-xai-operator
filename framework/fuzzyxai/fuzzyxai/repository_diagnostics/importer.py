from __future__ import annotations

import ast
import configparser
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .graph import EvidenceRef, RepositoryEdge, RepositoryGraph, RepositoryNode

FORBIDDEN_GOLD_FIELDS = frozenset(
    {
        "patch",
        "fix_commit",
        "changed_files",
        "changed_symbols",
        "gold_contracts",
        "gold_repair_atoms",
    }
)
MANIFEST_NAMES = frozenset(
    {
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "requirements.txt",
        "requirements-dev.txt",
    }
)
CONFIG_SUFFIXES = frozenset({".toml", ".ini", ".cfg", ".yaml", ".yml", ".json"})


@dataclass(frozen=True)
class RepositoryIncident:
    incident_id: str
    repository: str
    buggy_revision: str
    repository_root: Path
    failing_tests: tuple[str, ...]
    traceback: str = ""
    stdout: str = ""
    stderr: str = ""
    assertion_difference: str = ""

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> RepositoryIncident:
        leaked = FORBIDDEN_GOLD_FIELDS.intersection(payload)
        if leaked:
            raise ValueError(f"gold fields are forbidden: {sorted(leaked)}")
        return cls(
            incident_id=str(payload["incident_id"]),
            repository=str(payload["repository"]),
            buggy_revision=str(payload["buggy_revision"]),
            repository_root=Path(str(payload["repository_root"])).resolve(),
            failing_tests=tuple(str(item) for item in payload.get("failing_tests", ())),
            traceback=str(payload.get("traceback", "")),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            assertion_difference=str(payload.get("assertion_difference", "")),
        )


def _node_id(kind: str, path: str, symbol: str | None = None) -> str:
    suffix = f"::{symbol}" if symbol else ""
    return f"{kind}:{path}{suffix}"


def _evidence_id(kind: str, source: str, detail: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{source}\0{detail}".encode()).hexdigest()[:16]
    return f"evidence:{kind}:{digest}"


class _PythonIndex(ast.NodeVisitor):
    def __init__(self, repository: str, relative: str) -> None:
        self.repository = repository
        self.relative = relative
        self.nodes: list[RepositoryNode] = []
        self.edges: list[tuple[str, str, str, int]] = []
        self.scope: list[str] = []
        self.symbol_ids: dict[str, str] = {}
        self.calls: list[tuple[str, str, int]] = []
        self.imports: list[tuple[str, str | None, int]] = []
        self.artifact_accesses: list[tuple[str, str, str, int, str]] = []
        self.file_id = _node_id("file", relative)

    def _symbol(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def _symbol_id(self) -> str:
        return self.symbol_ids.get(self._symbol(), self.file_id)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend((alias.name, None, node.lineno) for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        self.imports.extend(
            (module, alias.name, node.lineno)
            for alias in node.names
        )

    def _visit_definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, kind: str) -> None:
        self.scope.append(node.name)
        symbol = self._symbol()
        identifier = _node_id(kind, self.relative, symbol)
        self.symbol_ids[symbol] = identifier
        semantic_tokens = tuple(
            sorted(
                {
                    item.id.lower()
                    for item in ast.walk(node)
                    if isinstance(item, ast.Name)
                }
                | {
                    item.attr.lower()
                    for item in ast.walk(node)
                    if isinstance(item, ast.Attribute)
                }
            )
        )[:200]
        self.nodes.append(
            RepositoryNode(
                identifier,
                kind,
                self.repository,
                self.relative,
                symbol,
                {
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "parameters": tuple(
                        argument.arg
                        for argument in getattr(getattr(node, "args", None), "args", ())
                    ),
                    "return_annotation": (
                        ast.unparse(node.returns)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.returns is not None
                        else None
                    ),
                    "semantic_tokens": semantic_tokens,
                    "decorators": tuple(
                        ast.unparse(item) for item in getattr(node, "decorator_list", ())
                    ),
                },
            )
        )
        self.edges.append((self.file_id, identifier, "contains", node.lineno))
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "fixture" if any("fixture" in ast.unparse(item) for item in node.decorator_list) else (
            "test" if node.name.startswith("test_") else "function"
        )
        self._visit_definition(node, kind)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "test" if node.name.startswith("test_") else "function"
        self._visit_definition(node, kind)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_definition(node, "class")

    def visit_Call(self, node: ast.Call) -> None:
        try:
            called = ast.unparse(node.func)
        except ValueError:
            called = "<dynamic>"
        self.calls.append((self._symbol_id(), called, node.lineno))
        short = called.rsplit(".", 1)[-1].lower()
        relation = ""
        kind = "serialized_artifact"
        if short in {"load", "loads", "read", "read_bytes", "read_text", "open_dataset"}:
            relation = "loads" if short in {"load", "loads", "open_dataset"} else "reads"
        elif short in {"dump", "dumps", "write", "write_bytes", "write_text", "save"}:
            relation = "serializes" if short in {"dump", "dumps", "save"} else "writes"
        elif short in {"load_model", "load_state_dict", "load_weights"}:
            relation = "loads"
            kind = "model_checkpoint"
        if short == "open" and node.args:
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            relation = "writes" if any(flag in mode for flag in ("w", "a", "+")) else "reads"
        if relation:
            target = ast.unparse(node.args[0]) if node.args else called
            self.artifact_accesses.append(
                (self._symbol_id(), target, relation, node.lineno, kind)
            )
        self.generic_visit(node)


class RepositoryStructureImporter:
    """Build a concrete graph from the buggy repository and runtime evidence."""

    def __init__(self, *, max_python_files: int = 4000) -> None:
        self.max_python_files = max_python_files

    def build(self, incident: RepositoryIncident) -> RepositoryGraph:
        root = incident.repository_root
        if not root.is_dir():
            raise FileNotFoundError(root)
        nodes: dict[str, RepositoryNode] = {}
        edges: dict[str, RepositoryEdge] = {}
        evidence: dict[str, EvidenceRef] = {}
        limitations: list[str] = []

        repository_id = "repository:root"
        nodes[repository_id] = RepositoryNode(
            repository_id,
            "repository",
            incident.repository,
            attributes={"root_name": root.name},
        )
        python_files = sorted(
            path
            for path in root.rglob("*.py")
            if not any(part in {".git", ".tox", ".venv", "venv", "site-packages"} for part in path.parts)
        )
        if len(python_files) > self.max_python_files:
            limitations.append(f"python_file_limit:{self.max_python_files}/{len(python_files)}")
            python_files = python_files[: self.max_python_files]
        module_by_name: dict[str, str] = {}
        call_records: list[tuple[str, str, str, int]] = []
        import_records: list[tuple[str, str, str | None, int]] = []
        artifact_records: list[tuple[str, str, str, str, int, str]] = []
        package_nodes: dict[str, str] = {}
        for path in python_files:
            relative = path.relative_to(root).as_posix()
            file_id = _node_id("file", relative)
            source = path.read_text(encoding="utf-8", errors="replace")
            source_ref = _evidence_id("source", relative, hashlib.sha256(source.encode()).hexdigest())
            evidence[source_ref] = EvidenceRef(source_ref, "source", relative, "buggy revision source")
            nodes[file_id] = RepositoryNode(
                file_id,
                "file",
                incident.repository,
                relative,
                attributes={"size": len(source), "sha256": hashlib.sha256(source.encode()).hexdigest()},
                evidence_refs=(source_ref,),
            )
            module_name = relative.removesuffix(".py").replace("/", ".")
            module_name = module_name.removesuffix(".__init__")
            package_name = module_name.rsplit(".", 1)[0] if "." in module_name else "<root>"
            package_id = package_nodes.setdefault(
                package_name,
                _node_id("package", package_name),
            )
            nodes.setdefault(
                package_id,
                RepositoryNode(
                    package_id,
                    "package",
                    incident.repository,
                    symbol=package_name,
                ),
            )
            package_edge = f"edge:{repository_id}->{package_id}:contains"
            edges.setdefault(
                package_edge,
                RepositoryEdge(package_edge, repository_id, package_id, "contains"),
            )
            module_id = _node_id("module", module_name)
            nodes[module_id] = RepositoryNode(
                module_id,
                "module",
                incident.repository,
                relative,
                module_name,
                evidence_refs=(source_ref,),
            )
            module_by_name[module_name] = module_id
            module_edge = f"edge:{package_id}->{module_id}:contains"
            edges[module_edge] = RepositoryEdge(
                module_edge,
                package_id,
                module_id,
                "contains",
                (source_ref,),
            )
            file_edge = f"edge:{module_id}->{file_id}:contains"
            edges[file_edge] = RepositoryEdge(
                file_edge,
                module_id,
                file_id,
                "contains",
                (source_ref,),
            )
            export_edge = f"edge:{file_id}->{module_id}:produces"
            edges[export_edge] = RepositoryEdge(
                export_edge,
                file_id,
                module_id,
                "produces",
                (source_ref,),
            )
            try:
                tree = ast.parse(source, filename=relative)
            except SyntaxError as exc:
                limitations.append(f"syntax_error:{relative}:{exc.lineno}")
                continue
            index = _PythonIndex(incident.repository, relative)
            index.visit(tree)
            for node in index.nodes:
                nodes[node.node_id] = RepositoryNode(
                    **{
                        **node.__dict__,
                        "evidence_refs": (source_ref,),
                    }
                )
            for source_id, target_id, relation, line in index.edges:
                edge_id = f"edge:{source_id}->{target_id}:{relation}"
                edges[edge_id] = RepositoryEdge(edge_id, source_id, target_id, relation, (source_ref,))
            import_records.extend(
                (file_id, module, imported_name, line)
                for module, imported_name, line in index.imports
            )
            call_records.extend((relative, caller, called, line) for caller, called, line in index.calls)
            artifact_records.extend(
                (relative, caller, target, relation, line, kind)
                for caller, target, relation, line, kind in index.artifact_accesses
            )

        self._add_imports(nodes, edges, evidence, module_by_name, import_records, incident.repository)
        self._add_calls(nodes, edges, evidence, call_records, incident.repository)
        self._add_artifact_accesses(nodes, edges, evidence, artifact_records, incident.repository)
        self._add_manifests(root, incident.repository, repository_id, nodes, edges, evidence)
        obligations = self._add_runtime(incident, root, nodes, edges, evidence)
        if not obligations:
            limitations.append("no_runtime_obligations")
        return RepositoryGraph(
            incident.repository,
            incident.buggy_revision,
            tuple(sorted(nodes.values(), key=lambda item: item.node_id)),
            tuple(sorted(edges.values(), key=lambda item: item.edge_id)),
            tuple(sorted(evidence.values(), key=lambda item: item.evidence_id)),
            tuple(obligations),
            tuple(limitations),
        )

    @staticmethod
    def _add_imports(
        nodes: dict[str, RepositoryNode],
        edges: dict[str, RepositoryEdge],
        evidence: dict[str, EvidenceRef],
        module_by_name: dict[str, str],
        records: list[tuple[str, str, str | None, int]],
        repository: str,
    ) -> None:
        symbol_by_name: dict[str, list[str]] = {}
        for node in nodes.values():
            if node.symbol:
                symbol_by_name.setdefault(
                    node.symbol.rsplit(".", 1)[-1],
                    [],
                ).append(node.node_id)
        for source, module, imported_name, line in records:
            symbol_targets = symbol_by_name.get(imported_name or "", ())
            target = (
                symbol_targets[0]
                if imported_name and len(symbol_targets) == 1
                else module_by_name.get(module.lstrip("."))
            )
            if target is None:
                imported = module.lstrip(".").split(".")[0] or "<relative>"
                target = _node_id("module", f"external:{imported}")
                nodes.setdefault(
                    target,
                    RepositoryNode(
                        target,
                        "module",
                        repository,
                        symbol=module,
                        attributes={"external": True},
                    ),
                )
            imported = f".{imported_name}" if imported_name else ""
            ref = _evidence_id("import", source, f"{line}:{module}{imported}")
            evidence[ref] = EvidenceRef(
                ref,
                "import",
                source,
                f"line {line}: {module}{imported}",
            )
            edge_id = f"edge:{source}->{target}:imports:{line}"
            edges[edge_id] = RepositoryEdge(edge_id, source, target, "imports", (ref,))

    @staticmethod
    def _add_calls(
        nodes: dict[str, RepositoryNode],
        edges: dict[str, RepositoryEdge],
        evidence: dict[str, EvidenceRef],
        records: list[tuple[str, str, str, int]],
        repository: str,
    ) -> None:
        by_short_name: dict[str, list[str]] = {}
        for node in nodes.values():
            if node.symbol:
                by_short_name.setdefault(node.symbol.rsplit(".", 1)[-1], []).append(node.node_id)
        for relative, caller, called, line in records:
            short = called.rsplit(".", 1)[-1]
            targets = by_short_name.get(short, ())
            if len(targets) != 1:
                continue
            target = targets[0]
            ref = _evidence_id("call", relative, f"{line}:{called}")
            evidence[ref] = EvidenceRef(ref, "call", relative, f"line {line}: {called}")
            edge_id = f"edge:{caller}->{target}:calls:{line}"
            edges[edge_id] = RepositoryEdge(edge_id, caller, target, "calls", (ref,))

    @staticmethod
    def _add_artifact_accesses(
        nodes: dict[str, RepositoryNode],
        edges: dict[str, RepositoryEdge],
        evidence: dict[str, EvidenceRef],
        records: list[tuple[str, str, str, str, int, str]],
        repository: str,
    ) -> None:
        for relative, caller, target, relation, line, kind in records:
            artifact_id = _node_id(kind, relative, target)
            ref = _evidence_id(relation, relative, f"{line}:{target}")
            evidence[ref] = EvidenceRef(
                ref,
                relation,
                relative,
                f"line {line}: {target}",
            )
            nodes.setdefault(
                artifact_id,
                RepositoryNode(
                    artifact_id,
                    kind,
                    repository,
                    relative,
                    target,
                    {"access_expression": target},
                    (ref,),
                ),
            )
            edge_id = f"edge:{caller}->{artifact_id}:{relation}:{line}"
            edges[edge_id] = RepositoryEdge(
                edge_id,
                caller,
                artifact_id,
                relation,
                (ref,),
            )

    @staticmethod
    def _add_manifests(
        root: Path,
        repository: str,
        repository_id: str,
        nodes: dict[str, RepositoryNode],
        edges: dict[str, RepositoryEdge],
        evidence: dict[str, EvidenceRef],
    ) -> None:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            if path.name not in MANIFEST_NAMES and path.suffix.lower() not in CONFIG_SUFFIXES:
                continue
            if any(part in {".git", ".tox", ".venv", "venv"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            file_id = _node_id("file", relative)
            ref = _evidence_id("configuration", relative, hashlib.sha256(text.encode()).hexdigest())
            evidence[ref] = EvidenceRef(ref, "configuration", relative, "pre-fix configuration")
            nodes.setdefault(
                file_id,
                RepositoryNode(file_id, "file", repository, relative, evidence_refs=(ref,)),
            )
            entries = RepositoryStructureImporter._configuration_entries(path, text)
            for key, value in entries:
                config_id = _node_id("configuration_key", relative, key)
                nodes[config_id] = RepositoryNode(
                    config_id,
                    "configuration_key",
                    repository,
                    relative,
                    key,
                    {"value": value},
                    (ref,),
                )
                edge_id = f"edge:{file_id}->{config_id}:configured_by"
                edges[edge_id] = RepositoryEdge(edge_id, file_id, config_id, "configured_by", (ref,))
                for dependency, constraint in RepositoryStructureImporter._dependencies(key, value):
                    dependency_id = _node_id("dependency", dependency)
                    nodes[dependency_id] = RepositoryNode(
                        dependency_id,
                        "dependency",
                        repository,
                        relative,
                        dependency,
                        {"constraint": constraint},
                        (ref,),
                    )
                    dependency_edge = (
                        f"edge:{repository_id}->{dependency_id}:depends_on:{key}"
                    )
                    edges[dependency_edge] = RepositoryEdge(
                        dependency_edge,
                        repository_id,
                        dependency_id,
                        "depends_on",
                        (ref,),
                    )
            edge_id = f"edge:{repository_id}->{file_id}:contains"
            edges[edge_id] = RepositoryEdge(
                edge_id,
                repository_id,
                file_id,
                "contains",
                (ref,),
            )

    @staticmethod
    def _configuration_entries(path: Path, text: str) -> tuple[tuple[str, str], ...]:
        try:
            if path.suffix.lower() == ".toml":
                payload = tomllib.loads(text)
                return tuple(
                    (f"{section}.{key}", str(value))
                    for section, values in payload.items()
                    if isinstance(values, dict)
                    for key, value in values.items()
                    if not isinstance(value, dict)
                )[:200]
            if path.suffix.lower() in {".ini", ".cfg"}:
                parser = configparser.ConfigParser()
                parser.read_string(text)
                return tuple(
                    (f"{section}.{key}", value)
                    for section in parser.sections()
                    for key, value in parser.items(section)
                )[:200]
        except (ValueError, configparser.Error):
            return ()
        return tuple(
            (match.group(1), match.group(2).strip())
            for match in re.finditer(r"(?m)^\s*([A-Za-z_][\w.-]*)\s*[:=]\s*(.+?)\s*$", text)
        )[:200]

    @staticmethod
    def _dependencies(key: str, value: str) -> tuple[tuple[str, str], ...]:
        if "dependenc" not in key.lower() and not any(
            marker in value for marker in ("==", ">=", "<=", "~=", "!=")
        ):
            return ()
        matches = re.findall(
            r"([A-Za-z0-9_.-]+)\s*((?:==|>=|<=|~=|!=|>|<)[^,'\"\]\s]+)?",
            value,
        )
        return tuple(
            (name, constraint or "")
            for name, constraint in matches
            if name[0].isalnum()
            and name.lower()
            not in {"dependencies", "optional", "python", "test", "tests", "dev", "all"}
        )[:100]

    @staticmethod
    def _add_runtime(
        incident: RepositoryIncident,
        root: Path,
        nodes: dict[str, RepositoryNode],
        edges: dict[str, RepositoryEdge],
        evidence: dict[str, EvidenceRef],
    ) -> list[str]:
        obligations: list[str] = []
        runtime_text = (
            f"{incident.traceback}\n{incident.stdout}\n"
            f"{incident.stderr}\n{incident.assertion_difference}"
        )
        for index, test_id in enumerate(incident.failing_tests):
            obligation = f"failing_test:{index}:{test_id}"
            obligations.append(obligation)
            test_node = next(
                (
                    node
                    for node in nodes.values()
                    if node.kind == "test"
                    and node.symbol
                    and node.symbol.rsplit(".", 1)[-1] in test_id
                ),
                None,
            )
            if test_node is None:
                test_node_id = _node_id("test", test_id)
                test_node = RepositoryNode(
                    test_node_id,
                    "test",
                    incident.repository,
                    symbol=test_id,
                )
                nodes[test_node_id] = test_node
            ref = _evidence_id("failing_test", incident.incident_id, test_id)
            evidence[ref] = EvidenceRef(
                ref,
                "failing_test",
                test_id,
                f"registered failing test: {test_id}",
            )
            runtime_id = _node_id("runtime_exception", incident.incident_id, str(index))
            nodes[runtime_id] = RepositoryNode(
                runtime_id,
                "runtime_exception",
                incident.repository,
                symbol=test_id,
                attributes={
                    "obligation": obligation,
                    "assertion_difference": incident.assertion_difference,
                },
                evidence_refs=(ref,),
            )
            edge_id = f"edge:{test_node.node_id}->{runtime_id}:fails_in"
            edges[edge_id] = RepositoryEdge(edge_id, test_node.node_id, runtime_id, "fails_in", (ref,))
        frames = [
            match.groups()
            for match in re.finditer(
                r'File "([^"]+)", line (\d+), in ([^\n]+)',
                runtime_text,
            )
        ]
        frames.extend(
            match.groups()
            for match in re.finditer(
                r"(?m)^([^:\n]+\.py):(\d+): in ([^\n]+)$",
                runtime_text,
            )
        )
        for frame_index, (raw_path, raw_line, function) in enumerate(frames):
            try:
                relative = Path(raw_path).resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                relative = raw_path.replace("\\", "/")
                if relative.startswith(str(root).replace("\\", "/")):
                    relative = relative[len(str(root)) :].lstrip("/")
            candidates = [
                node
                for node in nodes.values()
                if node.file_path == relative
                and (
                    node.symbol is None
                    or node.symbol.rsplit(".", 1)[-1] == function.strip()
                )
            ]
            target = next((node for node in candidates if node.symbol), None) or next(iter(candidates), None)
            if target is None:
                continue
            ref = _evidence_id("traceback", relative, f"{raw_line}:{function}")
            evidence[ref] = EvidenceRef(ref, "traceback", relative, f"line {raw_line} in {function}")
            for runtime_node in (node for node in nodes.values() if node.kind == "runtime_exception"):
                edge_id = f"edge:{target.node_id}->{runtime_node.node_id}:produces:{frame_index}"
                edges[edge_id] = RepositoryEdge(edge_id, target.node_id, runtime_node.node_id, "produces", (ref,))
        return obligations
