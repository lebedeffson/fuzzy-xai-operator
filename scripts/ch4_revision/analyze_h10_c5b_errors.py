#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from fuzzyxai.gold_repository import GoldRepairAtom, extract_gold


def _top_k_signature(raw: str) -> tuple[tuple[str, str], ...]:
    return tuple((str(item["node_id"]), str(item["contract"])) for item in json.loads(raw))


def analyze(source: Path) -> dict[str, object]:
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["incident_id"]][row["method"]] = row
    route = [methods["O_ROUTE"] for methods in grouped.values()]
    limitation_counts = Counter(
        limitation
        for row in route
        for limitation in json.loads(row["limitations"])
        if limitation
        in {
            "contract_family_not_supported_by_evidence",
            "no_structural_candidate",
        }
    )
    diagnosed = [row for row in route if row["status"] == "DIAGNOSED"]
    identical = sum(
        _top_k_signature(methods["O_ROUTE"]["top_k_candidates"]) == _top_k_signature(methods["B_GREEDY"]["top_k_candidates"]) for methods in grouped.values()
    )
    return {
        "analysis_type": "POSTHOC_ERROR_ANALYSIS_ONLY",
        "official_result_modified": False,
        "incident_count": len(grouped),
        "o_route_abstained": sum(row["status"] == "INSUFFICIENT_EVIDENCE" for row in route),
        "o_route_diagnosed": len(diagnosed),
        "o_route_false_localizations_among_diagnosed": sum(float(row["false_localization"]) == 1.0 for row in diagnosed),
        "limitation_counts": dict(sorted(limitation_counts.items())),
        "identical_top_k_incidents": identical,
        "candidate_generation_boundary": ("global optimization cannot recover a Gold atom absent from the shared retrieved candidate ranking"),
        "reuse_policy": "these incidents must not be rescored",
    }


def build_error_table(
    source: Path,
    evidence_archive: Path,
) -> list[dict[str, object]]:
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    route_rows = {row["incident_id"]: row for row in rows if row["method"] == "O_ROUTE"}
    output = []
    with ZipFile(evidence_archive) as archive:
        names = set(archive.namelist())
        for incident_id, row in sorted(route_rows.items()):
            prefix = f"incidents/held_out/{incident_id}"
            patch = _zip_text(archive, f"{prefix}/fix.patch")
            before = json.loads(_zip_text(archive, f"{prefix}/before_sources.json"))
            after = json.loads(_zip_text(archive, f"{prefix}/after_sources.json"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                gold = extract_gold(patch, before, after)
            candidates = json.loads(row["top_k_candidates"])
            gold_sources = {(atom.file_path, atom.symbol) for atom in gold.atoms}
            gold_contracts = {atom.contract for atom in gold.atoms}
            retrieved_top3 = any((candidate.get("file_path"), candidate.get("symbol")) in gold_sources for candidate in candidates)
            contract_supported = any(candidate.get("contract") in gold_contracts for candidate in candidates)
            limitations = tuple(json.loads(row["limitations"]))
            runtime_text = _incident_runtime_text(
                archive,
                names,
                incident_id,
            ).lower()
            graph_gap = _graph_gap(gold.atoms, before)
            runtime_gap = _runtime_gap(gold.atoms, runtime_text)
            repair_expressible = False
            error_class = _error_class(
                retrieved_top3,
                contract_supported,
                graph_gap,
                runtime_gap,
                repair_expressible,
            )
            output.append(
                {
                    "incident_id": incident_id,
                    "gold_file": json.dumps(sorted({atom.file_path for atom in gold.atoms})),
                    "gold_symbol": json.dumps(sorted({atom.symbol for atom in gold.atoms if atom.symbol is not None})),
                    "gold_contract": json.dumps(sorted(gold_contracts)),
                    "retrieved_top10": "NOT_ESTIMABLE_FROZEN_TOP3_ONLY",
                    "retrieved_top3": retrieved_top3,
                    "contract_supported": contract_supported,
                    "abstention_reason": json.dumps(limitations),
                    "graph_gap": graph_gap,
                    "runtime_gap": runtime_gap,
                    "repair_expressible": repair_expressible,
                    "repair_expressibility_reason": ("NO_INCIDENT_SPECIFIC_REGISTERED_REPAIR_IN_FROZEN_RESULT"),
                    "error_class": error_class,
                }
            )
    return output


def _zip_text(archive: ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8")


def _incident_runtime_text(
    archive: ZipFile,
    names: set[str],
    incident_id: str,
) -> str:
    prefix = f"runtime-held_out/evidence/{incident_id}"
    return "\n".join(
        _zip_text(archive, name)
        for name in (
            f"{prefix}/traceback.txt",
            f"{prefix}/stdout.txt",
            f"{prefix}/stderr.txt",
        )
        if name in names
    )


def _graph_gap(
    atoms: tuple[GoldRepairAtom, ...],
    before: dict[str, str],
) -> str:
    missing_files = sorted({atom.file_path for atom in atoms if atom.file_path not in before})
    if missing_files:
        return "GOLD_FILE_ABSENT_FROM_BUGGY_SOURCE_SNAPSHOT"
    symbols_by_file = {file_path: _source_symbols(source) for file_path, source in before.items()}
    missing_symbols = sorted(
        {f"{atom.file_path}:{atom.symbol}" for atom in atoms if atom.symbol and atom.symbol not in symbols_by_file.get(atom.file_path, set())}
    )
    if missing_symbols:
        return "GOLD_SYMBOL_ABSENT_FROM_BUGGY_SOURCE_SNAPSHOT"
    return "NONE_OBSERVED_IN_DISCLOSED_SOURCE_SNAPSHOT"


def _source_symbols(source: str) -> set[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return set()
    symbols: set[str] = set()
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def _visit(self, node: ast.AST, name: str) -> None:
            stack.append(name)
            symbols.add(".".join(stack))
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit(node, node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._visit(node, node.name)

    Visitor().visit(tree)
    return symbols


def _runtime_gap(
    atoms: tuple[GoldRepairAtom, ...],
    runtime_text: str,
) -> str:
    if not runtime_text.strip():
        return "RUNTIME_OUTPUT_MISSING"
    represented = any(
        atom.file_path.lower() in runtime_text or (atom.symbol is not None and atom.symbol.rsplit(".", 1)[-1].lower() in runtime_text) for atom in atoms
    )
    return "NONE_GOLD_SOURCE_OBSERVED_IN_RUNTIME_OUTPUT" if not represented else "GOLD_SOURCE_OBSERVED"


def _error_class(
    retrieved_top3: bool,
    contract_supported: bool,
    graph_gap: str,
    runtime_gap: str,
    repair_expressible: bool,
) -> str:
    if graph_gap != "NONE_OBSERVED_IN_DISCLOSED_SOURCE_SNAPSHOT":
        return "GRAPH_CONSTRUCTION_MISS"
    if not retrieved_top3:
        if runtime_gap != "GOLD_SOURCE_OBSERVED":
            return "INSUFFICIENT_RUNTIME_EVIDENCE"
        return "RETRIEVAL_MISS"
    if not contract_supported:
        return "CONTRACT_INFERENCE_MISS"
    if not repair_expressible:
        return "REPAIR_NOT_EXPRESSIBLE"
    raise AssertionError("post-hoc row has no registered error class")


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-archive", type=Path)
    parser.add_argument("--table-output", type=Path)
    args = parser.parse_args()
    result = analyze(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.evidence_archive or args.table_output:
        if args.evidence_archive is None or args.table_output is None:
            raise ValueError("--evidence-archive and --table-output must be used together")
        _write_table(
            args.table_output,
            build_error_table(
                args.source.resolve(),
                args.evidence_archive.resolve(),
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
