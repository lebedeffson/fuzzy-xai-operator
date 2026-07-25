from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldRepairAtom:
    file_path: str
    symbol: str | None
    contract: str
    operation: str


@dataclass(frozen=True)
class RepositoryGold:
    atoms: tuple[GoldRepairAtom, ...]
    changed_files: tuple[str, ...]
    changed_symbols: tuple[tuple[str, str], ...]
    changed_config_keys: tuple[tuple[str, str], ...] = ()
    changed_api_calls: tuple[tuple[str, str], ...] = ()
    scorer_version: str = "independent-ast-diff-gold-v2"


def _files(patch: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            right
            for _left, right in re.findall(
                r"^diff --git a/(.+?) b/(.+?)$",
                patch,
                flags=re.MULTILINE,
            )
        )
    )


def _symbols(source: str) -> dict[str, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    values: dict[str, str] = {}
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def _definition(self, node: ast.AST, name: str) -> None:
            stack.append(name)
            values[".".join(stack)] = ast.dump(node, include_attributes=False)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._definition(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._definition(node, node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._definition(node, node.name)

    Visitor().visit(tree)
    return values


def _calls(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    values = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            values.add(ast.unparse(node.func))
        except ValueError:
            continue
    return values


def _changed_lines(before: str, after: str) -> str:
    return "\n".join(
        line[2:]
        for line in difflib.ndiff(before.splitlines(), after.splitlines())
        if line.startswith(("+ ", "- "))
    )


def _changed_config_keys(before: str, after: str) -> tuple[str, ...]:
    changed = _changed_lines(before, after)
    return tuple(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(
                r"(?m)^\s*([A-Za-z_][\w.-]*)\s*[:=]",
                changed,
            )
        )
    )


def _contract(before: str, after: str, file_path: str) -> tuple[str, str]:
    suffix = file_path.lower()
    before_tree = _symbols(before)
    after_tree = _symbols(after)
    changed = _changed_lines(before, after)
    changed_calls = _calls(before) ^ _calls(after)
    changed_symbols = {
        name
        for name in set(before_tree) | set(after_tree)
        if before_tree.get(name) != after_tree.get(name)
    }
    combined = (
        f"{changed}\n{' '.join(sorted(changed_calls))}\n"
        f"{' '.join(sorted(changed_symbols))}"
    )
    if any(name in suffix for name in ("requirements", "pyproject", "setup.cfg", "tox.ini")):
        return "DEPENDENCY_VERSION", "update_dependency_constraint"
    if re.search(r"\b(?:pickle|json|yaml|loads?|dumps?|decode|encode)\b", combined):
        return "SERIALIZATION", "align_reader_writer"
    if re.search(
        r"\b(?:dtype|shape|columns?|fields?|schema|ndim|typeerror|"
        r"isinstance|arrays?|iterable|sequence|validators?)\b",
        combined,
        flags=re.IGNORECASE,
    ):
        return "DATA_CONTRACT", "align_data_schema"
    if re.search(r"\b(?:checkpoint|state_dict|load_model|weights)\b", combined):
        return "MODEL_LOADING", "align_model_loading"
    if re.search(r"\b(?:cache|metadata|path|artifact|checksum|digest)\b", combined):
        return "ARTIFACT_PROVENANCE", "restore_artifact_provenance"
    if before_tree != after_tree:
        return "CONFIGURATION", "change_program_symbol"
    return "CONFIGURATION", "change_configuration"


def extract_gold(
    patch: str,
    before_sources: dict[str, str],
    after_sources: dict[str, str],
) -> RepositoryGold:
    changed_files = _files(patch)
    atoms: list[GoldRepairAtom] = []
    changed_symbols: list[tuple[str, str]] = []
    changed_config_keys: list[tuple[str, str]] = []
    changed_api_calls: list[tuple[str, str]] = []
    for file_path in changed_files:
        before = before_sources.get(file_path, "")
        after = after_sources.get(file_path, "")
        before_symbols = _symbols(before)
        after_symbols = _symbols(after)
        config_keys = (
            _changed_config_keys(before, after)
            if file_path.lower().endswith(
                (".toml", ".ini", ".cfg", ".yaml", ".yml", ".json")
            )
            else ()
        )
        api_calls = tuple(sorted(_calls(before) ^ _calls(after)))
        symbols = tuple(
            sorted(
                name
                for name in set(before_symbols) | set(after_symbols)
                if before_symbols.get(name) != after_symbols.get(name)
            )
        )
        contract, operation = _contract(before, after, file_path)
        if not symbols and not config_keys:
            atoms.append(GoldRepairAtom(file_path, None, contract, operation))
        for symbol in symbols:
            changed_symbols.append((file_path, symbol))
            atoms.append(GoldRepairAtom(file_path, symbol, contract, operation))
        for key in config_keys:
            changed_config_keys.append((file_path, key))
            atoms.append(GoldRepairAtom(file_path, key, contract, operation))
        changed_api_calls.extend((file_path, call) for call in api_calls)
    return RepositoryGold(
        tuple(atoms),
        changed_files,
        tuple(changed_symbols),
        tuple(changed_config_keys),
        tuple(changed_api_calls),
    )
