from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

COLLECTION_LOCK_PATH = Path(
    "protocol/h10_c5c_evidence_retrieval/H10_C5C_DATA_COLLECTION_LOCK.json"
)
PROTOCOL_LOCK_PATH = Path(
    "protocol/h10_c5c_evidence_retrieval/H10_C5C_PROTOCOL_LOCK.json"
)

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_SAFE_PROJECT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_BUG = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class BugsInPyCandidate:
    project: str
    bug_id: str
    repository: str
    repository_url: str
    python_version: str
    buggy_commit: str
    fixed_commit: str
    test_file: str
    test_commands: tuple[tuple[str, ...], ...]
    patch_path: Path
    bug_root: Path
    selection_rank_sha256: str

    @property
    def incident_id(self) -> str:
        return f"bugsinpy-{self.project}-{self.bug_id}"


@dataclass(frozen=True)
class PreparedDevelopmentData:
    manifest_path: Path
    command_registry_path: Path
    source_registry_path: Path
    selection_report_path: Path
    incident_count: int
    repository_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _run(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _git_commit(repository: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repository).stdout.strip()


def _git_is_clean(repository: Path) -> bool:
    return not _run(
        ["git", "status", "--porcelain"],
        cwd=repository,
    ).stdout.strip()


def _materialize_registered_text(source: Path, destination: Path) -> str:
    payload = source.read_bytes()
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    elif payload.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"
    text = payload.decode(encoding)
    destination.write_text(
        text.replace("\r\n", "\n").replace("\r", "\n"),
        encoding="utf-8",
    )
    return encoding


def _parse_assignment_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if match is None:
            raise ValueError(f"unsupported assignment in {path}:{line_number}: {raw}")
        key, encoded = match.groups()
        encoded = encoded.strip()
        if not encoded:
            values[key] = ""
            continue
        try:
            parsed = ast.literal_eval(encoded)
        except (SyntaxError, ValueError):
            parsed = encoded
        if not isinstance(parsed, (str, int, float)):
            raise TypeError(f"unsupported value in {path}:{line_number}: {raw}")
        values[key] = str(parsed)
    return values


def canonical_repository(value: str) -> str:
    text = value.strip().removesuffix(".git").rstrip("/")
    if "://" in text:
        parsed = urlparse(text)
        text = parsed.path.strip("/")
    elif text.startswith("git@") and ":" in text:
        text = text.split(":", 1)[1]
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"cannot canonicalize repository: {value}")
    return "/".join(parts[-2:])


def _parse_python_major(version: str) -> int | None:
    match = re.match(r"\s*(\d+)", version)
    return int(match.group(1)) if match else None


def _instrumentable_command(command: tuple[str, ...]) -> bool:
    if not command:
        return False
    executable = Path(command[0]).name
    return (
        executable in {"pytest", "py.test", "nosetests", "nose2"}
        or re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable) is not None
        or executable.endswith(".py")
    )


def _parse_test_commands(path: Path) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = tuple(shlex.split(line, comments=True, posix=True))
        if not tokens:
            continue
        unsafe = {"|", "||", "&&", ";", ">", ">>", "<", "2>", "2>>"}
        if unsafe.intersection(tokens):
            raise ValueError(f"shell composition is not accepted in {path}:{line_number}")
        commands.append(tokens)
    return tuple(commands)


def _load_locks(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    protocol = json.loads((root / PROTOCOL_LOCK_PATH).read_text(encoding="utf-8"))
    collection = json.loads((root / COLLECTION_LOCK_PATH).read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_BEFORE_IMPLEMENTATION":
        raise ValueError("H10-C5c protocol lock is invalid")
    if collection.get("status") != "LOCKED_BEFORE_DEVELOPMENT_COLLECTION":
        raise ValueError("H10-C5c data collection lock is invalid")
    return protocol, collection


def _rank(collection_id: str, repository: str, bug_id: str) -> str:
    return hashlib.sha256(
        f"{collection_id}\0{repository}\0{bug_id}".encode()
    ).hexdigest()


def discover_bugsinpy_candidates(
    bugsinpy_root: Path,
    root: Path,
) -> tuple[BugsInPyCandidate, ...]:
    protocol, collection = _load_locks(root)
    projects_root = bugsinpy_root / "projects"
    if not projects_root.is_dir():
        raise FileNotFoundError(projects_root)
    excluded = {
        canonical_repository(str(value))
        for value in protocol["registered_h10_c5b_held_out_repositories"]
    }
    collection_id = str(collection["collection_id"])
    candidates: list[BugsInPyCandidate] = []
    for project_root in sorted(projects_root.iterdir()):
        if not project_root.is_dir() or not _SAFE_PROJECT.fullmatch(project_root.name):
            continue
        project_info_path = project_root / "project.info"
        bugs_root = project_root / "bugs"
        if not project_info_path.is_file() or not bugs_root.is_dir():
            continue
        project_info = _parse_assignment_file(project_info_path)
        repository_url = project_info.get("github_url", "").strip()
        if not repository_url:
            continue
        repository = canonical_repository(repository_url)
        if repository in excluded:
            continue
        for bug_root in sorted(bugs_root.iterdir(), key=lambda item: item.name):
            if not bug_root.is_dir() or not _SAFE_BUG.fullmatch(bug_root.name):
                continue
            required = (
                bug_root / "bug.info",
                bug_root / "bug_patch.txt",
                bug_root / "run_test.sh",
            )
            if not all(path.is_file() for path in required):
                continue
            bug_info = _parse_assignment_file(required[0])
            python_version = bug_info.get("python_version", "")
            if _parse_python_major(python_version) != 3:
                continue
            buggy_commit = bug_info.get("buggy_commit_id", "").strip()
            fixed_commit = bug_info.get("fixed_commit_id", "").strip()
            test_file = bug_info.get("test_file", "").strip()
            patch_text = required[1].read_text(encoding="utf-8", errors="replace")
            if (
                not buggy_commit
                or not fixed_commit
                or not test_file
                or not patch_text.strip()
            ):
                continue
            try:
                commands = _parse_test_commands(required[2])
            except ValueError:
                continue
            if not commands or not all(_instrumentable_command(command) for command in commands):
                continue
            candidates.append(
                BugsInPyCandidate(
                    project_root.name,
                    bug_root.name,
                    repository,
                    repository_url,
                    python_version,
                    buggy_commit,
                    fixed_commit,
                    test_file,
                    commands,
                    required[1].resolve(),
                    bug_root.resolve(),
                    _rank(collection_id, repository, bug_root.name),
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.repository,
                item.selection_rank_sha256,
                item.incident_id,
            ),
        )
    )


def select_balanced_development(
    candidates: tuple[BugsInPyCandidate, ...],
    *,
    target_incidents: int,
    minimum_repositories: int,
    maximum_per_repository: int,
) -> tuple[BugsInPyCandidate, ...]:
    grouped: dict[str, list[BugsInPyCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.repository].append(candidate)
    if len(grouped) < minimum_repositories:
        raise ValueError(
            f"eligible BugsInPy repositories are insufficient: {len(grouped)} < {minimum_repositories}"
        )
    repository_order = sorted(
        grouped,
        key=lambda repository: hashlib.sha256(repository.encode()).hexdigest(),
    )
    for repository in repository_order:
        grouped[repository].sort(
            key=lambda item: (item.selection_rank_sha256, item.incident_id)
        )
    selected: list[BugsInPyCandidate] = []
    round_index = 0
    while len(selected) < target_incidents:
        progressed = False
        for repository in repository_order:
            rows = grouped[repository]
            if round_index >= maximum_per_repository or round_index >= len(rows):
                continue
            selected.append(rows[round_index])
            progressed = True
            if len(selected) == target_incidents:
                break
        if not progressed:
            break
        round_index += 1
    repositories = {candidate.repository for candidate in selected}
    if len(selected) < target_incidents:
        raise ValueError(
            f"eligible BugsInPy incidents are insufficient: {len(selected)} < {target_incidents}"
        )
    if len(repositories) < minimum_repositories:
        raise ValueError(
            f"selected BugsInPy repositories are insufficient: {len(repositories)} < {minimum_repositories}"
        )
    return tuple(selected)


def _repository_cache_path(candidate: BugsInPyCandidate, cache_root: Path) -> Path:
    return cache_root / candidate.repository.replace("/", "__")


def _ensure_repository(
    candidate: BugsInPyCandidate,
    cache_root: Path,
    *,
    allow_network: bool,
) -> Path:
    repository = _repository_cache_path(candidate, cache_root)
    if not repository.is_dir():
        if not allow_network:
            raise FileNotFoundError(
                f"repository cache is absent for {candidate.repository}: {repository}"
            )
        repository.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--quiet",
                candidate.repository_url,
                str(repository),
            ]
        )
    for commit in (candidate.buggy_commit, candidate.fixed_commit):
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repository,
            capture_output=True,
            check=False,
        )
        if exists.returncode:
            if not allow_network:
                raise ValueError(
                    f"commit is absent from local cache for {candidate.incident_id}: {commit}"
                )
            _run(
                ["git", "fetch", "--quiet", "--depth=1", "origin", commit],
                cwd=repository,
            )
    return repository


def _safe_extract_git_archive(repository: Path, commit: str, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", commit],
        cwd=repository,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(output, filter="data")
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)


def _changed_files(patch_text: str) -> tuple[str, ...]:
    paths = []
    for left, right in _DIFF_HEADER.findall(patch_text):
        value = right if right != "/dev/null" else left
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe patch path: {value}")
        paths.append(path.as_posix())
    return tuple(dict.fromkeys(paths))


def _show_bytes(repository: Path, commit: str, relative: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def _show(repository: Path, commit: str, relative: str) -> str:
    content = _show_bytes(repository, commit, relative)
    return content.decode("utf-8", errors="replace") if content is not None else ""


def _safe_relative_paths(encoded: str) -> tuple[str, ...]:
    paths: list[str] = []
    for raw in encoded.split(";"):
        value = raw.strip()
        if not value:
            continue
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe benchmark path: {value}")
        paths.append(path.as_posix())
    return tuple(dict.fromkeys(paths))


def _overlay_exposing_tests(
    repository: Path,
    fixed_commit: str,
    test_file: str,
    buggy_root: Path,
) -> tuple[dict[str, str], ...]:
    overlays: list[dict[str, str]] = []
    for relative in _safe_relative_paths(test_file):
        content = _show_bytes(repository, fixed_commit, relative)
        if content is None:
            raise ValueError(
                f"registered exposing test is absent from fixed revision: {relative}"
            )
        destination = buggy_root / Path(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        overlays.append(
            {
                "path": relative,
                "fixed_test_sha256": hashlib.sha256(content).hexdigest(),
                "materialized_test_sha256": _sha256(destination),
            }
        )
    return tuple(overlays)


def _normalize_test_id(command: tuple[str, ...], fallback: str) -> str:
    selectors = [
        token
        for token in command
        if "::" in token or token.endswith(".py")
    ]
    return selectors[-1] if selectors else fallback


def _materialize_candidate(
    candidate: BugsInPyCandidate,
    *,
    output_root: Path,
    cache_root: Path,
    allow_network: bool,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    repository = _ensure_repository(
        candidate,
        cache_root,
        allow_network=allow_network,
    )
    incident_root = output_root / "incidents" / "development" / candidate.incident_id
    buggy_root = incident_root / "buggy"
    _safe_extract_git_archive(repository, candidate.buggy_commit, buggy_root)
    exposing_test_overlays = _overlay_exposing_tests(
        repository,
        candidate.fixed_commit,
        candidate.test_file,
        buggy_root,
    )
    patch_text = candidate.patch_path.read_text(encoding="utf-8", errors="replace")
    changed_files = _changed_files(patch_text)
    if not changed_files:
        raise ValueError(f"patch has no changed files: {candidate.incident_id}")
    before = {
        path: _show(repository, candidate.buggy_commit, path)
        for path in changed_files
    }
    after = {
        path: _show(repository, candidate.fixed_commit, path)
        for path in changed_files
    }
    incident_root.mkdir(parents=True, exist_ok=True)
    patch_output = incident_root / "fix.patch"
    before_output = incident_root / "before_sources.json"
    after_output = incident_root / "after_sources.json"
    runtime_output = incident_root / "runtime_events.jsonl"
    setup_source = candidate.bug_root / "setup.sh"
    requirements_source = candidate.bug_root / "requirements.txt"
    setup_output = incident_root / "setup.sh"
    requirements_output = incident_root / "requirements.txt"
    if setup_source.is_file():
        shutil.copy2(setup_source, setup_output)
    requirements_encoding = ""
    if requirements_source.is_file():
        requirements_encoding = _materialize_registered_text(
            requirements_source,
            requirements_output,
        )
    patch_output.write_text(patch_text, encoding="utf-8")
    before_output.write_text(
        json.dumps(before, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    after_output.write_text(
        json.dumps(after, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    test_ids = tuple(
        _normalize_test_id(command, candidate.test_file or candidate.incident_id)
        for command in candidate.test_commands
    )
    manifest_row: dict[str, object] = {
        "incident_id": candidate.incident_id,
        "repository": candidate.repository,
        "buggy_revision": candidate.buggy_commit,
        "repository_root": _portable(buggy_root, output_root),
        "failing_tests": test_ids,
        "split": "development",
        "patch_path": _portable(patch_output, output_root),
        "before_sources_path": _portable(before_output, output_root),
        "after_sources_path": _portable(after_output, output_root),
        "runtime_events_path": _portable(runtime_output, output_root),
        "runtime_evidence_status": "PENDING_COLLECTION",
        "source_benchmark": "BugsInPy",
        "source_project": candidate.project,
        "source_bug_id": candidate.bug_id,
        "source_python_version": candidate.python_version,
        "exposing_test_files": [item["path"] for item in exposing_test_overlays],
        "fixed_revision_sha256": hashlib.sha256(
            candidate.fixed_commit.encode()
        ).hexdigest(),
        "selection_rank_sha256": candidate.selection_rank_sha256,
    }
    command_row: dict[str, object] = {
        "incident_id": candidate.incident_id,
        "repository": candidate.repository,
        "python_version": candidate.python_version,
        "python_executable": (
            "python" + ".".join(candidate.python_version.split(".")[:2])
            if candidate.python_version
            else "python3"
        ),
        "commands": [
            {
                "test_id": test_id,
                "argv": list(command),
            }
            for test_id, command in zip(test_ids, candidate.test_commands, strict=True)
        ],
        "setup_script": (
            _portable(setup_output, output_root)
            if setup_output.is_file()
            else ""
        ),
        "requirements_path": (
            _portable(requirements_output, output_root)
            if requirements_output.is_file()
            else ""
        ),
    }
    source_row: dict[str, object] = {
        "incident_id": candidate.incident_id,
        "repository": candidate.repository,
        "repository_url": candidate.repository_url,
        "project_cache_commit": _git_commit(repository),
        "buggy_commit": candidate.buggy_commit,
        "fixed_commit": candidate.fixed_commit,
        "patch_sha256": _sha256(patch_output),
        "source_patch_sha256": _sha256(candidate.patch_path),
        "bug_info_sha256": _sha256(candidate.bug_root / "bug.info"),
        "run_test_sha256": _sha256(candidate.bug_root / "run_test.sh"),
        "setup_script_source_sha256": (
            _sha256(setup_source) if setup_source.is_file() else ""
        ),
        "setup_script_materialized_sha256": (
            _sha256(setup_output) if setup_output.is_file() else ""
        ),
        "requirements_source_sha256": (
            _sha256(requirements_source) if requirements_source.is_file() else ""
        ),
        "requirements_materialized_sha256": (
            _sha256(requirements_output) if requirements_output.is_file() else ""
        ),
        "requirements_source_encoding": requirements_encoding,
        "exposing_test_overlays": list(exposing_test_overlays),
        "repository_tree_sha256": hashlib.sha256(
            _run(["git", "rev-parse", f"{candidate.buggy_commit}^{{tree}}"], cwd=repository).stdout.strip().encode()
        ).hexdigest(),
    }
    return manifest_row, command_row, source_row


def prepare_bugsinpy_development(
    bugsinpy_root: Path,
    output_root: Path,
    cache_root: Path,
    root: Path,
    *,
    allow_network: bool = False,
) -> PreparedDevelopmentData:
    _protocol, collection = _load_locks(root)
    benchmark = collection["benchmark"]
    if not isinstance(benchmark, dict):
        raise TypeError("H10-C5c benchmark lock must be an object")
    bugsinpy_commit = _git_commit(bugsinpy_root)
    expected_commit = str(benchmark.get("commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("H10-C5c BugsInPy lock must contain a full commit SHA")
    if bugsinpy_commit != expected_commit:
        raise ValueError(
            "BugsInPy checkout does not match the locked commit: "
            f"{bugsinpy_commit} != {expected_commit}"
        )
    if not _git_is_clean(bugsinpy_root):
        raise ValueError("BugsInPy checkout must be clean before materialization")
    selection = collection["selection"]
    if not isinstance(selection, dict):
        raise TypeError("H10-C5c selection lock must be an object")
    candidates = discover_bugsinpy_candidates(bugsinpy_root, root)
    selected = select_balanced_development(
        candidates,
        target_incidents=int(selection["target_incidents"]),
        minimum_repositories=int(selection["minimum_repositories"]),
        maximum_per_repository=int(selection["maximum_incidents_per_repository"]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    command_rows = {}
    source_rows = []
    for candidate in selected:
        manifest_row, command_row, source_row = _materialize_candidate(
            candidate,
            output_root=output_root,
            cache_root=cache_root,
            allow_network=allow_network,
        )
        manifest_rows.append(manifest_row)
        command_rows[candidate.incident_id] = command_row
        source_rows.append(source_row)
    manifest_path = output_root / "H10_C5C_DEVELOPMENT_UNCOLLECTED.jsonl"
    command_registry_path = output_root / "H10_C5C_COMMAND_REGISTRY.json"
    source_registry_path = output_root / "H10_C5C_SOURCE_REGISTRY.json"
    selection_report_path = output_root / "H10_C5C_SELECTION_REPORT.json"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    command_registry_path.write_text(
        json.dumps(command_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_registry = {
        "collection_id": collection["collection_id"],
        "bugsinpy_repository": str(benchmark["repository"]),
        "bugsinpy_commit": bugsinpy_commit,
        "bugsinpy_commit_sha256": hashlib.sha256(
            bugsinpy_commit.encode()
        ).hexdigest(),
        "incidents": source_rows,
    }
    source_registry_path.write_text(
        json.dumps(source_registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "collection_id": collection["collection_id"],
        "status": "MATERIALIZED_AWAITING_RUNTIME_COLLECTION",
        "eligible_incidents": len(candidates),
        "eligible_repositories": len({item.repository for item in candidates}),
        "selected_incidents": len(selected),
        "selected_repositories": len({item.repository for item in selected}),
        "selected": [
            {
                "project": item.project,
                "bug_id": item.bug_id,
                "incident_id": item.incident_id,
                "repository": item.repository,
                "python_version": item.python_version,
                "buggy_commit": item.buggy_commit,
                "fixed_commit": item.fixed_commit,
                "test_file": item.test_file,
                "test_commands": [list(command) for command in item.test_commands],
                "selection_rank_sha256": item.selection_rank_sha256,
            }
            for item in selected
        ],
        "manifest_sha256": _sha256(manifest_path),
        "command_registry_sha256": _sha256(command_registry_path),
        "source_registry_sha256": _sha256(source_registry_path),
        "scientific_result": "NOT_EVALUATED",
    }
    selection_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreparedDevelopmentData(
        manifest_path,
        command_registry_path,
        source_registry_path,
        selection_report_path,
        len(selected),
        len({item.repository for item in selected}),
    )
