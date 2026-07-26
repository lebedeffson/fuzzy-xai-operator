from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fuzzyxai.experiments.h10_c5c_data import (
    COLLECTION_LOCK_PATH,
    PROTOCOL_LOCK_PATH,
    RUNTIME_COMPATIBILITY_AMENDMENT_PATH,
    RUNTIME_INSTALLATION_AMENDMENT_PATH,
    canonical_repository,
    discover_bugsinpy_candidates,
    prepare_bugsinpy_development,
    select_balanced_development,
)
from fuzzyxai.experiments.h10_c5c_readiness import (
    verify_h10_c5c_development_readiness,
)
from fuzzyxai.experiments.h10_c5c_runtime import collect_h10_c5c_runtime
from fuzzyxai.repository_diagnostics.runtime_events import load_runtime_events

ROOT = Path(__file__).resolve().parents[2]
AMENDMENT_PATH = Path(
    "protocol/h10_c5c_evidence_retrieval/H10_C5C_DATA_COLLECTION_AMENDMENT_001.json"
)
LOCKED_BUGSINPY_COMMIT = json.loads(
    (ROOT / COLLECTION_LOCK_PATH).read_text(encoding="utf-8")
)["benchmark"]["commit"]


def _run(*arguments: str, cwd: Path) -> str:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.email", "fixture@example.test", cwd=path)
    _run("git", "config", "user.name", "Fixture", cwd=path)


def _commit_all(path: Path, message: str) -> str:
    _run("git", "add", ".", cwd=path)
    _run("git", "commit", "-qm", message, cwd=path)
    return _run("git", "rev-parse", "HEAD", cwd=path)


def _root_with_locked_benchmark(tmp_path: Path, commit: str) -> Path:
    root = tmp_path / "locked-protocol"
    for relative in (
        PROTOCOL_LOCK_PATH,
        COLLECTION_LOCK_PATH,
        AMENDMENT_PATH,
        RUNTIME_COMPATIBILITY_AMENDMENT_PATH,
        RUNTIME_INSTALLATION_AMENDMENT_PATH,
    ):
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(source.read_text(encoding="utf-8"))
        if relative == COLLECTION_LOCK_PATH:
            data["benchmark"]["commit"] = commit
        target.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return root


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    bugsinpy = tmp_path / "BugsInPy"
    _git_init(bugsinpy)
    projects = bugsinpy / "projects"
    upstream_root = tmp_path / "upstream"
    for project_index in range(8):
        owner = upstream_root / f"owner{project_index}"
        upstream = owner / f"project{project_index}"
        _git_init(upstream)
        (upstream / "pkg").mkdir()
        (upstream / "pkg" / "core.py").write_text(
            "def value():\n    return 0\n",
            encoding="utf-8",
        )
        (upstream / "test_core.py").write_text(
            "from pkg.core import value\n\nassert value() == 1\n",
            encoding="utf-8",
        )
        buggy = _commit_all(upstream, "buggy")
        (upstream / "pkg" / "core.py").write_text(
            "def value():\n    return 1\n",
            encoding="utf-8",
        )
        (upstream / "test_core.py").write_text(
            "# exposing test from the fixed revision\n"
            "from pkg.core import value\n\nassert value() == 1\n",
            encoding="utf-8",
        )
        fixed = _commit_all(upstream, "fixed")
        patch = _run("git", "diff", buggy, fixed, "--", "pkg/core.py", cwd=upstream)
        project = projects / f"project{project_index}"
        (project / "bugs").mkdir(parents=True)
        (project / "project.info").write_text(
            f'github_url="file://{upstream}"\nstatus="OK"\ncause="N.A."\n',
            encoding="utf-8",
        )
        for bug_index in range(4):
            bug = project / "bugs" / str(bug_index + 1)
            bug.mkdir()
            (bug / "bug.info").write_text(
                f'python_version="{sys.version_info.major}.{sys.version_info.minor}"\n'
                f'buggy_commit_id="{buggy}"\n'
                f'fixed_commit_id="{fixed}"\n'
                'test_file="test_core.py"\n',
                encoding="utf-8",
            )
            (bug / "bug_patch.txt").write_text(patch, encoding="utf-8")
            (bug / "run_test.sh").write_text(
                "python test_core.py\n",
                encoding="utf-8",
            )
            if bug_index == 0:
                (bug / "requirements.txt").write_bytes(
                    "fixture-package==1.0\r\n".encode("utf-16")
                )
    _commit_all(bugsinpy, "benchmark")
    return bugsinpy, upstream_root


def test_canonical_repository_accepts_https_ssh_and_file_paths() -> None:
    assert canonical_repository("https://github.com/org/repo.git") == "org/repo"
    assert canonical_repository("git@github.com:org/repo.git") == "org/repo"
    assert canonical_repository("file:///tmp/org/repo") == "org/repo"


def test_bugsinpy_assignment_parser_accepts_spaces_around_equals(
    tmp_path: Path,
) -> None:
    bugsinpy, _upstream = _build_fixture(tmp_path)
    project_info = bugsinpy / "projects/project0/project.info"
    project_info.write_text(
        project_info.read_text(encoding="utf-8") + 'PYTHONPATH = "pkg"\n',
        encoding="utf-8",
    )
    candidates = discover_bugsinpy_candidates(bugsinpy, ROOT)
    assert candidates


def test_balanced_selection_meets_locked_counts(tmp_path: Path) -> None:
    bugsinpy, _upstream = _build_fixture(tmp_path)
    candidates = discover_bugsinpy_candidates(bugsinpy, ROOT)
    selected = select_balanced_development(
        candidates,
        target_incidents=30,
        minimum_repositories=8,
        maximum_per_repository=4,
    )
    assert len(selected) == 30
    assert len({item.repository for item in selected}) == 8
    per_repository = {
        repository: sum(item.repository == repository for item in selected)
        for repository in {item.repository for item in selected}
    }
    assert max(per_repository.values()) <= 4


def test_prepare_materializes_uncollected_development_manifest(
    tmp_path: Path,
) -> None:
    bugsinpy, _upstream = _build_fixture(tmp_path)
    fixture_commit = _run("git", "rev-parse", "HEAD", cwd=bugsinpy)
    protocol_root = _root_with_locked_benchmark(tmp_path, fixture_commit)
    result = prepare_bugsinpy_development(
        bugsinpy,
        tmp_path / "output",
        tmp_path / "cache",
        protocol_root,
        allow_network=True,
    )
    rows = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.incident_count == 30
    assert result.repository_count == 8
    assert len(rows) == 30
    assert all(row["runtime_evidence_status"] == "PENDING_COLLECTION" for row in rows)
    base = result.manifest_path.parent
    assert all((base / row["repository_root"]).is_dir() for row in rows)
    assert all((base / row["patch_path"]).is_file() for row in rows)
    assert all(not Path(row["repository_root"]).is_absolute() for row in rows)
    assert all(row["exposing_test_files"] == ["test_core.py"] for row in rows)
    for row in rows:
        materialized_test = base / row["repository_root"] / "test_core.py"
        assert materialized_test.read_text(encoding="utf-8").startswith(
            "# exposing test from the fixed revision"
        )
    source = json.loads(result.source_registry_path.read_text(encoding="utf-8"))
    assert source["bugsinpy_commit"] == fixture_commit
    assert source["bugsinpy_repository"].endswith("/BugsInPy.git")
    assert "bugsinpy_checkout" not in source
    assert len(source["incidents"]) == 30
    assert all(
        item["exposing_test_overlays"][0]["path"] == "test_core.py"
        for item in source["incidents"]
    )
    encoded_requirements = [
        item
        for item in source["incidents"]
        if item["requirements_source_encoding"] == "utf-16"
    ]
    assert encoded_requirements
    commands = json.loads(
        result.command_registry_path.read_text(encoding="utf-8")
    )
    assert all(
        command["build_toolchain"]
        == {
            "pip": "20.3.4",
            "setuptools": "44.1.1",
            "wheel": "0.36.2",
        }
        for command in commands.values()
    )
    assert all(
        command["requirements_install_options"]
        == ["--use-deprecated=legacy-resolver", "--no-build-isolation"]
        for command in commands.values()
    )
    assert all(
        command["requirements_install_strategy"]
        == "BENCHMARK_PER_LINE_TWO_PASS"
        for command in commands.values()
    )
    for item in encoded_requirements:
        materialized = base / commands[item["incident_id"]]["requirements_path"]
        payload = materialized.read_bytes()
        assert payload == b"fixture-package==1.0\n"
        assert (
            hashlib.sha256(payload).hexdigest()
            == item["requirements_materialized_sha256"]
        )
        assert (
            item["requirements_source_sha256"]
            != item["requirements_materialized_sha256"]
        )
    selection = json.loads(result.selection_report_path.read_text(encoding="utf-8"))
    assert all("patch_path" not in item for item in selection["selected"])
    assert all("bug_root" not in item for item in selection["selected"])
    assert str(bugsinpy.resolve()) not in result.source_registry_path.read_text(
        encoding="utf-8"
    )
    assert str(bugsinpy.resolve()) not in result.selection_report_path.read_text(
        encoding="utf-8"
    )


def test_prepare_rejects_unlocked_or_dirty_bugsinpy_checkout(
    tmp_path: Path,
) -> None:
    bugsinpy, _upstream = _build_fixture(tmp_path)
    fixture_commit = _run("git", "rev-parse", "HEAD", cwd=bugsinpy)
    mismatched_root = _root_with_locked_benchmark(tmp_path, "0" * 40)
    with pytest.raises(ValueError, match="does not match the locked commit"):
        prepare_bugsinpy_development(
            bugsinpy,
            tmp_path / "mismatch-output",
            tmp_path / "mismatch-cache",
            mismatched_root,
            allow_network=True,
        )

    protocol_root = _root_with_locked_benchmark(tmp_path, fixture_commit)
    (bugsinpy / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be clean"):
        prepare_bugsinpy_development(
            bugsinpy,
            tmp_path / "dirty-output",
            tmp_path / "dirty-cache",
            protocol_root,
            allow_network=True,
        )


def test_runtime_collector_emits_typed_per_test_events(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "src" / "core.py").write_text(
        "def explode():\n    value = 1\n    assert value == 2\n",
        encoding="utf-8",
    )
    (repository / "tests" / "test_core.py").write_text(
        "from src.core import explode\n\ndef test_failure():\n    explode()\n",
        encoding="utf-8",
    )
    (repository / "docs").mkdir()
    (repository / "docs" / "generated.md").symlink_to("_build/generated.md")
    auxiliary = tmp_path / "auxiliary"
    auxiliary.mkdir()
    for name, value in (
        ("fix.patch", "diff --git a/src/core.py b/src/core.py\n"),
        ("before.json", "{}\n"),
        ("after.json", "{}\n"),
    ):
        (auxiliary / name).write_text(value, encoding="utf-8")
    test_id = "tests/test_core.py::test_failure"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "fixture-1",
                "repository": "fixture/project",
                "buggy_revision": "buggy",
                "repository_root": str(repository),
                "failing_tests": [test_id],
                "split": "development",
                "patch_path": str(auxiliary / "fix.patch"),
                "before_sources_path": str(auxiliary / "before.json"),
                "after_sources_path": str(auxiliary / "after.json"),
                "runtime_events_path": str(auxiliary / "pending.jsonl"),
                "runtime_evidence_status": "PENDING_COLLECTION",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "fixture-1": {
                    "incident_id": "fixture-1",
                    "repository": "fixture/project",
                    "commands": [
                        {
                            "test_id": test_id,
                            "argv": ["pytest", "-q", test_id],
                        }
                    ],
                    "setup_script": "",
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "python_executable": sys.executable,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = collect_h10_c5c_runtime(
        manifest,
        registry,
        tmp_path / "runtime",
        timeout_seconds=60,
    )
    assert result.complete_incidents == 1
    enriched = json.loads(
        result.enriched_manifest_path.read_text(encoding="utf-8").strip()
    )
    assert enriched["runtime_evidence_status"] == "BUG_REPRODUCED_WITH_TRACE"
    assert not Path(enriched["runtime_events_path"]).is_absolute()
    events = load_runtime_events(
        result.enriched_manifest_path.parent / enriched["runtime_events_path"]
    )
    assert {event.kind for event in events} >= {"coverage", "traceback_frame"}
    assert {event.test_id for event in events} == {test_id}
    assert not any("patch" in event.detail.lower() for event in events)


def test_runtime_collector_isolates_unsafe_symlink_failure(
    tmp_path: Path,
) -> None:
    good_repository = tmp_path / "good"
    good_repository.mkdir()
    (good_repository / "test_failure.py").write_text(
        "raise AssertionError('expected failure')\n",
        encoding="utf-8",
    )
    bad_repository = tmp_path / "bad"
    bad_repository.mkdir()
    (bad_repository / "test_failure.py").write_text(
        "raise AssertionError('must not execute')\n",
        encoding="utf-8",
    )
    (bad_repository / "escape").symlink_to("../outside-secret")
    (tmp_path / "outside-secret").write_text("private\n", encoding="utf-8")

    rows = [
        {
            "incident_id": "fixture-good",
            "repository": "fixture/good",
            "repository_root": str(good_repository),
            "failing_tests": ["test_failure.py"],
            "split": "development",
            "runtime_evidence_status": "PENDING_COLLECTION",
        },
        {
            "incident_id": "fixture-bad",
            "repository": "fixture/bad",
            "repository_root": str(bad_repository),
            "failing_tests": ["test_failure.py"],
            "split": "development",
            "runtime_evidence_status": "PENDING_COLLECTION",
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    registered_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                row["incident_id"]: {
                    "python_version": registered_minor,
                    "python_executable": sys.executable,
                    "commands": [
                        {
                            "test_id": "test_failure.py",
                            "argv": ["python", "test_failure.py"],
                        }
                    ],
                    "setup_script": "",
                }
                for row in rows
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = collect_h10_c5c_runtime(
        manifest,
        registry,
        tmp_path / "runtime",
        timeout_seconds=30,
        max_workers=2,
    )
    assert result.complete_incidents == 1
    enriched = [
        json.loads(line)
        for line in result.enriched_manifest_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["incident_id"] for row in enriched] == [
        "fixture-good",
        "fixture-bad",
    ]
    assert enriched[0]["runtime_evidence_status"] == "BUG_REPRODUCED_WITH_TRACE"
    assert enriched[1]["runtime_evidence_status"] == "RUNTIME_INFRASTRUCTURE_ERROR"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    failed = report["evidence"][1]
    assert failed["error_type"] == "ValueError"
    assert "escapes the snapshot" in failed["error_message"]
    assert failed["gold_fields_present"] is False


def test_runtime_collector_supports_parallel_incident_collection(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "test_failure.py").write_text(
        "def fail():\n    raise AssertionError('expected failure')\n\nfail()\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "incident_id": f"fixture-parallel-{index}",
            "repository": "fixture/project",
            "repository_root": str(repository),
            "failing_tests": ["test_failure.py"],
            "split": "development",
            "runtime_evidence_status": "PENDING_COLLECTION",
        }
        for index in range(2)
    ]
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    registered_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                row["incident_id"]: {
                    "python_version": registered_minor,
                    "python_executable": sys.executable,
                    "commands": [
                        {
                            "test_id": "test_failure.py",
                            "argv": ["python", "test_failure.py"],
                        }
                    ],
                    "setup_script": "",
                }
                for row in rows
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = collect_h10_c5c_runtime(
        manifest,
        registry,
        tmp_path / "runtime",
        timeout_seconds=60,
        max_workers=2,
    )
    assert result.complete_incidents == 2
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["max_workers"] == 2
    assert all(
        item["status"] == "BUG_REPRODUCED_WITH_TRACE"
        for item in report["evidence"]
    )


def test_runtime_collector_prepares_isolated_environment_when_enabled(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "test_failure.py").write_text(
        "from setup_marker import READY\n"
        "assert READY is True\n"
        "raise AssertionError('expected benchmark failure')\n",
        encoding="utf-8",
    )
    setup_script = tmp_path / "setup.sh"
    setup_script.write_text(
        "printf 'READY = True\\n' > setup_marker.py\n",
        encoding="utf-8",
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("# intentionally empty\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "fixture-environment",
                "repository": "fixture/project",
                "repository_root": str(repository),
                "failing_tests": ["test_failure.py"],
                "split": "development",
                "runtime_evidence_status": "PENDING_COLLECTION",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registered_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    registry.write_text(
        json.dumps(
            {
                "fixture-environment": {
                    "python_version": registered_minor,
                    "python_executable": sys.executable,
                    "commands": [
                        {
                            "test_id": "test_failure.py",
                            "argv": ["python", "test_failure.py"],
                        }
                    ],
                    "setup_script": str(setup_script),
                    "requirements_path": str(requirements),
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = collect_h10_c5c_runtime(
        manifest,
        registry,
        tmp_path / "runtime",
        timeout_seconds=60,
        allow_setup=True,
    )
    assert result.complete_incidents == 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    evidence = report["evidence"][0]
    assert evidence["setup"]["status"] == "PASS"
    assert evidence["setup"]["requirements"]["status"] == "EMPTY"
    assert evidence["setup"]["setup_script"]["status"] == "PASS"
    assert evidence["python_runtime_exact"] is True


def test_runtime_collector_retains_failed_setup_diagnostics(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "test_failure.py").write_text(
        "raise AssertionError('must not execute after setup failure')\n",
        encoding="utf-8",
    )
    setup_script = tmp_path / "setup.sh"
    setup_script.write_text(
        "printf 'registered setup failed\\n' >&2\nexit 7\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "fixture-setup-failure",
                "repository": "fixture/project",
                "repository_root": str(repository),
                "failing_tests": ["test_failure.py"],
                "split": "development",
                "runtime_evidence_status": "PENDING_COLLECTION",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "fixture-setup-failure": {
                    "python_version": (
                        f"{sys.version_info.major}.{sys.version_info.minor}"
                    ),
                    "python_executable": sys.executable,
                    "commands": [
                        {
                            "test_id": "test_failure.py",
                            "argv": ["python", "test_failure.py"],
                        }
                    ],
                    "setup_script": str(setup_script),
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = collect_h10_c5c_runtime(
        manifest,
        registry,
        tmp_path / "runtime",
        timeout_seconds=30,
        allow_setup=True,
    )
    assert result.complete_incidents == 0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    evidence = report["evidence"][0]
    assert evidence["status"] == "ENVIRONMENT_SETUP_FAILED"
    assert evidence["setup"]["failure_stage"] == "setup_script"
    assert (
        "registered setup failed"
        in evidence["setup"]["setup_script"]["stderr_tail"]
    )


def test_runtime_collector_benchmark_strategy_records_requirement_failures(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "test_failure.py").write_text(
        "from setup_marker import READY\n"
        "assert READY is True\n"
        "raise AssertionError('expected benchmark failure')\n",
        encoding="utf-8",
    )
    setup_script = tmp_path / "setup.sh"
    setup_script.write_text(
        "printf 'READY = True\\n' > setup_marker.py\n",
        encoding="utf-8",
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("./missing-local-package\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "fixture-partial-requirements",
                "repository": "fixture/project",
                "repository_root": str(repository),
                "failing_tests": ["test_failure.py"],
                "split": "development",
                "runtime_evidence_status": "PENDING_COLLECTION",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "fixture-partial-requirements": {
                    "python_version": (
                        f"{sys.version_info.major}.{sys.version_info.minor}"
                    ),
                    "python_executable": sys.executable,
                    "commands": [
                        {
                            "test_id": "test_failure.py",
                            "argv": ["python", "test_failure.py"],
                        }
                    ],
                    "setup_script": str(setup_script),
                    "requirements_path": str(requirements),
                    "requirements_install_strategy": (
                        "BENCHMARK_PER_LINE_TWO_PASS"
                    ),
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = collect_h10_c5c_runtime(
        manifest,
        registry,
        tmp_path / "runtime",
        timeout_seconds=30,
        allow_setup=True,
    )
    assert result.complete_incidents == 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    evidence = report["evidence"][0]
    assert evidence["status"] == "BUG_REPRODUCED_WITH_TRACE"
    assert evidence["setup"]["status"] == "PASS"
    assert evidence["setup"]["requirements"]["status"] == "PARTIAL"
    assert evidence["setup"]["requirements"]["pre_setup"]["failed_count"] == 1
    assert evidence["setup"]["requirements"]["post_setup"]["failed_count"] == 1


def test_runtime_collector_fails_closed_on_missing_registered_python(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "test_failure.py").write_text(
        "raise AssertionError('expected failure')\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "fixture-runtime-missing",
                "repository": "fixture/project",
                "repository_root": str(repository),
                "failing_tests": ["test_failure.py"],
                "split": "development",
                "runtime_evidence_status": "PENDING_COLLECTION",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "fixture-runtime-missing": {
                    "python_version": "9.9",
                    "python_executable": "python9.9",
                    "commands": [
                        {
                            "test_id": "test_failure.py",
                            "argv": ["python", "test_failure.py"],
                        }
                    ],
                    "setup_script": "",
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = collect_h10_c5c_runtime(
        manifest,
        registry,
        tmp_path / "runtime",
        timeout_seconds=30,
    )
    assert result.complete_incidents == 0
    enriched = json.loads(
        result.enriched_manifest_path.read_text(encoding="utf-8").strip()
    )
    assert enriched["runtime_evidence_status"] == "PYTHON_RUNTIME_UNAVAILABLE"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["all_python_runtimes_exact"] is False
    assert report["evidence"][0]["commands"][0]["python_version_match"] is False


def test_development_readiness_requires_complete_locked_inputs(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime_events.jsonl"
    test_id = "tests/test_core.py::test_failure"
    runtime_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                {
                    "event_id": "coverage-1",
                    "test_id": test_id,
                    "kind": "coverage",
                    "source_file": "src/core.py",
                    "source_symbol": "explode",
                    "target_file": None,
                    "target_symbol": None,
                    "detail": "function entered",
                },
                {
                    "event_id": "trace-1",
                    "test_id": test_id,
                    "kind": "traceback_frame",
                    "source_file": "src/core.py",
                    "source_symbol": "explode",
                    "target_file": None,
                    "target_symbol": None,
                    "detail": "src/core.py:3",
                },
            )
        ),
        encoding="utf-8",
    )
    patch_path = tmp_path / "fix.patch"
    patch_path.write_text("diff --git a/src/core.py b/src/core.py\n", encoding="utf-8")
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text("{}\n", encoding="utf-8")
    after_path.write_text("{}\n", encoding="utf-8")
    repository_root = tmp_path / "repository"
    (repository_root / "tests").mkdir(parents=True)
    exposing_test = repository_root / "tests" / "test_core.py"
    exposing_test.write_text("def test_failure():\n    assert False\n", encoding="utf-8")
    exposing_test_hash = hashlib.sha256(exposing_test.read_bytes()).hexdigest()

    manifest_rows = []
    command_registry = {}
    source_rows = []
    for index in range(30):
        repository = f"owner{index % 8}/project{index % 8}"
        incident_id = f"fixture-{index:02d}"
        manifest_rows.append(
            {
                "incident_id": incident_id,
                "repository": repository,
                "buggy_revision": "buggy",
                "repository_root": str(repository_root),
                "failing_tests": [test_id],
                "split": "development",
                "patch_path": str(patch_path),
                "before_sources_path": str(before_path),
                "after_sources_path": str(after_path),
                "runtime_events_path": str(runtime_path),
                "runtime_evidence_status": "BUG_REPRODUCED_WITH_TRACE",
            }
        )
        command_registry[incident_id] = {
            "commands": [{"test_id": test_id, "argv": ["pytest", "-q", test_id]}],
            "build_toolchain": {
                "pip": "20.3.4",
                "setuptools": "44.1.1",
                "wheel": "0.36.2",
            },
            "requirements_install_options": [
                "--use-deprecated=legacy-resolver",
                "--no-build-isolation",
            ],
            "requirements_install_strategy": "BENCHMARK_PER_LINE_TWO_PASS",
        }
        source_rows.append(
            {
                "incident_id": incident_id,
                "repository": repository,
                "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
                "exposing_test_overlays": [
                    {
                        "path": "tests/test_core.py",
                        "fixed_test_sha256": exposing_test_hash,
                        "materialized_test_sha256": exposing_test_hash,
                    }
                ],
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    command_path = tmp_path / "commands.json"
    command_path.write_text(json.dumps(command_registry), encoding="utf-8")
    source_path = tmp_path / "sources.json"
    source_path.write_text(
        json.dumps(
            {
                "collection_id": "h10-c5c-bugsinpy-development-v1",
                "bugsinpy_commit": LOCKED_BUGSINPY_COMMIT,
                "incidents": source_rows,
            }
        ),
        encoding="utf-8",
    )
    runtime_report = tmp_path / "runtime_report.json"
    runtime_report.write_text(
        json.dumps(
            {
                "status": "DEVELOPMENT_RUNTIME_COMPLETE",
                "scientific_result": "NOT_EVALUATED",
                "total_incidents": 30,
                "complete_incidents": 30,
                "enriched_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "command_registry_sha256": hashlib.sha256(
                    command_path.read_bytes()
                ).hexdigest(),
                "all_python_runtimes_exact": True,
                "evidence": [
                    {
                        "incident_id": f"fixture-{index:02d}",
                        "python_runtime_exact": True,
                    }
                    for index in range(30)
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "readiness.json"
    result = verify_h10_c5c_development_readiness(
        manifest,
        command_path,
        source_path,
        runtime_report,
        output,
        ROOT,
    )
    assert result.status == "H10_C5C_DEVELOPMENT_READINESS_PASS"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scientific_result"] == "NOT_EVALUATED"
    assert report["development_scored"] is False
    assert all(report["checks"].values())
    assert all(
        row["runtime_compatibility_matches"] for row in report["per_incident"]
    )

    command_payload = json.loads(command_path.read_text(encoding="utf-8"))
    command_payload["fixture-00"]["build_toolchain"]["pip"] = "24.0"
    command_path.write_text(json.dumps(command_payload), encoding="utf-8")
    failed_toolchain = verify_h10_c5c_development_readiness(
        manifest,
        command_path,
        source_path,
        runtime_report,
        tmp_path / "readiness-wrong-toolchain.json",
        ROOT,
    )
    assert failed_toolchain.status == "H10_C5C_DEVELOPMENT_READINESS_FAIL"
    toolchain_report = json.loads(
        failed_toolchain.report_path.read_text(encoding="utf-8")
    )
    assert toolchain_report["checks"]["all_command_records_present"] is False
    assert (
        toolchain_report["per_incident"][0]["runtime_compatibility_matches"]
        is False
    )
    command_path.write_text(json.dumps(command_registry), encoding="utf-8")

    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload["bugsinpy_commit"] = "0" * 40
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")
    failed = verify_h10_c5c_development_readiness(
        manifest,
        command_path,
        source_path,
        runtime_report,
        tmp_path / "readiness-wrong-commit.json",
        ROOT,
    )
    assert failed.status == "H10_C5C_DEVELOPMENT_READINESS_FAIL"
    failed_report = json.loads(failed.report_path.read_text(encoding="utf-8"))
    assert failed_report["checks"]["bugsinpy_commit_matches_lock"] is False


def test_development_workflow_uses_only_the_locked_bugsinpy_commit() -> None:
    workflow = (ROOT / ".github/workflows/h10-c5c-development.yml").read_text(
        encoding="utf-8"
    )
    assert f"default: {LOCKED_BUGSINPY_COMMIT}" in workflow
    assert "default: master" not in workflow
    assert 'test "${#BUGSINPY_REF}" -eq 40' in workflow
    assert 'test "$BUGSINPY_REF" = "$locked_ref"' in workflow
    assert "h10-c5c-development-summary-${{ github.run_id }}" in workflow
    assert (
        "sha256sum h10-c5c-development-evidence.tgz"
        " > h10-c5c-development-evidence.tgz.sha256"
    ) in workflow
