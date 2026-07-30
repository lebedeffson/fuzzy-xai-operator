from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fuzzyxai.repository_diagnostics.runtime_events import (
    normalize_runtime_event_rows,
)

_TRACEBACK_FILE = re.compile(r'^\s*File "([^"]+)", line \d+, in ([^\n]+)\s*$')
_PYTEST_FRAME = re.compile(r"^\s*(.+?\.py):(\d+): in ([^\n]+)\s*$")
_PYTEST_LOCATION = re.compile(r"^\s*(.+?\.py):(\d+):(?:\s|$)")


@dataclass(frozen=True)
class RuntimeCollectionResult:
    enriched_manifest_path: Path
    report_path: Path
    complete_incidents: int
    total_incidents: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(base: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _validate_internal_symlinks(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = Path(os.readlink(path))
        if target.is_absolute():
            raise ValueError(f"absolute repository symlink is forbidden: {path}")
        resolved_target = (path.parent / target).resolve(strict=False)
        if not resolved_target.is_relative_to(resolved_root):
            raise ValueError(f"repository symlink escapes the snapshot: {path}")


def _normalize_command(argv: tuple[str, ...]) -> tuple[str, ...]:
    if not argv:
        raise ValueError("runtime command is empty")
    return argv


def _normalized_minor(value: object) -> str:
    match = re.match(r"\s*(\d+)\.(\d+)", str(value or ""))
    return f"{match.group(1)}.{match.group(2)}" if match else ""


def _interpreter_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-c", "import sys; print('{}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _select_interpreter(
    registered: dict[str, object],
    argv: tuple[str, ...],
    interpreter_map: Mapping[str, str] | None = None,
) -> tuple[str, str, str]:
    requested_version = _normalized_minor(registered.get("python_version"))
    requested = str(registered.get("python_executable", "")).strip()
    executable = Path(argv[0]).name if argv else ""
    candidates: list[str] = []
    if requested_version and interpreter_map is not None:
        mapped = str(interpreter_map.get(requested_version, "")).strip()
        if mapped:
            candidates.append(mapped)
    if requested:
        candidates.append(requested)
    if requested_version:
        candidates.extend((f"python{requested_version}", f"python{requested_version[0]}"))
    if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable):
        candidates.append(argv[0])
    if not requested_version:
        candidates.append(sys.executable)
    checked: list[str] = []
    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved is None:
            path = Path(candidate)
            resolved = str(path.resolve()) if path.is_file() else None
        if resolved is None:
            checked.append(candidate)
            continue
        version = _interpreter_version(resolved)
        resolved_minor = _normalized_minor(version)
        if requested_version and resolved_minor != requested_version:
            checked.append(f"{resolved}={version}")
            continue
        return resolved, version, requested_version
    detail = ", ".join(checked) if checked else "none"
    raise FileNotFoundError(
        "no exact Python interpreter is available for registered command "
        f"(requested={requested_version or 'unspecified'}, checked={detail}, argv={argv})"
    )


def _launcher_source(
    root: Path,
    event_dir: Path,
    test_id: str,
    argv: tuple[str, ...],
) -> str:
    return textwrap.dedent(
        f"""
        import atexit
        import collections
        import hashlib
        import itertools
        import json
        import os
        import runpy
        import sys
        import threading
        import time

        ROOT = os.path.realpath({str(root)!r})
        EVENT_DIR = {str(event_dir)!r}
        TEST_ID = {test_id!r}
        ARGV = {list(argv)!r}
        TAIL_LIMIT = 20000
        EVENTS = collections.deque()
        PREFIX_AGGREGATES = {{}}
        SEQUENCE = itertools.count()
        EVENT_LOCK = threading.Lock()
        DEPTH = threading.local()
        LAST_WRITER = {{}}

        def _relative(filename):
            if not filename or filename.startswith('<'):
                return None
            real = os.path.realpath(filename)
            try:
                if os.path.commonpath((ROOT, real)) != ROOT:
                    return None
            except ValueError:
                return None
            relative = os.path.relpath(real, ROOT).replace(os.sep, '/')
            if relative.startswith('../'):
                return None
            if any(
                part in {{
                    '.tox',
                    '.venv',
                    '__pycache__',
                    'dist-packages',
                    'site-packages',
                    'venv',
                }}
                for part in relative.split('/')
            ):
                return None
            return relative

        def _symbol(frame):
            value = getattr(frame.f_code, 'co_qualname', frame.f_code.co_name)
            return value.replace('.<locals>.', '.') if value not in ('<module>', '') else None

        def _safe_value(value):
            type_name = type(value).__module__ + '.' + type(value).__qualname__
            try:
                preview = repr(value)
            except Exception as error:
                preview = '<repr failed: ' + type(error).__name__ + '>'
            preview = preview[:160]
            encoded = (type_name + '\\0' + preview).encode('utf-8', errors='replace')
            payload = {{
                'type': type_name,
                'digest': 'sha256:' + hashlib.sha256(encoded).hexdigest(),
                'preview': preview,
                'object_id': 'run:' + hashlib.sha256(
                    (str(id(value)) + '\\0' + type_name).encode()
                ).hexdigest()[:20],
            }}
            try:
                payload['length'] = len(value)
            except Exception:
                pass
            shape = None
            if type_name.startswith(('numpy.', 'pandas.', 'torch.')):
                try:
                    shape = object.__getattribute__(value, 'shape')
                except Exception:
                    pass
            if shape is not None:
                try:
                    payload['shape'] = [int(item) for item in shape]
                except Exception:
                    payload['shape'] = str(shape)[:80]
            if isinstance(value, dict):
                payload['fields'] = sorted(str(key)[:80] for key in value)[:32]
            return payload

        def _prefix_record(event):
            key = event.pop('_aggregate_key')
            existing = PREFIX_AGGREGATES.get(key)
            if existing is None:
                encoded = repr(key).encode('utf-8', errors='replace')
                existing = dict(event)
                existing['event_id'] = 'aggregate-' + hashlib.sha256(encoded).hexdigest()[:24]
                existing['occurrence_count'] = 0
                existing['first_sequence_id'] = event['sequence_id']
                PREFIX_AGGREGATES[key] = existing
            existing['occurrence_count'] += event.get('occurrence_count', 1)
            existing['last_sequence_id'] = event['sequence_id']
            existing['sequence_id'] = existing['first_sequence_id']

        def _emit(kind, source_file, source_symbol=None, target_file=None, target_symbol=None, detail=''):
            key = (TEST_ID, kind, source_file, source_symbol, target_file, target_symbol, detail)
            with EVENT_LOCK:
                sequence_id = next(SEQUENCE)
                timestamp_ns = time.monotonic_ns()
                thread_id = threading.get_ident()
                call_depth = max(0, int(getattr(DEPTH, 'value', 0)))
                encoded = (
                    '\\0'.join('' if value is None else str(value) for value in key)
                    + '\\0' + str(sequence_id)
                    + '\\0' + str(timestamp_ns)
                ).encode('utf-8', errors='replace')
                event = {{
                    'event_id': 'probe-' + hashlib.sha256(encoded).hexdigest()[:24],
                    'sequence_id': sequence_id,
                    'timestamp_ns': timestamp_ns,
                    'thread_id': thread_id,
                    'call_depth': call_depth,
                    'test_id': TEST_ID,
                    'kind': kind,
                    'source_file': source_file,
                    'source_symbol': source_symbol,
                    'target_file': target_file,
                    'target_symbol': target_symbol,
                    'occurrence_count': 1,
                    'first_sequence_id': sequence_id,
                    'last_sequence_id': sequence_id,
                    'detail': detail,
                    '_aggregate_key': key,
                }}
                if len(EVENTS) >= TAIL_LIMIT:
                    _prefix_record(EVENTS.popleft())
                EVENTS.append(event)
                return event

        def _value_event(kind, frame, value, source_file, source_symbol):
            payload = _safe_value(value)
            detail = json.dumps(payload, sort_keys=True)
            event = _emit(kind, source_file, source_symbol, detail=detail)
            object_id = payload['object_id']
            writer = LAST_WRITER.get(object_id)
            if writer is not None and writer[:2] != (source_file, source_symbol):
                writer_file, writer_symbol, writer_sequence = writer
                flow = dict(payload)
                flow['writer_sequence_id'] = writer_sequence
                flow_detail = json.dumps(flow, sort_keys=True)
                _emit(
                    'last_writer',
                    writer_file,
                    writer_symbol,
                    source_file,
                    source_symbol,
                    flow_detail,
                )
                _emit(
                    'value_flow',
                    writer_file,
                    writer_symbol,
                    source_file,
                    source_symbol,
                    flow_detail,
                )
            return payload, event

        def _profile(frame, event, arg):
            target_file = _relative(frame.f_code.co_filename)
            if event == 'call':
                depth = max(0, int(getattr(DEPTH, 'value', 0)))
                DEPTH.value = depth + 1
                if target_file is None:
                    return
                target_symbol = _symbol(frame)
                _emit('coverage', target_file, target_symbol, detail='function entered')
                caller = frame.f_back
                if caller is not None:
                    source_file = _relative(caller.f_code.co_filename)
                    if source_file is not None:
                        _emit(
                            'call',
                            source_file,
                            _symbol(caller),
                            target_file,
                            target_symbol,
                            'runtime call',
                        )
                argument_count = (
                    frame.f_code.co_argcount
                    + frame.f_code.co_kwonlyargcount
                )
                for name in frame.f_code.co_varnames[:argument_count][:12]:
                    if name in frame.f_locals:
                        _value_event(
                            'argument_value',
                            frame,
                            frame.f_locals[name],
                            target_file,
                            target_symbol,
                        )
                return
            if event == 'return':
                DEPTH.value = max(0, int(getattr(DEPTH, 'value', 1)) - 1)
                if target_file is None:
                    return
                target_symbol = _symbol(frame)
                payload, value_event = _value_event(
                    'return_value',
                    frame,
                    arg,
                    target_file,
                    target_symbol,
                )
                LAST_WRITER[payload['object_id']] = (
                    target_file,
                    target_symbol,
                    value_event['sequence_id'],
                )
                return
        def _trace(frame, event, arg):
            if event != 'exception':
                return _trace
            target_file = _relative(frame.f_code.co_filename)
            if target_file is None:
                return _trace
            target_symbol = _symbol(frame)
            exception = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else arg
            _value_event(
                'exception',
                frame,
                exception,
                target_file,
                target_symbol,
            )
            if type(exception).__name__ == 'AssertionError':
                for name, value in list(frame.f_locals.items())[:12]:
                    operand = _safe_value(value)
                    operand['name'] = name
                    _emit(
                        'assertion_operand',
                        target_file,
                        target_symbol,
                        detail=json.dumps(operand, sort_keys=True),
                    )
            return _trace

        def _flush():
            if not EVENTS and not PREFIX_AGGREGATES:
                return
            os.makedirs(EVENT_DIR, exist_ok=True)
            path = os.path.join(EVENT_DIR, 'probe-' + str(os.getpid()) + '.jsonl')
            with open(path, 'w', encoding='utf-8') as stream:
                values = [
                    *sorted(
                        PREFIX_AGGREGATES.values(),
                        key=lambda item: item['first_sequence_id'],
                    ),
                    *list(EVENTS),
                ]
                for value in values:
                    value.pop('_aggregate_key', None)
                    stream.write(json.dumps(value, sort_keys=True) + '\\n')

        def _execute():
            if not ARGV:
                raise SystemExit(2)
            executable = os.path.basename(ARGV[0])
            if executable in ('pytest', 'py.test'):
                sys.argv = [executable] + ARGV[1:]
                runpy.run_module('pytest', run_name='__main__')
                return
            if executable in ('nosetests', 'nose2'):
                module = 'nose' if executable == 'nosetests' else 'nose2'
                sys.argv = [executable] + ARGV[1:]
                runpy.run_module(module, run_name='__main__')
                return
            if executable.startswith('python'):
                if len(ARGV) >= 3 and ARGV[1] == '-m':
                    sys.argv = [ARGV[2]] + ARGV[3:]
                    runpy.run_module(ARGV[2], run_name='__main__', alter_sys=True)
                    return
                if len(ARGV) >= 2:
                    sys.argv = ARGV[1:]
                    runpy.run_path(ARGV[1], run_name='__main__')
                    return
            if ARGV[0].endswith('.py'):
                sys.argv = ARGV
                runpy.run_path(ARGV[0], run_name='__main__')
                return
            raise SystemExit('unsupported instrumented command: ' + ' '.join(ARGV))

        sys.setprofile(_profile)
        threading.setprofile(_profile)
        sys.settrace(_trace)
        threading.settrace(_trace)
        atexit.register(_flush)
        _execute()
        """
    ).lstrip()


def _project_relative(path_text: str, root: Path) -> str | None:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if not candidate.exists() and not (root / relative).exists():
        return None
    return relative.as_posix()


def _traceback_events(
    text: str,
    *,
    root: Path,
    test_id: str,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    seen: set[tuple[str, str | None]] = set()
    for raw in text.splitlines():
        path_text: str | None = None
        symbol: str | None = None
        match = _TRACEBACK_FILE.match(raw)
        if match is not None:
            path_text, symbol = match.groups()
        else:
            match = _PYTEST_FRAME.match(raw)
            if match is not None:
                path_text, _line, symbol = match.groups()
            else:
                match = _PYTEST_LOCATION.match(raw)
                if match is not None:
                    path_text = match.group(1)
        if path_text is None:
            continue
        relative = _project_relative(path_text, root)
        if relative is None:
            continue
        normalized_symbol = symbol.strip() if symbol else None
        key = (relative, normalized_symbol)
        if key in seen:
            continue
        seen.add(key)
        encoded = f"{test_id}\0traceback_frame\0{relative}\0{normalized_symbol or ''}".encode()
        events.append(
            {
                "event_id": "trace-" + hashlib.sha256(encoded).hexdigest()[:24],
                "sequence_id": len(events),
                "timestamp_ns": time.monotonic_ns(),
                "thread_id": threading.get_ident(),
                "call_depth": 0,
                "test_id": test_id,
                "kind": "traceback_frame",
                "source_file": relative,
                "source_symbol": normalized_symbol,
                "target_file": None,
                "target_symbol": None,
                "occurrence_count": 1,
                "detail": raw.strip(),
            }
        )
    return events


def _assertion_difference(text: str) -> str:
    selected = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith(("E ", "E\t")):
            selected.append(stripped[1:].strip())
        elif stripped.startswith(("AssertionError", "assert ")):
            selected.append(stripped)
    return "\n".join(dict.fromkeys(selected))


def _failure_observation_events(
    text: str,
    *,
    test_id: str,
    traceback_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Convert pytest failure text into typed, safely summarized observations."""
    if not traceback_events:
        return []
    source = traceback_events[-1]
    source_file = str(source["source_file"])
    source_symbol = source.get("source_symbol")
    assertion = _assertion_difference(text)
    exception_type = "Exception"
    exception_preview = ""
    for raw in reversed(text.splitlines()):
        stripped = raw.strip()
        match = re.match(
            r"^(?:E\s+)?([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))"
            r"(?::\s*(.*))?$",
            stripped,
        )
        if match:
            exception_type = match.group(1)
            exception_preview = (match.group(2) or "")[:160]
            break

    observations = [
        _text_observation_event(
            test_id,
            "exception",
            source_file,
            source_symbol,
            exception_type,
            exception_preview,
        )
    ]
    if assertion:
        observations.append(
            _text_observation_event(
                test_id,
                "assertion_operand",
                source_file,
                source_symbol,
                "pytest_assertion_difference",
                assertion[:320],
            )
        )
    return observations


def _text_observation_event(
    test_id: str,
    kind: str,
    source_file: str,
    source_symbol: object,
    value_type: str,
    preview: str,
) -> dict[str, object]:
    detail = {
        "type": value_type,
        "preview": preview,
        "digest": "sha256:"
        + hashlib.sha256(preview.encode("utf-8", errors="replace")).hexdigest(),
    }
    encoded = (
        f"{test_id}\0{kind}\0{source_file}\0{source_symbol}\0"
        f"{detail['digest']}"
    ).encode()
    return {
        "event_id": "failure-" + hashlib.sha256(encoded).hexdigest()[:24],
        "sequence_id": 0,
        "timestamp_ns": time.monotonic_ns(),
        "thread_id": threading.get_ident(),
        "call_depth": 0,
        "test_id": test_id,
        "kind": kind,
        "source_file": source_file,
        "source_symbol": source_symbol,
        "target_file": None,
        "target_symbol": None,
        "occurrence_count": 1,
        "detail": json.dumps(detail, sort_keys=True),
    }


def _merge_probe_events(event_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(event_dir.glob("probe-*.jsonl")):
        rows.extend(_load_jsonl(path))
    rows.sort(
        key=lambda row: (
            int(row.get("timestamp_ns", 0)) <= 0,
            int(row.get("timestamp_ns", 0)),
            int(row.get("sequence_id", -1)),
        )
    )
    return normalize_runtime_event_rows(rows)


def _run_logged(
    arguments: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            env=dict(environment),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return {
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            "stdout_tail": completed.stdout[-8000:],
            "stderr_tail": completed.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        return {
            "status": "TIMEOUT",
            "returncode": 124,
            "timed_out": True,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            "stdout_tail": stdout[-8000:],
            "stderr_tail": stderr[-8000:],
        }


def _environment_python(environment_root: Path) -> Path:
    candidates = (
        environment_root / "bin" / "python",
        environment_root / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"virtual environment Python is absent: {environment_root}")


def _requirements_have_content(path: Path) -> bool:
    if not path.is_file():
        return False
    return any(
        line.strip() and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    )


def _prepare_environment(
    base_interpreter: str,
    registered: dict[str, object],
    sandbox: Path,
    environment_root: Path,
    *,
    timeout_seconds: int,
    allow_setup: bool,
) -> tuple[str | None, dict[str, object]]:
    base_version = _interpreter_version(base_interpreter)
    if not allow_setup:
        return base_interpreter, {
            "status": "DISABLED",
            "base_python_executable": base_interpreter,
            "base_python_version": base_version,
            "runtime_python_executable": base_interpreter,
            "runtime_python_version": base_version,
            "venv": {"status": "NOT_RUN"},
            "requirements": {"status": "NOT_RUN"},
            "setup_script": {"status": "NOT_RUN"},
        }

    environment = os.environ.copy()
    venv_result = _run_logged(
        [base_interpreter, "-m", "venv", str(environment_root)],
        cwd=sandbox,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    if venv_result["status"] != "PASS":
        return None, {
            "status": "FAIL",
            "failure_stage": "venv",
            "base_python_executable": base_interpreter,
            "base_python_version": base_version,
            "venv": venv_result,
            "requirements": {"status": "NOT_RUN"},
            "setup_script": {"status": "NOT_RUN"},
        }
    try:
        runtime_python = _environment_python(environment_root)
        runtime_version = _interpreter_version(str(runtime_python))
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return None, {
            "status": "FAIL",
            "failure_stage": "venv_python",
            "error": str(error),
            "base_python_executable": base_interpreter,
            "base_python_version": base_version,
            "venv": venv_result,
            "requirements": {"status": "NOT_RUN"},
            "setup_script": {"status": "NOT_RUN"},
        }
    requested_minor = _normalized_minor(registered.get("python_version"))
    if requested_minor and _normalized_minor(runtime_version) != requested_minor:
        return None, {
            "status": "FAIL",
            "failure_stage": "venv_python_version",
            "base_python_executable": base_interpreter,
            "base_python_version": base_version,
            "runtime_python_executable": str(runtime_python),
            "runtime_python_version": runtime_version,
            "requested_python_version": requested_minor,
            "venv": venv_result,
            "requirements": {"status": "NOT_RUN"},
            "setup_script": {"status": "NOT_RUN"},
        }

    runtime_environment = os.environ.copy()
    runtime_environment["H10_C5C_PYTHON"] = str(runtime_python)
    runtime_environment["PYTHON"] = str(runtime_python)
    runtime_environment["PATH"] = (
        str(runtime_python.parent)
        + os.pathsep
        + runtime_environment.get("PATH", "")
    )
    runtime_environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    build_toolchain = registered.get("build_toolchain", {})
    if not isinstance(build_toolchain, dict):
        return None, {
            "status": "FAIL",
            "failure_stage": "build_toolchain_registry",
            "error": "build_toolchain must be an object",
            "venv": venv_result,
            "build_toolchain": {"status": "INVALID"},
            "requirements": {"status": "NOT_RUN"},
            "setup_script": {"status": "NOT_RUN"},
        }
    if build_toolchain:
        required_tools = ("pip", "setuptools", "wheel")
        if not all(str(build_toolchain.get(name, "")).strip() for name in required_tools):
            return None, {
                "status": "FAIL",
                "failure_stage": "build_toolchain_registry",
                "error": "pip, setuptools, and wheel versions must be pinned",
                "venv": venv_result,
                "build_toolchain": {"status": "INVALID"},
                "requirements": {"status": "NOT_RUN"},
                "setup_script": {"status": "NOT_RUN"},
            }
        toolchain_result = _run_logged(
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                *(f"{name}=={build_toolchain[name]}" for name in required_tools),
            ],
            cwd=sandbox,
            environment=runtime_environment,
            timeout_seconds=timeout_seconds,
        )
    else:
        toolchain_result = {"status": "NOT_REGISTERED"}
    if toolchain_result["status"] not in {"PASS", "NOT_REGISTERED"}:
        return None, {
            "status": "FAIL",
            "failure_stage": "build_toolchain",
            "base_python_executable": base_interpreter,
            "base_python_version": base_version,
            "runtime_python_executable": str(runtime_python),
            "runtime_python_version": runtime_version,
            "venv": venv_result,
            "build_toolchain": toolchain_result,
            "requirements": {"status": "NOT_RUN"},
            "setup_script": {"status": "NOT_RUN"},
        }
    requirements_path = Path(str(registered.get("requirements_path", "")))
    if _requirements_have_content(requirements_path):
        install_options = registered.get("requirements_install_options", [])
        if not isinstance(install_options, list) or not all(
            isinstance(option, str) for option in install_options
        ):
            return None, {
                "status": "FAIL",
                "failure_stage": "requirements_options_registry",
                "error": "requirements_install_options must be a string list",
                "venv": venv_result,
                "build_toolchain": toolchain_result,
                "requirements": {"status": "NOT_RUN"},
                "setup_script": {"status": "NOT_RUN"},
            }
        requirements_result = _run_logged(
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                *install_options,
                "-r",
                str(requirements_path),
            ],
            cwd=sandbox,
            environment=runtime_environment,
            timeout_seconds=timeout_seconds,
        )
    elif requirements_path and requirements_path.is_file():
        requirements_result = {"status": "EMPTY", "returncode": 0, "timed_out": False}
    else:
        requirements_result = {"status": "NOT_REGISTERED", "returncode": None, "timed_out": False}
    if requirements_result["status"] not in {"PASS", "EMPTY", "NOT_REGISTERED"}:
        return None, {
            "status": "FAIL",
            "failure_stage": "requirements",
            "base_python_executable": base_interpreter,
            "base_python_version": base_version,
            "runtime_python_executable": str(runtime_python),
            "runtime_python_version": runtime_version,
            "venv": venv_result,
            "build_toolchain": toolchain_result,
            "requirements": requirements_result,
            "setup_script": {"status": "NOT_RUN"},
        }

    setup_script = Path(str(registered.get("setup_script", "")))
    if setup_script.is_file():
        setup_result = _run_logged(
            ["bash", str(setup_script)],
            cwd=sandbox,
            environment=runtime_environment,
            timeout_seconds=timeout_seconds,
        )
    elif str(registered.get("setup_script", "")).strip():
        setup_result = {"status": "MISSING", "returncode": None, "timed_out": False}
    else:
        setup_result = {"status": "NOT_REGISTERED", "returncode": None, "timed_out": False}
    if setup_result["status"] not in {"PASS", "NOT_REGISTERED"}:
        return None, {
            "status": "FAIL",
            "failure_stage": "setup_script",
            "base_python_executable": base_interpreter,
            "base_python_version": base_version,
            "runtime_python_executable": str(runtime_python),
            "runtime_python_version": runtime_version,
            "venv": venv_result,
            "build_toolchain": toolchain_result,
            "requirements": requirements_result,
            "setup_script": setup_result,
        }
    return str(runtime_python), {
        "status": "PASS",
        "base_python_executable": base_interpreter,
        "base_python_version": base_version,
        "runtime_python_executable": str(runtime_python),
        "runtime_python_version": runtime_version,
        "venv": venv_result,
        "build_toolchain": toolchain_result,
        "requirements": requirements_result,
        "setup_script": setup_result,
    }


def _collect_incident(
    row: dict[str, object],
    registered: dict[str, object],
    output_root: Path,
    manifest_base: Path,
    *,
    timeout_seconds: int,
    allow_setup: bool,
    interpreter_map: Mapping[str, str] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    incident_id = str(row["incident_id"])
    source_root = _resolve(manifest_base, row["repository_root"])
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    incident_output = output_root / incident_id
    incident_output.mkdir(parents=True, exist_ok=True)
    _validate_internal_symlinks(source_root)
    with tempfile.TemporaryDirectory(prefix=f"h10-c5c-{incident_id}-") as temporary:
        temporary_root = Path(temporary)
        sandbox = temporary_root / "repository"
        shutil.copytree(
            source_root,
            sandbox,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git",
                ".tox",
                ".venv",
                "venv",
                "__pycache__",
                ".pytest_cache",
            ),
        )
        commands = registered.get("commands", ())
        if not isinstance(commands, list) or not commands:
            raise ValueError(f"runtime commands are absent for {incident_id}")
        first_command = commands[0]
        if not isinstance(first_command, dict) or not isinstance(
            first_command.get("argv"),
            list,
        ):
            raise TypeError(f"invalid runtime command for {incident_id}")
        setup_interpreter = None
        setup_resolution_error = ""
        try:
            setup_interpreter, _setup_version, _setup_minor = _select_interpreter(
                registered,
                tuple(str(item) for item in first_command["argv"]),
                interpreter_map,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            setup_resolution_error = str(error)
        if setup_interpreter is None:
            runtime_interpreter = None
            setup_result = {
                "status": "PYTHON_RUNTIME_UNAVAILABLE",
                "returncode": None,
                "error": setup_resolution_error,
            }
        else:
            runtime_interpreter, setup_result = _prepare_environment(
                setup_interpreter,
                registered,
                sandbox,
                temporary_root / "environment",
                timeout_seconds=timeout_seconds,
                allow_setup=allow_setup,
            )
        combined_stdout: list[str] = []
        combined_stderr: list[str] = []
        all_events: list[dict[str, object]] = []
        command_results = []
        for index, command_record in enumerate(commands):
            if not isinstance(command_record, dict):
                raise TypeError(f"invalid runtime command for {incident_id}")
            test_id = str(command_record.get("test_id", "")).strip()
            raw_argv = command_record.get("argv", ())
            if not test_id or not isinstance(raw_argv, list):
                raise ValueError(f"invalid runtime command for {incident_id}")
            argv = _normalize_command(tuple(str(item) for item in raw_argv))
            requested_version = _normalized_minor(registered.get("python_version"))
            if runtime_interpreter is None:
                runtime_status = (
                    "PYTHON_RUNTIME_UNAVAILABLE"
                    if setup_result.get("status") == "PYTHON_RUNTIME_UNAVAILABLE"
                    else "ENVIRONMENT_SETUP_FAILED"
                )
                command_results.append(
                    {
                        "test_id": test_id,
                        "argv": list(argv),
                        "returncode": None,
                        "timed_out": False,
                        "runtime_status": runtime_status,
                        "requested_python_version": requested_version,
                        "resolved_python_executable": setup_result.get(
                            "runtime_python_executable"
                        ),
                        "resolved_python_version": setup_result.get(
                            "runtime_python_version"
                        ),
                        "python_version_match": bool(
                            setup_result.get("runtime_python_version")
                            and _normalized_minor(
                                setup_result.get("runtime_python_version")
                            )
                            == requested_version
                        ),
                        "error": setup_result.get("error", setup_result.get("failure_stage", "")),
                        "event_count": 0,
                    }
                )
                continue
            interpreter = runtime_interpreter
            resolved_version = _interpreter_version(interpreter)
            requested_minor = requested_version
            probe_root = temporary_root / f"probe-{index}"
            event_dir = temporary_root / f"events-{index}"
            probe_root.mkdir()
            launcher = probe_root / "runtime_launcher.py"
            launcher.write_text(
                _launcher_source(sandbox, event_dir, test_id, argv),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            try:
                completed = subprocess.run(
                    [interpreter, str(launcher)],
                    cwd=sandbox,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_seconds,
                )
                timed_out = False
                stdout = completed.stdout
                stderr = completed.stderr
                returncode = completed.returncode
            except subprocess.TimeoutExpired as error:
                timed_out = True
                stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
                stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
                returncode = 124
            combined_stdout.append(stdout)
            combined_stderr.append(stderr)
            events = _merge_probe_events(event_dir)
            all_events.extend(events)
            failure_text = f"{stdout}\n{stderr}"
            traceback_events = _traceback_events(
                failure_text,
                root=sandbox,
                test_id=test_id,
            )
            all_events.extend(traceback_events)
            all_events.extend(
                _failure_observation_events(
                    failure_text,
                    test_id=test_id,
                    traceback_events=traceback_events,
                )
            )
            command_results.append(
                {
                    "test_id": test_id,
                    "argv": list(argv),
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                    "runtime_status": "EXECUTED",
                    "requested_python_version": requested_minor,
                    "resolved_python_executable": interpreter,
                    "resolved_python_version": resolved_version,
                    "python_version_match": (
                        not requested_minor
                        or _normalized_minor(resolved_version) == requested_minor
                    ),
                    "event_count": len(events),
                }
            )
    stdout_text = "\n".join(combined_stdout)
    stderr_text = "\n".join(combined_stderr)
    merged_text = f"{stdout_text}\n{stderr_text}"
    event_rows = normalize_runtime_event_rows(all_events)
    event_kinds = {str(event["kind"]) for event in event_rows}
    runtime_unavailable = any(
        result.get("runtime_status") == "PYTHON_RUNTIME_UNAVAILABLE"
        for result in command_results
    )
    environment_setup_failed = any(
        result.get("runtime_status") == "ENVIRONMENT_SETUP_FAILED"
        for result in command_results
    )
    reproduced = any(
        result.get("returncode") is not None and int(result["returncode"]) != 0
        for result in command_results
    )
    has_trace = "traceback_frame" in event_kinds
    has_coverage = "coverage" in event_kinds
    if runtime_unavailable:
        status = "PYTHON_RUNTIME_UNAVAILABLE"
    elif environment_setup_failed:
        status = "ENVIRONMENT_SETUP_FAILED"
    elif reproduced and has_trace and has_coverage:
        status = "BUG_REPRODUCED_WITH_TRACE"
    elif not reproduced:
        status = "TEST_DID_NOT_FAIL"
    elif not has_trace:
        status = "BUG_REPRODUCED_WITHOUT_PROJECT_TRACE"
    else:
        status = "BUG_REPRODUCED_WITHOUT_EXECUTED_SLICE"
    stdout_path = incident_output / "stdout.txt"
    stderr_path = incident_output / "stderr.txt"
    traceback_path = incident_output / "traceback.txt"
    assertion_path = incident_output / "assertion_difference.txt"
    events_path = incident_output / "runtime_events.jsonl"
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    traceback_path.write_text(merged_text, encoding="utf-8")
    assertion_path.write_text(_assertion_difference(merged_text), encoding="utf-8")
    events_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in event_rows),
        encoding="utf-8",
    )
    enriched = dict(row)
    for field in (
        "repository_root",
        "patch_path",
        "before_sources_path",
        "after_sources_path",
    ):
        if str(row.get(field, "")).strip():
            enriched[field] = _relative(_resolve(manifest_base, row[field]), output_root)
    enriched.update(
        {
            "runtime_evidence_status": status,
            "stdout_path": _relative(stdout_path, output_root),
            "stderr_path": _relative(stderr_path, output_root),
            "traceback_path": _relative(traceback_path, output_root),
            "assertion_difference_path": _relative(assertion_path, output_root),
            "runtime_events_path": _relative(events_path, output_root),
        }
    )
    evidence = {
        "incident_id": incident_id,
        "repository": row["repository"],
        "status": status,
        "setup": setup_result,
        "commands": command_results,
        "python_runtime_exact": bool(command_results) and all(
            bool(result.get("python_version_match"))
            for result in command_results
        ),
        "event_count": len(event_rows),
        "event_kinds": sorted(event_kinds),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
        "traceback_sha256": _sha256(traceback_path),
        "assertion_difference_sha256": _sha256(assertion_path),
        "runtime_events_sha256": _sha256(events_path),
        "gold_fields_present": False,
    }
    return enriched, evidence


def _record_infrastructure_error(
    row: dict[str, object],
    output_root: Path,
    manifest_base: Path,
    error: Exception,
) -> tuple[dict[str, object], dict[str, object]]:
    incident_id = str(row["incident_id"])
    incident_output = output_root / incident_id
    incident_output.mkdir(parents=True, exist_ok=True)
    message = (
        str(error)
        .replace(str(manifest_base), "<manifest_root>")
        .replace(str(output_root), "<output_root>")
    )
    error_path = incident_output / "infrastructure_error.txt"
    events_path = incident_output / "runtime_events.jsonl"
    error_path.write_text(
        f"{type(error).__name__}: {message}\n",
        encoding="utf-8",
    )
    events_path.write_text("", encoding="utf-8")
    enriched = dict(row)
    for field in (
        "repository_root",
        "patch_path",
        "before_sources_path",
        "after_sources_path",
    ):
        if str(row.get(field, "")).strip():
            enriched[field] = _relative(_resolve(manifest_base, row[field]), output_root)
    enriched.update(
        {
            "runtime_evidence_status": "RUNTIME_INFRASTRUCTURE_ERROR",
            "runtime_events_path": _relative(events_path, output_root),
            "infrastructure_error_path": _relative(error_path, output_root),
        }
    )
    evidence = {
        "incident_id": incident_id,
        "repository": row["repository"],
        "status": "RUNTIME_INFRASTRUCTURE_ERROR",
        "python_runtime_exact": False,
        "event_count": 0,
        "event_kinds": [],
        "runtime_events_sha256": _sha256(events_path),
        "infrastructure_error_sha256": _sha256(error_path),
        "error_type": type(error).__name__,
        "error_message": message,
        "gold_fields_present": False,
    }
    return enriched, evidence


def collect_h10_c5c_runtime(
    manifest_path: Path,
    command_registry_path: Path,
    output_root: Path,
    *,
    timeout_seconds: int = 300,
    allow_setup: bool = False,
    interpreter_map: Mapping[str, str] | None = None,
    max_workers: int = 1,
) -> RuntimeCollectionResult:
    rows = _load_jsonl(manifest_path)
    manifest_base = manifest_path.parent.resolve()
    registry_base = command_registry_path.parent.resolve()
    registry = json.loads(command_registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise TypeError("H10-C5c command registry must be an object")
    identifiers = [str(row["incident_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("H10-C5c runtime manifest has duplicate incident IDs")
    if max_workers < 1:
        raise ValueError("max_workers must be at least one")
    output_root.mkdir(parents=True, exist_ok=True)

    def collect_row(
        row: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        incident_id = str(row["incident_id"])
        registered = registry.get(incident_id)
        if not isinstance(registered, dict):
            return (
                {**row, "runtime_evidence_status": "RUNTIME_COMMAND_NOT_REGISTERED"},
                {
                    "incident_id": incident_id,
                    "repository": row["repository"],
                    "status": "RUNTIME_COMMAND_NOT_REGISTERED",
                },
            )
        prepared_registered = dict(registered)
        for field in ("setup_script", "requirements_path"):
            raw = str(prepared_registered.get(field, "")).strip()
            if raw:
                prepared_registered[field] = str(_resolve(registry_base, raw))
        try:
            return _collect_incident(
                row,
                prepared_registered,
                output_root,
                manifest_base,
                timeout_seconds=timeout_seconds,
                allow_setup=allow_setup,
                interpreter_map=interpreter_map,
            )
        except Exception as error:  # noqa: BLE001
            return _record_infrastructure_error(
                row,
                output_root,
                manifest_base,
                error,
            )

    if max_workers == 1:
        collected = [collect_row(row) for row in rows]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            collected = list(executor.map(collect_row, rows))
    enriched_rows = [item[0] for item in collected]
    evidence_rows = [item[1] for item in collected]
    enriched_path = output_root / "H10_C5C_DEVELOPMENT_RUNTIME_ENRICHED.jsonl"
    report_path = output_root / "H10_C5C_RUNTIME_EVIDENCE_REPORT.json"
    enriched_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched_rows),
        encoding="utf-8",
    )
    complete = sum(
        str(row.get("runtime_evidence_status")) == "BUG_REPRODUCED_WITH_TRACE"
        for row in enriched_rows
    )
    report = {
        "status": (
            "DEVELOPMENT_RUNTIME_COMPLETE"
            if complete == len(enriched_rows)
            else "DEVELOPMENT_RUNTIME_INCOMPLETE"
        ),
        "scientific_result": "NOT_EVALUATED",
        "total_incidents": len(enriched_rows),
        "complete_incidents": complete,
        "incomplete_incidents": len(enriched_rows) - complete,
        "input_manifest_sha256": _sha256(manifest_path),
        "command_registry_sha256": _sha256(command_registry_path),
        "enriched_manifest_sha256": _sha256(enriched_path),
        "allow_setup": allow_setup,
        "max_workers": max_workers,
        "interpreter_map": dict(sorted((interpreter_map or {}).items())),
        "all_python_runtimes_exact": bool(evidence_rows) and all(
            bool(row.get("python_runtime_exact"))
            for row in evidence_rows
        ),
        "evidence": evidence_rows,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RuntimeCollectionResult(
        enriched_path,
        report_path,
        complete,
        len(enriched_rows),
    )
