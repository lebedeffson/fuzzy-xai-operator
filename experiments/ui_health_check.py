from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


@dataclass
class UICheckResult:
    app: str
    port: int
    url: str
    ok: bool
    status_code: int | None
    title_found: bool
    error: str | None
    elapsed_sec: float


def _fetch(url: str, timeout: float = 2.0) -> tuple[int, str]:
    with urlopen(url, timeout=timeout) as resp:  # nosec B310 (localhost only)
        code = int(resp.getcode())
        body = resp.read().decode('utf-8', errors='ignore')
        return code, body


def _resolve_python_bin() -> str:
    override = os.environ.get('UI_CHECK_PYTHON', '').strip()
    if override:
        return override
    candidates = [
        Path.home() / 'Code' / 'venv' / 'bin' / 'python',
        Path.cwd().parent / 'venv' / 'bin' / 'python',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


def _run_one(app_path: str, port: int, expected_title: str, wait_sec: float = 20.0) -> UICheckResult:
    url = f'http://127.0.0.1:{port}'
    env = dict(os.environ)
    env['PYTHONPATH'] = '.'
    start = time.time()
    stderr_file = subprocess.PIPE
    proc = subprocess.Popen(  # noqa: S603,S607
        [_resolve_python_bin(), app_path, '--port', str(port)],
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
        env=env,
        text=True,
    )
    status_code: int | None = None
    title_found = False
    err: str | None = None
    ok = False

    try:
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr_out = ''
                try:
                    if proc.stderr is not None:
                        stderr_out = (proc.stderr.read() or '').strip()
                except Exception:
                    stderr_out = ''
                err = f'process exited with code {proc.returncode}'
                if stderr_out:
                    err += f' | stderr: {stderr_out.splitlines()[-1]}'
                break
            try:
                code, body = _fetch(url)
                status_code = code
                title_found = expected_title in body
                if code == 200 and title_found:
                    ok = True
                    break
            except URLError:
                pass
            except Exception as exc:  # pragma: no cover
                err = f'{type(exc).__name__}: {exc}'
                break
            time.sleep(0.5)
        else:
            err = 'timeout waiting for HTTP 200 with expected title'
    finally:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    return UICheckResult(
        app=app_path,
        port=port,
        url=url,
        ok=ok,
        status_code=status_code,
        title_found=title_found,
        error=err,
        elapsed_sec=round(time.time() - start, 3),
    )


def run_all(out_dir: str | Path = 'reports', all_apps: bool = False) -> dict[str, object]:
    checks = [('apps/fuzzyxai_studio.py', 18097, 'FuzzyXAI Studio')]
    if all_apps:
        checks.extend(
            [
                ('apps/layered_demo.py', 18096, 'FuzzyXAI Layered Demo'),
                ('apps/defense_demo.py', 18085, 'FuzzyXAI Defense Demo'),
            ]
        )
    rows = [_run_one(app, port, title, wait_sec=35.0) for app, port, title in checks]
    ok = all(r.ok for r in rows)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        'ok': ok,
        'results': [asdict(r) for r in rows],
        'checked_at_epoch': time.time(),
    }
    (out_root / 'ui_health_check.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# UI health check', '', f"overall: `{'PASS' if ok else 'FAIL'}`", '', '| app | port | ok | status | title_found | elapsed_sec | error |', '|---|---:|---|---:|---|---:|---|']
    for r in rows:
        md.append(
            f"| `{r.app}` | {r.port} | {r.ok} | {r.status_code if r.status_code is not None else '-'} | {r.title_found} | {r.elapsed_sec:.3f} | {r.error or ''} |"
        )
    (out_root / 'ui_health_check.md').write_text('\n'.join(md), encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports')
    parser.add_argument('--all-apps', action='store_true', help='Also check legacy GUI apps')
    args = parser.parse_args()
    out = run_all(out_dir=args.out_dir, all_apps=bool(args.all_apps))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not bool(out['ok']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
