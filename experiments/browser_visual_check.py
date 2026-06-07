from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass
class VisualCheck:
    name: str
    ok: bool
    note: str
    screenshot: str | None = None


def _wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2.0) as resp:  # nosec B310 localhost
                if int(resp.getcode()) == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f'HTTP not ready: {url}')


def run_visual_check(port: int = 18097, out_dir: str | Path = 'reports/browser_visual_check') -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    url = f'http://127.0.0.1:{port}'

    env = dict(os.environ)
    env['PYTHONPATH'] = '.'
    app = subprocess.Popen(  # noqa: S603,S607
        [sys.executable, 'apps/fuzzyxai_studio.py', '--port', str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    results: list[VisualCheck] = []
    try:
        _wait_http(url)
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1600, 'height': 1000})
            page.goto(url, wait_until='networkidle', timeout=60000)

            # Home screenshot
            home_path = out / '01_studio_home.png'
            page.screenshot(path=str(home_path), full_page=True)
            results.append(VisualCheck('home', True, 'loaded', str(home_path)))

            section_checks = [
                ('case_controls', '1) Кейс и запуск', None),
                ('plan_editor', '2) ExplainPlan', None),
                ('what_if', '3) What-if', None),
                ('benchmark', '4) Benchmark', None),
                ('export', '5) Export', None),
                ('summary', 'Итог по кейсу', ('tab', 'Overview')),
                ('operator_trace', 'Operator trace (что берет из системы)', ('tab', 'Operators')),
                ('operator_flow', 'rows:', ('tab', 'Operators')),
                ('evidence_contract', 'Evidence contract', ('tab', 'Evidence')),
                ('ecosystem_registry', 'External module registry', ('tab', 'Evidence')),
                ('real_rows_improvements', 'Real rows improvements', ('tab', 'Evidence')),
            ]
            for i, (name, marker, nav) in enumerate(section_checks, start=2):
                ok = True
                note = 'ok'
                shot = out / f'{i:02d}_{name}.png'
                try:
                    if nav and nav[0] == 'tab':
                        page.get_by_role('tab', name=nav[1]).click(timeout=4000)
                        page.wait_for_timeout(300)
                    loc = page.get_by_text(marker, exact=False).first
                    loc.wait_for(timeout=7000)
                    loc.scroll_into_view_if_needed(timeout=3000)
                    page.wait_for_timeout(300)
                    page.screenshot(path=str(shot), full_page=True)
                except Exception as exc:  # pragma: no cover
                    ok = False
                    note = f'{type(exc).__name__}: {exc}'
                results.append(VisualCheck(name, ok, note, str(shot) if ok else None))

            browser.close()

    finally:
        if app.poll() is None:
            try:
                app.send_signal(signal.SIGTERM)
                app.wait(timeout=5)
            except Exception:
                app.kill()

    ok_all = all(r.ok for r in results)
    payload = {
        'ok': ok_all,
        'url': url,
        'results': [asdict(r) for r in results],
    }
    (out / 'browser_visual_check.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# Browser visual check', '', f"overall: `{'PASS' if ok_all else 'FAIL'}`", '', f'- url: `{url}`', '']
    for r in results:
        line = f"- {r.name}: {'PASS' if r.ok else 'FAIL'} ({r.note})"
        if r.screenshot:
            line += f" -> `{r.screenshot}`"
        md.append(line)
    (out / 'browser_visual_check.md').write_text('\n'.join(md), encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18097)
    parser.add_argument('--out-dir', default='reports/browser_visual_check')
    args = parser.parse_args()
    out = run_visual_check(port=args.port, out_dir=args.out_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
