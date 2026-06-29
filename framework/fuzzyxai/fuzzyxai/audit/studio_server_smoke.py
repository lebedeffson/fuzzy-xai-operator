from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    if _port_is_open(args.port):
        raise SystemExit(f"port {args.port} is already in use")

    cmd = [sys.executable, "apps/fuzzyxai_studio.py", "--port", str(args.port)]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    try:
        deadline = time.time() + args.timeout
        url = f"http://127.0.0.1:{args.port}"
        while time.time() < deadline:
            if proc.poll() is not None:
                output += proc.stdout.read() if proc.stdout else ""
                raise SystemExit(f"studio server exited early:\n{output}")
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    body = response.read(4000).decode("utf-8", errors="ignore")
                    if response.status == 200 and ("FuzzyXAI" in body or "NiceGUI" in body or "<html" in body.lower()):
                        print("studio-server-smoke: PASS")
                        return
            except Exception:
                time.sleep(0.4)
        output += proc.stdout.read() if proc.stdout else ""
        raise SystemExit(f"studio server did not answer on {url}:\n{output}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
