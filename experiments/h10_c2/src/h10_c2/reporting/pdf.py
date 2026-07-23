from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def markdown_to_pdf(source: Path, target: Path) -> None:
    lines = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        lines.extend(wrap(raw, width=100) or [""])
    target.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(target) as pdf:
        for start in range(0, len(lines), 48):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.94, "\n".join(lines[start : start + 48]), va="top", family="DejaVu Sans", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)

