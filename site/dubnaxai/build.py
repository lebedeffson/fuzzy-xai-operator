from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
DATA = ROOT / "src" / "data"
PUBLIC = ROOT / "public"


def load(name: str):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def card(title: str, body: str) -> str:
    return f'<article class="card"><h3>{title}</h3><p>{body}</p></article>'


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    if (DIST / "screenshots").exists():
        shutil.rmtree(DIST / "screenshots")
    shutil.copytree(PUBLIC / "screenshots", DIST / "screenshots")
    models = load("models")
    methods = load("methods")
    demos = load("demos")
    researchers = load("researchers")
    publications = load("publications")
    model_html = "".join(card(m.get("model_id", "model"), f"scenario: {m.get('scenario_id')}<br>status: {m.get('model_status')}<br>{m.get('not_a_claim','')}") for m in models)
    method_html = "".join(card(m["title"], f"{m['kind']}<br>{m['description']}") for m in methods)
    demo_html = "".join(card(d["name"], f"{d['artifact_type']} / {d['artifact_status']}<br>evidence: {d['evidence_level']}") for d in demos)
    research_html = "".join(card(r["name"], f"{r['role']}<br>{r['focus']}") for r in researchers)
    pub_html = "".join(card(p["title"], f"{p['type']} / {p['status']}") for p in publications)
    shots = ["00_ecosystem_main.png", "02_hybrid_xiris_input_eye.png", "07_ecg_signal_input.png", "10_gd_anfis_shap_workspace.png", "11_beacon_xai_workspace.png", "12_gis_integro_workspace.png"]
    gallery = "".join(f'<figure><img src="screenshots/{s}" alt="{s}"><figcaption>{s}</figcaption></figure>' for s in shots)
    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DubnaXAI</title><link rel="stylesheet" href="style.css"></head>
<body><header><h1>DubnaXAI</h1><p>Исследовательская экосистема: FuzzyXAI framework, модели, методы и прикладные сценарии.</p>
<nav><a href="#models">Модели</a><a href="#methods">Методы</a><a href="#demos">Демонстрации</a><a href="#operators">Операторы</a><a href="#publications">Публикации</a></nav></header>
<main>
<section id="ecosystem"><h2>Экосистема</h2><p>Сайт отделён от библиотеки и читает структурированные данные из <code>src/data/*.json</code>.</p>{gallery}</section>
<section id="researchers"><h2>Исследователи</h2><div class="grid">{research_html}</div></section>
<section id="models"><h2>Модели</h2><div class="grid">{model_html}</div></section>
<section id="methods"><h2>Методы</h2><div class="grid">{method_html}</div></section>
<section id="demos"><h2>Демонстрации</h2><div class="grid">{demo_html}</div></section>
<section id="operators"><h2>Панель операторов</h2><p>E_k -> T_ij -> F -> Delta -> rho -> действие -> доказательный след.</p></section>
<section id="publications"><h2>Публикации</h2><div class="grid">{pub_html}</div></section>
</main><footer>DubnaXAI static site build.</footer></body></html>"""
    (DIST / "index.html").write_text(html, encoding="utf-8")
    (DIST / "style.css").write_text("""body{margin:0;font-family:Arial,sans-serif;background:#f7f7f4;color:#17202a}header{padding:48px 64px;background:#12343b;color:white}nav{display:flex;gap:18px;flex-wrap:wrap}nav a{color:white}main{padding:32px 64px}section{margin:0 0 42px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}.card{background:white;border:1px solid #d8ddd8;border-radius:8px;padding:16px}figure{display:inline-block;width:31%;min-width:280px;margin:8px;background:white;border:1px solid #ddd;padding:8px}img{width:100%;height:auto}code{background:#eee;padding:2px 5px}footer{padding:24px 64px;background:#e8ece7}""", encoding="utf-8")
    print(DIST / "index.html")


if __name__ == "__main__":
    main()
