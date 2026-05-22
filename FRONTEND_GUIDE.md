# NiceGUI Frontend Guide

This project uses **NiceGUI** as a local professional GUI for the doctoral demonstration of chapters 2 and 3.

## Run

```bash
pip install -r requirements.txt
python apps/defense_demo.py --port 8085
```

Open `http://localhost:8085`.

The full technical dashboard is still available:

```bash
python apps/nicegui_dashboard.py --port 8080
```

Optional desktop-like mode for the technical dashboard:

```bash
pip install pywebview
python apps/nicegui_dashboard.py --native
```

## Defense Demo

`apps/defense_demo.py` is the main presentation interface. It is intentionally narrow:

1. `ExplainPlan`: membership functions built from data.
2. `E_k`: selected representation class, uncertainty, reduction loss, rules, and deterministic text.
3. `Composition`: graph, semantic disagreement, `D_ij`, `L_ext`, and `I(E_G)`.

Use this for the defense and screenshots.

## Technical Dashboard

`apps/nicegui_dashboard.py` keeps the broader development tools: CSV builder, FML synthesizer, reports, thesis validation, and session export.

## What to show at the defense

1. Open `http://localhost:8085`.
2. Move the risk slider and show how `E_k` changes.
3. Toggle conflict and show `D_ij`.
4. Download `defense_demo_report.json`.

## Notes

The GUI does not replace the formal definitions. It demonstrates that the formal constructions are computable and reproducible.


## v6 polishing

### Theme switch
The header contains a dark/light switch powered by `ui.dark_mode`. The action is written to the session report and CSV audit log.

### Interactive plots
The representation page and ExplainPlan page use Plotly figures through `ui.plotly`, so supports can be zoomed, inspected and exported from the browser context.

### Optional LLM explanation
The Explain Instance page contains an optional LLM switch. If `OPENAI_API_KEY` is absent or the OpenAI package is not installed, the system falls back to deterministic template generation and stores this fact in the trace.

### Local audit log
Every important GUI action is appended to `reports/nicegui_session_log.csv` and can be downloaded from the Session page.

### Native packaging
See `BUILD_NATIVE.md` and `fuzzyxai_gui.spec` for the desktop build route.


## v8 stability fixes

The NiceGUI dashboard is now synthetic-first. It does not require a user-uploaded CSV or existing reports at first launch. The startup state contains sample medical-like data, a generated ExplainPlan, a demo `E_k^ext`, and a demo composition graph.

Pages hardened in v8:

- **ExplainPlan**: sample data button also builds a valid plan; CSV upload remains optional.
- **Explain Instance**: always has a default explanation; slider changes can auto-refresh.
- **Composition Graph**: default graph is generated automatically; conflict mode produces `D_ij`.
- **FML Synthesizer**: selected uncertainty types and levels persist in GUI state.
- **Reports**: missing report files show instructions instead of errors.
- **Thesis Demo**: validation and demo are wrapped in visible GUI error handling.

If a GUI handler fails, the error is shown with `ui.notify` and also appended to the session report as `gui_error`.
