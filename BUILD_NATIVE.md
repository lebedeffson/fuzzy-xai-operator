# Native build guide

The NiceGUI dashboard can be launched as a local web app or as a desktop-like native window.

## Local GUI

```bash
pip install -r requirements.txt
python apps/nicegui_dashboard.py
```

## Native window

```bash
pip install -r requirements.txt
pip install -r requirements-native.txt
python apps/nicegui_dashboard.py --native
```

## One-file executable draft

PyInstaller builds are environment-specific. Use this as a starting point and test the result on the target OS.

```bash
pip install -r requirements.txt
pip install -r requirements-native.txt
pyinstaller fuzzyxai_gui.spec
```

The executable is produced in `dist/`. If your OS blocks the first launch, run it once from the terminal to inspect missing dynamic libraries.

## Notes

- The GUI is local and does not require cloud deployment.
- Optional LLM generation requires `OPENAI_API_KEY` and `pip install -r requirements-llm.txt`.
- If no key is present, the app falls back to deterministic template explanations.
