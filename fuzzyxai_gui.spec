# PyInstaller spec for the local NiceGUI dashboard.
# Build from repository root:
#   pyinstaller fuzzyxai_gui.spec

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

ROOT = Path.cwd()

hiddenimports = collect_submodules('nicegui') + collect_submodules('fuzzyxai')

a = Analysis(
    ['apps/nicegui_dashboard.py'],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        ('apps/assets', 'apps/assets'),
        ('reports', 'reports'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['streamlit'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FuzzyXAI_Doctoral_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
