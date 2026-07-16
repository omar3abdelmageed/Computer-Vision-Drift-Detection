# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules
from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).resolve().parent
datas = [
    (str(PROJECT_ROOT / "streamlit_app.py"), "."),
    (str(PROJECT_ROOT / ".streamlit" / "config.toml"), ".streamlit"),
]
streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all("streamlit")
datas += streamlit_datas

hiddenimports = ["streamlit_app", *streamlit_hiddenimports]
for package in ("application", "background_worker", "core", "dashboard", "database", "model_management"):
    hiddenimports += collect_submodules(package)

excludes = [
    "IPython",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "cefpython3",
    "gtk",
    "jupyter",
    "notebook",
    "pytest",
    "tensorboard",
    "tkinter",
]

a = Analysis(
    [str(PROJECT_ROOT / "desktop_launcher.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=streamlit_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CVMonitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="CVMonitor",
)
