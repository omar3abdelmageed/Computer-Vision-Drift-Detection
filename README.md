# Computer Vision Model Monitoring

A trusted-local Streamlit desktop application for registering YOLO models, validating datasets, building baselines, monitoring watched folders, reviewing predictions, and inspecting drift and alerts.

This branch intentionally keeps the existing Streamlit interface. It requires no account or sign-in and stores application state in a local SQLite database.

Setup installer:
https://mega.nz/file/DToSUbzY#PwChOryc-Dw6us0WYWT8S8lvckbmpktiLSAoHIvgM-E

## App flow

1. Create or select a model.
2. Choose object detection or classification.
3. Register and validate a local YOLO dataset.
4. Register a local YOLO `.pt` model.
5. Check model/dataset compatibility.
6. Build the training baseline.
7. Configure a watched folder.
8. Start, pause, resume, or stop monitoring.
9. Review predictions, feedback, drift, and alerts in the existing dashboard.

## Local data

The default runtime directory is `%LOCALAPPDATA%\CVMonitor` on Windows:

- `data\cv_monitor.sqlite3`: primary SQLite database.
- `data\cv_monitor.sqlite3-wal` and `-shm`: SQLite runtime files.
- `backups\`: database backups.
- `models\<sha256>\`: explicitly trusted model copies.
- `logs\desktop.log`: desktop launcher and Streamlit server output.
- `logs\worker\`: background worker logs.
- `runtime\worker\`: worker PID, heartbeat, and owner state.

Set `APP_DATA_DIR` to use a different root during development or testing.

SQLite uses foreign keys, a five-second busy timeout, WAL journaling, and versioned migrations from [database/migrations.py](database/migrations.py). Queries use parameter values and allowlisted table/column identifiers.

## Run locally

Install the project and its desktop, test, and build dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -e ".[desktop,test,build]"
```

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Run in the desktop WebView2 wrapper:

```powershell
.\venv\Scripts\python.exe desktop_launcher.py
```

Run one worker cycle:

```powershell
.\venv\Scripts\python.exe -m background_worker.worker --once
```

Run tests:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

## Install or update on Windows

Run `dist\installer\CVMonitor-0.1.0-Setup.exe` and follow the installer. CV Monitor is installed per user under `%LOCALAPPDATA%\Programs\CVMonitor`; administrator access is not required. The installer can also create Start menu and desktop shortcuts.

Close CV Monitor before installing an update. Installing a newer build over the existing installation replaces application files but preserves the SQLite database, trusted model copies, logs, and other runtime data under `%LOCALAPPDATA%\CVMonitor`.

## Build the Windows desktop bundle

```powershell
.\scripts\build_desktop.ps1
```

The script runs the regression suite and creates the bundle at `dist\CVMonitor`. Use `-SkipTests` only when the tests were already run against the same commit.

Compile [packaging/windows/CVMonitor.iss](packaging/windows/CVMonitor.iss) with Inno Setup to create the per-user installer:

```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" "packaging\windows\CVMonitor.iss"
```

If Inno Setup is installed elsewhere, replace the executable path. PowerShell requires the leading `&` when a quoted executable path contains spaces. The installer is written to `dist\installer\CVMonitor-0.1.0-Setup.exe`.

The desktop wrapper binds Streamlit to `127.0.0.1` only, retains Streamlit XSRF protection, disables static serving and telemetry, and opens the dashboard in a native WebView2 window.

## Detection overlap monitoring

Detection results remain stored and counted as raw model predictions. For presentation, boxes with normalized IoU of at least `0.80` are grouped into physical locations. The image shows one representative box per location, while the review table retains every prediction as independently reviewable subrows such as `4a` and `4b`. Different classes at one location are reported as ambiguity; repeated matching classes are reported as same-class duplicates. This does not alter model inference, confidence metrics, drift calculations, or persisted predictions.

## Troubleshooting

- If the desktop window shows a redacted Streamlit error, close the app and inspect `%LOCALAPPDATA%\CVMonitor\logs\desktop.log` for the original traceback.
- Worker startup and inference errors are recorded in `%LOCALAPPDATA%\CVMonitor\logs\worker`.
- Only one desktop instance is allowed. Close the existing CV Monitor process before launching it again.
- The installer uses Microsoft Edge WebView2. When the optional bundled WebView2 bootstrapper is present, setup installs it automatically; otherwise install the WebView2 Runtime separately on systems that do not already provide it.

## Main structure

```text
streamlit_app.py             Existing Streamlit entry point and tabs
dashboard/                   Existing views and UI components
model_management/            Dataset/model validation and baseline creation
background_worker/           Folder scanning, inference, drift, retention, supervision
database/
  local.py                   SQLite connection, backup, integrity, migration lifecycle
  migrations.py              Versioned local schema
  repositories.py            Parameterized repository interface
core/                        Shared types, runtime paths, trusted model handling
desktop_launcher.py          Loopback Streamlit + WebView2 desktop host
packaging/                   PyInstaller and Inno Setup configuration
tests/                       Unit, regression, worker, UI-helper, and SQLite tests
```

## Security boundary

This is a single-user local application. Removing authentication means anyone who can access the Windows account and app-data directory can use the application and read its database. OS account security and filesystem permissions are therefore the access boundary. Model files remain opt-in: a model is copied into a hash-addressed managed directory before Ultralytics loads it.
