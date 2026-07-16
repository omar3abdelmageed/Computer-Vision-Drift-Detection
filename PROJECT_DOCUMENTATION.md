# Project architecture

## Product constraint

The desktop branch preserves the original Streamlit dashboard and its model-management and live-monitoring workflows. The only product-level changes are:

- no authentication screen or account requirement;
- SQLite persistence on the local machine;
- local worker access to the same SQLite database;
- packaging inside a loopback-only WebView2 desktop window.

## Startup

`streamlit_app.main()` configures the page, opens the singleton `LocalDatabase`, and renders the existing `Model Management` and `Live Sessions` views. View functions receive the database through the same repository boundary previously used by the application.

`desktop_launcher.py` enforces one desktop instance, reserves an ephemeral loopback port, launches Streamlit as a child process, waits for health, and hosts it in pywebview. The child receives a new random Streamlit cookie secret and writes server output to `%LOCALAPPDATA%\CVMonitor\logs\desktop.log`. Closing the desktop window terminates owned child processes and releases the single-instance mutex.

The packaged configuration disables Streamlit development mode, telemetry, file watching, static serving, detailed client errors, and external binding. CORS and XSRF protection remain enabled. The server listens only on `127.0.0.1`.

## Persistence

`database/local.py` owns connections and migrations. Every connection enables foreign keys and a busy timeout. Writable connections use WAL and `synchronous=NORMAL` for responsive concurrent UI/worker access.

`database/repositories.py` provides list/get/insert/update/delete/upsert/filter/count/ordered-latest operations. It:

- allowlists table names and validates columns against SQLite metadata;
- parameterizes values;
- creates UUIDs and timestamps locally;
- encodes structured fields as JSON;
- round-trips Boolean columns;
- wraps writes in `BEGIN IMMEDIATE` transactions.

`database/migrations.py` is the source of truth for the schema. `PRAGMA user_version` records the installed version. Before future upgrades of an existing database, initialization runs an integrity check and creates a pre-migration backup.

## Worker

The supervisor preserves PID validation, owner identity, heartbeat tracking, stale-worker recovery, log rotation, and explicit stop behavior. It starts `background_worker.worker` without credentials or token handoff. The worker opens the same database path from `APP_DATA_DIR` or the Windows default runtime directory.

Session leases remain in the database to prevent duplicate processing. Folder readiness, image hashing, inference status, prediction persistence, drift calculations, alert deduplication, feedback, and retention keep their existing behavior.

## Detection presentation

Object-detection predictions are clustered during rendering when normalized boxes have IoU greater than or equal to `0.80`. Clustering is class agnostic and preserves input order. The highest-confidence member supplies the displayed representative box. Competing classes mark an ambiguous location; matching classes mark a same-class duplicate location.

This layer is presentation and monitoring only. Every raw prediction remains unchanged in SQLite, retains its own feedback control, and continues to contribute independently to prediction counts, confidence metrics, drift, and alerts. Historical sessions are clustered dynamically and require no migration or reprocessing.

## Runtime data

`core/runtime_paths.py` resolves runtime state outside the installation directory. The default is `%LOCALAPPDATA%\CVMonitor`; tests and development can override it with `APP_DATA_DIR`.

The per-user installer places immutable application files under `%LOCALAPPDATA%\Programs\CVMonitor`. Updating the installation does not remove runtime data. Database files, managed model copies, backups, logs, and worker state therefore survive application upgrades and reinstalls unless the user deletes the application-data directory explicitly.

## Packaging

`scripts/build_desktop.ps1` runs pytest using an isolated temporary directory and then builds `packaging/CVMonitor.spec` with PyInstaller. The specification includes the complete Streamlit runtime because Streamlit loads several server and script-runner modules dynamically; reducing it to static frontend assets alone produces runtime import failures. Project packages are collected explicitly, while unused training, notebook, test, and GUI modules remain excluded where safe.

`packaging/windows/CVMonitor.iss` turns `dist\CVMonitor` into a per-user Inno Setup installer at `dist\installer\CVMonitor-0.1.0-Setup.exe`. It installs the desktop bundle, adds optional shortcuts, and runs the bundled WebView2 bootstrapper when that vendor file is available.

## Verification

The test suite covers the existing dashboard helpers and monitoring behavior plus SQLite migration state, WAL/foreign-key configuration, JSON/Boolean/YAML round trips, identifier validation, constraints, concurrent reads during writes, detection clustering, desktop launcher behavior, and required packaged Streamlit modules. Release builds should additionally be smoke-tested by starting the frozen executable and loading the rendered dashboard, because a successful health endpoint alone does not execute the Streamlit application script.
