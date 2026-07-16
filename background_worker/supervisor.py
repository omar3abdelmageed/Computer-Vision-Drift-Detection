from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from socket import gethostname
from typing import Any

from database.repositories import Repository
from core.runtime_paths import runtime_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATHS = runtime_paths()
WORKER_DIR = APP_PATHS.worker
PID_FILE = WORKER_DIR / "worker.pid"
HEARTBEAT_FILE = WORKER_DIR / "heartbeat.json"
OWNER_FILE = WORKER_DIR / "worker_owner.json"
LOG_DIR = APP_PATHS.logs / "worker"
STARTUP_CHECK_SECONDS = 1.5
WORKER_MODULE = "background_worker.worker"
MAX_LOG_FILES = 10


@dataclass(frozen=True)
class WorkerStartResult:
    started: bool
    already_running: bool
    pid: int | None = None
    owner_id: str | None = None
    error: str | None = None
    log_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.started or self.already_running


@dataclass(frozen=True)
class WorkerStopResult:
    stopped: bool
    attempted_pids: tuple[int, ...] = ()
    owner_id: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def ensure_worker_running(interval_seconds: int = 30, client=None) -> WorkerStartResult:
    try:
        owner_id = worker_owner_id()
        existing_pid = read_worker_pid()
        if existing_pid:
            try:
                existing_worker_is_alive = is_worker_process(existing_pid)
            except Exception:
                existing_worker_is_alive = False
            heartbeat = read_heartbeat()
            heartbeat_stale = heartbeat_is_stale(heartbeat, interval_seconds)
            if existing_worker_is_alive and not heartbeat_stale:
                return WorkerStartResult(started=False, already_running=True, pid=existing_pid, owner_id=owner_id)
            mark_locally_owned_sessions_stale(client, owner_id)

        if existing_pid:
            clear_worker_pid()

        WORKER_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        rotate_worker_logs()
        log_path = next_worker_log_path()
        log_handle = log_path.open("a", encoding="utf-8")
        command = worker_command() + [
            "--interval-seconds",
            str(max(1, int(interval_seconds))),
            "--worker-owner-id",
            owner_id,
        ]
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=worker_creation_flags(),
            text=False,
        )
        log_handle.close()
        time.sleep(STARTUP_CHECK_SECONDS)
        if process.poll() is not None:
            clear_worker_pid()
            return WorkerStartResult(
                started=False,
                already_running=False,
                pid=process.pid,
                owner_id=owner_id,
                error=f"Background worker exited during startup with code {process.returncode}. Check {log_path}.",
                log_path=str(log_path),
            )
        write_worker_pid(process.pid)
        return WorkerStartResult(started=True, already_running=False, pid=process.pid, owner_id=owner_id, log_path=str(log_path))
    except Exception as exc:
        return WorkerStartResult(started=False, already_running=False, error=str(exc))


def stop_worker(client=None, reason: str = "session_paused_or_stopped", timeout_seconds: float = 3.0) -> WorkerStopResult:
    try:
        owner_id = worker_owner_id()
        pids = worker_candidate_pids()
        stopped_any = False
        for pid in pids:
            try:
                if not is_worker_process(pid):
                    continue
                stopped_any = stop_worker_process(pid, timeout_seconds=timeout_seconds) or stopped_any
            except Exception:
                continue
        clear_worker_pid()
        write_stopped_heartbeat(owner_id, reason)
        clear_locally_owned_running_worker_fields(client, owner_id)
        return WorkerStopResult(stopped=stopped_any, attempted_pids=tuple(pids), owner_id=owner_id)
    except Exception as exc:
        return WorkerStopResult(stopped=False, error=str(exc))


def worker_candidate_pids() -> list[int]:
    candidates: list[int] = []
    pid = read_worker_pid()
    if pid:
        candidates.append(pid)
    heartbeat = read_heartbeat() or {}
    try:
        heartbeat_pid = int(heartbeat.get("pid") or 0)
    except (TypeError, ValueError):
        heartbeat_pid = 0
    if heartbeat_pid > 0 and heartbeat_pid not in candidates:
        candidates.append(heartbeat_pid)
    return candidates


def stop_worker_process(pid: int, timeout_seconds: float = 3.0) -> bool:
    psutil_result = stop_worker_process_with_psutil(pid, timeout_seconds)
    if psutil_result is not None:
        return psutil_result
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, SystemError, ValueError):
        return False
    deadline = time.time() + max(0.1, timeout_seconds)
    while time.time() < deadline:
        if not is_worker_process(pid):
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except (OSError, SystemError, ValueError, AttributeError):
        return False


def stop_worker_process_with_psutil(pid: int, timeout_seconds: float) -> bool | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=max(0.1, timeout_seconds))
        return True
    except psutil.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=max(0.1, timeout_seconds))
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired, OSError):
            return False
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False


def worker_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS


def worker_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker-child"]
    return [worker_python_executable(), "-m", WORKER_MODULE]


def read_worker_pid() -> int | None:
    try:
        raw_pid = PID_FILE.read_text(encoding="utf-8").strip()
        pid = int(raw_pid)
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


def write_worker_pid(pid: int) -> None:
    PID_FILE.write_text(str(pid), encoding="utf-8")


def clear_worker_pid() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        return


def is_process_alive(pid: int) -> bool:
    return is_worker_process(pid)


def is_worker_process(pid: int) -> bool:
    psutil_process = psutil_worker_process(pid)
    if psutil_process is not None:
        return psutil_process
    try:
        os.kill(pid, 0)
    except (OSError, SystemError, ValueError):
        return False
    return False


def psutil_worker_process(pid: int) -> bool | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(pid)
        command_line = process.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError, SystemError, ValueError):
        return False
    joined_command = " ".join(command_line).replace("\\", "/")
    return WORKER_MODULE in joined_command or "background_worker/worker.py" in joined_command or "--worker-child" in joined_command


def worker_python_executable() -> str:
    if os.name == "nt":
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def next_worker_log_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return LOG_DIR / f"worker-{stamp}.log"


def worker_owner_id() -> str:
    try:
        payload = json.loads(OWNER_FILE.read_text(encoding="utf-8"))
        owner_id = payload.get("owner_id")
        if owner_id:
            return str(owner_id)
    except (OSError, json.JSONDecodeError):
        pass
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    root_hash = sha256(str(PROJECT_ROOT).encode("utf-8")).hexdigest()[:12]
    owner_id = f"{gethostname()}-{root_hash}"
    OWNER_FILE.write_text(json.dumps({"owner_id": owner_id}, indent=2), encoding="utf-8")
    return owner_id


def write_heartbeat(pid: int, interval_seconds: int, started_at: str, owner_id: str | None = None) -> None:
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "owner_id": owner_id or worker_owner_id(),
        "interval_seconds": interval_seconds,
        "started_at": started_at,
        "last_seen_at": datetime.now(timezone.utc).isoformat(),
    }
    HEARTBEAT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_stopped_heartbeat(owner_id: str, reason: str) -> None:
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": None,
        "owner_id": owner_id,
        "status": "stopped",
        "reason": reason,
        "last_seen_at": datetime.now(timezone.utc).isoformat(),
    }
    HEARTBEAT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_heartbeat() -> dict[str, Any] | None:
    try:
        return json.loads(HEARTBEAT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def heartbeat_is_stale(heartbeat: dict[str, Any] | None, interval_seconds: int) -> bool:
    if not heartbeat:
        return True
    last_seen_at = heartbeat.get("last_seen_at")
    if not last_seen_at:
        return True
    try:
        last_seen = datetime.fromisoformat(str(last_seen_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    max_age = max(3 * max(1, int(interval_seconds)), 60)
    return (datetime.now(timezone.utc) - last_seen).total_seconds() > max_age


def mark_locally_owned_sessions_stale(client, owner_id: str) -> None:
    if client is None:
        return
    sessions = Repository(client, "monitoring_sessions")
    stale_payload = {
        "worker_status": "stale",
        "last_worker_error": "Worker process disappeared.",
        "worker_lease_expires_at": datetime.now(timezone.utc).isoformat(),
    }
    for session in sessions.where(worker_owner_id=owner_id, status="running_live_monitoring"):
        if session.get("worker_status") in {"running", "idle", "caught_up"}:
            sessions.update(session["id"], stale_payload)


def clear_locally_owned_running_worker_fields(client, owner_id: str) -> None:
    if client is None:
        return
    sessions = Repository(client, "monitoring_sessions")
    payload = {
        "worker_pid": None,
        "worker_heartbeat_at": None,
        "worker_lease_expires_at": datetime.now(timezone.utc).isoformat(),
    }
    for session in sessions.where(worker_owner_id=owner_id, status="running_live_monitoring"):
        sessions.update(session["id"], payload)


def rotate_worker_logs(max_files: int = MAX_LOG_FILES) -> None:
    if not LOG_DIR.exists():
        return
    logs = sorted(LOG_DIR.glob("worker-*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in logs[max_files:]:
        try:
            path.unlink()
        except OSError:
            continue
