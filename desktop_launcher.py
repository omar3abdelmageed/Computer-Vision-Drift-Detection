from __future__ import annotations

import argparse
import ctypes
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from core.runtime_paths import runtime_paths


APP_TITLE = "Computer Vision Model Monitoring"
LOOPBACK = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 45


def bundled_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_root() -> Path:
    bundle_path = getattr(sys, "_MEIPASS", None)
    return Path(bundle_path) if bundle_path else Path(__file__).resolve().parent


def app_script_path() -> Path:
    return resource_root() / "streamlit_app.py"


def reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK, 0))
        return int(listener.getsockname()[1])


def child_command(mode: str, *args: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, mode, *args]
    return [sys.executable, str(Path(__file__).resolve()), mode, *args]


def start_streamlit_child(port: int) -> subprocess.Popen:
    paths = runtime_paths().create()
    log_path = paths.logs / "desktop.log"
    log_handle = log_path.open("a", encoding="utf-8")
    command = child_command("--streamlit-child", "--port", str(port))
    child_environment = os.environ.copy()
    child_environment["STREAMLIT_SERVER_COOKIE_SECRET"] = secrets.token_urlsafe(48)
    process = subprocess.Popen(
        command,
        cwd=str(bundled_root()),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        env=child_environment,
    )
    log_handle.close()
    return process


def wait_until_ready(process: subprocess.Popen, url: str, timeout_seconds: int = STARTUP_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"The local application server exited with code {process.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)
    raise RuntimeError("The local application server did not become ready in time.")


def stop_process_tree(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        import psutil

        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=5)
        for remaining in alive:
            remaining.kill()
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def acquire_single_instance_mutex():
    if os.name != "nt":
        return None
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\CVMonitorDesktop")
    if ctypes.windll.kernel32.GetLastError() == 183:
        ctypes.windll.kernel32.CloseHandle(handle)
        raise RuntimeError("CV Monitor is already running.")
    return handle


def release_single_instance_mutex(handle) -> None:
    if handle and os.name == "nt":
        ctypes.windll.kernel32.CloseHandle(handle)


def run_desktop() -> None:
    mutex = acquire_single_instance_mutex()
    server = None
    try:
        port = reserve_loopback_port()
        url = f"http://{LOOPBACK}:{port}"
        server = start_streamlit_child(port)
        wait_until_ready(server, url)
        try:
            import webview
        except ImportError:
            webbrowser.open(url)
            server.wait()
            return
        webview.create_window(APP_TITLE, url, width=1440, height=920, min_size=(1024, 700))
        webview.start(gui="edgechromium")
    finally:
        stop_process_tree(server)
        release_single_instance_mutex(mutex)


def run_streamlit_child(port: int) -> None:
    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        str(app_script_path()),
        "--server.address",
        LOOPBACK,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.enableCORS",
        "true",
        "--server.enableXsrfProtection",
        "true",
        "--server.fileWatcherType",
        "none",
        "--server.enableStaticServing",
        "false",
        "--browser.gatherUsageStats",
        "false",
        "--client.showErrorDetails",
        "none",
        "--client.toolbarMode",
        "minimal",
    ]
    raise SystemExit(streamlit_cli.main())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--streamlit-child", action="store_true")
    parser.add_argument("--worker-child", action="store_true")
    parser.add_argument("--port", type=int)
    args, remaining = parser.parse_known_args()
    args.remaining = remaining
    return args


def main() -> None:
    args = parse_args()
    if args.worker_child:
        from background_worker.worker import main as worker_main

        sys.argv = ["background_worker.worker", *args.remaining]
        worker_main()
        return
    if args.streamlit_child:
        if not args.port:
            raise RuntimeError("Desktop server configuration is incomplete.")
        run_streamlit_child(args.port)
        return
    run_desktop()


if __name__ == "__main__":
    main()
