from __future__ import annotations

import socket
import tomllib
from pathlib import Path

import desktop_launcher
from background_worker import supervisor, worker
from core import runtime_paths


def test_runtime_paths_use_configured_app_data(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app-data"))

    paths = runtime_paths.runtime_paths().create()

    assert paths.root == (tmp_path / "app-data").resolve()
    assert paths.logs.is_dir()
    assert paths.worker.is_dir()
    assert paths.models.is_dir()


def test_desktop_reserves_loopback_port():
    port = desktop_launcher.reserve_loopback_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((desktop_launcher.LOOPBACK, port))


def test_source_child_command_uses_launcher(monkeypatch):
    monkeypatch.delattr(desktop_launcher.sys, "frozen", raising=False)

    command = desktop_launcher.child_command("--streamlit-child", "--port", "1234")

    assert command[0] == desktop_launcher.sys.executable
    assert Path(command[1]).name == "desktop_launcher.py"
    assert command[2:] == ["--streamlit-child", "--port", "1234"]


def test_frozen_worker_command_uses_same_executable(monkeypatch):
    monkeypatch.setattr(supervisor.sys, "frozen", True, raising=False)

    assert supervisor.worker_command() == [supervisor.sys.executable, "--worker-child"]


def test_worker_opens_local_database_without_authentication():
    source = Path(worker.__file__).read_text(encoding="utf-8")

    assert "create_local_database()" in source
    assert "auth-stdin" not in source
    assert "access_token" not in source


def test_frozen_resource_root_uses_meipass(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop_launcher.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert desktop_launcher.app_script_path() == tmp_path / "streamlit_app.py"


def test_streamlit_desktop_config_is_loopback_only():
    with Path(".streamlit/config.toml").open("rb") as handle:
        config = tomllib.load(handle)

    assert config["global"]["developmentMode"] is False
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["enableCORS"] is True
    assert config["server"]["enableXsrfProtection"] is True
    assert config["server"]["enableStaticServing"] is False
    assert config["browser"]["gatherUsageStats"] is False


def test_streamlit_app_has_no_authentication_gate_and_uses_sqlite():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "create_local_database()" in source
    assert "render_auth_screen" not in source
    assert "supabase" not in source.lower()
    assert '["Model Management", "Live Sessions"]' in source


def test_desktop_bundle_keeps_streamlit_startup_modules():
    source = Path("packaging/CVMonitor.spec").read_text(encoding="utf-8")

    assert 'collect_all("streamlit")' in source
    assert '"streamlit.hello",' not in source
    assert '"streamlit.testing",' not in source
