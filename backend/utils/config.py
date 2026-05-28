from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: str | Path = "secrets/.env") -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path)


@dataclass(frozen=True)
class AppConfig:
    supabase_url: str | None
    supabase_anon_key: str | None
    supabase_service_role_key: str | None
    local_workspace_dir: Path
    app_env: str
    log_level: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_env_file()
        return cls(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY"),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            local_workspace_dir=Path(os.getenv("LOCAL_WORKSPACE_DIR", "workspace")),
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
