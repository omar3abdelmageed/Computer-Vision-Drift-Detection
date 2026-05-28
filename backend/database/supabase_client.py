from __future__ import annotations

from backend.utils.config import AppConfig


def create_supabase_client(use_service_role: bool = False):
    config = AppConfig.from_env()
    key = config.supabase_service_role_key if use_service_role and config.supabase_service_role_key else config.supabase_anon_key
    if not config.supabase_url or not key:
        return None
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("Install supabase to use database persistence.") from exc
    return create_client(config.supabase_url, key)
