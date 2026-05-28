from __future__ import annotations

import streamlit as st


def status_card(title: str, status: str | None, detail: str = "") -> None:
    status = status or "pending"
    if status in {"valid", "ok", "completed", "baseline_completed"}:
        st.success(f"{title}: {status}. {detail}")
    elif status in {"warning", "running", "not_started", "draft", "pending", "uploaded"}:
        st.warning(f"{title}: {status}. {detail}")
    elif status in {"invalid", "failed", "critical"}:
        st.error(f"{title}: {status}. {detail}")
    else:
        st.info(f"{title}: {status}. {detail}")
