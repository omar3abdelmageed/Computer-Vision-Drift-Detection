from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_image(path: str | Path | None, caption: str | None = None) -> None:
    if not path:
        st.info("No image selected.")
        return
    path = Path(path)
    if not path.exists():
        st.warning(f"Image not found: {path}")
        return
    st.image(str(path), caption=caption or path.name, use_container_width=True)
