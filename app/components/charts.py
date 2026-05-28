from __future__ import annotations

import pandas as pd
import streamlit as st


def line_chart(rows: list[dict], x: str, y: str, color: str | None = None) -> None:
    if not rows:
        st.info("No chart data available.")
        return
    st.line_chart(pd.DataFrame(rows), x=x, y=y, color=color)
