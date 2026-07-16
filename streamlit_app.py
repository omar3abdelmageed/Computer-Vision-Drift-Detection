from __future__ import annotations

import streamlit as st

from dashboard.views.live_sessions import render_live_sessions_tab
from dashboard.views.model_management import render_model_registration_tab
from database.local import create_local_database


def main() -> None:
    st.set_page_config(page_title="CV Model Monitoring", layout="wide")
    st.title("Computer Vision Model Monitoring")

    database = create_local_database()
    tab_registration, tab_sessions = st.tabs(["Model Management", "Live Sessions"])
    with tab_registration:
        render_model_registration_tab(database)
    with tab_sessions:
        render_live_sessions_tab(database)


if __name__ == "__main__":
    main()
