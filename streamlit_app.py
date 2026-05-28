from __future__ import annotations

import streamlit as st

from app.tabs.live_sessions import render_live_sessions_tab
from app.tabs.model_registration import render_model_registration_tab
from backend.database.supabase_client import create_supabase_client
from backend.utils.config import AppConfig


def supabase_is_configured() -> bool:
    config = AppConfig.from_env()
    return bool(config.supabase_url and config.supabase_anon_key)


def render_auth_screen() -> None:
    st.title("Computer Vision Model Monitoring")

    if not supabase_is_configured():
        st.subheader("Connect Supabase")
        st.info("Add Supabase credentials to secrets/.env before signing in.")
        st.code(
            "SUPABASE_URL=...\n"
            "SUPABASE_ANON_KEY=...\n"
            "SUPABASE_SERVICE_ROLE_KEY=...",
            language="dotenv",
        )
        st.stop()

    auth_client = create_supabase_client(use_service_role=False)
    if auth_client is None:
        st.error("Supabase client could not be created.")
        st.stop()

    st.subheader("Sign in")
    auth_tab, signup_tab = st.tabs(["Sign In", "Create Account"])

    with auth_tab:
        with st.form("sign_in"):
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            try:
                response = auth_client.auth.sign_in_with_password({"email": email, "password": password})
            except Exception as exc:
                st.error(f"Sign in failed: {exc}")
            else:
                st.session_state["auth_user"] = response.user
                st.session_state["auth_session"] = response.session
                st.rerun()

    with signup_tab:
        with st.form("sign_up"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            try:
                response = auth_client.auth.sign_up({"email": email, "password": password})
            except Exception as exc:
                st.error(f"Account creation failed: {exc}")
            else:
                st.success("Account created. Check your email if confirmation is enabled, then sign in.")
                if response.user and response.session:
                    st.session_state["auth_user"] = response.user
                    st.session_state["auth_session"] = response.session
                    st.rerun()

    st.stop()


def render_authenticated_header() -> None:
    user = st.session_state.get("auth_user")
    email = getattr(user, "email", None) or "Signed in"
    left, right = st.columns([1, 0.2])
    left.title("Computer Vision Model Monitoring")
    if right.button("Sign out"):
        auth_client = create_supabase_client(use_service_role=False)
        if auth_client is not None:
            auth_client.auth.sign_out()
        st.session_state.pop("auth_user", None)
        st.session_state.pop("auth_session", None)
        st.rerun()
    st.caption(email)


def main() -> None:
    st.set_page_config(page_title="CV Model Monitoring", layout="wide")

    if "auth_user" not in st.session_state:
        render_auth_screen()

    render_authenticated_header()
    client = create_supabase_client(use_service_role=True)
    if client is None:
        st.error("Supabase persistence is unavailable. Check secrets/.env.")
        st.stop()

    tab_registration, tab_sessions = st.tabs(["Model Management", "Live Sessions"])
    with tab_registration:
        render_model_registration_tab(client)
    with tab_sessions:
        render_live_sessions_tab(client)


if __name__ == "__main__":
    main()
