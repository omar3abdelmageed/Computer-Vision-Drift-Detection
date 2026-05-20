import streamlit as st
from src.auth.supabase_auth import login, sign_up

def render_auth_view():
    """Renders the authentication view (Login / Sign Up)."""
    st.title("CV Drift Monitoring Dashboard")
    st.markdown("Please log in or sign up to continue.")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if login(email, password):
                    st.rerun()

    with tab2:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                sign_up(email, password)
