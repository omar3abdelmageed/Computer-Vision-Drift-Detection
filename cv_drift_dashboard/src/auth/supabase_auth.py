import streamlit as st
from src.database.client import supabase
from src.auth.session_state import set_user, clear_user

def sign_up(email, password):
    """Register a new user with Supabase."""
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            st.success("Sign up successful! You may be able to login immediately.")
            return True
        return False
    except Exception as e:
        st.error(f"Sign up error: {str(e)}")
        return False

def login(email, password):
    """Authenticate an existing user."""
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user and response.session:
            set_user(response.user, response.session.access_token)
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def logout():
    """Log out the current user."""
    try:
        supabase.auth.sign_out()
        clear_user()
    except Exception as e:
        st.error(f"Logout error: {str(e)}")
