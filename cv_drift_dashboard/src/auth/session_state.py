import streamlit as st

def init_session_state():
    """Initialize necessary session state variables for authentication."""
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "auth_token" not in st.session_state:
        st.session_state["auth_token"] = None
    if "current_model_id" not in st.session_state:
        st.session_state["current_model_id"] = None

def set_user(user_data, token):
    """Set the authenticated user session."""
    st.session_state["user"] = user_data
    st.session_state["auth_token"] = token

def clear_user():
    """Clear the user session (Logout)."""
    st.session_state["user"] = None
    st.session_state["auth_token"] = None
    st.session_state["current_model_id"] = None

def get_current_user():
    """Get the currently logged-in user."""
    return st.session_state.get("user")
