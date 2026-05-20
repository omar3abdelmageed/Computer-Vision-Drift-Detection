import streamlit as st

# Configure page settings first
st.set_page_config(
    page_title="CV Drift Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.auth.session_state import init_session_state, get_current_user
from src.auth.supabase_auth import logout
from src.ui.views_auth import render_auth_view
from src.ui.views_upload import render_upload_view
from src.ui.views_dashboard import render_dashboard_view

def main():
    # Initialize session
    init_session_state()
    
    user = get_current_user()
    
    if not user:
        render_auth_view()
    else:
        # Sidebar Navigation
        with st.sidebar:
            st.title("Navigation")
            st.write(f"Logged in as: {user.email}")
            
            nav_selection = st.radio(
                "Go to",
                ["Live Dashboard", "Manage Models"],
                index=0
            )
            
            st.divider()
            if st.button("Logout"):
                logout()
                st.rerun()
                
        # Main content area
        if nav_selection == "Live Dashboard":
            render_dashboard_view()
        elif nav_selection == "Manage Models":
            render_upload_view()

if __name__ == "__main__":
    main()
