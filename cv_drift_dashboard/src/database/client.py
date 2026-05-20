import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def init_connection() -> Client:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error("Failed to initialize Supabase connection. Please check your secrets.")
        st.stop()

supabase = init_connection()
