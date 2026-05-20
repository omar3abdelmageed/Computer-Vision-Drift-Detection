import streamlit as st

def kpi_card(title: str, value: str, sub_text: str = "", is_alert: bool = False):
    """Renders a single KPI scorecard using markdown/HTML for better styling."""
    color = "#d9534f" if is_alert else "#5cb85c"
    html = f"""
    <div style='padding: 15px; border-radius: 8px; border: 1px solid #ddd; background-color: white; margin-bottom: 10px;'>
        <h4 style='margin: 0; color: #555;'>{title}</h4>
        <h2 style='margin: 10px 0; color: {color};'>{value}</h2>
        <p style='margin: 0; font-size: 12px; color: #888;'>{sub_text}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
