import streamlit as st
import plotly.express as px
import pandas as pd
from src.database.models_db import get_models_by_user, get_drift_metrics
from src.auth.session_state import get_current_user
from src.ui.components import kpi_card

def render_dashboard_view():
    """Renders the main dashboard for drift visualization."""
    user = get_current_user()
    models = get_models_by_user(user.id)
    
    if not models:
        st.info("You haven't uploaded any models yet. Go to 'Manage Models' to get started.")
        return
        
    st.header("Live Production Dashboard")
    
    # Model Selection
    model_options = {m["id"]: m["name"] for m in models}
    selected_model_id = st.selectbox("Select Model", options=list(model_options.keys()), format_func=lambda x: model_options[x])
    
    if not selected_model_id:
        return
        
    model_info = next(m for m in models if m["id"] == selected_model_id)
    thresholds = model_info.get("thresholds", {"data": 0.05, "prediction": 0.1, "concept": 50})
    
    # Action button for live evaluation simulation
    if st.button("Start Live Production Evaluation"):
        st.info("Simulating production batch evaluation... (Integration with analytics pipeline goes here)")
        
    # Fetch metrics
    metrics = get_drift_metrics(selected_model_id)
    
    if not metrics:
        st.warning("No drift metrics available for this model yet.")
        return
        
    df = pd.DataFrame(metrics)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    latest = df.iloc[-1]
    
    st.divider()
    
    # KPI Scorecard
    st.subheader("Current Drift Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        is_alert = latest["data_drift_score"] > (1.0 - thresholds["data"])
        kpi_card("Data Drift", f"{latest['data_drift_score']:.2f}", "Alert if > Threshold", is_alert)
    with col2:
        is_alert = latest["prediction_drift_score"] > thresholds["prediction"]
        kpi_card("Prediction Drift", f"{latest['prediction_drift_score']:.2f}", "KL Divergence", is_alert)
    with col3:
        is_alert = latest["concept_drift_score"] > (thresholds["concept"] / 100.0)
        kpi_card("Concept Drift", f"{latest['concept_drift_score']:.2f}", "Page-Hinkley Proxy", is_alert)
        
    st.divider()
    
    # Drill-down charts
    tab1, tab2, tab3 = st.tabs(["Data Drift Trend", "Prediction Drift Trend", "Concept Drift Trend"])
    
    with tab1:
        fig = px.line(df, x="timestamp", y="data_drift_score", title="Data Drift Over Time")
        fig.add_hline(y=(1.0 - thresholds["data"]), line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)
        if latest["data_drift_score"] > (1.0 - thresholds["data"]):
            st.error("**Insights & Actionable Solutions:** Data drift detected due to structural image changes. Action: Check physical camera lighting, alignment, or retrain with data augmentation.")
            
    with tab2:
        fig = px.line(df, x="timestamp", y="prediction_drift_score", title="Prediction Drift Over Time")
        fig.add_hline(y=thresholds["prediction"], line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        fig = px.line(df, x="timestamp", y="concept_drift_score", title="Concept Drift Over Time")
        st.plotly_chart(fig, use_container_width=True)
