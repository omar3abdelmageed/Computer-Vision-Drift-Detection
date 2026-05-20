import streamlit as st
from src.database.models_db import insert_model
from src.auth.session_state import get_current_user

def render_upload_view():
    """Renders the view for onboarding new models and datasets."""
    st.header("Register New Model")
    user = get_current_user()
    
    with st.form("onboarding_form"):
        model_name = st.text_input("Model Name")
        model_type = st.selectbox("Model Type", ["Classification", "Object Detection"])
        classes = st.text_input("Class Labels (comma-separated)")
        
        st.subheader("Drift Thresholds")
        data_drift_thresh = st.slider("Data Drift Threshold (p-value)", 0.01, 0.10, 0.05)
        pred_drift_thresh = st.slider("Prediction Drift Threshold (KL Divergence)", 0.01, 0.50, 0.10)
        concept_drift_thresh = st.slider("Concept Drift Threshold (Page-Hinkley)", 10, 100, 50)
        
        # File uploaders
        model_file = st.file_uploader("Upload PyTorch Model (.pt)", type=["pt"])
        dataset_files = st.file_uploader("Upload Baseline Dataset (.jxr, .wdp, .bmp)", type=["jxr", "wdp", "jpg", "png", "bmp"], accept_multiple_files=True)
        
        submitted = st.form_submit_button("Register Model")
        if submitted:
            if not model_name or not model_file:
                st.error("Please provide a model name and upload the model file.")
            else:
                # Mock storage path (in a real app, upload via supabase.storage)
                model_data = {
                    "user_id": user.id,
                    "name": model_name,
                    "model_type": model_type,
                    "classes": [c.strip() for c in classes.split(",")] if classes else [],
                    "thresholds": {
                        "data": data_drift_thresh,
                        "prediction": pred_drift_thresh,
                        "concept": concept_drift_thresh
                    },
                    "storage_path": f"{user.id}/{model_file.name}",
                    "dataset_path": f"{user.id}/dataset"
                }
                insert_model(model_data)
                st.success("Model registered successfully!")
