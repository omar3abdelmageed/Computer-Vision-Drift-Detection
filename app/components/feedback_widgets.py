from __future__ import annotations

import streamlit as st


def render_feedback_form(feedback_repo, images_repo, predictions_repo, selected_model: dict, session: dict | None) -> None:
    st.subheader("8. Human Feedback")
    if not session:
        st.info("Select a session before submitting feedback.")
        return
    images = images_repo.where(session_id=session["id"])
    if not images:
        st.info("No images are available for review.")
        return
    image_options = {row.get("filename", row["id"]): row for row in images}
    image = image_options[st.selectbox("Image", list(image_options.keys()))]
    predictions = predictions_repo.where(image_id=image["id"])
    prediction_id = predictions[0]["id"] if predictions else None
    task = selected_model.get("selected_task_type")
    with st.form("feedback_form"):
        if task == "classification":
            feedback_type = st.selectbox("Feedback", ["approve", "reject", "corrected_label"])
            corrected_class = st.text_input("Corrected class")
            payload = {"corrected_class": corrected_class} if corrected_class else {}
        else:
            feedback_type = st.selectbox("Feedback", ["approve", "reject", "false_positive", "missed_object", "wrong_class"])
            payload = {}
        comment = st.text_area("Comment")
        if st.form_submit_button("Submit feedback"):
            feedback_repo.insert({"image_id": image["id"], "prediction_id": prediction_id, "feedback_type": feedback_type, "corrected_payload": payload, "comment": comment})
            st.success("Feedback submitted.")
