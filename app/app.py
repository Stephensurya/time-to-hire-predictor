# app/app.py

import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Time to Hire Predictor",
    layout="wide"
)

st.title("⏳ Time to Hire Prediction")
st.caption("Predict the expected hiring duration based on role, process, and candidate inputs.")

st.divider()

# -----------------------------
# Load model artifacts
# -----------------------------
MODEL_PATH = "model/time_to_hire_model.pkl"
FREQ_PATH = "model/job_role_frequency.pkl"
LOG_PATH = "data/prediction_logs.csv"

pipeline = joblib.load(MODEL_PATH)
job_role_freq = joblib.load(FREQ_PATH)

# -----------------------------
# Input Layout (3 Columns)
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📌 Role Details")
    job_family = st.selectbox(
        "Job Family *",
        ["Technology", "Sales", "HR", "Operations"]
    )

    job_role = st.text_input(
        "Job Role *",
        placeholder="e.g. Data Engineer"
    )

    job_band = st.selectbox(
        "Job Band *",
        ["Band_1", "Band_2", "Band_3", "Band_4", "Band_5", "Band_6"]
    )

    experience = st.number_input(
        "Experience Required (Years) *",
        min_value=0,
        max_value=30,
        value=3
    )

with col2:
    st.subheader("🏢 Location & Source")
    city = st.selectbox(
        "City *",
        ["Bangalore", "Hyderabad", "Chennai", "Tokyo", "Osaka", "Beijing", "Shanghai"]
    )

    candidate_source = st.selectbox(
        "Candidate Source *",
        ["LinkedIn", "Naukri", "Referral", "Consultant", "Other"]
    )

    assessment = st.selectbox(
        "Assessment Included *",
        ["Yes", "No"]
    )

    notice = st.selectbox(
        "Notice Period (Days) *",
        [30, 45, 60, 90]
    )

with col3:
    st.subheader("🧪 Interview Process")
    panel_days = st.number_input(
        "Panel Availability (Days) *",
        min_value=0,
        max_value=30,
        value=3
    )

    rounds = st.number_input(
        "Total Interview Rounds *",
        min_value=1,
        max_value=5,
        value=2
    )

st.divider()

# -----------------------------
# Predict Button (Centered)
# -----------------------------
predict_col = st.columns([3, 2, 3])[1]

with predict_col:
    predict_clicked = st.button("🚀 Predict Time to Hire", use_container_width=True)

# -----------------------------
# Prediction Output (Card)
# -----------------------------
if predict_clicked:

    if job_role.strip() == "":
        st.warning("Please enter a Job Role.")
    else:
        job_role_encoded = job_role_freq.get(job_role, 0)

        input_df = pd.DataFrame([{
            "Job_Family": job_family,
            "Job_Role": job_role_encoded,
            "Job_Band": job_band,
            "Experience_Required_Years": experience,
            "City": city,
            "Interview_Panel_Availability": panel_days,
            "Total_Interview_Rounds": rounds,
            "Assessment_Included": assessment,
            "Candidate_Source": candidate_source,
            "Notice_Period_Days": notice,
        }])

        prediction = pipeline.predict(input_df)[0]

        st.markdown(
            f"""
            <div style="
                background-color:#f0f9ff;
                padding:25px;
                border-radius:12px;
                border-left:6px solid #1f77b4;
                text-align:center;
                margin-top:20px;
            ">
                <h3 style="color:#1f77b4;">⏱ Estimated Time to Hire</h3>
                <h1 style="margin:0;">{int(prediction)} Days</h1>
                <p style="color:gray;margin-top:10px;">
                    Prediction based on historical hiring patterns
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Log prediction
        # -----------------------------
        log_df = input_df.copy()
        log_df["Predicted_Time_to_Hire"] = int(prediction)
        log_df["Prediction_Timestamp"] = datetime.now()

        os.makedirs("data", exist_ok=True)
        log_df.to_csv(
            LOG_PATH,
            mode="a",
            header=not os.path.exists(LOG_PATH),
            index=False
        )
