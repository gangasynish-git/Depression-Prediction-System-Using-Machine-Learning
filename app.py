import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIG

st.set_page_config(
    page_title="Depression Prediction System",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# LOAD MODEL
model = joblib.load("depression_pipeline.pkl")

# CUSTOM CSS

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title-text {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        color: #2c3e50;
    }
    .subtitle-text {
        text-align: center;
        color: #6c757d;
        font-size: 18px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# HEADER
st.markdown('<div class="title-text">Depression Prediction System</div>', unsafe_allow_html=True)

st.image("depression.avif", use_container_width=True)

st.markdown("---")

# INPUT SECTIONS

st.subheader("Personal Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 60, 25)
    city = st.selectbox(
        "City",
        sorted([
            "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai",
            "Pune", "Patna", "Jaipur", "Ahmedabad", "Indore", "Bhopal",
            "Lucknow", "Kanpur", "Faridabad", "Ghaziabad", "Meerut",
            "Surat", "Vadodara", "Rajkot", "Nagpur", "Nashik",
            "Visakhapatnam", "Varanasi", "Srinagar", "Ludhiana","Kerala"
        ])
    )

with col2:
    profession = st.selectbox(
        "Profession",
        ["Student", "Working Professional", "Doctor", "Engineer", "Teacher", "Other"]
    )
    degree = st.selectbox(
        "Degree",
        ["Class 12", "BSc", "BA", "BCA", "BTech", "BCom",
         "MSc", "MA", "MCA", "MTech", "MBA", "PhD", "Other"]
    )
    family_history = st.selectbox(
        "Family History of Mental Illness",
        ["Yes", "No"]
    )

st.markdown("---")

st.subheader("Academic & Work Details")
col3, col4 = st.columns(2)

with col3:
    academic_pressure = st.slider("Academic Pressure", 0, 5, 3)
    study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)

with col4:
    work_pressure = st.slider("Work Pressure", 0, 5, 2)
    job_satisfaction = st.slider("Job Satisfaction", 0, 5, 3)
    work_hours = st.slider("Work / Study Hours", 0, 15, 6)

st.markdown("---")

st.subheader("Lifestyle & Mental Health")

col5, col6 = st.columns(2)

with col5:
    sleep_duration = st.selectbox(
        "Sleep Duration",
        ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    )
    diet = st.selectbox(
        "Dietary Habits",
        ["Healthy", "Moderate", "Unhealthy"]
    )

with col6:
    suicidal = st.selectbox("Ever had suicidal thoughts?", ["Yes", "No"])
    financial_stress = st.selectbox(
        "Financial Stress",
        ["Low", "Medium", "High"]
    )

st.markdown("---")

# INPUT DATAFRAME
input_df = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "City": [city],
    "Profession": [profession],
    "Academic Pressure": [academic_pressure],
    "Work Pressure": [work_pressure],
    "CGPA": [cgpa],
    "Study Satisfaction": [study_satisfaction],
    "Job Satisfaction": [job_satisfaction],
    "Sleep Duration": [sleep_duration],
    "Dietary Habits": [diet],
    "Degree": [degree],
    "Have you ever had suicidal thoughts ?": [suicidal],
    "Work/Study Hours": [work_hours],
    "Financial Stress": [financial_stress],
    "Family History of Mental Illness": [family_history]
})

# PREDICTION

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Depression Risk", use_container_width=True):
    prediction = model.predict(input_df)[0]

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
        confidence = f"Confidence: {prob:.2%}"
    else:
        confidence = ""

    if prediction == 1:
        st.error(f"**High Risk of Depression**")
    else:
        st.success(f"**Low Risk of Depression**")

st.markdown("---")
st.caption("This tool is for educational purposes only and not a medical diagnosis.")

