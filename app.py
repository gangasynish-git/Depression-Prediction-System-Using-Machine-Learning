import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Depression Prediction System", layout="centered")

# load model and dataset
model = joblib.load("depression_pipeline.pkl")
data = pd.read_csv("depression.csv")

# simple sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    ["Dataset", "Prediction System"]
)

# -------------------- DATASET PAGE --------------------
if page == "Dataset":

    st.title("Depression Dataset")

    st.write("Preview of dataset")
    st.dataframe(data.head())

    st.write("Shape of dataset")
    st.write("Rows:", data.shape[0])
    st.write("Columns:", data.shape[1])

    st.write("Columns in dataset")
    st.write(list(data.columns))

    st.write("Summary statistics")
    st.dataframe(data.describe(include="all"))

    st.write("Missing values")
    st.dataframe(data.isnull().sum())


# -------------------- PREDICTION PAGE --------------------
else:

    st.title("Depression Prediction System")

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 15, 60, 25)
        city = st.selectbox("City", sorted([
            "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata",
            "Mumbai", "Pune", "Jaipur", "Ahmedabad", "Lucknow", "Kerala"
        ]))

    with col2:
        profession = st.selectbox("Profession",
                                  ["Student", "Working Professional", "Doctor",
                                   "Engineer", "Teacher", "Other"])
        degree = st.selectbox("Degree",
                              ["Class 12", "BSc", "BA", "BCA", "BTech",
                               "MSc", "MCA", "MBA", "PhD", "Other","BPharm","MBBS","BDS","MDS"])
        family_history = st.selectbox("Family History of Mental Illness",
                                      ["Yes", "No"])

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

    st.subheader("Lifestyle")

    col5, col6 = st.columns(2)

    with col5:
        sleep_duration = st.selectbox("Sleep Duration",
                                      ["Less than 5 hours", "5-6 hours",
                                       "7-8 hours", "More than 8 hours"])
        diet = st.selectbox("Dietary Habits",
                            ["Healthy", "Moderate", "Unhealthy"])

    with col6:
        suicidal = st.selectbox("Ever had suicidal thoughts?",
                                ["Yes", "No"])
        financial_stress = st.selectbox("Financial Stress",
                                        ["Low", "Medium", "High"])

    if st.button("Predict"):

        input_data = pd.DataFrame({
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

        result = model.predict(input_data)[0]

        if hasattr(model.named_steps["classifier"], "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]
            st.write("Prediction confidence:", round(prob * 100, 2), "%")

        if result == 1:
            st.error("High Risk of Depression")
        else:
            st.success("Low Risk of Depression")

    st.caption("For educational purposes only. Not a medical diagnosis.")




