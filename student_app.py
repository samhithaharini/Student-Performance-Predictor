import logging

# Keep the logger level higher to avoid noisy Streamlit warnings when
# someone accidentally runs the module in a non-Streamlit context.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_artifacts():
    try:
        model = joblib.load('student_performance_model.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        return model, feature_cols
    except FileNotFoundError as e: 
        st.warning("Model files not found. Form will display but predictions won't work.")
        return None, None
    except Exception as e:
        st.warning(f"Error loading model: {e}")
        return None, None

model, FINAL_FEATURE_COLUMNS = load_artifacts()


#st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("Student Performance Predictor")
st.markdown("Enter student characteristics to predict their final score.")


with st.form("prediction_form"):
    st.subheader("Academic and Demographic Factors") 
    col1, col2, col3 = st.columns(3)
    with col1:
        previous_score = st.number_input("Previous Semester Score", min_value=0.0, max_value=100.0, value=75.0, step=0.1, help="Score from the prior semester.")
        attendance = st.slider("Attendance Percentage", 50.0, 100.0, 80.0, 0.1, help="Student's average attendance.")
        study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=50.0, value=20.0, step=0.1)
        library_usage = st.slider("Library Usage per Week", 0, 15, 5)
        
   
    with col2:
        gender = st.selectbox("Gender", ['Female', 'Male'])
        school_type = st.selectbox("School Type", ['Private', 'Public'])
        # Include all possible categories from your training data
        parental_education = st.selectbox("Parental Education", ['Postgraduate', 'High School', 'Graduate']) 
        family_income = st.number_input("Family Income", min_value=0.0, value=50000.0, step=1000.0)
        internet_access = st.selectbox("Internet Access", ['Yes', 'No'])
        
  
    with col3:
        sleep_hours = st.slider("Sleep Hours", 4.0, 10.0, 7.0, 0.1)
        travel_time = st.slider("Travel Time (Hours)", 0.5, 5.0, 1.5, 0.1)
        test_anxiety = st.slider("Test Anxiety Level (1-10)", 1.0, 10.0, 5.0, 0.1)
        motivation = st.slider("Motivation Level (1-10)", 1.0, 10.0, 7.5, 0.1)
        peer_influence = st.slider("Peer Influence (1-10)", 1.0, 10.0, 5.0, 0.1)

        teacher_feedback = st.selectbox("Teacher Feedback", ['Good', 'Excellent', 'Average', 'Poor']) 
        tutoring = st.selectbox("Tutoring Classes", ['No', 'Yes'])
        sports = st.selectbox("Sports Activity", ['Yes', 'No'])
        extra_curricular = st.selectbox("Extra Curricular", ['Yes', 'No'])

    submitted = st.form_submit_button("Predict Final Score", type="primary")


if submitted:
    
    raw_input_data = {
        'Study_Hours_per_Week': study_hours,
        'Attendance_Percentage': attendance,
        'Previous_Sem_Score': previous_score,
        'Parental_Education': parental_education,
        'Internet_Access': internet_access,
        'Family_Income': family_income,
        'Tutoring_Classes': tutoring,
        'Sports_Activity': sports,
        'Extra_Curricular': extra_curricular,
        'School_Type': school_type,
        'Sleep_Hours': sleep_hours,
        'Travel_Time': travel_time,
        'Test_Anxiety_Level': test_anxiety,
        'Peer_Influence': peer_influence,
        'Teacher_Feedback': teacher_feedback,
        'Motivation_Level': motivation,
        'Library_Usage_per_Week': library_usage,
        'Gender': gender
    }

    input_df = pd.DataFrame([raw_input_data])

    
    categorical_cols = input_df.select_dtypes(include='object').columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    
    final_input = pd.DataFrame(0, index=[0], columns=FINAL_FEATURE_COLUMNS)

    
    for col in input_encoded.columns:
        if col in FINAL_FEATURE_COLUMNS:
            final_input[col] = input_encoded[col]

    
    try:
        prediction = model.predict(final_input)[0]
        
        
        st.success(f"## Predicted Final Score: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed. Please check your model artifacts. Error: {e}")

        #streamlit run student_app.py 