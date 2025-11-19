# Students Performance Prediction using Linear Regression

A complete end-to-end Machine Learning project that uses the **Linear Regression** algorithm to predict a student’s academic performance based on key behavioral, demographic, and study-related features.  
This project includes dataset preprocessing, EDA, model training, evaluation, model saving, and a Streamlit app for real-time performance prediction.

---

## Project Overview

This project aims to predict **final student performance scores** using the following features:

- gender  
- age  
- study_time  
- absences  
- parental_education  
- test_preparation  
- past_scores  
- hours_of_sleep  
- final_grade *(Target)*  

The dataset is cleaned, encoded, scaled, and used to train a **Linear Regression model**, which is later integrated with a simple Streamlit UI.

---

## Features

- Load & clean dataset  
- Exploratory Data Analysis (visualizations included)  
- Label encoding and feature transformation  
- Train-test split  
- Linear Regression Model Training  
- Model Evaluation (MAE, MSE, RMSE, R² Score)  
- Save trained model as `.pkl`  
- Streamlit GUI for predictions  
- Lightweight & production-ready code  

---

## How the Model Works

### 1️⃣ Data Preprocessing
- Handle missing values  
- Label encoding for:  
  - gender  
  - parental_education  
  - test_preparation  
- Standardization using **StandardScaler**  
- Splitting dataset into training and testing sets  

### 2️⃣ Model Training
Algorithm used:  
**Linear Regression (from scikit-learn)**  

Train-test split:  
**80% Training, 20% Testing**

### 3️⃣ Model Evaluation
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  
- EDA visualizations (distributions, correlations, scatter plots)

---

## ▶️ Running the Project Locally

### Step 1: Install Dependencies

pip install -r requirements.txt


### Step 2: Train the Model (Optional)

python train_student_performance.py

This generates:

student_performance_model.pkl

scaler.pkl

### Step 3: Run the Streamlit App

-streamlit run app.py

-Deploying on Streamlit Cloud

-Push all project files to GitHub

-Visit: https://share.streamlit.io

-Click New App

-Select your GitHub repository

-Choose app.py as the main file

-Deploy

###Technologies Used

-Python

-Pandas

-NumPy

-Matplotlib

-Seaborn

-Scikit-Learn

-Streamlit

-Joblib
