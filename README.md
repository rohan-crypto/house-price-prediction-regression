House Price Prediction using Machine Learning

Overview

This project predicts house prices using regression techniques based on features such as square footage, number of bedrooms, lot size, year built, and neighborhood quality.
The goal of this project is to build an end-to-end machine learning pipeline — from data exploration and model training to deployment using a Streamlit web application.

Tech Stack

Python
Pandas, NumPy – Data handling
Matplotlib, Seaborn – Visualization
Scikit-learn – Model training & evaluation
Streamlit – Web application
Pickle – Model persistence

Dataset

Source: Kaggle (House Price Regression Dataset)
Rows: 1,000

Features:

Square_Footage
Num_Bedrooms
Num_Bathrooms
Year_Built
Lot_Size
Garage_Size
Neighborhood_Quality

Target:
House_Price

The dataset contains no missing values, making it suitable for regression modeling.

Project Workflow

Data loading and exploratory data analysis (EDA)
Feature selection and train-test split
Model training using:
Linear Regression
Ridge Regression
Lasso Regression

Model evaluation using:
RMSE
R² score
Cross-validation
Feature importance analysis
Saving trained model and scaler using Pickle
Deployment using Streamlit

Model Performance

Best R² Score: ~0.99
Cross-Validation R²: ~0.998
RMSE: ~10,000 – 20,000 (depending on model)

Feature importance analysis showed Square Footage as the most influential predictor.


Streamlit Web App

The Streamlit app allows users to:
Input house features using sliders and number inputs
Get real-time house price predictions
Reset inputs using session state handling

Run the app:

streamlit run app.py

# How to Run Locally
## Clone repository
git clone https://github.com/rohan-crypto/house-price-prediction-regression.git

cd House-Price-Prediction-Regression

## Create virtual environment
python3 -m venv venv
source venv/bin/activate

## Install dependencies
pip install -r requirements.txt

## Run Streamlit app
streamlit run app.py


Key Learnings

End-to-end ML workflow implementation
Regression model comparison and evaluation
Feature importance interpretation
Model serialization using Pickle
Building interactive ML web apps with Streamlit
Managing application state in Streamlit

Future Improvements

Add confidence intervals to predictions
Deploy app on Streamlit Cloud
Add model comparison toggle in UI
Improve feature engineering
Use real-world housing datasets
