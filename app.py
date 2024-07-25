# predict.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocess import preprocess_data

# Load the pre-trained model
model = joblib.load('xgboost_model.pkl')

# Streamlit app with a blue theme
st.set_page_config(page_title='Yearly Amount Spent Prediction', page_icon=':moneybag:', layout='wide')

# CSS to inject a blue color theme
st.markdown("""
    <style>
        .main {
            background-color: #121212;
            color: #e0e0e0;
        }
        .sidebar .sidebar-content {
            background-color: #333333;
            color: #e0e0e0;
        }
        .stButton>button {
            color: #e0e0e0;
            background-color: #555555;
        }
        .stTextInput>div>input {
            background-color: #1e1e1e;  /* Dark input background */
            color: #e0e0e0;  /* Light text color in inputs */
        }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title('Yearly Amount Spent Prediction')
st.write("Input the details to predict the yearly amount spent:")

# User inputs for prediction
avg_session_length = st.number_input('Avg. Session Length', min_value=0.0, value=34.5, step=0.1)
time_on_app = st.number_input('Time on App', min_value=0.0, value=12.5, step=0.1)
time_on_website = st.number_input('Time on Website', min_value=0.0, value=39.5, step=0.1)
length_of_membership = st.number_input('Length of Membership', min_value=0.0, value=4.0, step=0.1)

# Prediction button
if st.button('Predict'):
    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        'Avg. Session Length': [avg_session_length],
        'Time on App': [time_on_app],
        'Time on Website': [time_on_website],
        'Length of Membership': [length_of_membership]
    })

    # Preprocess the new data (although outlier removal might not be necessary for a single input)
    new_data_processed = preprocess_data(new_data)

    # Make prediction
    prediction = model.predict(new_data_processed)
    
    # Display the prediction
    st.write(f'Predicted Yearly Amount Spent: ${prediction[0]:.2f}')

# Footer
st.markdown("""
    <footer>
        <div style="text-align:center; margin-top:20px;">
            <p>Developed by VictorCode</p>
        </div>
    </footer>
    """, unsafe_allow_html=True)
