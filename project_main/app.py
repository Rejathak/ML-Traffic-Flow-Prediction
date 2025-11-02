import streamlit as st
import joblib
import pandas as pd
import json
import warnings
from datetime import datetime

# --- *** FIX IS HERE *** ---
# st.set_page_config() must be the first Streamlit command.
st.set_page_config(page_title="Traffic Volume Predictor", layout="centered")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. LOAD ARTIFACTS (CACHE THIS!) ---
@st.cache_resource
def load_artifacts():
    """Loads all the necessary files for the app."""
    try:
        model = joblib.load("traffic_model.joblib")
        
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
            
        # *** This file seems to be missing, causing the error ***
        with open('data_ranges.json', 'r') as f:
            data_ranges = json.load(f)
            
        return model, model_columns, data_ranges
    except FileNotFoundError:
        st.error("ERROR: Model files not found (e.g., 'data_ranges.json'). Please run train_model.py first.")
        return None, None, None

model, model_columns, data_ranges = load_artifacts()

# --- 2. PREDICTION LOGIC ---

def classify_traffic(volume, thresholds):
    """Converts a number into a human-readable label."""
    if volume <= thresholds['low']:
        return "LOW"
    elif volume <= thresholds['high']:
        return "MEDIUM"
    else:
        return "HIGH"

def process_and_predict_manual(temp, weather_main, dt_obj):
    """
    Takes user inputs and builds the feature row for prediction.
    """
    data_row = {col: 0 for col in model_columns} # Start with empty row

    # A. Date/Time features
    data_row['hour'] = dt_obj.hour
    
    # === THIS IS THE FIXED LINE ===
    # The correct datetime method is .weekday() (which you've already fixed!)
    data_row['day_of_week'] = dt_obj.weekday()  
    # === END OF FIX ===
    
    data_row['month'] = dt_obj.month
    data_row['year'] = dt_obj.year

    # B. Temp feature
    data_row['temp'] = temp

    # C. Weather feature
    weather_main_col = f"weather_main_{weather_main}"
    if weather_main_col in data_row:
        data_row[weather_main_col] = 1
    else:
        st.warning(f"Weather '{weather_main}' was not in training data. Ignoring.")

    # D. Predict
    try:
        data_df = pd.DataFrame([data_row], columns=model_columns)
        prediction = model.predict(data_df)
        return prediction[0]
    except Exception as e:
        # We'll show the actual error to help debug
        st.error(f"Error during prediction: {e}")
        return None


# --- 3. BUILD THE STREAMLIT UI ---

st.title("🚦 Interstate Traffic Volume Predictor")

# --- *** FIX IS HERE *** ---
# Only run the app logic IF the models loaded successfully.
if model is not None and model_columns is not None and data_ranges is not None:
    
    # This line was causing the error. It's now safely inside the 'if' block.
    traffic_thresholds = data_ranges.get('traffic_thresholds', {'low': 2000, 'high': 4500})
    
    st.markdown("Enter the conditions below to predict the traffic volume.")

    # Get the validation ranges
    temp_min = data_ranges['temp']['min']
    temp_max = data_ranges['temp']['max']
    weather_categories = data_ranges['weather_main']['categories']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date and Time inputs
        date_input = st.date_input("Select Date")
        time_input = st.time_input("Select Time")
    
    with col2:
        # Temp input (with validation)
        temp_input = st.number_input(
            f"Temperature (in Kelvin, {temp_min:.0f}-{temp_max:.0f})",
            min_value=0.0,  # Allow 0K but will show warning
            value=289.0,
            step=1.0
        )
        
        # Weather input (dropdown box)
        weather_input = st.selectbox(
            "Select Weather Condition",
            options=weather_categories
        )

    # --- Prediction Button ---
    if st.button("Predict Traffic Volume", use_container_width=True):
        
        # --- 1. Validation ---
        if not (temp_min <= temp_input <= temp_max):
            st.warning(f"Temperature {temp_input}K is outside the model's reliable range "
                       f"({temp_min:.0f}K - {temp_max:.0f}K). Prediction may be inaccurate.")
        
        # --- 2. Process Data ---
        try:
            # Combine date and time into a single datetime object
            dt_obj = datetime.combine(date_input, time_input)
            
            # --- 3. Get Prediction ---
            prediction = process_and_predict_manual(temp_input, weather_input, dt_obj)
            
            if prediction is not None:
                # --- 4. Display Results ---
                traffic_label = classify_traffic(prediction, traffic_thresholds)
                
                st.success(f"**Predicted Traffic Volume:** `{prediction:.0f}` vehicles")
                st.subheader(f"(This is {traffic_label} traffic)")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    # This will be shown if the artifacts failed to load
    st.warning("Application is not ready. Please check the error message above.")

