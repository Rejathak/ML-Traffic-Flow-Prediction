import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime

# Set page config FIRST!
st.set_page_config(page_title="Traffic Volume Predictor", page_icon="🚦", layout="centered")


# Load model and columns
@st.cache_resource
def load_model():
    model = joblib.load("traffic_model.joblib")
    with open("model_columns.json", "r") as f:
        model_columns = json.load(f)
    test_set_df = pd.read_csv("user_test_set.csv")
    return model, model_columns, test_set_df

model, model_columns, test_set_df = load_model()
average_traffic = test_set_df['traffic_volume'].mean()

def process_and_predict(user_input):
    data_row = {col: 0 for col in model_columns}
    dt = pd.to_datetime(user_input['date_time'])
    data_row['hour'] = dt.hour
    data_row['day_of_week'] = dt.dayofweek
    data_row['month'] = dt.month
    data_row['year'] = dt.year
    # If temp is None, use test set mean (in Kelvin)
    if user_input['temp'] is None:
        data_row['temp'] = float(test_set_df['temp'].mean())
    else:
        data_row['temp'] = user_input['temp'] + 273.15  # Convert to Kelvin if needed
    # If weather_main is None, use most common
    weather_main = user_input['weather_main']
    if weather_main and f"weather_main_{weather_main}" in model_columns:
        data_row[f"weather_main_{weather_main}"] = 1
    elif f"weather_main_{test_set_df['weather_main'].mode()[0]}" in model_columns:
        data_row[f"weather_main_{test_set_df['weather_main'].mode()[0]}"] = 1
    # Set holiday to None by default
    if "holiday_None" in data_row:
        data_row["holiday_None"] = 1
    # Use test set average for clouds_all
    data_row['clouds_all'] = test_set_df['clouds_all'].mean()
    data_df = pd.DataFrame([data_row], columns=model_columns)
    prediction = model.predict(data_df)
    return prediction[0]

def traffic_comment(predicted, average):
    ratio = predicted / average
    if ratio < 0.7:
        return "🟢 **Traffic is less than average.**"
    elif ratio < 1.2:
        return "🟡 **Traffic is around average.**"
    else:
        return "🔴 **Traffic is higher than average.**"

# --- Streamlit UI ---
st.set_page_config(page_title="Traffic Volume Predictor", page_icon="🚦", layout="centered")
st.markdown(
    """
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .stButton button {
        background-color: #0072C6;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="big-font">🚦 Traffic Volume Predictor</div>', unsafe_allow_html=True)
st.write(
    "Predict hourly traffic volume for a given date and time. "
    "Temperature and weather are optional; if left blank, the app will use typical values for you."
)

st.markdown("## Enter Details to Predict Traffic Volume:")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Date", datetime.now().date())
    time = st.time_input("Time", datetime.now().time())
with col2:
    temp_placeholder = float(test_set_df['temp'].mean()) - 273.15
    temp = st.number_input(
        "Temperature (°C) _(optional)_", 
        min_value=-50.0, max_value=60.0, value=temp_placeholder, step=0.1, format="%.1f"
    )
    temp_input = temp if st.checkbox("I know the temperature", value=False) else None

    weather_options = sorted({col.replace("weather_main_", "") for col in model_columns if col.startswith("weather_main_")})
    weather_main = st.selectbox("Weather _(optional)_", [""] + weather_options)
    weather_input = weather_main if weather_main else None

st.markdown("")

if st.button("🚗 Predict Traffic"):
    if date is None or time is None:
        st.error("Please enter both date and time.")
    else:
        user_input = {
            "date_time": f"{date} {time}",
            "temp": temp_input,
            "weather_main": weather_input
        }
        predicted_traffic = process_and_predict(user_input)
        st.success(f"### 🚦 Predicted Traffic Volume: **{predicted_traffic:.0f}** vehicles/hour")
        st.markdown(traffic_comment(predicted_traffic, average_traffic))
        st.info("Prediction uses typical temperature/weather values if left blank.")

st.markdown("---")