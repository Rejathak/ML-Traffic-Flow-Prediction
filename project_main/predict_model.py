import joblib
import pandas as pd
import json
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. LOAD MODEL, COLUMNS, AND THE TEST SET "MENU" ---
print("Loading model, column blueprint, and test set...")
try:
    model = joblib.load("traffic_model.joblib")
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    test_set_df = pd.read_csv("user_test_set.csv")
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("One or more files ('traffic_model.joblib', 'model_columns.json', 'user_test_set.csv') not found.")
    print("Please run train_model.py first to create these files.")
    exit()

print("Model and test set loaded successfully.")

# Calculate average traffic volume for user-friendly output
average_traffic = test_set_df['traffic_volume'].mean()

def process_and_predict_from_input(user_input_dict):
    data_row = {col: 0 for col in model_columns}
    # Time features
    try:
        dt = pd.to_datetime(user_input_dict['date_time'], dayfirst=True)
        data_row['hour'] = dt.hour
        data_row['day_of_week'] = dt.dayofweek
        data_row['month'] = dt.month
        data_row['year'] = dt.year
    except Exception as e:
        print(f"Error processing date/time: {e}")
        return None
    # Numeric features
    try:
        data_row['temp'] = float(user_input_dict['temp'])
    except Exception:
        # If user didn't provide temp, use average from test set
        data_row['temp'] = test_set_df['temp'].mean()
    # Optional weather info (one-hot)
    weather_main = (user_input_dict.get('weather_main') or '').strip()

    # Set weather_main
    if weather_main:
        for col in model_columns:
            if col.startswith('weather_main_') and col.lower() == f"weather_main_{weather_main}".lower():
                data_row[col] = 1
                break
    # Set holiday to None by default (or you can prompt for it)
    if 'holiday_None' in data_row:
        data_row['holiday_None'] = 1

    # Use test set average for clouds_all if not provided
    data_row['clouds_all'] = test_set_df['clouds_all'].mean()

    try:
        data_df = pd.DataFrame([data_row], columns=model_columns)
        prediction = model.predict(data_df)
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def traffic_comment(predicted, average):
    ratio = predicted / average
    if ratio < 0.7:
        return "Traffic is less than average."
    elif ratio < 1.2:
        return "Traffic is around average."
    else:
        return "Traffic is higher than average."

# --- 3. INTERACTIVE PREDICTION LOOP ---
if __name__ == "__main__":

    print("\n--- Interactive Traffic Prediction ---")
    print("You can choose a row from the test set, or enter your own values for prediction.")

    test_set_df = test_set_df.reset_index(drop=True)
    print("\nHere are the first 5 rows from the test set ('user_test_set.csv'):")
    try:
        print(test_set_df.head().to_markdown(numalign="left", stralign="left"))
    except ImportError:
        print("\nNOTE: 'tabulate' package not found. Printing a basic table.")
        print(test_set_df.head())

    total_rows = len(test_set_df)
    print(f"\n(Total rows in test set: {total_rows})")

    while True:
        print("\nOptions:")
        print("  [0] Predict using a row from the test set")
        print("  [1] Enter your own data for prediction")
        print("  [q] Quit")
        choice = input("Enter your choice: ").strip().lower()
        if choice == 'q':
            break
        elif choice == '0':
            try:
                row_num_str = input(f"\nEnter a row number (0 to {total_rows - 1}) to predict: ")
                row_num = int(row_num_str)
                if not (0 <= row_num < total_rows):
                    print(f"Error: Please enter a number between 0 and {total_rows - 1}.")
                    continue
                selected_row = test_set_df.iloc[row_num]
                actual_traffic = selected_row['traffic_volume']
                predicted_traffic = process_and_predict_from_input(selected_row)
                print("\n--- Prediction Comparison ---")
                print(f"Data for row:     {row_num}")
                print(f"Time:             {selected_row['date_time']}")
                print(f"Temperature:      {selected_row['temp']} °C")
                print(f"Weather:          {selected_row['weather_main']}")
                print("\nRESULTS:")
                print(f" > Actual Traffic:    {actual_traffic}")
                print(f" > Predicted Traffic: {predicted_traffic:.0f}")
                # Percentage difference calculation
                if actual_traffic == 0:
                    print(f" > Accuracy:          Cannot compute percentage difference (actual value is 0)")
                else:
                    percent_diff = abs(predicted_traffic - actual_traffic) / actual_traffic * 100
                    print(f" > Accuracy:          {100 - percent_diff:.2f}% (Percentage similarity between prediction and actual value)")
                    print(f" > Percentage Difference: {percent_diff:.2f}%")
                print(f" > {traffic_comment(predicted_traffic, average_traffic)}")
            except Exception as e:
                print(f"An error occurred: {e}")
        elif choice == '1':
            print("Enter your data below. (Leave temperature or weather blank if not known)")
            user_input = {}
            date_str = input("Date (YYYY-MM-DD): ")
            time_str = input("Time (HH:MM, 24-hour): ")
            try:
                user_input['date_time'] = f"{date_str} {time_str}"
                # Validate date/time
                _ = pd.to_datetime(user_input['date_time'], dayfirst=True)
            except Exception:
                print("Invalid date or time format. Try again.")
                continue
            temp = input("Temperature (°C, optional): ").strip()
            user_input['temp'] = temp if temp else None
            weather_main = input("Weather main (e.g. Clear, Clouds, Rain) [optional]: ").strip()
            user_input['weather_main'] = weather_main if weather_main else None
            predicted_traffic = process_and_predict_from_input(user_input)
            print("\n--- Prediction Result ---")
            print(f"Predicted Traffic Volume: {predicted_traffic:.0f}")
            print(f"{traffic_comment(predicted_traffic, average_traffic)}")
        else:
            print("Invalid choice. Please select 0, 1, or q.")

    print("\nExiting program.")