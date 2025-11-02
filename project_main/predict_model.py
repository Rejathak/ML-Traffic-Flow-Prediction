# predict.py

import joblib
import pandas as pd
import json
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. GLOBAL: LOAD ALL ARTIFACTS ---
try:
    model = joblib.load("traffic_model.joblib")
    
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
        
    with open('data_ranges.json', 'r') as f:
        data_ranges = json.load(f)
        
    # Load traffic thresholds
    traffic_thresholds = data_ranges.get('traffic_thresholds', 
                                         {'low': 2000, 'high': 4500})
        
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("One or more files ('traffic_model.joblib', 'model_columns.json', 'data_ranges.json') not found.")
    print("Please run train_model.py first to create these files.")
    exit()

# --- 2. PREDICTION FUNCTIONS ---

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
    Mode 2: Takes user inputs and builds the feature row.
    """
    data_row = {col: 0 for col in model_columns}

    # A. Date/Time features
    data_row['hour'] = dt_obj.hour
    data_row['day_of_week'] = dt_obj.dayofweek
    data_row['month'] = dt_obj.month
    data_row['year'] = dt_obj.year

    # B. Temp feature
    data_row['temp'] = temp

    # === C. Weather feature ===
    weather_main_col = f"weather_main_{weather_main}"
    if weather_main_col in data_row:
        data_row[weather_main_col] = 1

    # D. Predict
    try:
        data_df = pd.DataFrame([data_row], columns=model_columns)
        prediction = model.predict(data_df)
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def process_and_predict_from_row(selected_row_series):
    """
    Mode 1: Takes a full row from the test_set CSV and processes it.
    """
    data_row = {col: 0 for col in model_columns}

    try:
        dt = pd.to_datetime(selected_row_series['date_time'], dayfirst=True)
        data_row['hour'] = dt.hour
        data_row['day_of_week'] = dt.dayofweek
        data_row['month'] = dt.month
        data_row['year'] = dt.year
    except Exception as e:
        print(f"Error processing date_time: {e}")
        return None

    data_row['temp'] = selected_row_series['temp']
    
    # === Add weather_main ===
    weather_main_col = f"weather_main_{selected_row_series['weather_main']}"
    if weather_main_col in data_row:
        data_row[weather_main_col] = 1

    try:
        data_df = pd.DataFrame([data_row], columns=model_columns)
        prediction = model.predict(data_df)
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- 3. HELPER FUNCTIONS FOR MODES ---

def get_numeric_input(prompt, min_val, max_val):
    """Helper to get a number from the user, with validation."""
    while True:
        val_str = input(prompt)
        if val_str.lower() == 'q':
            return 'q'
        try:
            val_float = float(val_str)
            if not (min_val <= val_float <= max_val):
                print("="*50)
                print(f"--- WARNING: INPUT OUTSIDE TRAINING RANGE ---")
                print(f"Your input: {val_float}K")
                print(f"Model was trained on data between: {min_val:.2f}K and {max_val:.2f}K")
                print("The prediction will be unreliable.")
                print("="*50)
            return val_float
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_categorical_input(prompt, allowed_categories):
    """Helper to get categorical input, with validation."""
    while True:
        val_str = input(prompt)
        if val_str.lower() == 'q':
            return 'q'
        
        if val_str not in allowed_categories:
            print("="*50)
            print(f"--- WARNING: UNKNOWN CATEGORY ---")
            print(f"Your input: '{val_str}' was not in the training data.")
            print(f"The model will treat this as an 'unknown' category.")
            print("Prediction will be less reliable.")
            print(f"Known categories are: {', '.join(allowed_categories)}")
            print("="*50)
        return val_str

def get_date_time_input(date_prompt, time_prompt):
    """Helper to get date and time separately. Allows future dates."""
    while True:
        date_str = input(date_prompt)
        if date_str.lower() == 'q':
            return 'q'
        
        time_str = input(time_prompt)
        if time_str.lower() == 'q':
            return 'q'
        
        val_str = f"{date_str} {time_str}"
        try:
            dt = pd.to_datetime(val_str, format='%d-%m-%Y %H:%M')
            return dt
        except ValueError:
            print("Invalid date/time format. Please use 'DD-MM-YYYY' and 'HH:MM'.")
            print("Please try entering both again.")

def run_comparison_mode():
    """Runs the "Compare with Test Data" mode."""
    print("\n--- Mode 1: Compare with Test Data ---")
    
    if not os.path.exists("user_test_set.csv"):
        print("\n--- ERROR ---")
        print("File 'user_test_set.csv' not found.")
        print("Please run train_model.py to generate this file.")
        return

    test_set_df = pd.read_csv("user_test_set.csv").reset_index(drop=True)
    
    print("\nHere are the first 5 rows from the test set:")
    try:
        print(test_set_df.head().to_markdown(numalign="left", stralign="left"))
    except ImportError:
        print("(Note: 'tabulate' not installed. Printing basic table.)")
        print(test_set_df.head())
    
    total_rows = len(test_set_df)
    print(f"\n(Total rows in test set: {total_rows})")

    while True:
        try:
            row_num_str = input(f"\nEnter a row number (0 to {total_rows - 1}) to predict (or 'b' to go back): ")
            
            if row_num_str.lower() == 'b':
                break
            row_num = int(row_num_str)
            if not (0 <= row_num < total_rows):
                print(f"Error: Please enter a number between 0 and {total_rows - 1}.")
                continue
                
            selected_row = test_set_df.iloc[row_num]
            actual_traffic = selected_row['traffic_volume']
            
            predicted_traffic = process_and_predict_from_row(selected_row)
            
            if predicted_traffic is not None:
                traffic_label = classify_traffic(predicted_traffic, traffic_thresholds)
                
                print("\n--- Prediction Comparison ---")
                print(f"Data for row:     {row_num}")
                print(f"Time:             {selected_row['date_time']}")
                print(f"Weather:          {selected_row['weather_main']}") 
                
                difference = predicted_traffic - actual_traffic
                percent_error_str = "N/A (Actual was 0)"
                if actual_traffic != 0:
                    percent_error = (difference / actual_traffic) * 100
                    percent_error_str = f"{percent_error:.2f}%"
                
                print("\nRESULTS:")
                print(f" > Actual Traffic:    {actual_traffic}")
                print(f" > Predicted Traffic: {predicted_traffic:.0f} (This is {traffic_label} traffic)")
                print(f" > Difference:        {difference:.0f} vehicles")
                print(f" > Percent Error:     {percent_error_str}")

        except ValueError:
            print("Error: Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def run_manual_mode():
    """Runs the "Manual Data Entry" mode (simplified)."""
    print("\n--- Mode 2: Manual Data Entry ---")
    
    temp_min = data_ranges['temp']['min']
    temp_max = data_ranges['temp']['max']
    temp_prompt = f"Enter Temperature (in Kelvin, e.g., {temp_min:.0f}-{temp_max:.0f}): "
    
    # === ADDED weather_main prompts ===
    weather_categories = data_ranges['weather_main']['categories']
    weather_prompt = f"Enter Weather Main (e.g., {', '.join(weather_categories[:3])}): "
    
    date_prompt = "Enter Date (DD-MM-YYYY): "
    time_prompt = "Enter Time (HH:MM): "

    while True:
        print("\nEnter new data for prediction (or type 'q' at any prompt to quit):")
        
        temp = get_numeric_input(temp_prompt, temp_min, temp_max)
        if temp == 'q': break
        
        # === ADDED: Ask for weather_main ===
        weather_main = get_categorical_input(weather_prompt, weather_categories)
        if weather_main == 'q': break
        
        dt_obj = get_date_time_input(date_prompt, time_prompt)
        if dt_obj == 'q': break

        # === UPDATED: Pass weather_main to the function ===
        predicted_traffic = process_and_predict_manual(temp, weather_main, dt_obj)
        
        if predicted_traffic is not None:
            traffic_label = classify_traffic(predicted_traffic, traffic_thresholds)
            
            print("\n" + "-" * 40)
            print(f"  The Predicted Traffic Volume is: {predicted_traffic:.0f} (This is {traffic_label} traffic)")
            print("-" * 40 + "\n")
        
        again = input("Predict again with new data? (y/n): ")
        if again.lower() != 'y':
            break

# --- 4. MAIN SCRIPT ---
if __name__ == "__main__":
    
    print("\n" + "=" * 40)
    print("     Traffic Volume Prediction")
    print("=" * 40)
    print("Model loaded successfully.")
    
    while True:
        print("\n--- Main Menu ---")
        print("Choose a prediction mode:")
        print("  1: Compare with Test Data")
        print("  2: Enter Manual Data")
        print("  q: Quit")
        
        choice = input("Enter choice (1, 2, or q): ")
        
        if choice == '1':
            run_comparison_mode()
        elif choice == '2':
            run_manual_mode()
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or q.")

    print("\nExiting program.")