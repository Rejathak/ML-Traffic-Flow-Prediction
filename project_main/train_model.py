# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import json
import time
import numpy as np

# --- Configuration ---
DATA_FILE = "Metro_Interstate_Traffic_Volume.csv"
TEST_SET_FILE = "user_test_set.csv"
MODEL_FILE = "traffic_model.joblib"
COLUMNS_FILE = "model_columns.json"
RANGES_FILE = "data_ranges.json"

def load_data(file_path):
    """Loads and validates the initial CSV data."""
    print(f"[Step 1/9] Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"File not found: '{file_path}'. Please check the file name.")
        return None
    print("           ...Data loaded successfully.")
    return df

def save_data_validation_artifacts(df):
    """Finds and saves the min/max for temp and traffic thresholds."""
    print(f"[Step 2/9] Saving data validation ranges to '{RANGES_FILE}'...")
    
    # 1. Temperature Range
    temp_min = float(df[df['temp'] > 0]['temp'].min())
    temp_max = float(df['temp'].max())
    
    # 2. Traffic Thresholds for labels
    low_thresh = float(df['traffic_volume'].quantile(0.33))
    high_thresh = float(df['traffic_volume'].quantile(0.66))
    
    # 3. Weather Main Categories (will be saved in the next step)
    
    ranges = {
        'temp': {'min': temp_min, 'max': temp_max},
        'traffic_thresholds': {'low': low_thresh, 'high': high_thresh},
        'weather_main': {'categories': []} # Will be filled by next function
    }
    
    print(f"           ...Thresholds set: Low (<{low_thresh:.0f}), High (>{high_thresh:.0f})")
    return ranges # Return ranges to be updated

def engineer_features(df):
    """Engineers new time-based features from the 'date_time' column."""
    print("[Step 3/9] Engineering features from 'date_time'...")
    try:
        df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
    except ValueError as e:
        print(f"\n--- ERROR ---")
        print(f"Date conversion error: {e}")
        return None
        
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df = df.drop('date_time', axis=1) # Drop original
    print("           ...Time features (hour, day_of_week, etc.) created.")
    return df

def select_and_preprocess(df, ranges):
    """Keeps high-importance features AND 'weather_main' for interaction."""
    print("[Step 4/9] Selecting features and preprocessing...")
    
    # === UPDATED: We are now keeping 'weather_main' ===
    features_to_keep = [
        'traffic_volume',
        'temp',
        'weather_main',  # <-- THE COMPROMISE
        'hour',
        'day_of_week',
        'month',
        'year'
    ]
    
    # Save the categories for 'weather_main' before we drop other columns
    if 'weather_main' in df.columns:
        ranges['weather_main']['categories'] = list(df['weather_main'].unique())
    
    # Drop all other low-importance "noise" columns
    original_columns = list(df.columns)
    for col in original_columns:
        if col not in features_to_keep:
            df = df.drop(col, axis=1)
            
    print(f"           ...Features removed. Kept: {list(df.columns)}")

    # NOW, we must one-hot encode the 'weather_main' column
    print("[Step 5/9] Applying one-hot encoding to 'weather_main'...")
    df = pd.get_dummies(df, columns=['weather_main'])
    print("           ...'weather_main' converted to numeric.")
    
    # Now that we're done, save the complete ranges file
    with open(RANGES_FILE, 'w') as f:
        json.dump(ranges, f, indent=4)
    print("           ...Validation file saved successfully.")
    
    return df

def save_prediction_artifacts(X, test_indices):
    """Saves the column blueprint and the user-facing test set."""
    print(f"[Step 6/9] Saving column blueprint to '{COLUMNS_FILE}'...")
    column_list = list(X.columns)
    with open(COLUMNS_FILE, 'w') as f:
        json.dump(column_list, f)

    print(f"[Step 7/9] Saving 'unseen' test set to '{TEST_SET_FILE}'...")
    df_original = pd.read_csv(DATA_FILE)
    df_test_set_for_user = df_original.iloc[test_indices]
    df_test_set_for_user.to_csv(TEST_SET_FILE, index=False)

def main():
    """Main function to orchestrate the model training pipeline."""
    print("=" * 60)
    print("--- MODEL TRAINING PIPELINE (Compromise Version) ---")
    print("=" * 60)
    
    df = load_data(DATA_FILE)
    if df is None: return

    ranges = save_data_validation_artifacts(df)

    df = engineer_features(df)
    if df is None: return

    # New combined step
    df_processed = select_and_preprocess(df, ranges)

    print("[Step 8/9] Separating features (X) and target (y)...")
    y = df_processed['traffic_volume']
    X = df_processed.drop('traffic_volume', axis=1)

    print("[Step 9/9] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    save_prediction_artifacts(X, X_test.index)

    print("\n[Step 10/10] TRAINING MODEL: GradientBoostingRegressor")
    print("           (This will take 5-10 minutes)...")
    start_time = time.time()
    
    model = GradientBoostingRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.7,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"\nTraining complete. Total time: {(end_time - start_time) / 60:.2f} minutes")

    print("\n--- PERFORMANCE EVALUATION ---")
    print("Evaluating model performance on the 'unseen' test set...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f" > R-squared (R^2):               {r2:.4f} (or {r2*100:.2f}%)")
    print(f" > Mean Absolute Error (MAE):     {mae:.2f} vehicles")
    print(f" > Mean Squared Error (MSE):      {mse:.2f}")
    print(f" > Root Mean Squared Error (RMSE):{rmse:.2f} vehicles")

    print(f"\nSaving the final trained model to '{MODEL_FILE}'...")
    joblib.dump(model, MODEL_FILE)

    print("\n" + "=" * 60)
    print("--- PIPELINE COMPLETE ---")
    print("=" * 60)

if __name__ == "__main__":
    main()