import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Data Loading and Initial Exploration 
print("Loading the dataset...")
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
print("Preprocessing the data...")
# Convert 'date_time' column to datetime objects
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')

# Feature Engineering
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek  # Monday=0, Sunday=6
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year

# drop 'weather_description' as 'weather_main' is a more general categorysss
df = df.drop(['date_time', 'weather_description'], axis=1)

# create new columns for each category in 'holiday' and 'weather_main'
df_encoded = pd.get_dummies(df, columns=['holiday', 'weather_main'], drop_first=True)

# Feature and Target Selection 
print("Defining features and target variable...")

# what we want to predict
y = df_encoded['traffic_volume']

# the inputs to the model
X = df_encoded.drop('traffic_volume', axis=1)

# Splitting the Data
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
print("Training the Random Forest Regressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)

# Train the model on the training data
model.fit(X_train, y_train)

# Model Evaluation 
print("Evaluating the model...")
y_pred = model.predict(X_test)

# Calculate performance metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
oob = model.oob_score_

# Print the results
print("\n--- Model Performance ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Out-of-Bag (OOB) Score: {oob:.4f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("-------------------------\n")


# Visualization of Results
print("Generating visualizations...")

# a) Scatter plot of Actual vs. Predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.3, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs. Predicted Traffic Volume', fontsize=16)
plt.xlabel('Actual Volume', fontsize=12)
plt.ylabel('Predicted Volume', fontsize=12)
plt.grid(True)
plt.show()

# b) Feature Importance plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(15)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_features, y=top_features.index, palette='viridis')
plt.title('Top 15 Feature Importances', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.show()

print("Program finished.")