# to import the required library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# to load and read the dataset
file_path = "D:/RAK/Programs/ML Py Project/Metro_Interstate_Traffic_Volume.csv"
df = pd.read_csv(file_path)

# Parse datetime
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

# Create output folder
out_dir = "poster_outputs"
plot_dir = os.path.join(out_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------
# 2. Dataset Summary
# -------------------------------
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn Data Types:\n", df.dtypes)
print("\nSample Data (first 8 rows):\n", df.head(8))

# -------------------------------
# 3. Basic Statistics
# -------------------------------
numeric_cols = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]

stats = df[numeric_cols].describe().T
stats["mode"] = [df[c].mode(dropna=True).iloc[0] for c in numeric_cols]
stats["range"] = stats["max"] - stats["min"]
stats = stats[["mean", "50%", "mode", "min", "max", "range", "std"]]
stats = stats.rename(columns={"50%": "median", "std": "std_dev"})

print("\nBasic Statistics:\n", stats.round(2))

# Save stats
stats.to_csv(os.path.join(out_dir, "basic_statistics.csv"))

# -------------------------------
# 4. Visualizations
# -------------------------------

# Histogram - traffic volume
plt.figure()
plt.hist(df["traffic_volume"].dropna(), bins=30, edgecolor="black")
plt.title("Traffic Volume Distribution")
plt.xlabel("Traffic Volume (vehicles/hour)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "hist_traffic_volume.png"))
plt.close()

# Bar chart - avg traffic by hour
df["hour"] = df["date_time"].dt.hour
hourly_mean = df.groupby("hour")["traffic_volume"].mean()
plt.figure()
hourly_mean.plot(kind="bar")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Traffic Volume")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "bar_avg_by_hour.png"))
plt.close()

# Pie chart - weather conditions
top_weather = df["weather_main"].value_counts().head(6)
plt.figure()
top_weather.plot(kind="pie", autopct="%1.0f%%")
plt.ylabel("")
plt.title("Top Weather Conditions (Proportion)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "pie_weather.png"))
plt.close()

# Boxplot - traffic by day of week
df["dow"] = df["date_time"].dt.dayofweek
box_data = [df.loc[df["dow"] == d, "traffic_volume"].dropna() for d in range(7)]
plt.figure()
plt.boxplot(box_data, labels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], showfliers=True)
plt.title("Traffic Volume by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Traffic Volume")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "box_traffic_by_dow.png"))
plt.close()

# Line chart - daily avg & rolling mean
df["date"] = df["date_time"].dt.date
daily = df.groupby("date")["traffic_volume"].mean()
rolling = daily.rolling(window=7, min_periods=3).mean()
plt.figure()
plt.plot(daily.index, daily.values, label="Daily Mean")
plt.plot(rolling.index, rolling.values, label="7-day Rolling Mean")
plt.title("Daily Average Traffic Volume")
plt.xlabel("Date")
plt.ylabel("Avg Traffic Volume")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "line_daily_avg.png"))
plt.close()

# Scatter - temp vs traffic
plt.figure()
plt.scatter(df["temp"], df["traffic_volume"], alpha=0.2, s=10)
plt.title("Temperature vs Traffic Volume")
plt.xlabel("Temperature (Kelvin)")
plt.ylabel("Traffic Volume")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "scatter_temp_vs_traffic.png"))
plt.close()

print(f"\n✅ All outputs saved in folder: {out_dir}")