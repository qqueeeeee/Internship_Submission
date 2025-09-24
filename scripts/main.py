import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('../outputs', exist_ok=True)

# -------------------------
# 1. Simulated Datasets
# -------------------------
np.random.seed(42)
traffic_data = pd.DataFrame({
    'hour': np.tile(np.arange(0, 24), 10),
    'day_of_week': np.repeat(np.arange(1, 11), 24),
    'traffic_volume': np.random.randint(50, 500, size=240)
})
mining_data = pd.DataFrame({
    'temperature': np.random.uniform(20, 80, 200),
    'pressure': np.random.uniform(1, 10, 200),
    'humidity': np.random.uniform(30, 90, 200),
    'quality_score': np.random.uniform(50, 100, 200)
})

# -------------------------
# 2. Basic Statistics
# -------------------------
print("Traffic Dataset Statistics:\n", traffic_data.describe())
print("\nMining Dataset Statistics:\n", mining_data.describe())

# -------------------------
# 3. Visualizations
# -------------------------
plt.figure(figsize=(8,5))
sns.histplot(traffic_data['traffic_volume'], bins=20, kde=True)
plt.title("Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.savefig('../outputs/traffic_histogram.png')
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(x='temperature', y='quality_score', data=mining_data)
plt.title("Temperature vs Quality Score")
plt.xlabel("Temperature")
plt.ylabel("Quality Score")
plt.savefig('../outputs/mining_scatter.png')
plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(mining_data.corr(), annot=True, cmap='coolwarm')
plt.title("Mining Dataset Correlation")
plt.savefig('../outputs/mining_correlation.png')
plt.close()

# -------------------------
# 4. Probability Examples
# -------------------------
prob_high_traffic = (traffic_data['traffic_volume'] > 400).mean()
prob_low_quality = (mining_data['quality_score'] < 60).mean()
print(f"Probability of traffic volume > 400: {prob_high_traffic:.2f}")
print(f"Probability of mining quality score < 60: {prob_low_quality:.2f}")

# -------------------------
# 5. Machine Learning Models
# -------------------------
# Traffic Forecasting
X_traffic = traffic_data[['hour', 'day_of_week']]
y_traffic = traffic_data['traffic_volume']
X_train, X_test, y_train, y_test = train_test_split(X_traffic, y_traffic, test_size=0.2, random_state=42)
traffic_model = LinearRegression()
traffic_model.fit(X_train, y_train)
y_pred_traffic = traffic_model.predict(X_test)
print("\nTraffic Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_traffic))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_traffic)))
print("R2 Score:", r2_score(y_test, y_pred_traffic))

# Mining Quality Prediction
X_mining = mining_data[['temperature', 'pressure', 'humidity']]
y_mining = mining_data['quality_score']
X_train, X_test, y_train, y_test = train_test_split(X_mining, y_mining, test_size=0.2, random_state=42)
mining_model = DecisionTreeRegressor(random_state=42)
mining_model.fit(X_train, y_train)
y_pred_mining = mining_model.predict(X_test)
print("\nMining Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_mining))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_mining)))
print("R2 Score:", r2_score(y_test, y_pred_mining))

# -------------------------
# 6. Predictions Example
# -------------------------
sample_traffic = np.array([[9, 3]])
predicted_traffic = traffic_model.predict(sample_traffic)
print(f"\nPredicted traffic volume at 9 AM on day 3: {predicted_traffic[0]:.2f}")

sample_mining = np.array([[50, 5, 60]])
predicted_quality = mining_model.predict(sample_mining)
print(f"Predicted mining quality score: {predicted_quality[0]:.2f}")
