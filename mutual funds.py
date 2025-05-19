import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "/mnt/data/comprehensive_mutual_funds_data.csv"
df = pd.read_csv('comprehensive_mutual_funds_data.csv')

# Convert non-numeric columns (sortino, alpha, sd, beta, sharpe) to numeric
for col in ['sortino', 'alpha', 'sd', 'beta', 'sharpe']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop fund manager (not relevant for prediction)
#df.drop(['fund_manager'], axis=1, inplace=True)

# Handle missing values: Fill NaN with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical variables
categorical_cols = ['category', 'sub_category', 'amc_name']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Define Features and Target Variable
features = ['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha',
            'sd', 'beta', 'sharpe', 'risk_level', 'rating', 'category', 'sub_category']
target = 'returns_5yr'  # Predicting 1-year return

X = df[features]
y = df[target]

# Split dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train ML Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}



model_scores = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_scores[name] = {'MSE': mse, 'R2': r2}
    print(f"\n{name} Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

# --- Visualization: Model Comparison ---
score_df = pd.DataFrame(model_scores).T
score_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# Predict Future Returns (Example: Last 5 funds)
future_pred = models[ "Random Forest"].predict(X_test_scaled[-10:])
print("\nPredicted Returns for Last 10 Funds:", future_pred)



# Plot Actual vs Predicted for Best Model (XGBoost)
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label="Actual Returns", marker='o')
plt.plot(models[ "Random Forest"].predict(X_test_scaled)[:50], label="Predicted Returns", linestyle='dashed', marker='x')
plt.legend()
plt.title("Actual vs Predicted Mutual Fund Returns")
plt.xlabel("Fund Index")
plt.ylabel("5-Year Return (%)")
plt.show()
