import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# Load dataset
df = pd.read_csv("synthetic_real_estate_data.csv")

# Feature selection
categorical_features = ["City", "District", "Condition", "Market_Trend", "Energy_Efficiency", "Carbon_Footprint"]
numerical_features = ["Area_m2", "Rooms", "Year_Built", "Floor", "Interest_Rate", "Inflation", "Green_Access"]
target_variable = "Total_Price"

# Preprocessing
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_columns = encoder.get_feature_names_out(categorical_features)
categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_columns)

scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df[numerical_features])
numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_features)

X = pd.concat([numerical_df, categorical_df], axis=1)
y = df[target_variable]

# Train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Calculate MAE (Mean Absolute Error)
y_pred_linear = linear_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# Print results in a readable format
print("\n========================================")
print("  📊 OCENA MODELI PREDYKCJI CEN NIERUCHOMOŚCI  ")
print("========================================\n")
print(f"📈 Średni błąd dla prostego modelu matematycznego (Regresja Liniowa): {mae_linear:.2f} PLN")
print(f"🤖 Średni błąd dla modelu analizującego dane (XGBoost): {mae_xgb:.2f} PLN")
print("\n(Uwaga: niższy błąd oznacza dokładniejszy model)\n")



