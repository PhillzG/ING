from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("synthetic_real_estate_data.csv")

# **1. Usunięcie wartości odstających (outliers)**
df = df[df["Total_Price"] < df["Total_Price"].quantile(0.99)]  # Usuwamy górne 1%
df["Price_per_m2"] = df["Total_Price"] / df["Area_m2"]

# Feature selection
categorical_features = ["City", "District", "Condition", "Market_Trend", "Energy_Efficiency", "Carbon_Footprint"]
numerical_features = ["Area_m2", "Rooms", "Year_Built", "Floor", "Interest_Rate", "Inflation", "Green_Access"]
target_variable = "Price_per_m2"

# **2. Normalizacja danych (RobustScaler zamiast StandardScaler)**
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_columns = encoder.get_feature_names_out(categorical_features)
categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_columns)

scaler = RobustScaler()  # Lepiej radzi sobie z wartościami odstającymi
numerical_scaled = scaler.fit_transform(df[numerical_features])
numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_features)

X = pd.concat([numerical_df, categorical_df], axis=1)
y = df[target_variable]

# Train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# **3. Ulepszony XGBoost (Hyperparameter Tuning)**
xgb_model = XGBRegressor(
    n_estimators=300,        # Więcej drzew dla lepszego dopasowania
    learning_rate=0.05,      # Mniejsze tempo uczenia, stabilniejsze predykcje
    max_depth=6,             # Optymalna głębokość drzewa
    subsample=0.8,           # Ograniczenie danych do uniknięcia przeuczenia
    colsample_bytree=0.8,    # Użycie tylko części cech dla lepszej generalizacji
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Calculate MAE (Mean Absolute Error)
y_pred_linear = linear_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print(f"📉 Średni błąd dla prostego modelu matematycznego: {mae_linear:.2f} PLN/m²")
print(f"🤖 Średni błąd dla modelu analizującego dane: {mae_xgb:.2f} PLN/m²")

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', cities=df["City"].unique().tolist(), districts=df["District"].unique().tolist())

@app.route('/get_districts')
def get_districts():
    district_mapping = df.groupby("City")["District"].unique().apply(list).to_dict()
    return jsonify(district_mapping)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert input into DataFrame
        input_data = pd.DataFrame([data])

        # Encode and scale data
        input_cat_encoded = encoder.transform(input_data[categorical_features])
        input_num_scaled = scaler.transform(input_data[numerical_features])
        input_final = np.hstack((input_num_scaled, input_cat_encoded))

        # Predictions
        price_per_m2_linear = float(linear_model.predict(input_final)[0])
        price_per_m2_xgb = float(xgb_model.predict(input_final)[0])

        # Calculate total price
        area = float(data["Area_m2"])
        total_price_linear = price_per_m2_linear * area
        total_price_xgb = price_per_m2_xgb * area

        return jsonify({
            "Cena za m² (Regresja Liniowa)": round(price_per_m2_linear, 2),
            "Cena za m² (XGBoost)": round(price_per_m2_xgb, 2),
            "Całkowita cena (Regresja Liniowa)": round(total_price_linear, 2),
            "Całkowita cena (XGBoost)": round(total_price_xgb, 2),
            "Średni błąd prostego modelu": round(mae_linear, 2),
            "Średni błąd zaawansowanego modelu": round(mae_xgb, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
