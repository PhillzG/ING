from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Wczytanie danych z CSV
df = pd.read_csv("synthetic_real_estate_data.csv")

# Wybór cech i zmiennych docelowych
categorical_features = ["City", "District", "Condition", "Market_Trend", "Energy_Efficiency", "Carbon_Footprint"]
numerical_features = ["Area_m2", "Rooms", "Year_Built", "Floor", "Interest_Rate", "Inflation", "Green_Access"]
target_variable = "Total_Price"

# Przetwarzanie zmiennych kategorycznych
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_columns = encoder.get_feature_names_out(categorical_features)
categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_columns)

scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df[numerical_features])
numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_features)

X = pd.concat([numerical_df, categorical_df], axis=1)
y = df[target_variable]

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modeli
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', cities=df["City"].unique().tolist(), districts=df["District"].unique().tolist())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_data = pd.DataFrame([data])
    input_cat_encoded = encoder.transform(input_data[categorical_features])
    input_num_scaled = scaler.transform(input_data[numerical_features])
    input_final = np.hstack((input_num_scaled, input_cat_encoded))

    price_linear = linear_model.predict(input_final)[0]
    price_xgb = xgb_model.predict(input_final)[0]

    return jsonify({
        "Cena Regresji Liniowej": round(price_linear, 2),
        "Cena XGBoost": round(price_xgb, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
