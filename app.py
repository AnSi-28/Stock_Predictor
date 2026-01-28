import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
MAX_DAYS = 15  # keep short horizon for reliability

# =========================
# LOAD MODELS
# =========================
linear_model = joblib.load("saved_models/linear_model.pkl")
lstm_model = load_model(
    "saved_models/lstm_model.h5",
    compile=False
)
scaler = joblib.load("saved_models/scaler.pkl")
config = joblib.load("saved_models/config.pkl")

SEQUENCE_LENGTH = config["sequence_length"]
STOCK = config["stock"]
FEATURES = config["features"]

app = Flask(__name__)

# =========================
# HELPER: Prepare Latest Data
# =========================
def get_latest_data():
    data = yf.download(STOCK, start="2015-01-01")
    data = data[['Close', 'Volume']]
    data.dropna(inplace=True)

    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(5).std()

    data.dropna(inplace=True)
    return data

# =========================
# LSTM PREDICTION (Return-based)
# =========================
def predict_lstm(n_days):
    data = get_latest_data()

    scaled = scaler.transform(data[FEATURES].values)

    last_sequence = scaled[-SEQUENCE_LENGTH:]
    last_close = float(data['Close'].iloc[-1])

    predictions = []

    for _ in range(n_days):

        input_seq = last_sequence.reshape(1, SEQUENCE_LENGTH, len(FEATURES))
        pred_scaled = lstm_model.predict(input_seq, verbose=0)

        # Extract scalar safely
        pred_scaled_value = float(pred_scaled.flatten()[0])

        # Build fake row for inverse scaling
        fake_row = np.zeros((1, len(FEATURES)))
        fake_row[0, 0] = pred_scaled_value

        inv = scaler.inverse_transform(fake_row)
        predicted_return = float(inv[0, 0])
        predicted_return = max(min(predicted_return, 0.05), -0.05)

        # Convert return → price
        next_price = last_close * (1 + predicted_return)
        predictions.append(float(next_price))

        # Build next feature row properly
        volume_scalar = float(data['Volume'].iloc[-1])
        volatility_scalar = float(data['Return'].tail(5).std())

        new_row = np.array([[predicted_return,
                             volume_scalar,
                             volatility_scalar]])

        new_scaled = scaler.transform(new_row)

        last_sequence = np.vstack((last_sequence[1:], new_scaled))

        last_close = next_price

    return predictions


# =========================
# LINEAR REGRESSION (Return-based)
# =========================
def predict_lr(n_days):
    data = get_latest_data()

    data['Time'] = np.arange(len(data))

    last_time = int(data['Time'].iloc[-1])
    last_close = float(data['Close'].iloc[-1])

    future_times = np.arange(last_time + 1,
                             last_time + 1 + n_days).reshape(-1, 1)

    predicted_returns = linear_model.predict(future_times)

    predictions = []
    current_price = last_close

    for r in predicted_returns:
        r = float(r)  # force scalar
        current_price = current_price * (1 + r)
        predictions.append(float(current_price))

    return predictions  # pure Python list

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        model_type = request.args.get("model", "lstm")
        n_days = int(request.args.get("days", 1))

        if n_days > MAX_DAYS:
            return jsonify({"error": f"Max {MAX_DAYS} days allowed"}), 400

        data = get_latest_data()
        historical_prices = data['Close'].tail(30).values.flatten().tolist()

        if model_type == "lstm":
            preds = predict_lstm(n_days)
        elif model_type == "lr":
            preds = predict_lr(n_days)
        else:
            return jsonify({"error": "Invalid model"}), 400

        return jsonify({
            "stock": STOCK,
            "model": model_type,
            "days": n_days,
            "historical": historical_prices,
            "predictions": preds
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
