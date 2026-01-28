import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIG
# =========================
STOCK = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
SEQUENCE_LENGTH = 60
MODELS_DIR = "saved_models"

os.makedirs(MODELS_DIR, exist_ok=True)

# =========================
# 1️⃣ DOWNLOAD DATA
# =========================
print("Downloading stock data...")
data = yf.download(STOCK, start=START_DATE, end=END_DATE)

data = data[['Close', 'Volume']]
data.dropna(inplace=True)

# =========================
# 2️⃣ FEATURE ENGINEERING
# =========================
print("Preparing features...")

# Percentage return
data['Return'] = data['Close'].pct_change()

# Rolling volatility
data['Volatility'] = data['Return'].rolling(window=5).std()

data.dropna(inplace=True)

features = ['Return', 'Volume', 'Volatility']
target = 'Return'

# =========================
# 3️⃣ LINEAR REGRESSION
# =========================
print("Training Linear Regression...")

data['Time'] = np.arange(len(data))

X_lr = data[['Time']]
y_lr = data[target]

split = int(len(data) * 0.8)

X_train_lr, X_test_lr = X_lr[:split], X_lr[split:]
y_train_lr, y_test_lr = y_lr[:split], y_lr[split:]

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

lr_preds = lr_model.predict(X_test_lr)

print("LR MSE:", mean_squared_error(y_test_lr, lr_preds))

joblib.dump(lr_model, os.path.join(MODELS_DIR, "linear_model.pkl"))

# =========================
# 4️⃣ LSTM TRAINING
# =========================
print("Training LSTM...")

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

X_lstm = []
y_lstm = []

for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X_lstm.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y_lstm.append(scaled_data[i, 0])  # Return column index

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

split = int(len(X_lstm) * 0.8)

X_train_lstm = X_lstm[:split]
X_test_lstm = X_lstm[split:]
y_train_lstm = y_lstm[:split]
y_test_lstm = y_lstm[split:]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(features))),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

lstm_preds = model.predict(X_test_lstm)

print("LSTM MSE:", mean_squared_error(y_test_lstm, lstm_preds))

# Save LSTM
model.save(os.path.join(MODELS_DIR, "lstm_model.h5"))

# Save scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# Save config
config = {
    "sequence_length": SEQUENCE_LENGTH,
    "stock": STOCK,
    "features": features,
    "target": target
}

joblib.dump(config, os.path.join(MODELS_DIR, "config.pkl"))

print("\n✅ Training Complete.")
print("Models saved in:", MODELS_DIR)
