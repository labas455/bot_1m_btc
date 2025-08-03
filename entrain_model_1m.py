# 📦 Imports
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from keras.metrics import MeanSquaredError

# 🔑 Binance API (TEST ou réel)
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
client = Client(api_key, api_secret)

# 📥 Télécharger données 1m (ex: 7 jours)
def fetch_binance_data(symbol="BTCUSDT", interval="1m", days=7):
    klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "close"]]
    return df

# 📐 Créer les séquences pour LSTM
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 🔧 Paramètres
SEQ_LEN = 60
EPOCHS = 10
BATCH_SIZE = 32

# ⚙️ Pipeline
df = fetch_binance_data()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df["close"].values.reshape(-1, 1))

X, y = create_sequences(scaled, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 🧠 Modèle LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse",metrics=[MeanSquaredError()])

# 📁 Enregistrer les meilleurs poids
checkpoint = ModelCheckpoint("lstm_1m_model.h5", monitor="loss", save_best_only=True)

# 🚀 Entraînement
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint])

# 📉 Courbe de la perte
plt.plot(history.history["loss"])
plt.title("Loss du modèle LSTM 1m")
plt.xlabel("Épochs")
plt.ylabel("MSE")
plt.grid()
plt.show()
