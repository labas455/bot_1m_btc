from binance.client import Client
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from datetime import datetime
import uuid

from dotenv import load_dotenv
import os


from keras.metrics import MeanSquaredError
from keras.saving import register_keras_serializable
from keras.metrics import mean_squared_error

@register_keras_serializable()
def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

load_dotenv()  # charge les variables depuis .env
# === Clés API Binance TESTNET ===
api_key = os.environ.get("API_KEY")
api_secret =  os.environ.get("API_SECRET")

if not api_key or not api_secret:
    raise ValueError("❌ API key or secret not found in environment variables.")
client = Client(api_key, api_secret)
client.API_URL = 'https://testnet.binance.vision/api'

# === Charger le modèle entraîné pour le 1m ===
model = load_model("lstm_1m_model.h5",custom_objects={'mse': mse})  # 📁 Le modèle que tu viens d'entraîner
SEQ_LEN = 60

# === Journalisation (log) ===

last_action = None
last_trade_price = None

# === Journal avec profit/perte
LOG_FILE = "log_trades_1m.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,current_price,predicted_price,action,profit,session_id\n")



session_id = str(uuid.uuid4())[:8]
def log_trade(timestamp, current_price, predicted_price, action, profit=0):
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp},{current_price:.2f},{predicted_price:.2f},{action},{profit:.2f},{session_id}\n")



# === Fonction pour récupérer les données 1m ===
def get_live_data(symbol="BTCUSDT", interval="1m", lookback=SEQ_LEN+1):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    closes = [float(candle[4]) for candle in klines]
    return np.array(closes)

# === Prédiction du prix futur ===
scaler = MinMaxScaler()

def predict_next_price(prices):
    scaled = scaler.fit_transform(prices.reshape(-1, 1))
    sequence = scaled[-SEQ_LEN:]
    X_input = np.expand_dims(sequence, axis=0)
    pred_scaled = model.predict(X_input, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    return pred[0][0]

# === Prise de décision ===
def decide_trade(current_price, predicted_price, threshold=0.00):
    if predicted_price > current_price * (1 + threshold):
        return "BUY"
    elif predicted_price < current_price * (1 - threshold):
        return "SELL"
    else:
        return "HOLD"

# === Envoi de l'ordre sur testnet ===
def place_order(symbol, side, quantity):
    order = client.order_market(symbol=symbol, side=side, quantity=quantity)
    print(f"🟢 Order placed: {side} {quantity} {symbol}")
    return order


# === Boucle principale toutes les 60 secondes ===
SYMBOL = "BTCUSDT"
TRADE_QTY = 0.001


try:
    prices = get_live_data()
    current_price = prices[-1]
    predicted_price = predict_next_price(prices)
    decision = decide_trade(current_price, predicted_price)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    profit = 0

    # 💰 Calcul de profit si SELL après un BUY
    if decision == "SELL" and last_action == "BUY":
        profit = (current_price - last_trade_price) * TRADE_QTY
    elif decision == "BUY" and last_action == "SELL":
        profit = (last_trade_price - current_price) * TRADE_QTY

    log_trade(timestamp, current_price, predicted_price, decision, profit)

    print(f"💡 {timestamp} | Price: {current_price:.2f}, Pred: {predicted_price:.2f}, Action: {decision}, Profit: {profit:.2f}")

    if decision in ["BUY", "SELL"]:
        place_order(SYMBOL, decision, TRADE_QTY)
        last_action = decision
        last_trade_price = current_price

    time.sleep(60)

except Exception as e:
    print("⚠️ Error:", e)
    time.sleep(10)
