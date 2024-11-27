import yfinance as yf
import pandas as pd
import numpy as np
from agente import Agent  # Suponiendo que tienes una clase `Agent` implementada
from enviroment import EnhancedTradingEnv  # Suponiendo que tienes el entorno implementado

# Descargar datos históricos de Yahoo Finance
ticker = "AAPL"  # Cambia por el ticker deseado
data = yf.Ticker(ticker).history(start="2010-01-01", end=None)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Procesar y calcular métricas
data['Close'] = data['Close'].astype(float)
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_Ratio'] = data['Close'] / data['SMA_20']

ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_hist'] = data['MACD'] - data['Signal_Line']

delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

low_14 = data['Close'].rolling(window=14).min()
high_14 = data['Close'].rolling(window=14).max()
data['%K'] = (data['Close'] - low_14) / (high_14 - low_14) * 100
data['%D'] = data['%K'].rolling(window=3).mean()
data['SO_diff'] = data['%K'] - data['%D']

data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# Guardar el DataFrame procesado
data.to_csv('aapl_data_with_metrics.csv', index=False)

# Filtrar las columnas clave para usar en el entorno
important_columns = [
    'Close', 'SMA_Ratio', 'MACD_hist', 'RSI', 'SO_diff', 'Returns'
]
filtered_data = data.dropna(subset=important_columns).reset_index(drop=True)

# Crear el entorno y el agente
env = EnhancedTradingEnv(real_data=[filtered_data], simulated_datasets=[])
agent = Agent(env)

# Entrenar al agente
agent.train()

# Guardar el modelo entrenado
agent.save_model("trading_agent_model.keras")
