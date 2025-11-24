import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from prophet import Prophet
import xgboost as xgb
import requests
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import pytz
import numpy as np

# --- Configuración ---
st.set_page_config(layout="wide", page_title="Bitcoin AI Híbrido")

# --- Funciones Auxiliares ---

@st.cache_data(ttl=300)
def get_fear_greed_index():
    """Obtiene el sentimiento del mercado desde API alternativa"""
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url)
        data = response.json()
        return data['data'][0]
    except:
        return {"value": "50", "value_classification": "Neutral"}

@st.cache_data(ttl=300)
def load_data():
    df = yf.download('BTC-USD', start='2018-01-01', end=date.today().strftime("%Y-%m-%d"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

# --- Procesamiento ---
with st.spinner('Inicializando motores de IA (Prophet + XGBoost)...'):
    df = load_data()
    fng = get_fear_greed_index()

# Calculo Indicadores (Features para XGBoost)
df['SMA_50'] = ta.sma(df['Close'], length=50)
df['SMA_200'] = ta.sma(df['Close'], length=200)
df['RSI'] = ta.rsi(df['Close'], length=14)
df['Volatility'] = df['Close'].rolling(20).std()
df.dropna(inplace=True) # XGBoost no quiere valores nulos

# --- MODELO 1: PROPHET (Tendencia) ---
df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
m_prophet = Prophet(daily_seasonality=True)
m_prophet.fit(df_prophet)
future_prophet = m_prophet.make_future_dataframe(periods=30) # 30 días
forecast_prophet = m_prophet.predict(future_prophet)

# --- MODELO 2: XGBOOST (Técnico/ML) ---
# Preparamos datos para ML
df_ml = df.copy()
df_ml['Target'] = df_ml['Close'].shift(-1) # Predecir el cierre de mañana
features = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'Volatility']
df_ml.dropna(inplace=True)

X = df_ml[features]
y = df_ml['Target']

# Entrenamos XGBoost (Regresor)
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model_xgb.fit(X, y)

# Predicción recursiva con XGBoost (Simulación 30 días)
future_xgb_prices = []
last_row = df_ml.iloc[-1][features].copy()

for _ in range(30):
    pred = model_xgb.predict(pd.DataFrame([last_row], columns=features))[0]
    future_xgb_prices.append(pred)
    # Actualizamos features simulados para el siguiente paso (básico)
    last_row['Close'] = pred
    # Nota: Recalcular RSI/SMA real es complejo en loop, usamos aproximación estática para demo
    
dates_future = forecast_prophet['ds'].tail(30).values

# --- INTERFAZ ---

# Barra Lateral
st.sidebar.title("🧠 Cerebro Digital")
st.sidebar.info(f"**Sentimiento Actual:**\n\n# {fng['value']}\n{fng['value_classification']}")
st.sidebar.markdown("---")
st.sidebar.write("Modelos Activos:")
st.sidebar.success("✅ Prophet (Meta): Tendencias Estacionales")
st.sidebar.success("✅ XGBoost: Patrones Técnicos")

# Título
st.title('₿ Bitcoin: Modelo Híbrido de Predicción')
st.markdown("Este panel combina **Estadística (Prophet)** con **Machine Learning (XGBoost)** y **Análisis de Sentimiento**.")

# Métricas
col1, col2, col3, col4 = st.columns(4)
last_price = df['Close'].iloc[-1]
col1.metric("Precio Actual", f"${last_price:,.0f}")
col2.metric("Sentimiento Mercado", f"{fng['value_classification']}")
col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
col4.metric("Predicción XGBoost (Mañana)", f"${future_xgb_prices[0]:,.0f}")

# Gráfico Principal
fig = go.Figure()

# 1. Historia
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historia Real", line=dict(color='cyan', width=2)))

# 2. Prophet
fig.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat'], name="Modelo Prophet (Tendencia)", line=dict(color='green', dash='dot')))

# 3. XGBoost
fig.add_trace(go.Scatter(x=dates_future, y=future_xgb_prices, name="Modelo XGBoost (Técnico)", line=dict(color='orange', width=3)))

fig.update_layout(title="Comparativa de Modelos: Tendencia vs Técnico", template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

# Explicación
st.markdown("### 🤖 Análisis de Modelos")
st.info("""
**¿Por qué dos líneas de predicción?**
* **Línea Verde (Prophet):** Es conservadora. Mira el calendario y la historia general.
* **Línea Naranja (XGBoost):** Es agresiva. Mira el RSI, la volatilidad reciente y las medias móviles. Según tu investigación, este modelo suele tener mejor precisión a corto plazo (10-20% superior).
""")

# Tabla comparativa
st.subheader("📋 Proyección a 7 Días")
proyeccion = pd.DataFrame({
    'Fecha': pd.to_datetime(dates_future[:7]).strftime('%Y-%m-%d'),
    'Prophet (Estable)': [f"${x:,.0f}" for x in forecast_prophet['yhat'].tail(30).values[:7]],
    'XGBoost (Reactivo)': [f"${x:,.0f}" for x in future_xgb_prices[:7]]
})
st.table(proyeccion)
