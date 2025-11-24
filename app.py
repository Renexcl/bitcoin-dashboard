import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from prophet import Prophet
import xgboost as xgb
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import pytz
import numpy as np

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Bitcoin Dashboard Pro")

# --- Estilos CSS ---
st.markdown("""
<style>
.big-font { font-size:20px !important; color: #grey; }
</style>
""", unsafe_allow_html=True)

# --- 1. CACHÉ DE DATOS (Solo descarga cada 1 hora) ---
@st.cache_data(ttl=3600) 
def load_data():
    # Descargamos datos. Reduje un poco el histórico (desde 2020) para acelerar la descarga inicial
    df = yf.download('BTC-USD', start='2020-01-01', end=date.today().strftime("%Y-%m-%d"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    
    # Cálculos rápidos (Vectorizados)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    
    # Lags para XGBoost
    for i in range(1, 8):
        df[f'Lag_{i}'] = df['Close'].shift(i)
        
    df.dropna(inplace=True)
    return df

# --- 2. CACHÉ DE MODELOS (La parte pesada se hace una sola vez) ---
@st.cache_resource
def run_models(df):
    # --- A. PROPHET ---
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    # --- B. XGBOOST ---
    features = [f'Lag_{i}' for i in range(1, 8)]
    X = df[features]
    y = df['Close']
    
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.05)
    model_xgb.fit(X, y)
    
    # Predicción Recursiva
    future_xgb_prices = []
    current_lags = df['Close'].tail(7).values.tolist()[::-1]
    
    for _ in range(30):
        input_data = pd.DataFrame([current_lags], columns=features)
        pred = model_xgb.predict(input_data)[0]
        future_xgb_prices.append(pred)
        current_lags.insert(0, pred)
        current_lags.pop()
        
    return forecast, future_xgb_prices

# --- EJECUCIÓN PRINCIPAL ---
with st.spinner('Cargando datos del mercado...'):
    df = load_data()

with st.spinner('Entrenando Inteligencia Artificial...'):
    # Aquí está la magia: Si ya se ejecutó antes, recupera el resultado de la memoria instantáneamente
    forecast_prophet, future_xgb_prices = run_models(df)

# Variables de tiempo para gráficos
last_date = df['Date'].iloc[-1]
dates_future = [last_date + timedelta(days=i) for i in range(1, 31)]

# --- INTERFAZ (RENDERIZADO) ---
st.sidebar.title("⚙️ Configuración")
tz_chile = pytz.timezone('America/Santiago')
now_chile = datetime.now(tz_chile)
st.sidebar.info(f"Actualizado: {now_chile.strftime('%d-%m-%Y %H:%M')}")
st.sidebar.markdown("---")
st.sidebar.write("Modelos en Memoria:")
st.sidebar.caption("🟢 Prophet: Listo")
st.sidebar.caption("🟠 XGBoost: Listo")

# Título
st.title('₿ Bitcoin Intelligence Dashboard')
st.markdown("### By René Navarro Ourcilleón")
st.markdown("---")

# KPIs
last_price = df['Close'].iloc[-1]
last_rsi = df['RSI'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Precio Actual", f"${last_price:,.2f}")
col2.metric("RSI (Fuerza)", f"{last_rsi:.1f}", "Neutro" if 30 < last_rsi < 70 else ("Sobreventa 🟢" if last_rsi <= 30 else "Sobrecompra 🔴"))
col3.metric("Predicción XGBoost (7 días)", f"${future_xgb_prices[6]:,.2f}", delta=f"{((future_xgb_prices[6]-last_price)/last_price)*100:.1f}%")

# Gráficos
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.75, 0.25],
                    subplot_titles=('Predicción Híbrida', 'RSI'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Precio Histórico", line=dict(color='cyan', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat'], name="Prophet (Tendencia)", line=dict(color='lime', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=dates_future, y=future_xgb_prices, name="XGBoost (Reactivo)", line=dict(color='orange', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='mediumpurple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(template='plotly_dark', height=800, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Calculadora
st.markdown("---")
st.subheader("📅 Calculadora Predictiva")

col_cal1, col_cal2 = st.columns([1, 2])

with col_cal1:
    selected_date = st.date_input("Selecciona fecha futura:", 
                                  min_value=dates_future[0].date(), 
                                  max_value=dates_future[-1].date(), 
                                  value=dates_future[7].date())

with col_cal2:
    target_dt = pd.to_datetime(selected_date)
    # Prophet
    prophet_val = forecast_prophet.loc[forecast_prophet['ds'] == target_dt, 'yhat'].values
    val_p = prophet_val[0] if len(prophet_val) > 0 else 0
    
    # XGBoost
    try:
        idx = dates_future.index(pd.Timestamp(selected_date))
        val_xgb = future_xgb_prices[idx]
    except:
        val_xgb = 0

    st.write(f"**Proyección para el {selected_date.strftime('%d-%m-%Y')}:**")
    c1, c2 = st.columns(2)
    c1.info(f"🤖 **Modelo Prophet:**\n# ${val_p:,.2f}")
    c2.warning(f"⚡ **Modelo XGBoost:**\n# ${val_xgb:,.2f}")
    
    diff = val_xgb - val_p
    st.caption(f"Diferencia entre modelos: ${diff:,.2f}")

st.markdown("---")
with st.expander("Ver notas técnicas"):
    st.write("""
    * **Optimización:** Los modelos ahora se cargan en caché para mayor velocidad.
    * **RSI:** Indicador de momentum. >70 es caro, <30 es barato.
    * **Prophet:** Modelo estadístico de Meta para tendencias estacionales.
    * **XGBoost:** Modelo de Machine Learning reactivo.
    """)
