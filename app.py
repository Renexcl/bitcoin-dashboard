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
import requests

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Bitcoin Dashboard Pro")

# --- Estilos CSS ---
st.markdown("""
<style>
.big-font { font-size:20px !important; color: #grey; }
a { text-decoration: none; color: #ff4b4b !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 1. FUNCIÓN DE CARGA DE DATOS (ROBUSTA) ---
@st.cache_data(ttl=3600) 
def load_data():
    data_source = "Binance API"
    df = pd.DataFrame()

    # --- INTENTO 1: BINANCE ---
    try:
        url = "https://api.binance.com/api/v3/klines"
        symbol = "BTCUSDT"
        interval = "1d"
        
        # Solo últimos 4 años para evitar timeouts
        start_date = datetime.now() - timedelta(days=1460)
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        
        # Límite de seguridad: 5 llamadas máximo para no bloquear
        for _ in range(5):
            if start_ts >= end_ts: break
            
            params = {'symbol': symbol, 'interval': interval, 'limit': 1000, 'startTime': start_ts}
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code != 200: break
            
            data = response.json()
            if not data or isinstance(data, dict): break
            
            all_data.extend(data)
            start_ts = data[-1][6] + 1
        
        if len(all_data) > 100:
            df = pd.DataFrame(all_data, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                'Taker Buy Base', 'Taker Buy Quote', 'Ignore'
            ])
            df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
            df['Close'] = df['Close'].astype(float)
            df = df[['Date', 'Close']]
    except Exception as e:
        print(f"Binance falló: {e}")

    # --- INTENTO 2: YAHOO FINANCE (RESPALDO) ---
    # Si Binance falló o trajo tabla vacía, usamos Yahoo
    if df.empty or len(df) < 10:
        data_source = "Yahoo Finance (Backup)"
        df = yf.download('BTC-USD', start='2020-01-01', end=date.today().strftime("%Y-%m-%d"))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        # Aseguramos nombres de columnas estándar
        if 'Date' not in df.columns and 'index' in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)

    # --- PROCESAMIENTO COMÚN ---
    # Cálculos Técnicos
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    
    # Lags para XGBoost
    for i in range(1, 8):
        df[f'Lag_{i}'] = df['Close'].shift(i)
        
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df, data_source

# --- 2. CACHÉ DE MODELOS ---
@st.cache_resource
def run_models(df):
    # A. PROPHET
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    # B. XGBOOST
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
with st.spinner('Conectando con Mercados Globales...'):
    df, source_name = load_data()
    
    # Validación final de seguridad
    if df.empty:
        st.error("Error crítico: No se pudieron descargar datos de ninguna fuente.")
        st.stop()
        
    forecast_prophet, future_xgb_prices = run_models(df)

last_date = df['Date'].iloc[-1]
dates_future = [last_date + timedelta(days=i) for i in range(1, 31)]

# --- INTERFAZ GRÁFICA ---
st.sidebar.title("⚙️ Configuración")
tz_chile = pytz.timezone('America/Santiago')
now_chile = datetime.now(tz_chile)
st.sidebar.info(f"Actualizado: {now_chile.strftime('%d-%m-%Y %H:%M')}")
st.sidebar.markdown("---")
st.sidebar.write("Fuente de Datos:")
if "Binance" in source_name:
    st.sidebar.success(f"🟡 **{source_name}**")
else:
    st.sidebar.warning(f"🔵 **{source_name}**")
    
st.sidebar.markdown("---")
st.sidebar.caption("Modelos Activos:")
st.sidebar.caption("🟢 Prophet: Activo")
st.sidebar.caption("🟠 XGBoost: Activo")

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
    prophet_val = forecast_prophet.loc[forecast_prophet['ds'] == target_dt, 'yhat'].values
    val_p = prophet_val[0] if len(prophet_val) > 0 else 0
    
    try:
        idx = dates_future.index(pd.Timestamp(selected_date))
        val_xgb = future_xgb_prices[idx]
    except:
        val_xgb = 0

    st.write(f"**Proyección para el {selected_date.strftime('%d-%m-%Y')}:**")
    c1, c2 = st.columns(2)
    c1.info(f"🤖 **Prophet (Meta):**\n# ${val_p:,.2f}")
    c2.warning(f"⚡ **XGBoost (ML):**\n# ${val_xgb:,.2f}")
    st.caption(f"Diferencia: ${val_xgb - val_p:,.2f}")

# Biblioteca Técnica
st.markdown("---")
st.header("📚 Documentación y Metodología")
if "Yahoo" in source_name:
    st.info("ℹ️ Nota: Binance API no respondió desde este servidor (posible bloqueo regional). Se utilizaron datos de Yahoo Finance automáticamente.")
else:
    st.success("✅ Conexión exitosa con Binance API.")

with st.expander("🟢 Modelo 1: Facebook Prophet (Ver Detalles)"):
    st.markdown("""
    **¿Qué es?** Un algoritmo diseñado por el equipo de Core Data Science de Meta.
    * **Enfoque:** Descompone el precio en tendencias (anuales, semanales) y feriados.
    * **Uso en este panel:** Define la "trayectoria base" o inercia del mercado.
    🔗 [Prophet Docs](https://facebook.github.io/prophet/)
    """)

with st.expander("🟠 Modelo 2: XGBoost (Ver Detalles)"):
    st.markdown("""
    **¿Qué es?** "Extreme Gradient Boosting". Algoritmo líder en Machine Learning competitivo.
    * **Enfoque:** Árboles de decisión secuenciales que corrigen errores previos.
    * **Uso en este panel:** Reactivo a los últimos 7 días de precio.
    🔗 [XGBoost Docs](https://xgboost.readthedocs.io/)
    """)
