import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from prophet import Prophet
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import pytz
# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Monitor Bitcoin IA By René Navarro Ourcilleón")
# --- Barra Lateral (Configuración y Fecha) ---
st.sidebar.title("⚙️ Panel de Control")
st.sidebar.markdown("---")
# Hora en Chile
now_utc = datetime.now(pytz.utc)
tz_chile = pytz.timezone('America/Santiago')
now_chile = now_utc.astimezone(tz_chile)

st.sidebar.info(f"📅 **Última Actualización:**\n\n{now_chile.strftime('%d-%m-%Y')}\n⏱️ {now_chile.strftime('%H:%M:%S')} (Chile)")
st.sidebar.markdown("---")
st.sidebar.write("Este modelo descarga datos en tiempo real de Yahoo Finance y recalcula predicciones con IA.")

# --- Título Principal ---
st.title('₿ Monitor Estratégico de Bitcoin')
st.markdown(f"> *Análisis técnico automatizado e Inteligencia Artificial aplicada a BTC.*")

# --- Cargar Datos ---
@st.cache_data(ttl=300)
def load_data():
    # Descargamos data
    df = yf.download('BTC-USD', start='2018-01-01', end=date.today().strftime("%Y-%m-%d"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

with st.spinner('Conectando con el mercado...'):
    df = load_data()

# --- Cálculos Técnicos ---
df['SMA_50'] = ta.sma(df['Close'], length=50)
df['SMA_200'] = ta.sma(df['Close'], length=200)
df['RSI'] = ta.rsi(df['Close'], length=14)

last_price = df['Close'].iloc[-1]
last_rsi = df['RSI'].iloc[-1]
last_sma200 = df['SMA_200'].iloc[-1]

# --- DIAGNÓSTICO (KPIs) ---
st.markdown("### 📊 Diagnóstico Actual del Mercado")
col1, col2, col3 = st.columns(3)

col1.metric("Precio Actual (USD)", f"${last_price:,.2f}")

trend_status = "ALCISTA (Bullish) 🟢" if last_price > last_sma200 else "BAJISTA (Bearish) 🔴"
col2.metric("Tendencia Largo Plazo", trend_status)

if last_rsi > 70:
    rsi_status = "SOBRECOMPRA (Caro) 🔴"
elif last_rsi < 30:
    rsi_status = "SOBREVENTA (Barato) 🟢"
else:
    rsi_status = "NEUTRAL (Estable) ⚪"
col3.metric("Fuerza RSI (0-100)", f"{last_rsi:.1f}", rsi_status)

with st.expander("ℹ️ Ayuda para interpretar los indicadores"):
    st.markdown("""
    * **Tendencia:** Basada en si el precio está por encima o debajo de la media de 200 días.
    * **RSI:** Mide la "temperatura". >70 es muy caliente (posible caída), <30 es muy frío (posible rebote).
    """)

# --- ENTRENAMIENTO IA ---
df_train = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

m = Prophet(daily_seasonality=True)
m.fit(df_train)
# Predecimos 90 días
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

# --- VIS
