import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from prophet import Prophet
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from datetime import date, datetime
import pytz

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Monitor Bitcoin IA")

# --- Barra Lateral (Configuración y Fecha) ---
st.sidebar.title("⚙️ Panel de Control")
st.sidebar.markdown("---")

# 1. Obtener fecha y hora actual (Zona Horaria Chile/Santiago para ti, o UTC)
# Usamos UTC y luego convertimos si es necesario, aquí lo dejo en local del servidor o UTC directo
now_utc = datetime.now(pytz.utc)
# Si quieres hora de Chile restamos 3 o 4 horas, o usamos librería pytz para ser exactos
tz_chile = pytz.timezone('America/Santiago')
now_chile = now_utc.astimezone(tz_chile)

st.sidebar.info(f"📅 **Última Actualización:**\n\n{now_chile.strftime('%d-%m-%Y')}\n⏱️ {now_chile.strftime('%H:%M:%S')} (Hora Chile)")

st.sidebar.markdown("---")
st.sidebar.write("Este modelo descarga datos en tiempo real de Yahoo Finance y recalcula predicciones con IA.")

# --- Título Principal ---
st.title('₿ Monitor Estratégico de Bitcoin by René Navarro Ourcilleón')
st.markdown(f"> *Análisis técnico automatizado e Inteligencia Artificial aplicada a BTC.*")

# --- Cargar Datos ---
@st.cache_data(ttl=300) # Cache de 5 minutos para no saturar
def load_data():
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

# Obtener últimos valores para el resumen
last_price = df['Close'].iloc[-1]
last_rsi = df['RSI'].iloc[-1]
last_sma50 = df['SMA_50'].iloc[-1]
last_sma200 = df['SMA_200'].iloc[-1]

# --- SECCIÓN DE EXPLICACIÓN DINÁMICA ---
# Aquí el código "piensa" y te da una conclusión escrita
st.markdown("### 📊 Diagnóstico Actual del Mercado")
col1, col2, col3 = st.columns(3)

# 1. Precio
col1.metric("Precio Actual (USD)", f"${last_price:,.2f}")

# 2. Tendencia (Lógica: Precio vs SMA 200)
trend_status = "ALCISTA (Bullish) 🟢" if last_price > last_sma200 else "BAJISTA (Bearish) 🔴"
col2.metric("Tendencia Largo Plazo", trend_status, delta_color="normal")

# 3. Estado RSI (Lógica Semáforo)
if last_rsi > 70:
    rsi_status = "SOBRECOMPRA (Caro) 🔴"
elif last_rsi < 30:
    rsi_status = "SOBREVENTA (Barato) 🟢"
else:
    rsi_status = "NEUTRAL (Estable) ⚪"
col3.metric("Fuerza RSI (0-100)", f"{last_rsi:.1f}", rsi_status)

# Explicación desplegable para principiantes
with st.expander("ℹ️ ¿Cómo leer estos datos? (Clic para ver explicación)"):
    st.markdown("""
    **1. Medias Móviles (Líneas Naranja y Roja):**
    * Nos dicen la "salud" de la tendencia. 
    * Si el Precio (Azul) está sobre la línea Roja (SMA 200), estamos en un mercado sano a largo plazo.
    
    **2. RSI (Gráfico Morado inferior):**
    * Es el "velocímetro" del mercado.
    * **Arriba de 70:** El precio subió muy rápido, podría caer pronto (Peligro).
    * **Debajo de 30:** El precio cayó demasiado, podría rebotar pronto (Oportunidad).
    
    **3. Predicción IA (Línea Punteada Verde):**
    * Es el camino matemático más probable según el algoritmo 'Prophet' de Facebook.
    * La "sombra" verde claro muestra el margen de error. Cuanto más ancha, más incierto es el futuro.
    """)

# --- Predicción IA ---
df_train = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

# --- Visualización Gráfica ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.1, row_heights=[0.7, 0.3],
                    subplot_titles=('Evolución de Precio + Proyección IA', 'Indicador de Fuerza (RSI)'))

# Gráfico Precio
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Precio Real", line=dict(color='cyan')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="Tendencia Base (SMA 200)", line=dict(color='red', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicción IA", line=dict(color='#00ff00', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], showlegend=False, line=dict(width=0), hoverinfo='skip'), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', line=dict(width=0), name="Margen Error"), row=1, col=1)

# Gráfico RSI
fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(template='plotly_dark', height=700, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)
