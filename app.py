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
st.set_page_config(layout="wide", page_title="Monitor Bitcoin IA")

# --- Barra Lateral ---
st.sidebar.title("⚙️ Panel de Control")
st.sidebar.markdown("---")

# Hora Chile
now_utc = datetime.now(pytz.utc)
tz_chile = pytz.timezone('America/Santiago')
now_chile = now_utc.astimezone(tz_chile)

st.sidebar.info(f"📅 **Última Actualización:**\n\n{now_chile.strftime('%d-%m-%Y')}\n⏱️ {now_chile.strftime('%H:%M:%S')} (Chile)")
st.sidebar.markdown("---")
st.sidebar.write("Datos de Yahoo Finance en tiempo real.")

# --- Título ---
st.title('₿ Monitor Estratégico de Bitcoin')
st.markdown(f"> *Análisis técnico automatizado e Inteligencia Artificial aplicada.*")

# --- Cargar Datos ---
@st.cache_data(ttl=300)
def load_data():
    df = yf.download('BTC-USD', start='2018-01-01', end=date.today().strftime("%Y-%m-%d"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

with st.spinner('Procesando datos del mercado...'):
    df = load_data()

# --- Cálculos ---
df['SMA_200'] = ta.sma(df['Close'], length=200) # Tendencia Largo Plazo
df['RSI'] = ta.rsi(df['Close'], length=14)      # Fuerza

last_price = df['Close'].iloc[-1]
last_rsi = df['RSI'].iloc[-1]
last_sma200 = df['SMA_200'].iloc[-1]

# --- KPIs ---
st.markdown("### 📊 Estado del Mercado")
col1, col2, col3 = st.columns(3)
col1.metric("Precio Actual", f"${last_price:,.2f}")
trend = "ALCISTA (Bullish) 🟢" if last_price > last_sma200 else "BAJISTA (Bearish) 🔴"
col2.metric("Tendencia (SMA 200)", trend)
rsi_st = "SOBRECOMPRA 🔴" if last_rsi > 70 else "SOBREVENTA 🟢" if last_rsi < 30 else "NEUTRAL ⚪"
col3.metric("RSI (Fuerza)", f"{last_rsi:.1f}", rsi_st)

# --- IA Prophet ---
df_train = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

# --- Gráficos ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3],
                    subplot_titles=('Análisis de Precio y Predicción', 'Oscilador RSI'))

# Gráfico 1
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Precio Real (Cierre)", line=dict(color='cyan')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="Tendencia (SMA 200)", line=dict(color='red', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicción IA", line=dict(color='#00ff00', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], showlegend=False, line=dict(width=0), hoverinfo='skip'), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', line=dict(width=0), name="Rango Probable"), row=1, col=1)

# Gráfico 2
fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(template='plotly_dark', height=700)
st.plotly_chart(fig, use_container_width=True)

# --- Calculadora ---
st.subheader("🔮 Calculadora de Precio Futuro")
c1, c2 = st.columns([1, 2])
with c1:
    target = st.date_input("Consultar fecha:", min_value=df['Date'].iloc[-1].date()+timedelta(days=1), max_value=forecast['ds'].iloc[-1].date())
with c2:
    row = forecast[forecast['ds'] == pd.to_datetime(target)]
    if not row.empty:
        st.success(f"Estimación para {target.strftime('%d-%m-%Y')}: **${row['yhat'].values[0]:,.2f}** (Rango: ${row['yhat_lower'].values[0]:,.0f} - ${row['yhat_upper'].values[0]:,.0f})")

# --- SECCIÓN EDUCATIVA (NUEVA) ---
st.markdown("---")
st.header("📚 Glosario Técnico y Metodología")

with st.expander("1. Explicación de las Líneas del Gráfico (Leyenda)"):
    st.markdown("""
    * **🔵 Precio Real (Línea Cian):** Es el valor de cierre diario de Bitcoin verificado en el mercado.
    * **🔴 Tendencia / SMA 200 (Línea Roja):** Es el *Promedio Móvil Simple* de los últimos 200 días. 
        * *Interpretación:* Funciona como un "piso" o "techo" psicológico. Si el precio está por encima, se considera una tendencia general alcista (Bullish). Si cae por debajo, entramos en territorio bajista (Bearish).
    * **🟢 Predicción IA (Punteada Verde):** Es el precio matemático más probable calculado por el algoritmo.
    * **🟢 Rango Probable (Sombra Verde):** Ningún modelo conoce el futuro exacto. Esta sombra representa el intervalo de confianza del 80%. El precio real debería caer dentro de esta sombra la mayoría de las veces.
    """)

with st.expander("2. ¿Qué es el RSI? (Índice de Fuerza Relativa)"):
    st.markdown("""
    El **RSI** es un indicador tipo "termómetro" que va del 0 al 100.
    * **🔥 Sobrecompra (> 70):** Línea roja punteada. Significa que el precio ha subido muy rápido y muy fuerte. Estadísticamente, aumenta la probabilidad de que la gente empiece a vender y el precio baje (corrección).
    * **🧊 Sobreventa (< 30):** Línea verde punteada. Significa que el precio ha caído exageradamente. Aumenta la probabilidad de que los inversores vean una "oferta" y empiecen a comprar (rebote).
    """)

with st.expander("3. ¿Qué modelo predictor utiliza este panel?"):
    st.markdown("""
    Este panel utiliza **Facebook Prophet**, un algoritmo de series temporales de código abierto desarrollado por el equipo de Ciencia de Datos de Meta.
    
    **¿Por qué usamos Prophet para Bitcoin?**
    A diferencia de modelos financieros tradicionales, Prophet está diseñado para detectar:
    1.  **Tendencias no lineales:** Bitcoin no sube en línea recta; tiene curvas de adopción.
    2.  **Estacionalidad:** Detecta patrones que se repiten (ej: comportamiento los fines de semana vs. días hábiles, o ciclos anuales).
    3.  **Resistencia a ruido:** Ignora picos aislados (outliers) que podrían confundir a otros modelos más simples.
    """)
