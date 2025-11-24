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

# --- VISUALIZACIÓN GRÁFICA ---
st.markdown("---")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.1, row_heights=[0.7, 0.3],
                    subplot_titles=('Evolución de Precio + Proyección IA', 'Indicador de Fuerza (RSI)'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Precio Real", line=dict(color='cyan')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="Tendencia Base (SMA 200)", line=dict(color='red', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicción IA", line=dict(color='#00ff00', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], showlegend=False, line=dict(width=0), hoverinfo='skip'), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', line=dict(width=0), name="Margen Error"), row=1, col=1)

fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(template='plotly_dark', height=700)
st.plotly_chart(fig, use_container_width=True)

# --- NUEVA SECCIÓN: CALCULADORA DE PRECIO FUTURO ---
st.markdown("---")
st.subheader("🔮 Calculadora de Precio Futuro")
st.markdown("Selecciona una fecha futura para consultar la predicción exacta del modelo.")

# Contenedor para la calculadora
with st.container():
    col_calc_1, col_calc_2 = st.columns([1, 2])
    
    with col_calc_1:
        # Definir límites del selector de fecha
        last_date_in_history = df['Date'].iloc[-1].date()
        max_prediction_date = forecast['ds'].iloc[-1].date()
        
        # Selector de fecha
        target_date = st.date_input(
            "Selecciona la fecha:",
            min_value=last_date_in_history + timedelta(days=1),
            max_value=max_prediction_date,
            value=last_date_in_history + timedelta(days=7)
        )
    
    with col_calc_2:
        # Buscar la predicción para la fecha seleccionada
        # Convertimos target_date a datetime para buscar en el dataframe
        target_datetime = pd.to_datetime(target_date)
        prediction_row = forecast[forecast['ds'] == target_datetime]
        
        if not prediction_row.empty:
            predicted_price = prediction_row['yhat'].values[0]
            lower_bound = prediction_row['yhat_lower'].values[0]
            upper_bound = prediction_row['yhat_upper'].values[0]
            
            st.success(f"📅 **Para el {target_date.strftime('%d-%m-%Y')}, el modelo estima:**")
            
            # Mostrar métricas grandes
            c1, c2, c3 = st.columns(3)
            c1.metric("Precio Estimado", f"${predicted_price:,.2f}")
            c2.metric("Mínimo Probable", f"${lower_bound:,.2f}")
            c3.metric("Máximo Probable", f"${upper_bound:,.2f}")
            
            st.caption(f"*Nota: El 'Mínimo' y 'Máximo' representan el intervalo de confianza del modelo. Hay un 80% de probabilidad estadística de que el precio real caiga dentro de ese rango.*")
        else:
            st.warning("⚠️ La fecha seleccionada está fuera del rango de predicción disponible.")

st.markdown("---")
