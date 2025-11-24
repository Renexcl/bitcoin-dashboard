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

# --- Estilos CSS para ajustar el título ---
st.markdown("""
<style>
.big-font { font-size:20px !important; color: #grey; }
</style>
""", unsafe_allow_html=True)

# --- 1. Cargar Datos y Funciones ---
@st.cache_data(ttl=300)
def load_data():
    df = yf.download('BTC-USD', start='2019-01-01', end=date.today().strftime("%Y-%m-%d"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

with st.spinner('Cargando modelos y procesando datos...'):
    df = load_data()

# --- 2. Ingeniería de Características (Calculadora de Indicadores) ---
# RSI y Medias para el gráfico histórico
df['RSI'] = ta.rsi(df['Close'], length=14)
df['SMA_200'] = ta.sma(df['Close'], length=200)

# PREPARACIÓN PARA XGBOOST (Lógica de Lags para evitar línea plana)
# Creamos columnas con los precios de los 7 días anteriores
for i in range(1, 8):
    df[f'Lag_{i}'] = df['Close'].shift(i)

df.dropna(inplace=True) # Eliminar filas vacías por los lags

# --- 3. MODELO 1: PROPHET (Tendencia Estacional) ---
df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
m_prophet = Prophet(daily_seasonality=True)
m_prophet.fit(df_prophet)
future_prophet = m_prophet.make_future_dataframe(periods=30)
forecast_prophet = m_prophet.predict(future_prophet)

# --- 4. MODELO 2: XGBOOST (Dinámico Recursivo) ---
# Entrenamos para predecir el precio de "Hoy" usando los 7 días "Anteriores"
features = [f'Lag_{i}' for i in range(1, 8)]
X = df[features]
y = df['Close']

model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05)
model_xgb.fit(X, y)

# Bucle de Predicción Futura (30 días)
future_xgb_prices = []
# Tomamos los últimos 7 precios reales conocidos para empezar a predecir
current_lags = df['Close'].tail(7).values.tolist()[::-1] # Invertimos para que coincida con Lag_1, Lag_2...

for _ in range(30):
    # Predecir el siguiente día
    input_data = pd.DataFrame([current_lags], columns=features)
    pred = model_xgb.predict(input_data)[0]
    future_xgb_prices.append(pred)
    
    # Actualizar los lags: Quitamos el más viejo, agregamos la nueva predicción al inicio
    current_lags.insert(0, pred)
    current_lags.pop()

# Crear dataframe de fechas futuras para XGBoost
last_date = df['Date'].iloc[-1]
dates_future = [last_date + timedelta(days=i) for i in range(1, 31)]

# --- INTERFAZ GRÁFICA ---

# Barra Lateral
st.sidebar.title("⚙️ Configuración")
tz_chile = pytz.timezone('America/Santiago')
now_chile = datetime.now(tz_chile)
st.sidebar.info(f"Actualizado: {now_chile.strftime('%d-%m-%Y %H:%M')}")
st.sidebar.markdown("---")
st.sidebar.write("Modelos Activos:")
st.sidebar.caption("🟢 Prophet: Tendencia Base")
st.sidebar.caption("🟠 XGBoost: Reactivo (ML)")

# Encabezado Principal
st.title('₿ Bitcoin Intelligence Dashboard')
st.markdown("### By René Navarro Ourcilleón")
st.markdown("---")

# Métricas Clave (KPIs)
last_price = df['Close'].iloc[-1]
last_rsi = df['RSI'].iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Precio Actual", f"${last_price:,.2f}")
col2.metric("RSI (Fuerza)", f"{last_rsi:.1f}", "Neutro" if 30 < last_rsi < 70 else ("Sobreventa 🟢" if last_rsi <= 30 else "Sobrecompra 🔴"))
col3.metric("Predicción XGBoost (7 días)", f"${future_xgb_prices[6]:,.2f}", delta=f"{((future_xgb_prices[6]-last_price)/last_price)*100:.1f}%")

# --- GRÁFICOS COMPUESTOS ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.75, 0.25],
                    subplot_titles=('Predicción Híbrida de Precio', 'Oscilador RSI'))

# Gráfico 1: Precio + Modelos
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Precio Histórico", line=dict(color='cyan', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat'], name="Prophet (Tendencia)", line=dict(color='lime', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=dates_future, y=future_xgb_prices, name="XGBoost (Reactivo)", line=dict(color='orange', width=2)), row=1, col=1)

# Gráfico 2: RSI
fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='mediumpurple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.add_shape(type="rect", x0=df['Date'].iloc[0], x1=dates_future[-1], y0=30, y1=70, 
              fillcolor="gray", opacity=0.1, layer="below", line_width=0, row=2, col=1)

fig.update_layout(template='plotly_dark', height=800, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- CALCULADORA DE FECHA (Restaurada y Mejorada) ---
st.markdown("---")
st.subheader("📅 Calculadora Predictiva")

col_cal1, col_cal2 = st.columns([1, 2])

with col_cal1:
    # Selector de fecha limitado a los 30 días de predicción
    selected_date = st.date_input(
        "Selecciona una fecha futura:",
        min_value=dates_future[0].date(),
        max_value=dates_future[-1].date(),
        value=dates_future[7].date()
    )

with col_cal2:
    # Buscar valores en ambos modelos
    target_dt = pd.to_datetime(selected_date)
    
    # Valor Prophet
    prophet_val = forecast_prophet.loc[forecast_prophet['ds'] == target_dt, 'yhat'].values
    val_p = prophet_val[0] if len(prophet_val) > 0 else 0
    
    # Valor XGBoost
    try:
        idx = dates_future.index(pd.Timestamp(selected_date))
        val_xgb = future_xgb_prices[idx]
    except:
        val_xgb = 0

    # Mostrar Tarjetas
    st.write(f"**Proyección para el {selected_date.strftime('%d-%m-%Y')}:**")
    c1, c2 = st.
