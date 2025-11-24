import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# --- Configuración de la Página Web ---
st.set_page_config(layout="wide", page_title="Monitor Bitcoin IA")

st.title('₿ Monitor de Bitcoin: Inteligencia Artificial & Análisis Técnico')
st.markdown("Este panel se actualiza automáticamente descargando datos en tiempo real.")

# --- 1. Cargar Datos ---
@st.cache_data
def load_data():
    df = yf.download('BTC-USD', start='2017-01-01', end=date.today().strftime("%Y-%m-%d"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

with st.spinner('Descargando datos del mercado...'):
    df = load_data()

# --- 2. Cálculos ---
df['SMA_50'] = ta.sma(df['Close'], length=50)
df['SMA_200'] = ta.sma(df['Close'], length=200)
df['RSI'] = ta.rsi(df['Close'], length=14)

# --- 3. Predicción IA ---
st.subheader("🤖 Predicción de Tendencia (90 días)")
df_train = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

# --- 4. Gráficos ---
# Creamos el gráfico combinado
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.1, row_heights=[0.7, 0.3],
                    subplot_titles=('Precio + Proyección', 'Fuerza del Mercado (RSI)'))

# Precio y Medias
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Precio", line=dict(color='cyan')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="SMA 50", line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="SMA 200", line=dict(color='red')), row=1, col=1)

# Predicción
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicción IA", line=dict(color='#00ff00', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], showlegend=False, line=dict(width=0), hoverinfo='skip'), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', line=dict(width=0), name="Rango Error"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(template='plotly_dark', height=800)

# Mostrar en la web
st.plotly_chart(fig, use_container_width=True)
