import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('📈 Stock Price Prediction')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'GC=F', 'BTC-USD', '^GSPC', 'RELIANCE.NS')
selected_stock = st.selectbox('Enter Stock Ticker', stocks)

n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365

@st.cache_data  # Fixed caching issue
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return None

data_load_state = st.text('⏳ Loading data...')
data = load_data(selected_stock)
data_load_state.text('✅ Data loaded successfully!')

if data is not None:
    st.subheader('📊 Recent Stock Data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
        fig.layout.update(title_text='Stock Price Over Time', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Train Prophet Model
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df_train)
    
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    # Show forecast
    st.subheader('📈 Forecast Data')
    st.write(forecast.tail())

    st.write(f'📉 Forecast for {n_years} years')
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.write("🔎 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
else:
    st.error("⚠️ Could not load data. Please check the ticker or try again later.")
