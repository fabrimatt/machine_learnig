# -*- coding: utf-8 -*-

#instalar o yfinance
pip install yfinance

#import bibliotecas
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '${:,.2f}'.format

hj = datetime.today().strftime('%Y-%m-%d')
data_ini = '2016-01-01'
df_eth = yf.download('ETH-USD', data_ini, hj)
df_eth.tail()

df_eth.reset_index(inplace=True)

df_eth

df = df_eth[["Date", "Adj Close"]]
df.rename(columns = {'Date': 'ds', 'Adj Close': 'y' }, inplace=True)

df

# Grafico Preço de fechamento
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y = df['y']))

model = Prophet( seasonality_mode='multiplicative')
model.fit(df)

#criar df com datas no futuro
df_futuro = model.make_future_dataframe(periods=60)
df_futuro.tail (60)

#previsao
previsao = model.predict(df_futuro)
previsao

previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(60)

#grafico
plot_plotly(model, previsao)

plot_components_plotly(model, previsao)