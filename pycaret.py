# -*- coding: utf-8 -*-

#instalando as bibliotecas
!pip install pycaret == 2.1.2
!pip install yfinance

from pycaret.utils import enable_colab
enable_colab()

#importar as bibliotecas
import yfinance as yf
import pandas as pd

#escolher uma ação
df = yf.Ticker('RADL3.SA')
#escolher o intervalo de dados
raia = df.history(period='2y')
raia

#retirando os campos
raia = raia.drop(['Dividends','Stock Splits'], axis=1)
raia

#criando novos campos
raia['MM7d'] = raia['Close'].rolling(window=7).mean().round(2)
raia['MM30d'] = raia['Close'].rolling(window=30).mean().round(2)
raia

#5 dias para previsao
raia_prever = raia.tail(5)
raia_prever

#retirar os ultimos 5 dias do df
raia.drop(raia.tail(5).index, inplace=True)
raia

#empurra para frente os valores das ações
raia['Close'] = raia['Close'].shift(-1)
raia

#Retirar os nulos
raia.dropna(inplace=True)
raia

#drop index
raia.reset_index(drop=True, inplace=True)
raia_prever.reset_index(drop=True, inplace=True)

raia

#import regression lib pycaret
from pycaret.regression import *
setup(data= raia, target='Close', session_id=123)

top3 = compare_models(n_select=3)

print(top3)

models()

ridge = create_model('ridge', fold=10)

lar = create_model('lar', fold=10)

br = create_model('br', fold=10)

#Tunning
ridge_params = { 'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]}
tunne_ridge = tune_model(ridge, n_iter=1000, optimize='RMSE', custom_grid=ridge_params)

tunne_lar = tune_model(lar, n_iter=1000, optimize = 'RMSE')

tunne_br = tune_model(br, n_iter=1000, optimize = 'RMSE')

#Grafico erros
plot_model(tunne_ridge, plot='error')

plot_model(tunne_ridge, plot='feature')

#Testando com dados de treinameto
predict_model(tunne_ridge)

#Finalizar o modelo
final_ridge_model = finalize_model(tunne_ridge)

#Previsao
prev = predict_model(final_ridge_model, data=raia_prever)
prev

#Salvando o modelo para utilizar com dados novos
save_model(final_ridge_model, 'Modelo Final Ridge Pycaret')

#Dados novos
novo_dado = yf.download('RADL3.SA', period='45d')
novo_dado

#retira campos
novo_dado = novo_dado.drop('Adj Close',axis = 1)
#retirar index
novo_dado.reset_index(drop=True, inplace=True)
#criar novos campos
novo_dado['MM7d'] = novo_dado['Close'].rolling(window=7).mean().round(2)
novo_dado['MM30d'] = novo_dado['Close'].rolling(window=30).mean().round(2)
novo_dado

novo_dado = novo_dado.tail(1)
novo_dado

#Reutilizando o modelo
saved_final_ridge_model = load_model('Modelo Final Ridge Pycaret')

#Prevendo novo dado
nova_previsao = predict_model(saved_final_ridge_model, data=novo_dado)
nova_previsao.head()