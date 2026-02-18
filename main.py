## ANTIGO APP.PY, AGORA É MAIN.PY, PARA *TESTES LOCAIS*. O APP.PY VAI SER USADO PARA O STREAMLIT!!!
## SE DESEJA APENAS FAZER TESTES USE ESTE ARQUIVO, SE DESEJA RODAR EM WEBAPP USE O APP.PY!!!

import torch
import torch.serialization
# solução de erros do pytorch
torch.serialization.default_restore_location = lambda storage, loc: storage
torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)


import os
import matplotlib.pyplot as plt # visualização de data
import yfinance as yf # dados financeiros
import pandas as pd # manipulação de data

# machine learning
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from neuralprophet import NeuralProphet

# TRATAMENTO DOS DADOS!!!
# -----------------------------------------------------------------------

# váriaveis da empresa
cod = "NVDA"
begin = "2020-01-01"
end = "2026-01-01"
arquivo_modelo = f"modelo_{cod}.pt"

# dados históricos
dados = yf.download(cod, start=begin, end=end, multi_level_index=False)

# formatação de data para formato esperado pelo neuralprophet
dados = dados[["Close"]].reset_index()

# transformação das colunas para ds (datas) e y ("valores")
dados.columns = ["ds", "y"]

# print(dados)

# PREVISÕES!!!
# -----------------------------------------------------------------------

# se o arquivo existe, carrega. se não, treina o modelo
if os.path.exists(arquivo_modelo):
    print("--- Carregando modelo salvo... ---")
    modelo = torch.load(arquivo_modelo)
else:
    print("--- Treinando modelo (isso só acontece uma vez)... ---")
    modelo = NeuralProphet(learning_rate=0.01)
    modelo.fit(dados)
    torch.save(modelo, arquivo_modelo)

# dataframe com previsões futuras para um periodo de 1 ano
dados_futuros = modelo.make_future_dataframe(dados, periods=365)

# previsões para o futuro
prev_futuras = modelo.predict(dados_futuros)

# previsões do passado
prev_hist = modelo.predict(dados)

# modelo.plot_components(prev_futuras) (só funciona com o jupyter!!!)

# CRIAÇÃO DA IMAGEM!!!
# -----------------------------------------------------------------------

plt.figure(figsize=(12, 6))

# linha vermelha é a previsão do que já passou
plt.plot(
    prev_hist["ds"], 
    prev_hist["yhat1"], 
    label="Previsões históricas", 
    c="r"
)

# linha azul é a previsão do futuro (os 365 dias)
plt.plot(
    prev_futuras["ds"], 
    prev_futuras["yhat1"], 
    label="Previsões futuras (Próximos 365 dias)", 
    c="b"
)

# linha verde são os dados reais de fechamento das ações
plt.plot(
    dados["ds"], 
    dados["y"], 
    label="Valores reais", 
    c="g", 
)

plt.legend()
plt.show()

# AVALIAÇÃO DO MODELO COM MÉTRICA R2!!!
# -----------------------------------------------------------------------

print("\nAVALIAÇÃO DO MODELO COM R2:")
print(r2_score(y_true=prev_hist["y"], y_pred=prev_hist["yhat1"]))

# MAPE PARA MEDIR MÉDIA DE ERRO EM PORCENTAGEM (erro médio absoluto)!!!
# -----------------------------------------------------------------------

print("\nMÉDIA DE ERRO EM PORCENTAGEM:")
print(mean_absolute_percentage_error(y_pred=prev_hist["yhat1"], y_true=prev_hist["y"]) * 100, "\n")