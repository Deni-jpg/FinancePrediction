import streamlit as st
import yfinance as yf
import pandas as pd
import torch
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Configuração da página
st.set_page_config(page_title="Finance Predictor", layout="wide")

# Solução de segurança do PyTorch
torch.serialization.default_restore_location = lambda storage, loc: storage
torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)

# --- SISTEMA DE IDIOMAS ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'

def toggle_lang():
    st.session_state.lang = 'EN' if st.session_state.lang == 'PT' else 'PT'

texts = {
    'PT': {
        'title': "📈 Dashboard de Previsão Financeira",
        'subtitle': "Previsão de ações utilizando Machine Learning (NeuralProphet)",
        'sidebar_header': "Configurações",
        'ticker_label': "Digite o Ticker (ex: NVDA):",
        'slider_label': "Anos de dados históricos:",
        'days_label': "Dias para prever:",
        'btn_train': "Treinar e Prever",
        'btn_lang': "Change to English 🇺🇸",
        'error': "Erro: Dados não encontrados.",
        'metrics_current': "Preço Atual",
        'metrics_target': "Alvo Previsto",
        'legend_hist': "Previsões históricas",
        'legend_future': "Previsões futuras (Próximos 365 dias)",
        'legend_real': "Valores reais",
        'eval_r2': "AVALIAÇÃO DO MODELO COM R2:",
        'eval_mape': "MÉDIA DE ERRO EM PORCENTAGEM (MAPE):",
        'performance': "📊 Performance do Modelo"
    },
    'EN': {
        'title': "📈 Finance Prediction Dashboard",
        'subtitle': "Stock prediction using Machine Learning (NeuralProphet)",
        'sidebar_header': "Settings",
        'ticker_label': "Enter Ticker (e.g., NVDA):",
        'slider_label': "Years of historical data:",
        'days_label': "Days to predict:",
        'btn_train': "Train and Predict",
        'btn_lang': "Mudar para Português 🇧🇷",
        'error': "Error: Data not found.",
        'metrics_current': "Current Price",
        'metrics_target': "Target Price",
        'legend_hist': "Historical Predictions",
        'legend_future': "Future Predictions (Next 365 days)",
        'legend_real': "Real Values",
        'eval_r2': "MODEL EVALUATION (R2 SCORE):",
        'eval_mape': "MEAN ABSOLUTE PERCENTAGE ERROR (MAPE):",
        'performance': "📊 Model Performance"
    }
}

L = texts[st.session_state.lang]

st.title(L['title'])
st.subheader(L['subtitle'])

with st.sidebar:
    st.header(L['sidebar_header'])
    ticker = st.text_input(L['ticker_label'], value="").upper()
    periodo_anos = st.slider(L['slider_label'], 1, 10, 5)
    dias_previsao = st.number_input(L['days_label'], min_value=30, max_value=730, value=365)
    btn_treinar = st.button(L['btn_train'])

if btn_treinar:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner('Processing...'):
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=periodo_anos)
            
            try:
                dados = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
                
                if dados is None or dados.empty:
                    st.error(L['error'])
                else:
                    # 1. Preparação e Treino
                    dados_formatados = dados[["Close"]].reset_index()
                    dados_formatados.columns = ["ds", "y"]

                    modelo = NeuralProphet(learning_rate=0.01)
                    modelo.fit(dados_formatados, freq='D')
                    
                    # 2. Previsões
                    dados_futuros = modelo.make_future_dataframe(dados_formatados, periods=dias_previsao)
                    prev_futuras = modelo.predict(dados_futuros)
                    prev_hist = modelo.predict(dados_formatados)
                    
                    # 3. Gráfico (Tudo indentado dentro do else onde os dados existem)
                    plt.style.use('default')
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    ax.plot(prev_hist["ds"], prev_hist["yhat1"], label=L['legend_hist'], color="red")
                    ax.plot(prev_futuras["ds"].values, prev_futuras["yhat1"].values, label=L['legend_future'], color="blue", linewidth=0.6)
                    ax.plot(dados_formatados["ds"], dados_formatados["y"], label=L['legend_real'], color="green", alpha=0.8)

                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    # 4. Métricas de Preço
                    ultimo_preco = float(dados['Close'].iloc[-1])
                    preco_previsto = float(prev_futuras['yhat1'].iloc[-1])
                    variacao = ((preco_previsto - ultimo_preco) / ultimo_preco) * 100

                    col1, col2 = st.columns(2)
                    col1.metric(L['metrics_current'], f"${ultimo_preco:.2f}")
                    col2.metric(L['metrics_target'], f"${preco_previsto:.2f}", f"{variacao:.2f}%")

                    st.divider()

                    # 5. Avaliação (R2 e MAPE)
                    st.subheader(L['performance'])
                    r2 = r2_score(y_true=prev_hist["y"], y_pred=prev_hist["yhat1"])
                    mape = mean_absolute_percentage_error(y_true=prev_hist["y"], y_pred=prev_hist["yhat1"]) * 100

                    m_col1, m_col2 = st.columns(2)
                    m_col1.write(f"**{L['eval_r2']}**")
                    m_col1.info(f"{r2:.4f}")
                    
                    m_col2.write(f"**{L['eval_mape']}**")
                    m_col2.info(f"{mape:.2f}%")
                    
            except Exception as e:
                st.error(f"⚠️ Erro: {e}")

st.sidebar.button(L['btn_lang'], on_click=toggle_lang)