import torch
import torch.serialization

torch.serialization.default_restore_location = lambda storage, loc: storage
torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)

import os
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from neuralprophet import NeuralProphet

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="FinancePrediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------------------------------------------------
# IDIOMAS
# -----------------------------------------------------------------------
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'

def toggle_lang():
    st.session_state.lang = 'EN' if st.session_state.lang == 'PT' else 'PT'

T = {
    'PT': {
        'subtitle':   "PREVISÃO DE AÇÕES COM MACHINE LEARNING",
        'sb_title':   "Parâmetros",
        'ticker':     "Ticker (ex: NVDA)",
        'start':      "Data inicial (AAAA-MM-DD)",
        'end':        "Data final (AAAA-MM-DD)",
        'days':       "Dias para prever",
        'btn_train':  "Treinar e Prever",
        'btn_lang':   "🇺🇸 English",
        'error':      "Erro: dados não encontrados.",
        'cur_price':  "PREÇO ATUAL",
        'tgt_price':  "ALVO PREVISTO",
        'hist_pred':  "Previsões Históricas",
        'fut_pred':   "Previsão — Próximos {} dias",
        'real_val':   "Valores Reais",
        'perf_title': "PERFORMANCE DO MODELO",
        'r2_label':   "R² SCORE",
        'mape_label': "MAPE (ERRO MÉDIO)",
        'loading':    "Treinando modelo...",
        'dev_credit': "Desenvolvido por",
    },
    'EN': {
        'subtitle':   "AI-POWERED STOCK MARKET PREDICTION",
        'sb_title':   "Parameters",
        'ticker':     "Ticker (e.g., NVDA)",
        'start':      "Start date (YYYY-MM-DD)",
        'end':        "End date (YYYY-MM-DD)",
        'days':       "Days to predict",
        'btn_train':  "Train & Predict",
        'btn_lang':   "🇧🇷 Português",
        'error':      "Error: data not found.",
        'cur_price':  "CURRENT PRICE",
        'tgt_price':  "TARGET PRICE",
        'hist_pred':  "Historical Predictions",
        'fut_pred':   "Forecast — Next {} days",
        'real_val':   "Real Values",
        'perf_title': "MODEL PERFORMANCE",
        'r2_label':   "R² SCORE",
        'mape_label': "MAPE (MEAN ERROR)",
        'loading':    "Training model...",
        'dev_credit': "Developed by",
    }
}
L = T[st.session_state.lang]

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.markdown(f"""
<div class="fp-header">
    <div class="fp-title">FinancePrediction</div>
    <div class="fp-subtitle">{L['subtitle']}</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {L['sb_title']}")
    cod           = st.text_input(L['ticker'], value="NVDA").upper()
    begin         = st.text_input(L['start'],  value="2020-01-01")
    end_date_str  = st.text_input(L['end'],    value="2026-01-01")
    dias_previsao = st.number_input(L['days'], min_value=30, max_value=730, value=365)

    st.markdown("---")
    btn_treinar = st.button(L['btn_train'])
    st.button(L['btn_lang'], on_click=toggle_lang)

    # ── CRÉDITO NO RODAPÉ DA SIDEBAR ──────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div class="dev-credit">
        {L['dev_credit']} <a href="https://linktr.ee/danielcoelho" target="_blank">Daniel Coelho</a>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------
# EXECUÇÃO
# -----------------------------------------------------------------------
if btn_treinar:
    if not cod:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(L['loading']):
            try:
                # DADOS — igual ao main.py
                dados = yf.download(cod, start=begin, end=end_date_str, multi_level_index=False)

                if dados is None or dados.empty:
                    st.error(L['error'])
                else:
                    dados = dados[["Close"]].reset_index()
                    dados.columns = ["ds", "y"]

                    # MODELO — igual ao main.py
                    arquivo_modelo = f"modelo_{cod}.pt"
                    if os.path.exists(arquivo_modelo):
                        modelo = torch.load(arquivo_modelo)
                    else:
                        modelo = NeuralProphet(learning_rate=0.01)
                        modelo.fit(dados)
                        torch.save(modelo, arquivo_modelo)

                    # PREVISÕES — igual ao main.py
                    dados_futuros = modelo.make_future_dataframe(dados, periods=int(dias_previsao))
                    prev_futuras  = modelo.predict(dados_futuros)
                    prev_hist     = modelo.predict(dados)

                    # ── GRÁFICO PLOTLY ────────────────────────────────────
                    BG      = "#111318"
                    CARD_BG = "#16191f"
                    GRID    = "#2a2d35"
                    GREEN   = "#34d399"
                    RED     = "#f87171"
                    BLUE    = "#60a5fa"
                    MUTED   = "#6b7280"
                    TEXT    = "#d1d5db"

                    fig = go.Figure()

                    # Linha vermelha — previsões históricas
                    fig.add_trace(go.Scatter(
                        x=prev_hist["ds"],
                        y=prev_hist["yhat1"],
                        name=L['hist_pred'],
                        line=dict(color=RED, width=1.5),
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>",
                    ))

                    # Linha azul — previsões futuras (igual ao main.py, sem filtro)
                    fig.add_trace(go.Scatter(
                        x=prev_futuras["ds"],
                        y=prev_futuras["yhat1"],
                        name=L['fut_pred'].format(int(dias_previsao)),
                        line=dict(color=BLUE, width=2),
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>",
                    ))

                    # Linha verde — valores reais
                    fig.add_trace(go.Scatter(
                        x=dados["ds"],
                        y=dados["y"],
                        name=L['real_val'],
                        line=dict(color=GREEN, width=1.2),
                        opacity=0.9,
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>",
                    ))

                    fig.update_layout(
                        plot_bgcolor=CARD_BG,
                        paper_bgcolor=BG,
                        font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
                        title=dict(
                            text=f"<b>{cod}</b>  ·  {begin} → +{int(dias_previsao)}d forecast",
                            font=dict(size=12, color=MUTED),
                            x=1, xanchor="right",
                        ),
                        legend=dict(
                            bgcolor=BG, bordercolor=GRID, borderwidth=1,
                            font=dict(size=11), orientation="h",
                            x=0, y=1.08,
                        ),
                        xaxis=dict(
                            gridcolor=GRID, gridwidth=0.5,
                            tickcolor=MUTED, tickfont=dict(color=MUTED, size=10),
                            linecolor=GRID,
                            showspikes=True, spikecolor=MUTED,
                            spikethickness=1, spikedash="dot",
                        ),
                        yaxis=dict(
                            gridcolor=GRID, gridwidth=0.5,
                            tickcolor=MUTED, tickfont=dict(color=MUTED, size=10),
                            linecolor=GRID, tickprefix="$",
                            showspikes=True, spikecolor=MUTED,
                            spikethickness=1, spikedash="dot",
                        ),
                        hovermode="x unified",
                        hoverlabel=dict(
                            bgcolor=CARD_BG, bordercolor=GRID,
                            font=dict(family="JetBrains Mono", color=TEXT, size=11),
                        ),
                        margin=dict(l=10, r=10, t=50, b=10),
                        height=480,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # ── MÉTRICAS DE PREÇO ─────────────────────────────────
                    ultimo_preco   = float(dados["y"].iloc[-1])
                    preco_previsto = float(prev_futuras["yhat1"].iloc[-1])
                    variacao       = ((preco_previsto - ultimo_preco) / ultimo_preco) * 100
                    delta_class    = "metric-delta-pos" if variacao >= 0 else "metric-delta-neg"
                    delta_icon     = "▲" if variacao >= 0 else "▼"

                    mc1, mc2, mc3 = st.columns([1, 1, 2])
                    with mc1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{L['cur_price']}</div>
                            <div class="metric-value">${ultimo_preco:.2f}</div>
                        </div>""", unsafe_allow_html=True)
                    with mc2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{L['tgt_price']}</div>
                            <div class="metric-value">${preco_previsto:.2f}</div>
                            <div class="{delta_class}">{delta_icon} {abs(variacao):.2f}%</div>
                        </div>""", unsafe_allow_html=True)

                    # ── PERFORMANCE DO MODELO ─────────────────────────────
                    st.markdown(f'<div class="section-title">{L["perf_title"]}</div>',
                                unsafe_allow_html=True)

                    r2   = r2_score(y_true=prev_hist["y"], y_pred=prev_hist["yhat1"])
                    mape = mean_absolute_percentage_error(
                               y_true=prev_hist["y"], y_pred=prev_hist["yhat1"]) * 100

                    pc1, pc2, pc3 = st.columns([1, 1, 2])
                    with pc1:
                        st.markdown(f"""
                        <div class="perf-card">
                            <div class="perf-label">{L['r2_label']}</div>
                            <div class="perf-value">{r2:.4f}</div>
                        </div>""", unsafe_allow_html=True)
                    with pc2:
                        st.markdown(f"""
                        <div class="perf-card">
                            <div class="perf-label">{L['mape_label']}</div>
                            <div class="perf-value">{mape:.2f}%</div>
                        </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Error: {e}")