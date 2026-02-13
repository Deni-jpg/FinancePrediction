📈 Finance Prediction
=====================================

Este projeto utiliza a biblioteca **NeuralProphet** (baseada em PyTorch) para analisar dados históricos de ações e gerar previsões de preços para os próximos **365 dias**.

🚀 Funcionalidades
------------------

-   **Coleta de Dados Real** Integração com a API do **Yahoo Finance** para obter cotações históricas atualizadas.

-   **Treinamento Otimizado** O modelo é treinado apenas uma vez. Após o treino inicial, ele é salvo em um arquivo `.pt` e carregado automaticamente nas próximas execuções.

-   **Compatibilidade com PyTorch 2.6+** Implementação de um *bypass* para contornar mudanças recentes nos protocolos de segurança do PyTorch.

-   **Visualização Gráfica Completa** Geração de gráficos comparando dados reais, ajuste histórico e projeção futura.

-   **Métricas de Performance** Avaliação utilizando:

    -   **R² Score**

    -   **MAPE (Erro Médio Percentual Absoluto)**

🛠️ Tecnologias Utilizadas
--------------------------

-   **Python**

-   **NeuralProphet** -- Modelagem de séries temporais explicável

-   **yFinance** -- Coleta de dados financeiros em tempo real

-   **Pandas** -- Manipulação de dados

-   **Matplotlib** -- Visualização gráfica

-   **Scikit-Learn** -- Métricas de avaliação

📋 Como Executar
----------------

### 1\. Clone o repositório

bash

```
git clone https://github.com/Deni-jpg/FinancePrediction.git

```

### 2\. Instale as dependências

bash

```
pip install neuralprophet yfinance pandas matplotlib scikit-learn

```

### 3\. Execute o projeto

bash

```
python main.py

```

📊 Estrutura do Gráfico
-----------------------

O gráfico gerado pelo script contém três camadas principais:

| Cor | Representação |
| --- | --- |
| 🟩 **Verde** | Valores reais do preço de fechamento |
| 🔴 **Vermelho** | Previsões históricas (ajuste do modelo aos dados passados) |
| 🔵 **Azul** | Projeção futura para os próximos 365 dias |
