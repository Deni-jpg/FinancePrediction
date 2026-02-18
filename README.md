📈 FinancePrediction
====================

A **web application** for AI-powered stock price forecasting, built with **NeuralProphet** (PyTorch) and **Streamlit**.

![Finance Prediction Graph](/images/Figure_1.png)
*(Nvidia 2020--2027 forecast)*

> [!IMPORTANT]
>
> **Educational Use Only:** This project is intended for educational purposes and should not be used as professional financial advice.
>
> **Version Requirement:** This project requires **Python 3.11.9** for optimal compatibility with all specified libraries.

* * * * *

🚀 Features
-----------

-   **Interactive Web UI:** Full dashboard built with Streamlit --- no code needed to run predictions.
-   **Interactive Chart:** Zoom, pan and hover over data points powered by Plotly.
-   **Real Data Collection:** Integration with the **Yahoo Finance API** to retrieve up-to-date historical quotes.
-   **Configurable Parameters:** Set any ticker, date range, and forecast horizon directly from the sidebar.
-   **Optimized Training:** The model is trained only once per ticker. After the initial training, it is saved as a `.pt` file and loaded automatically on subsequent runs.
-   **PyTorch 2.6+ Compatibility:** Bypass for recent changes in PyTorch security protocols.
-   **Performance Metrics:** Model evaluation using:
    -   **R² Score**
    -   **MAPE (Mean Absolute Percentage Error)**
-   **PT / EN Language Toggle:** Full Portuguese and English support.

* * * * *

🛠️ Technologies
----------------

| Library | Purpose |
| --- | --- |
| **Python 3.11.9** | Runtime |
| **Streamlit** | Web application framework |
| **NeuralProphet** | Explainable time-series forecasting |
| **Plotly** | Interactive chart |
| **yFinance** | Real-time financial data |
| **Pandas** | Data manipulation |
| **Scikit-Learn** | Evaluation metrics (R², MAPE) |
| **PyTorch** | NeuralProphet backend |

* * * * *

📋 How to Run
-------------

### 1\. Clone the repository

bash

```
git clone https://github.com/Deni-jpg/FinancePrediction.git
cd FinancePrediction
```

### 2\. Install dependencies

bash

```
pip install -r requirements.txt
```

### 3\. Run the web app

bash

```
streamlit run app.py
```

> To run the original local script instead:
>
> bash
>
> ```
> python main.py
> ```

* * * * *

📊 Chart Legend
---------------

| Color | Representation |
| --- | --- |
| 🟩 **Green** | Actual closing price values |
| 🔴 **Red** | Historical predictions (model fit to past data) |
| 🔵 **Blue** | Future projection for the configured forecast period |
