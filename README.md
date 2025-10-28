# ⚡ Electricity Trading — Sentiment-Driven ΔPrice Strategy (EPEX DE-LU)

This project implements a **machine-learning-driven quantitative trading strategy** for the German-Dutch EPEX day-ahead electricity market.  
It predicts **next-day electricity price changes (ΔPrice)** by combining market fundamentals (TTF gas prices, load, volatility) with **news sentiment analysis (FinBERT)**.

---

## 🚀 Overview

| Aspect | Description |
|--------|--------------|
| **Goal** | Forecast next-day electricity price movements (ΔPrice) |
| **Market** | EPEX DE-LU (Day-Ahead Power Market) |
| **Model** | XGBoost Regression |
| **Sentiment Source** | Energy-related news headlines (FinBERT embeddings) |
| **Evaluation** | Cumulative PnL, Sharpe Ratio, Drawdown |

---

## ⚙️ Pipeline Summary

| Step | Description |
|------|--------------|
| **1️⃣ Data Acquisition** | Load market, TTF gas, and energy-related news data |
| **2️⃣ Energy Proxy** | Generate features from demand/load and price volatilities |
| **3️⃣ Sentiment Layer** | Transform news data into sentiment features using FinBERT |
| **4️⃣ Feature Engineering** | Combine market and sentiment variables into daily features |
| **5️⃣ Modeling** | Train XGBoost model for ΔPrice regression |
| **6️⃣ Backtesting** | Evaluate cumulative PnL, Sharpe ratio, and risk metrics |
| **7️⃣ Evaluation** | Compare with Buy-and-Hold benchmark |

---

## 🧮 Formulas

### Target (regression)
\[
fwd1\_dprice_t = p_t - p_{t-1} \quad [€ / MWh]
\]

### Sentiment & Load normalization
\[
load\_z_t = \frac{load_t - \mu_{30}(load)}{\sigma_{30}(load)}, \quad dprice\_vol\_{10,t} = \sigma_{10}(\Delta P)
\]

### Gas market driver
\[
ttf\_d1_t = TTF_t - TTF_{t-1}
\]

### Final model input set
\[
X_t = [mean\_sent_{t-k}, load\_z_t, dprice\_vol\_{10,t}, ttf\_d1_t]
\]

---

## 🧠 How to Run
🧠 How to Run
1️⃣ Clone the repository
git clone https://github.com/bhosalesiddharth30/Energy-Market-Sentiment-Analysis-and-Price-Forecasting.git
cd Energy-Market-Sentiment-Analysis-and-Price-Forecasting

2️⃣ Create and activate a virtual environment

On Windows:

python -m venv .venv
.venv\Scripts\activate


On Linux/Mac:

python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the notebook
jupyter notebook notebooks/energy_sentiment_analysis.ipynb

🧠 Tech Stack

Language: Python 3.10+

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, torch, transformers (FinBERT), yfinance, tqdm

Tools: Jupyter Notebook, VS Code

Data Sources: EPEX Spot (OPSD), TTF gas prices, energy news feeds

📊 Example Backtest Result (Simulated)
Metric	Buy & Hold	Strategy
CAGR	−1.85 %	19.71 %
Sharpe Ratio	0.21	0.84

Result shows predictive power of combining sentiment and fundamentals in electricity trading.

📜 License
MIT License © 2025 Siddharth Sunil Bhosale
For educational and research use only.
Data sourced from public EPEX / OPSD datasets and energy news feeds.