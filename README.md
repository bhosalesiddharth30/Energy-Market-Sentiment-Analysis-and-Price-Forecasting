# ⚡ Electricity Trading — Sentiment-Driven ΔPrice Strategy (EPEX DE-LU)

This project implements a **machine-learning-driven quantitative trading strategy** for the **German day-ahead power market**.  
It predicts **next-day electricity price movements (€/MWh)** by combining **news sentiment analysis (FinBERT)** with **energy fundamentals** such as system load and gas prices.

---

## 🚀 Overview

| Aspect | Description |
|--------|--------------|
| **Goal** | Forecast next-day price change (ΔPrice) using sentiment and market features |
| **Market** | EPEX DE-LU day-ahead electricity market |
| **Model** | XGBoost Regression on engineered features |
| **Sentiment Source** | Energy & market-related headlines analyzed with FinBERT |
| **Evaluation** | Cumulative PnL (€/MWh), Sharpe ratio, and drawdown |

---

## 🧩 Pipeline Summary

| Step | Description |
|------|--------------|
| **1. Data Acquisition** | Downloaded power-market fundamentals from [OPSD](https://data.open-power-system-data.org) (German load, day-ahead prices). |
| **2. Energy Proxy** | Added **Dutch TTF gas futures** (`TTF=F` via Yahoo Finance) as an energy price driver. |
| **3. Sentiment Layer** | Applied **FinBERT transformer** to energy-related headlines to compute daily sentiment scores. |
| **4. Feature Engineering** | Built lag features for sentiment, load, volatility, and TTF gas deltas. |
| **5. Modeling** | Trained an **XGBoost Regressor** on forward price changes (ΔPrice €/MWh). |
| **6. Backtesting** | Simulated a simple **points-PnL strategy**: trade when |predicted move| > quantile threshold (0.7). |
| **7. Evaluation** | Reported **Sharpe ratio, cumulative PnL (€/MWh)**, and **drawdown**. |

---

## 📈 Key Results

| Metric | Value | Interpretation |
|--------|--------|----------------|
| **Trades executed** | 54 | selective, low-frequency trading |
| **Sharpe Ratio** | 2.10 | strong risk-adjusted performance |
| **Cumulative PnL (€/MWh)** | +150 | steady positive trend |
| **Drawdown** | — | large due to normalization; can be rescaled with capital sizing |

💡 *Result shows predictive power of combining sentiment and market fundamentals in electricity trading.*

---

## 🧮 Formulas

**Target (regression):**

$$
fwd1\_dprice_t = p_t - p_{t-1} \quad [€/MWh]
$$

---

**Sentiment & Load normalization:**

$$
load\_z_t = \frac{load_t - \mu_{30}(load)}{\sigma_{30}(load)}, 
\quad dprice\_vol\_{10,t} = \sigma_{10}(\Delta P)
$$

---

**Gas market driver:**

$$
ttf\_d1_t = TTF_t - TTF_{t-1}
$$

---

**Final model input set:**

$$
X_t = [mean\_sent_{t-k}, load\_z_t, dprice\_vol\_{10,t}, ttf\_d1_t]
$$

---

## ⚙️ How to Run

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/Energy_Sentiment_Trading.git
cd Energy_Sentiment_Trading

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run notebook
jupyter notebook energy_sentiment_analysis.ipynb
🧠 Tech Stack
Category	Tools
Language	Python
Data	pandas, numpy, yfinance, OPSD
ML Framework	XGBoost
NLP	Hugging Face Transformers (FinBERT)
Visualization	Matplotlib, Seaborn
Backtesting	pandas-based signal simulation

🧩 Folder Structure
bash
Copy code
Energy_Sentiment_Trading/
│
├── data/
│   ├── raw/                # OPSD, TTF, and news CSVs
│   ├── processed/          # cleaned and feature-engineered data
│
├── notebooks/
│   └── energy_sentiment_analysis.ipynb
│
├── requirements.txt
└── README.md
🔍 Possible Extensions
Add real EPEX or ENTSO-E day-ahead forecasts as features

Use Transformer models (e.g., GPT or Llama-based) for contextual sentiment scoring

Integrate trading cost model for realistic PnL

Extend to intraday trading using shorter time resolutions

⚠️ Disclaimer
This is a research/educational project. It is not investment advice and does not account for transaction costs, liquidity, or market impact.