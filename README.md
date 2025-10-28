# ⚡ Electricity Trading — Sentiment-Driven ΔPrice Strategy (EPEX DE-LU)

This project implements a **machine-learning-driven quantitative trading strategy** for the German–Luxembourg electricity day-ahead market (EPEX DE-LU).  
It predicts **next-day electricity price changes** (`ΔPrice`) using **FinBERT-based sentiment analysis** on energy-related news combined with **market fundamentals** such as gas and load indicators.

---

## 🚀 Overview

| Aspect | Description |
|--------|--------------|
| **Goal** | Forecast next-day electricity price change (ΔPrice) |
| **Market** | EPEX DE-LU (Day-Ahead Market) |
| **Model** | XGBoost Regression with engineered sentiment + market features |
| **Sentiment Source** | Energy-related news headlines analyzed with FinBERT |
| **Evaluation** | Cumulative strategy returns & Sharpe ratio vs Buy & Hold benchmark |

---

## 🧩 Pipeline Summary

| Step | Description |
|------|--------------|
| **1️⃣ Data Acquisition** | Collected day-ahead prices (OPSD), gas (TTF), load data, and energy-sector news. |
| **2️⃣ Energy Proxy** | Added gas price and load as market drivers for energy price forecasting. |
| **3️⃣ Sentiment Layer** | Extracted daily sentiment using FinBERT (`transformers`). |
| **4️⃣ Feature Engineering** | Combined sentiment, volatility, and fundamental signals. |
| **5️⃣ Modeling** | Trained XGBoost Regressor on engineered features. |
| **6️⃣ Backtesting** | Simulated trading based on predicted price direction (ΔPrice). |
| **7️⃣ Evaluation** | Compared CAGR and Sharpe ratio vs Buy & Hold. |

---

## 📊 Core Formulas (Plain Text Version for GitHub)

**Target (regression):**  
`fwd1_dprice_t = p_t − p_(t−1)`    *(units: €/MWh)*  

---

### 📈 Sentiment & Load Normalization

**Formula 1 – Load Normalization:**  
`load_z_t = (load_t − mu_30(load)) / sigma_30(load)`  
→ **load_z**<sub>t</sub> = (**load**<sub>t</sub> − **μ**<sub>30</sub>(load)) / **σ**<sub>30</sub>(load)

**Formula 2 – Daily Price Volatility:**  
`dprice_vol_10_t = sigma_10(ΔP)`  
→ **dprice_vol_10**<sub>t</sub> = **σ**<sub>10</sub>(ΔP)  where ΔP<sub>t</sub> = p<sub>t</sub> − p<sub>t−1</sub>

---

### 🔥 Gas Market Driver

`ttf_d1_t = TTF_t − TTF_(t−1)`  
→ **ttf_d1**<sub>t</sub> = **TTF**<sub>t</sub> − **TTF**<sub>t−1</sub>

---

### 🧠 Final Model Input Set

`X_t = [mean_sent_(t−k), load_z_t, dprice_vol_10_t, ttf_d1_t]`  
→ **X**<sub>t</sub> = [ **mean_sent**<sub>t−k</sub>, **load_z**<sub>t</sub>, **dprice_vol_10**<sub>t</sub>, **ttf_d1**<sub>t</sub> ]

---

### 🔢 Notation

| Symbol | Meaning |
|---------|----------|
| **μ**<sub>30</sub>(·) | 30-day rolling mean |
| **σ**<sub>30</sub>(·), **σ**<sub>10</sub>(·) | 30-day / 10-day rolling standard deviation |
| **mean_sent**<sub>t−k</sub> | Trailing k-day mean of daily sentiment |

---

## ⚙️ How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/bhosalesiddharth30/Energy-Market-Sentiment-Analysis-and-Price-Forecasting.git
   cd Energy-Market-Sentiment-Analysis-and-Price-Forecasting
Create and activate a virtual environment

bash
Copy code
python -m venv .venv
.\.venv\Scripts\activate    # on Windows
# source .venv/bin/activate # on Linux/Mac
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the notebook

bash
Copy code
jupyter notebook notebooks/energy_sentiment_analysis.ipynb
🧠 Tech Stack
Python 3.10+

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, torch, transformers (FinBERT), yfinance, tqdm

Tools: Jupyter Notebook, VS Code

Data: EPEX Spot (OPSD), TTF gas prices, energy news feeds

📈 Example Backtest Result (Simulated)
Metric	Strategy	Buy & Hold
CAGR	−1.85 %	19.71 %
Sharpe Ratio	0.84 (Strategy) vs 0.21 (B&H)	

🧾 License
MIT License © 2025 Siddharth Sunil Bhosale
For educational and research use only. Data sourced from public EPEX / OPSD datasets and energy news feeds.