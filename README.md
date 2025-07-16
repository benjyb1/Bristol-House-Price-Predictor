# 🏡 House Price Prediction in the Bristol Area using Machine Learning

This project uses machine learning techniques — including ARIMAX and LSTM models — to predict house prices across Bristol, incorporating both spatial and demographic data. Built in collaboration with the housing charity **Baggator**, the tool aims to offer a robust and interpretable way to understand and anticipate housing market trends.
This project was done for the housing charity Baggator.
[View Full Report (PDF)](MDM3_House_Prices.pdf)

---

## 📈 Overview

We combine **temporal analysis**, **spatial correlation**, and **demographic statistics** to create predictive models that capture the complex dynamics of the Bristol housing market. The system not only forecasts future prices with high accuracy, but also adapts intelligently to market shocks such as financial crashes.

---

## 🔍 Key Features

- **📊 ARIMAX model** — Predicts house prices across Bristol wards using historical prices and demographic data.
- **📉 Financial Crash Detection** — Identifies financial crises and dynamically adjusts predictions to account for sudden changes.
- **🧠 LSTM Forecasting** — Long Short-Term Memory models capture temporal dependencies in house price trends.
- **🌍 Spatial Signal Detection** — Identifies wards that influence price changes in other areas through spatial correlation analysis.
- **🔬 PCA Preprocessing** — Reduces dimensionality of demographic data for more efficient and robust training.

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** PyTorch, scikit-learn, pandas, statsmodels
- **Models:** ARIMAX, LSTM
- **Tools:** PCA, time-series analysis, spatial correlation mapping

---

## 🚀 Getting Started

> **Note:** This repo currently does not include a UI — it's focused on the modeling and analysis pipeline.

1. Clone the repo:
```bash
git clone https://github.com/YOUR-USERNAME/house-price-bristol.git
cd house-price-bristol
