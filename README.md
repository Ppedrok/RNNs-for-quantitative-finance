# Multi-Horizon Volatility Forecasting with RNNs

## Overview
This project implements a **multi-horizon volatility forecasting model** using a **sequence-to-sequence GRU architecture**.  
The model takes as input volatility-related features (absolute returns, rolling standard deviations, ATR) and predicts the **log-realized volatility** at horizons from 1 up to 10 days.  

The repository includes:
- Exploratory Data Analysis (EDA) on multiple assets  
- Feature engineering for volatility predictors  
- Sequence-to-sequence GRU model for multi-horizon forecasting  
- Walk-forward out-of-sample evaluation with embargo  
- Visualization of true vs. predicted volatility  

---

## Methodology
- **Target**: realized volatility  
  \[
  RV_{t,k} = \sqrt{\sum_{j=1}^{k} r_{t+j}^2}, \quad y_{t,k} = \log(RV_{t,k})
  \]  
- **Architecture**: GRU → GRU → TimeDistributed Dense layer  
- **Loss**: Huber (robust to outliers)  
- **Evaluation**: chronological train/validation/test split (70/15/15).  
  Future work will extend this to a full **walk-forward OOS evaluation** in order to better replicate production settings.

---

## Results
The model is able to track volatility clustering and regime shifts, providing a **term structure of risk forecasts** (1–10 days).  

---

## Applications
- Risk management (VaR, Expected Shortfall)  
- Trading and portfolio allocation (volatility targeting)  
- Options pricing (realized vs implied volatility)  

---

## How to Run
Clone the repo and install dependencies:
```bash
git clone https://github.com/username/repo-name.git
cd repo-name
pip install -r requirements.txt

