# F1 ML Race Outcome Predictor

A machine learning model trained on real 2026 Formula 1 race data that 
predicts race finishing positions using grid position, qualifying pace, 
track temperature, weather conditions, and driver identity.

## What it does

This script loads race and qualifying data from all 8 rounds of the 2026 
F1 season, trains three machine learning models, and produces six charts:

- Model accuracy comparison — Random Forest vs Gradient Boosting vs 
  Linear Regression by Mean Absolute Error
- Predicted vs actual positions — scatter plot showing model accuracy 
  with ±1 and ±2 position error bands
- Feature importance — which factors most strongly predict race outcome
- Austrian GP prediction vs actual — head to head comparison for top 10
- 5-fold cross validation — proving model robustness across data splits
- Grid vs finish correlation — the fundamental relationship the ML learns

## Models Used

- Random Forest Regressor (100 estimators)
- Gradient Boosting Regressor (100 estimators)
- Linear Regression (baseline)

## Key Findings

- Grid position is the strongest predictor at ~62% feature importance
- Qualifying position adds ~32% additional predictive power
- Driver identity contributes ~5% — some drivers consistently outperform grid
- Grid vs finish position correlation: 0.624
- Best model MAE: ~4.7 positions (Linear Regression)
- All models achieve consistent results across 5-fold cross validation

## Tech Stack

- Python
- [FastF1](https://github.com/theOehrly/Fast-F1) — official F1 race data
- Scikit-Learn — Random Forest, Gradient Boosting, Linear Regression, 
  cross validation
- Matplotlib — visualisation dashboard
- Pandas — multi-race data aggregation
- NumPy — numerical processing

## How to Run

1. Install dependencies: 
   `pip install fastf1 matplotlib pandas numpy scikit-learn`
2. Run the script: `python race_predictor.py`
3. Dashboard will display and save as `race_predictor.png`

## Why This Project

Race outcome prediction is the frontier of modern motorsport analytics. 
Teams use ML models to simulate thousands of race scenarios before and 
during each event — informing strategy calls, pit stop timing, and 
risk assessment. This project demonstrates the full ML pipeline from 
real-world data collection through model training, evaluation, and 
prediction on unseen data.

## Author

Hamna Shahzad — Electrical Engineering Student | Aspiring Motorsport Engineer
