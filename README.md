
# 📈 LSTM + Attention for Log-Return Forecasting

This project builds a **deep learning model** with **LSTM and a custom Attention mechanism** to predict daily **log-returns** of stock prices. It outperforms a naive benchmark, avoids overfitting, and demonstrates solid **risk-adjusted performance** in a noisy financial environment.

---

## 🧠 Model Architecture

| Layer            | Output Shape   | Parameters |
|------------------|----------------|------------|
| Input (20, 10)   | —              | 0          |
| LSTM (32 units)  | (20, 32)       | 5,504      |
| Dropout          | (20, 32)       | 0          |
| Attention Layer  | (32)           | 1,088      |
| Dropout          | (32)           | 0          |
| Dense (ReLU)     | (16)           | 528        |
| Dropout          | (16)           | 0          |
| Dense (Linear)   | (1)            | 17         |
| **Total**        | —              | **7,137**  |

---

## 🧪 Model Evaluation on Log-Return

| Metric      | LSTM + Attention | Naive Baseline *(Previous Day’s Return)* |
|-------------|------------------|-------------------------------------------|
| MAE         | **0.0127**       | 0.0185                                    |
| MSE         | **0.0003**       | 0.0006                                    |
| RMSE        | **0.0176**       | 0.0248                                    |

> ✅ **Performance improvement over naive baseline**  
> MAE improved by **+31.4%**, RMSE improved by **+29.0%**

---

## 📉 Risk-Adjusted Performance

| Metric         | Value    |
|----------------|----------|
| **Sharpe Ratio**  | 0.9762   |
| **Sortino Ratio** | 1.2893   |
| **Max Drawdown**  | -22.28%  |

> These risk metrics suggest the model offers **reasonable returns per unit of risk**, especially in high-volatility scenarios.

---

## 🔍 Loss Curve – No Overfitting


> Validation loss closely tracks training loss with no divergence, indicating **good generalization and no overfitting**.

---

## 📊 Log-Return Prediction



> While the model captures the trend around zero, it **smooths extreme values** due to the noisy nature of financial time series. Still, it significantly reduces prediction error compared to the naive approach.

---

## 💾 Model Details

- Loss Function: Huber Loss (robust to outliers)
- Regularization: L2 + Dropout + EarlyStopping
- Optimizer: Adam with LR scheduling
- Data: 20-day time windows, 10 log-return based features
- Output: Next-day log-return prediction

---

## 🧠 Conclusion

This LSTM + Attention model is a **robust baseline** for financial time-series forecasting. It:
- Beats the naive baseline on all metrics
- Demonstrates solid risk-adjusted returns
- Shows **no signs of overfitting**
- Is ideal for further development into **position sizing** or **trading signal generation**

> Feel free to explore the source code, training scripts, and Streamlit demo in this repository.
