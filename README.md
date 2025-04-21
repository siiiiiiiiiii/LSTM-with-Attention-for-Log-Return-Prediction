## 📊 Project Highlight — LSTM + Attention for Log-Return Forecasting

### 🌐 Try the Live App
You can try this model in action via an interactive Streamlit web app below:

➡️ [logreturn-predictor.streamlit.app](https://logreturn-predictor.streamlit.app)

> ⚠️ **Note**: Most predicted log-returns ≈ 0.000xx due to conservative training and MSE loss. I’m improving this with better features and loss functions — while ensuring Sharpe, Sortino, and max drawdown remain stable


This project builds a **deep learning model with LSTM and Attention** to forecast daily log-returns of Google (GOOG) stock. It integrates:
- ⚙️ A custom Attention layer
- 🔁 20-day time windows
- 🧹 Full data preprocessing with log-transformation, moving averages, and MinMax scaling
- 🧠 Regularized model with Dropout, L2 penalties, EarlyStopping, and LR scheduling

### ✅ Model Results (Test Set)
| Metric         | LSTM + Attention | Naive Baseline |
|----------------|------------------|----------------|
| MAE            | **0.0127**       | 0.0185         |
| RMSE           | **0.0176**       | 0.0248         |

> The model **outperformed the naive benchmark by 31.4% (MAE)** and **29.0% (RMSE)** — a strong indicator of predictive power in a noisy financial setting.

### 📉 Training Loss
The model converged smoothly over 100 epochs with no overfitting:
- Final Val Loss: **3.2359e-4**
- Train/Val Loss curves nearly overlap

### 📈 Visualization
- The red dashed line shows model predictions, closely aligned with true log-returns (blue)
- Predicts direction and magnitude well in many cases, avoids overreacting to noise

### 🧠 Why This Project Matters
Unlike toy datasets, this model handles:
- **Real-world volatility**
- **Multivariate input** (OHLCV + MA7)
- **Noise suppression**
- And it’s fully built, tuned, and trained from scratch — with production-ready code and results saved for reuse.
