import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# --- Custom Attention layer ---
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="att_u",
            shape=(input_shape[-1],),
            initializer="random_normal",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)         # (batch, time_steps)
        alphas = tf.nn.softmax(vu)                   # (batch, time_steps)
        context = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return context

# --- Load the trained .h5 model ---
@st.cache_resource
def load_trained_model():
    return load_model(
        "LSTM with Attention for Log-Return Prediction.h5",
        custom_objects={"Attention": Attention}
    )

model = load_trained_model()

# --- Streamlit App ---
st.title("📈 Next‑Day Log‑Return Predictor (LSTM + Attention)")

symbol = st.text_input("Enter stock ticker (e.g. AAPL, GOOG):", "GOOG").upper()

if st.button("Predict"):
    df = yf.download(
        symbol,
        start="2015-01-01",
        end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    if df.empty:
        st.error(f"No data for '{symbol}'")
        st.stop()

    # feature engineering
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df["Close_Log"] = np.log(df["Close"])
    df["LogReturn"] = df["Close_Log"].diff()
    df["MA7"] = df["Close"].rolling(7).mean()
    df.dropna(inplace=True)

    feats = ["Open","High","Low","Close","Volume","MA7"]
    data = df[feats].values

    # scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # sequence
    ts = 20
    if len(scaled) < ts:
        st.error("Not enough data (need ≥20 days).")
        st.stop()
    X = scaled[-ts:].reshape(1, ts, len(feats))

    # predict
    p = model.predict(X)[0][0]
    st.metric("Next‑Day Log‑Return", f"{p:.5f}")

    # signal
    thr = 0.001
    if p > thr:
        st.success("Signal: BUY")
    elif p < -thr:
        st.error("Signal: SELL")
    else:
        st.info("Signal: HOLD")

    # chart
    st.subheader("Recent Log‑Returns")
    st.line_chart(df["LogReturn"].tail(100))
