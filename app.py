import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import load_data, prepare_data, train_model, predict_future

st.title("🦠 AI Epidemic Predictor & Risk Dashboard")
st.write("Developed by Bhavya Sree Sai for Codecure AI Hackathon 2026")
# Load data
df = load_data()

# Dropdown for country
country = st.selectbox("🌍 Select Country", df['Country'].unique())

# Prepare data
df_country = prepare_data(df, country)

# Train model
model = train_model(df_country)

# Predict future cases
last_day = df_country['Days'].max()
future_days, predictions = predict_future(model, last_day)

# Plot graph
st.subheader("📈 Actual vs Predicted Cases")

plt.figure()
plt.plot(df_country['Days'], df_country['Confirmed'], label="Actual Cases")
plt.plot(future_days, predictions, label="Predicted Cases")
plt.xlabel("Days")
plt.ylabel("Cases")
plt.legend()
st.pyplot(plt)

# Risk Level
latest_cases = df_country['Confirmed'].iloc[-1]

if latest_cases < 100000:
    risk = "🟢 Low Risk"
elif latest_cases < 1000000:
    risk = "🟡 Medium Risk"
else:
    risk = "🔴 High Risk"

st.subheader(f"⚠️ Current Risk Level: {risk}")

# Show predictions
st.subheader("📊 Next 7 Days Prediction")
st.write(predictions)
