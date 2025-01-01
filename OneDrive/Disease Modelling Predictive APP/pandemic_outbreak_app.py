import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Title and Description
st.title("Pandemic Outbreak Prediction App")
st.write("""
This app simulates the spread of an infectious disease using the **SIR model**
and predicts future infection rates using machine learning.

Use the sidebar to input the parameters for the simulation and prediction.
""")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
population = st.sidebar.number_input("Population Size", value=1000000, step=1000)
initial_infected = st.sidebar.number_input("Initial Infected", value=10, step=1)
reproduction_number = st.sidebar.number_input("Reproduction Number (R0)", value=2.5, step=0.1)
recovery_rate = st.sidebar.number_input("Recovery Rate (γ)", value=0.1, step=0.01)
days = st.sidebar.number_input("Days to Simulate", value=30, step=1)
prediction_days = st.sidebar.number_input("Days to Predict", value=7, step=1)

# Validate Inputs
if population <= 0 or initial_infected <= 0 or recovery_rate <= 0 or days <= 0:
    st.error("All input values must be positive. Please update the inputs.")
    st.stop()

# SIR Model Implementation
@st.cache_data
def sir_model(S, I, R, beta, gamma, population, days):
    SIR = {"Susceptible": [S], "Infected": [I], "Recovered": [R]}
    for _ in range(days):
        dS = -beta * S * I / population
        dI = beta * S * I / population - gamma * I
        dR = gamma * I
        S, I, R = S + dS, I + dI, R + dR
        SIR["Susceptible"].append(S)
        SIR["Infected"].append(I)
        SIR["Recovered"].append(R)
    return pd.DataFrame(SIR)

@st.cache_data
def predict_infection_rate(data, days_to_predict=7):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["Infected"].values
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(data), len(data) + days_to_predict).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

# Derived Parameters
beta = reproduction_number * recovery_rate
initial_susceptible = population - initial_infected
initial_recovered = 0

# SIR Model Simulation
sir_data = sir_model(
    S=initial_susceptible,
    I=initial_infected,
    R=initial_recovered,
    beta=beta,
    gamma=recovery_rate,
    population=population,
    days=days
)

# Display SIR Simulation Results
st.subheader("SIR Model Simulation")
st.write("Simulation Parameters:")
st.write(f"Population: {population}, Initial Infected: {initial_infected}, β: {beta:.4f}, γ: {recovery_rate:.4f}")
st.line_chart(sir_data.rename(columns={
    "Susceptible": "Susceptible Population",
    "Infected": "Infected Population",
    "Recovered": "Recovered Population"
}))

# Predict Future Infections
st.subheader("Infection Rate Prediction")
predicted_infections = predict_infection_rate(sir_data, days_to_predict=prediction_days)
future_dates = pd.date_range(start=pd.Timestamp.today(), periods=prediction_days).date
prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Infections": predicted_infections})
st.write(prediction_df)

# Visualization of Predictions
# Visualization of Predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(sir_data)), sir_data["Infected"], label="Actual Infected", color="blue")
ax.plot(range(len(sir_data), len(sir_data) + prediction_days), predicted_infections, 
        label="Predicted Infected", linestyle="--", color="red")
ax.set_title("Actual vs Predicted Infections", fontsize=16)
ax.set_xlabel("Days", fontsize=12)
ax.set_ylabel("Infected Cases", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend()
st.pyplot(fig)

