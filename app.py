import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Set page configuration
st.set_page_config(page_title="Symbiotic AI Demo", layout="wide")

# Title and description
st.title("Symbiotic AI Project Demo")
st.write("Experience the power of AI-driven symbiotic systems that enhance human capabilities.")

# User Interface
st.header("User Interaction")
name = st.text_input("Enter your name:", "John Doe")
age = st.slider("Your age:", 18, 80, 30)
activity_level = st.selectbox("Select your activity level:", ["Sedentary", "Moderate", "Active"])

# Wearable Device Integration
st.header("Wearable Device Data")
heart_rate = st.number_input("Current heart rate (bpm)", min_value=50, max_value=200, value=75)
steps = st.number_input("Steps taken today", min_value=0, step=100, value=5000)
sleep_quality = st.slider("Sleep quality (1-10)", 1, 10, 7)

# Simulate a dataset
df = pd.DataFrame({
    "Age": [25, 35, 45, 55, 65],
    "Activity Level": ["Sedentary", "Moderate", "Active", "Sedentary", "Moderate"],
    "Heart Rate": [70, 80, 75, 65, 85],
    "Steps": [3000, 7000, 10000, 2000, 5000],
    "Sleep Quality": [5, 7, 8, 4, 6]
})

# Append the user's data
df = df.append({
    "Age": age,
    "Activity Level": activity_level,
    "Heart Rate": heart_rate,
    "Steps": steps,
    "Sleep Quality": sleep_quality
}, ignore_index=True)

# AI Algorithms
st.header("AI Insights")
# One-hot encode the "Activity Level" column
encoder = OneHotEncoder()
X = df[["Age", "Heart Rate", "Steps"]]
X_encoded = encoder.fit_transform(df[["Activity Level"]]).toarray()
X = np.hstack((X, X_encoded))

model = LinearRegression()
y = df["Sleep Quality"]
model.fit(X, y)

# Predict for the user's data
user_data = np.array([[age, heart_rate, steps] + list(encoder.transform([[activity_level]]).toarray()[0])])
predicted_sleep_quality = model.predict(user_data)[0]
st.write(f"Based on your data, your predicted sleep quality is: {predicted_sleep_quality:.2f}")

# Neural Interface Simulation
st.header("Neural Interface Simulation")
if st.button("Activate Neural Interface"):
    st.write("Connecting to neural interface...")
    # Simulate neural interface functionality
    neural_signal = np.random.normal(0, 1, 100)
    st.line_chart(neural_signal)
    st.write("Neural signals detected. Interpreting data...")
    # Simulate AI response to neural signals
    st.write("AI has interpreted your neural signals and adjusted the symbiotic system accordingly.")

# Results and Visualizations
st.header("Symbiotic AI Performance")
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5], [5000, 6000, 7000, 8000, 9000], label="Steps")
ax.plot([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], label="Sleep Quality")
ax.set_xlabel("Time (weeks)")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)
st.write("The symbiotic AI system has helped you increase your steps and improve your sleep quality over time.")
