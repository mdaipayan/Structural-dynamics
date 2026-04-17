import streamlit as st
import numpy as np
import pandas as pd
import time
import math

# Set page configuration
st.set_page_config(page_title="Real-Time Damping Visualization", layout="wide")

st.title("🏗️ Structural Dynamics: Real-Time Damping")
st.markdown("Adjust the system parameters in the sidebar and click **Start Simulation** to watch the oscillation dynamics draw in real-time.")

# --- SIDEBAR: System Parameters ---
st.sidebar.header("System Parameters")
m = st.sidebar.slider("Mass (m) [kg]", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
k = st.sidebar.slider("Stiffness (k) [N/m]", min_value=10.0, max_value=1000.0, value=200.0, step=10.0)
c = st.sidebar.slider("Damping Coefficient (c) [Ns/m]", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
x0 = st.sidebar.slider("Initial Displacement (x0) [m]", min_value=-10.0, max_value=10.0, value=5.0, step=0.5)
v0 = 0.0  # Initial velocity assumed to be zero for simplicity

# --- PHYSICS CALCULATIONS ---
omega_n = math.sqrt(k / m)                  # Natural frequency
c_critical = 2 * math.sqrt(k * m)           # Critical damping coefficient
zeta = c / c_critical                       # Damping ratio

# Display calculated properties
col1, col2, col3 = st.columns(3)
col1.metric("Natural Frequency (ω_n)", f"{omega_n:.2f} rad/s")
col2.metric("Damping Ratio (ζ)", f"{zeta:.2f}")

if zeta < 1:
    col3.metric("System State", "Underdamped")
    omega_d = omega_n * math.sqrt(1 - zeta**2)
elif zeta == 1:
    col3.metric("System State", "Critically Damped")
    omega_d = 0
else:
    col3.metric("System State", "Overdamped")
    omega_d = 0

st.divider()

# --- REAL-TIME VISUALIZATION ---
start_button = st.button("▶ Start Real-Time Simulation", type="primary")
chart_placeholder = st.empty() # Placeholder to update the chart dynamically

if start_button:
    t_data = []
    x_data = []
    
    dt = 0.05          # Time step
    total_time = 15.0  # Total simulation time in seconds

    for t in np.arange(0, total_time, dt):
        # Calculate displacement based on damping ratio
        if zeta < 1: # Underdamped
            A = x0
            B = (v0 + zeta * omega_n * x0) / omega_d
            x = math.exp(-zeta * omega_n * t) * (A * math.cos(omega_d * t) + B * math.sin(omega_d * t))
        elif zeta == 1: # Critically damped
            A = x0
            B = v0 + omega_n * x0
            x = (A + B * t) * math.exp(-omega_n * t)
        else: # Overdamped
            s1 = -omega_n * (zeta - math.sqrt(zeta**2 - 1))
            s2 = -omega_n * (zeta + math.sqrt(zeta**2 - 1))
            A = (v0 - s2 * x0) / (s1 - s2)
            B = (s1 * x0 - v0) / (s1 - s2)
            x = A * math.exp(s1 * t) + B * math.exp(s2 * t)
            
        t_data.append(t)
        x_data.append(x)
        
        # Create dataframe for Streamlit line chart
        df = pd.DataFrame({"Displacement [m]": x_data}, index=t_data)
        
        # Inject the updated chart into the placeholder
        chart_placeholder.line_chart(df, y="Displacement [m]")
        
        # Pause execution to create the real-time drawing effect
        time.sleep(dt)
