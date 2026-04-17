import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Pendulum Damping Animation", layout="wide")

st.title("🏗️ Structural Dynamics: Pendulum & Graph Animation")
st.markdown("Adjust the parameters in the sidebar. The simulation builds the frames dynamically. Click **▶ Play Animation** on the chart to watch the pendulum and the graph sync perfectly.")

# --- SIDEBAR: System Parameters ---
st.sidebar.header("System Parameters")
m = st.sidebar.slider("Mass (m) [kg]", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
k = st.sidebar.slider("Stiffness (k) [N/m]", min_value=10.0, max_value=1000.0, value=200.0, step=10.0)
c = st.sidebar.slider("Damping Coefficient (c) [Ns/m]", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
x0 = st.sidebar.slider("Initial Displacement (x0) [m]", min_value=-10.0, max_value=10.0, value=8.0, step=0.5)
v0 = 0.0  # Initial velocity

# --- PHYSICS CALCULATIONS ---
omega_n = math.sqrt(k / m)                  
c_critical = 2 * math.sqrt(k * m)           
zeta = c / c_critical                       

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

# --- GENERATE DATA ARRAY ---
# Instead of a loop with time.sleep, we pre-calculate the math 
# and let the browser animate it smoothly.
dt = 0.05
total_time = 15.0
t_data = np.arange(0, total_time, dt)
x_data = []

for t in t_data:
    if zeta < 1: 
        A = x0
        B = (v0 + zeta * omega_n * x0) / omega_d
        x = math.exp(-zeta * omega_n * t) * (A * math.cos(omega_d * t) + B * math.sin(omega_d * t))
    elif zeta == 1: 
        A = x0
        B = v0 + omega_n * x0
        x = (A + B * t) * math.exp(-omega_n * t)
    else: 
        s1 = -omega_n * (zeta - math.sqrt(zeta**2 - 1))
        s2 = -omega_n * (zeta + math.sqrt(zeta**2 - 1))
        A = (v0 - s2 * x0) / (s1 - s2)
        B = (s1 * x0 - v0) / (s1 - s2)
        x = A * math.exp(s1 * t) + B * math.exp(s2 * t)
    x_data.append(x)

# --- ANIMATION SETUP ---
# Visual variables for the pendulum
L = 12.0 # Pendulum visual string length
pivot_x, pivot_y = 0.0, 5.0
y0 = pivot_y - math.sqrt(L**2 - x_data[0]**2)

# Create side-by-side subplots
fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], 
                    subplot_titles=("Physical Pendulum", "Oscillation Graph"))

# 1. Base Traces (Starting Position)
fig.add_trace(go.Scatter(x=[pivot_x, x_data[0]], y=[pivot_y, y0], mode='lines', line=dict(width=4, color='gray')), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[y0], mode='markers', marker=dict(size=35, color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[t_data[0]], y=[x_data[0]], mode='lines', line=dict(width=3, color='blue')), row=1, col=2)

# 2. Build Animation Frames
frames = []
for i in range(0, len(t_data), 2):  # Skipping every other frame makes it render slightly faster
    xi = x_data[i]
    # Keep the pendulum from breaking math boundaries if initial displacement is wild
    yi = pivot_y - math.sqrt(max(0.1, L**2 - xi**2)) 
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=[pivot_x, xi], y=[pivot_y, yi]), # Update String
            go.Scatter(x=[xi], y=[yi]),                   # Update Mass
            go.Scatter(x=t_data[:i+1], y=x_data[:i+1])    # Update Graph
        ],
        traces=[0, 1, 2] # Map the updates to the 3 base traces we made above
    ))

fig.frames = frames

# 3. Layout Formatting and Play Buttons
fig.update_layout(
    height=550,
    showlegend=False,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.1, y=1.15,
        buttons=[
            dict(label="▶ Play Animation",
                 method="animate",
                 args=[None, dict(frame=dict(duration=50, redraw=False), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
        ]
    )]
)

# Lock axes so the graph doesn't resize constantly during playback
fig.update_xaxes(range=[-15, 15], title="X Position", row=1, col=1)
fig.update_yaxes(range=[-10, 6], title="Y Position", row=1, col=1)
fig.update_xaxes(range=[0, total_time], title="Time (s)", row=1, col=2)
fig.update_yaxes(range=[-abs(x0)-2, abs(x0)+2], title="Displacement [m]", row=1, col=2)

# Render
st.plotly_chart(fig, use_container_width=True)
