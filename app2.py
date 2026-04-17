import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Structural Dynamics App", layout="wide")

st.title("🏗️ Structural Dynamics: SDOF Oscillation")
st.markdown("Adjust parameters below. The graph above tracks **Displacement vs. Time (Vertical)**, perfectly aligned with the **Pendulum's horizontal swing** below.")

# ---------------------------------------------------------
# 2. SIDEBAR PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("System Parameters")
m = st.sidebar.slider("Mass (m) [kg]", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
k = st.sidebar.slider("Stiffness (k) [N/m]", min_value=10.0, max_value=1000.0, value=200.0, step=10.0)
c = st.sidebar.slider("Damping Coefficient (c) [Ns/m]", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
x0 = st.sidebar.slider("Initial Displacement (x0) [m]", min_value=-10.0, max_value=10.0, value=5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")
# NEW FEATURE: User can edit the total simulation time
total_time = st.sidebar.slider("Total Time [s]", min_value=5.0, max_value=50.0, value=10.0, step=1.0)

v0 = 0.0  # Initial velocity
num_points = 100  # Keeping it at 100 points ensures smooth web animation regardless of time length

# ---------------------------------------------------------
# 3. PHYSICS CALCULATIONS
# ---------------------------------------------------------
omega_n = math.sqrt(k / m)                  # Natural frequency (rad/s)
c_critical = 2 * math.sqrt(k * m)           # Critical damping coefficient
zeta = c / c_critical                       # Damping ratio

# Display system state on the dashboard
col1, col2, col3 = st.columns(3)
col1.metric("Natural Frequency (ω_n)", f"{omega_n:.2f} rad/s")
col2.metric("Damping Ratio (ζ)", f"{zeta:.2f}")

if zeta < 1:
    col3.metric("System State", "Underdamped")
elif zeta == 1:
    col3.metric("System State", "Critically Damped")
else:
    col3.metric("System State", "Overdamped")

# Create exactly 100 time steps across the user-defined total_time
time_array = np.linspace(0, total_time, num_points)
x_data = []

# Generate Analytical Data
for t in time_array:
    if zeta < 1: # Underdamped
        omega_d = omega_n * math.sqrt(1 - zeta**2)
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
        
    x_data.append(x)

# ---------------------------------------------------------
# 4. ANIMATION SETUP & GEOMETRY
# ---------------------------------------------------------
max_disp = np.max(np.abs(x_data))
if max_disp == 0: max_disp = 1.0 

# Pendulum visual constraints
L = max_disp * 1.5 
pivot_x, pivot_y = 0.0, L 
y0 = pivot_y - math.sqrt(L**2 - x_data[0]**2)

# Subplots: 2 Rows, 1 Column. shared_xaxes=True ensures perfectly aligned displacement!
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08,
                    subplot_titles=("Time-History Graph (Time on Vertical Axis)", "Oscillating Pendulum"))

# --- BASE TRACES (Frame 0) ---
# Trace 0: Graph line (Growing upward over time)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[time_array[0]], mode='lines', line=dict(color='blue', width=3)), row=1, col=1)
# Trace 1: Graph moving dot (Current position)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[time_array[0]], mode='markers', marker=dict(color='red', size=12)), row=1, col=1)

# Trace 2: Pendulum String
fig.add_trace(go.Scatter(x=[0, x_data[0]], y=[pivot_y, y0], mode='lines', line=dict(color='gray', width=4)), row=2, col=1)
# Trace 3: Pendulum Bob (Mass)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[y0], mode='markers', marker=dict(color='red', size=35)), row=2, col=1)

# --- BUILD ANIMATION FRAMES ---
frames = []
for i in range(len(time_array)):
    xi = x_data[i]
    ti = time_array[i]
    yi = pivot_y - math.sqrt(max(0.01, L**2 - xi**2)) 
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=x_data[:i+1], y=time_array[:i+1]), # Graph line extends up
            go.Scatter(x=[xi], y=[ti]),                     # Graph dot moves
            go.Scatter(x=[0, xi], y=[pivot_y, yi]),         # String updates
            go.Scatter(x=[xi], y=[yi])                      # Mass moves
        ],
        traces=[0, 1, 2, 3] 
    ))

fig.frames = frames

# ---------------------------------------------------------
# 5. LAYOUT & RENDERING
# ---------------------------------------------------------
fig.update_layout(
    height=800, # Taller layout to fit both stacked charts
    showlegend=False,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05, y=1.05,
        buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=False), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
        ]
    )]
)

disp_padding = max_disp * 1.2 

# Graph Axes (Row 1) - X is Displacement, Y is dynamically set to total_time
fig.update_xaxes(range=[-disp_padding, disp_padding], row=1, col=1)
fig.update_yaxes(range=[0, total_time], title="Time [seconds]", row=1, col=1)

# Pendulum Axes (Row 2) - X is Displacement, Y is Height
fig.update_xaxes(range=[-disp_padding, disp_padding], title="Horizontal Displacement [m]", row=2, col=1)
fig.update_yaxes(range=[0, L + (L*0.1)], title="Vertical Space", row=2, col=1, showticklabels=False)

st.plotly_chart(fig, use_container_width=True)
