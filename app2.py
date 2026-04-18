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

st.title("🏗️ Structural Dynamics: Single Fixed Column")
st.markdown("Adjust parameters below. The model represents a **single continuous column** fixed at the base with a circular mass at the free end (acting like an inverted pendulum).")

# ---------------------------------------------------------
# 2. SIDEBAR PARAMETERS & DATA IMPORT
# ---------------------------------------------------------
st.sidebar.header("1. Data Source")
data_mode = st.sidebar.radio(
    "Choose how to generate oscillation data:",
    ("Simulate Physics", "Upload Custom CSV")
)

st.sidebar.markdown("---")

if data_mode == "Simulate Physics":
    st.sidebar.header("System Parameters")
    m = st.sidebar.slider("Mass (m) [kg]", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    k = st.sidebar.slider("Stiffness (k) [N/m]", min_value=10.0, max_value=1000.0, value=200.0, step=10.0)
    c = st.sidebar.slider("Damping Coefficient (c) [Ns/m]", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
    x0 = st.sidebar.slider("Initial Displacement (x0) [m]", min_value=-10.0, max_value=10.0, value=5.0, step=0.5)

    st.sidebar.header("Simulation Settings")
    total_time = st.sidebar.slider("Total Time [s]", min_value=5.0, max_value=50.0, value=10.0, step=1.0)
    
    v0 = 0.0  
    num_points = 150  

    # --- PHYSICS CALCULATIONS ---
    omega_n = math.sqrt(k / m)                  
    c_critical = 2 * math.sqrt(k * m)           
    zeta = c / c_critical                       

    col1, col2, col3 = st.columns(3)
    col1.metric("Natural Frequency (ω_n)", f"{omega_n:.2f} rad/s")
    col2.metric("Damping Ratio (ζ)", f"{zeta:.2f}")

    if zeta < 1:
        col3.metric("System State", "Underdamped")
    elif zeta == 1:
        col3.metric("System State", "Critically Damped")
    else:
        col3.metric("System State", "Overdamped")

    time_array = np.linspace(0, total_time, num_points)
    x_data = []

    for t in time_array:
        if zeta < 1: 
            omega_d = omega_n * math.sqrt(1 - zeta**2)
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
        
    st.sidebar.markdown("---")
    st.sidebar.header("Export Data")
    results_df = pd.DataFrame({
        "Time [s]": np.round(time_array, 3),
        "Displacement [m]": np.round(x_data, 4)
    })
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Oscillation Data (CSV)",
        data=csv_data,
        file_name='single_column_data.csv',
        mime='text/csv'
    )

elif data_mode == "Upload Custom CSV":
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            time_col = next((col for col in df.columns if "time" in col.lower()), None)
            disp_col = next((col for col in df.columns if "disp" in col.lower()), None)

            if time_col is None or disp_col is None:
                st.error("CSV must contain columns containing the words 'Time' and 'Displacement'.")
                st.stop()
                
            time_array = df[time_col].values
            x_data = df[disp_col].values
            total_time = np.max(time_array)
            
            if len(time_array) > 200:
                step = len(time_array) // 150
                time_array = time_array[::step]
                x_data = x_data[::step]
                st.sidebar.warning(f"Data was large. Downsampled to {len(time_array)} frames for smooth animation.")
                
            st.sidebar.success("Custom CSV loaded successfully!")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("👈 Please upload a CSV file in the sidebar to view the animation.")
        st.stop()

# ---------------------------------------------------------
# 5. REAL-TIME CALCULATION & COLUMN GEOMETRY
# ---------------------------------------------------------
max_disp = np.max(np.abs(x_data))
if max_disp == 0: max_disp = 1.0 

# Structural Proportions
L = max_disp * 1.5        
pivot_x, pivot_y = 0.0, 0.0  # Fixed at the very bottom
y0 = math.sqrt(max(0.01, L**2 - x_data[0]**2)) # Mass height

dt_seconds = total_time / max(1, len(time_array) - 1)
frame_duration_ms = int(dt_seconds * 1000)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08,
                    subplot_titles=("Time-History Graph", "Single Fixed Column (Circular Free End)"))

# --- BASE TRACES (Frame 0) ---
# Graph Line and Dot
fig.add_trace(go.Scatter(x=[x_data[0]], y=[time_array[0]], mode='lines', line=dict(color='blue', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[time_array[0]], mode='markers', marker=dict(color='red', size=12)), row=1, col=1)

ground_width = max_disp * 1.5
# Ground (Thick black line)
fig.add_trace(go.Scatter(x=[-ground_width, ground_width], y=[0, 0], mode='lines', line=dict(color='black', width=8)), row=2, col=1)
# Single Column (Rod pointing up from ground)
fig.add_trace(go.Scatter(x=[pivot_x, x_data[0]], y=[pivot_y, y0], mode='lines', line=dict(color='gray', width=6)), row=2, col=1)
# Oscillating Mass (Circular bob at the top)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[y0], mode='markers', marker=dict(color='firebrick', size=40, symbol='circle')), row=2, col=1)

# --- BUILD ANIMATION FRAMES ---
frames = []
for i in range(len(time_array)):
    xi = x_data[i]
    ti = time_array[i]
    # Calculate upward position of swinging mass
    yi = math.sqrt(max(0.01, L**2 - xi**2)) 
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=x_data[:i+1], y=time_array[:i+1]), # Graph updates
            go.Scatter(x=[xi], y=[ti]),                     # Graph dot updates
            go.Scatter(x=[-ground_width, ground_width], y=[0, 0]), # Ground (Static)
            go.Scatter(x=[pivot_x, xi], y=[pivot_y, yi]),   # Column updates
            go.Scatter(x=[xi], y=[yi])                      # Circular mass updates
        ],
        traces=[0, 1, 2, 3, 4] 
    ))

fig.frames = frames

# ---------------------------------------------------------
# 6. LAYOUT & RENDERING
# ---------------------------------------------------------
fig.update_layout(
    height=800, 
    showlegend=False,
    autosize=True,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05, y=1.05,
        buttons=[
            dict(label="▶ Play in Real-Time", method="animate", args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=False), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
        ]
    )]
)

disp_padding = max_disp * 1.5 

# Graph Axes
fig.update_xaxes(range=[-disp_padding, disp_padding], row=1, col=1)
fig.update_yaxes(range=[0, total_time], title="Time [seconds]", row=1, col=1)

# Structure Axes (1:1 Aspect Ratio locked)
fig.update_xaxes(range=[-disp_padding, disp_padding], title="Horizontal Displacement [m]", row=2, col=1)
fig.update_yaxes(range=[-L*0.1, L + (L*0.2)], title="Elevation", row=2, col=1, showticklabels=False, scaleanchor="x", scaleratio=1)

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
