import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data-Driven Pendulum", layout="wide")

st.title("📊 Data-Driven Oscillation Animation")
st.markdown("Upload a CSV file containing `Time` and `Displacement` data, or use the generated sine-wave sample data. Click **▶ Play Animation** on the chart to watch.")

# --- SIDEBAR: Data Management ---
st.sidebar.header("1. Data Input")

# Generate Sample Data (Sine Wave)
sample_time = np.arange(0, 10, 0.05)
# Sine wave with amplitude of 5 and frequency of 1 rad/s
sample_disp = 5 * np.sin(2 * np.pi * 0.5 * sample_time) 
sample_df = pd.DataFrame({"Time": sample_time, "Displacement": sample_disp})

# Provide a download button for the sample CSV
csv_data = sample_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Sample Sine Wave CSV",
    data=csv_data,
    file_name='sine_wave_sample.csv',
    mime='text/csv',
)

st.sidebar.markdown("---")
st.sidebar.header("2. Upload Your CSV")
st.sidebar.markdown("CSV must have columns named **Time** and **Displacement**.")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# --- DATA PROCESSING ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if "Time" not in df.columns or "Displacement" not in df.columns:
            st.error("CSV must contain 'Time' and 'Displacement' columns.")
            st.stop()
        st.success("Custom CSV loaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    df = sample_df
    st.info("Currently using the default Sample Sine Wave data. Upload a CSV on the left to use your own data.")

# Extract arrays for animation
t_data = df["Time"].values
x_data = df["Displacement"].values

# Downsample if the CSV is huge to prevent browser crash during animation
if len(t_data) > 300:
    step = len(t_data) // 300
    t_data = t_data[::step]
    x_data = x_data[::step]
    st.warning(f"Data was large. Downsampled to {len(t_data)} frames for smooth web animation.")

# --- ANIMATION SETUP ---
max_disp = np.max(np.abs(x_data))
if max_disp == 0:
    max_disp = 1.0

# Visual variables for the pendulum
# Make string length slightly longer than max displacement so it looks realistic
L = max_disp * 1.5 
pivot_x, pivot_y = 0.0, L
y0 = pivot_y - math.sqrt(L**2 - x_data[0]**2)

# Create side-by-side subplots
# Pendulum X-axis (horizontal) represents displacement. 
# Graph Y-axis (vertical) represents displacement.
fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], 
                    subplot_titles=("Physical Pendulum (X = Displacement)", "Data Graph (Y = Displacement)"))

# 1. Base Traces (Starting Position)
# Trace 0: Pendulum String
fig.add_trace(go.Scatter(x=[pivot_x, x_data[0]], y=[pivot_y, y0], mode='lines', line=dict(width=4, color='gray')), row=1, col=1)
# Trace 1: Pendulum Bob (Mass)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[y0], mode='markers', marker=dict(size=35, color='red')), row=1, col=1)
# Trace 2: Oscillation Line
fig.add_trace(go.Scatter(x=[t_data[0]], y=[x_data[0]], mode='lines', line=dict(width=3, color='blue')), row=1, col=2)
# Trace 3: Sync Marker (The moving dot on the graph)
fig.add_trace(go.Scatter(x=[t_data[0]], y=[x_data[0]], mode='markers', marker=dict(size=12, color='red')), row=1, col=2)

# 2. Build Animation Frames
frames = []
for i in range(len(t_data)):
    xi = x_data[i]
    ti = t_data[i]
    # Calculate pendulum vertical position (ensure math domain is safe)
    yi = pivot_y - math.sqrt(max(0.01, L**2 - xi**2)) 
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=[pivot_x, xi], y=[pivot_y, yi]), # Update String
            go.Scatter(x=[xi], y=[yi]),                   # Update Mass
            go.Scatter(x=t_data[:i+1], y=x_data[:i+1]),   # Update Line Graph
            go.Scatter(x=[ti], y=[xi])                    # Update Graph Sync Dot
        ],
        traces=[0, 1, 2, 3] # Map the updates to the 4 base traces
    ))

fig.frames = frames

# 3. Layout Formatting and Play Buttons
fig.update_layout(
    height=600,
    showlegend=False,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.1, y=1.15,
        buttons=[
            dict(label="▶ Play Animation",
                 method="animate",
                 args=[None, dict(frame=dict(duration=40, redraw=False), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
        ]
    )]
)

# Set strictly matching axis limits so the visuals sync perfectly
disp_padding = max_disp * 1.2

# Pendulum Axes
fig.update_xaxes(range=[-disp_padding, disp_padding], title="Displacement [m]", row=1, col=1)
fig.update_yaxes(range=[0, L + (L*0.1)], title="Vertical Height [m]", row=1, col=1)

# Graph Axes
fig.update_xaxes(range=[t_data[0], t_data[-1]], title="Time [s]", row=1, col=2)
fig.update_yaxes(range=[-disp_padding, disp_padding], title="Displacement [m]", row=1, col=2)

# Render
st.plotly_chart(fig, use_container_width=True)
