import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------------
# Configure the Streamlit page to be wide so the side-by-side graphs fit nicely.
st.set_page_config(page_title="Data-Driven Pendulum", layout="wide")

st.title("📊 Data-Driven Oscillation Animation")
st.markdown("Upload a CSV file containing `Time` and `Displacement` data, or use the generated sine-wave sample data. Click **▶ Play Animation** on the chart to watch.")

# ---------------------------------------------------------
# 2. SIDEBAR: DATA INPUT & MANAGEMENT
# ---------------------------------------------------------
st.sidebar.header("1. Data Input")

# --- Generate Default Sample Data ---
# We create a sample sine wave so the app works immediately even without an upload.
sample_time = np.arange(0, 10, 0.05) # Time array from 0 to 10 seconds, stepping by 0.05
# Formula: Amplitude * sin(2 * pi * frequency * time)
sample_disp = 5 * np.sin(2 * np.pi * 0.5 * sample_time) 
# Combine arrays into a Pandas DataFrame for easy handling
sample_df = pd.DataFrame({"Time": sample_time, "Displacement": sample_disp})

# Provide a button to download the sample data as a CSV
csv_data = sample_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Sample Sine Wave CSV",
    data=csv_data,
    file_name='sine_wave_sample.csv',
    mime='text/csv',
)

st.sidebar.markdown("---")

# --- Handle User Uploads ---
st.sidebar.header("2. Upload Your CSV")
st.sidebar.markdown("CSV must have columns named **Time** and **Displacement**.")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# ---------------------------------------------------------
# 3. DATA PROCESSING
# ---------------------------------------------------------
# Decide whether to use the uploaded file or the default sample data
if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        
        # Check if the required columns exist
        if "Time" not in df.columns or "Displacement" not in df.columns:
            st.error("CSV must contain 'Time' and 'Displacement' columns.")
            st.stop() # Stop the app from running further if the data is bad
        st.success("Custom CSV loaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    # Fallback to the sample data
    df = sample_df
    st.info("Currently using the default Sample Sine Wave data. Upload a CSV on the left to use your own data.")

# Extract the columns into raw numpy arrays for faster math processing
t_data = df["Time"].values
x_data = df["Displacement"].values

# --- Data Optimization ---
# Web browsers will crash if an animation has too many frames (e.g., thousands).
# If the CSV has more than 300 data points, we downsample it by skipping rows.
if len(t_data) > 300:
    step = len(t_data) // 300
    t_data = t_data[::step]
    x_data = x_data[::step]
    st.warning(f"Data was large. Downsampled to {len(t_data)} frames for smooth web animation.")

# ---------------------------------------------------------
# 4. ANIMATION SETUP & GEOMETRY
# ---------------------------------------------------------
# Find the maximum displacement to scale our visualization correctly
max_disp = np.max(np.abs(x_data))
if max_disp == 0:
    max_disp = 1.0 # Prevent division-by-zero or zero-length visuals if data is flat

# --- Pendulum Geometry ---
# To make it look realistic, the string length (L) should be longer than the maximum swing.
L = max_disp * 1.5 
pivot_x, pivot_y = 0.0, L # The top attachment point of the pendulum

# Calculate the initial vertical position (y0) of the mass using Pythagorean theorem:
# L^2 = x^2 + y_drop^2  ->  y_drop = sqrt(L^2 - x^2)
y0 = pivot_y - math.sqrt(L**2 - x_data[0]**2)

# --- Subplot Initialization ---
# Create side-by-side charts. Left (40% width) for pendulum, Right (60% width) for graph.
fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], 
                    subplot_titles=("Physical Pendulum (X = Displacement)", "Data Graph (Y = Displacement)"))

# ---------------------------------------------------------
# 5. BASE TRACES (Frame 0)
# ---------------------------------------------------------
# Plotly animations require "base traces" representing the very first frame of the animation.
# Later, we will update these specific traces.

# Trace 0: The Pendulum String (a line from the pivot to the mass)
fig.add_trace(go.Scatter(x=[pivot_x, x_data[0]], y=[pivot_y, y0], mode='lines', line=dict(width=4, color='gray')), row=1, col=1)

# Trace 1: The Pendulum Mass (a large red dot)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[y0], mode='markers', marker=dict(size=35, color='red')), row=1, col=1)

# Trace 2: The static line graph showing the path taken so far
fig.add_trace(go.Scatter(x=[t_data[0]], y=[x_data[0]], mode='lines', line=dict(width=3, color='blue')), row=1, col=2)

# Trace 3: A red dot on the graph that moves in sync with the pendulum mass
fig.add_trace(go.Scatter(x=[t_data[0]], y=[x_data[0]], mode='markers', marker=dict(size=12, color='red')), row=1, col=2)

# ---------------------------------------------------------
# 6. BUILD ANIMATION FRAMES
# ---------------------------------------------------------
# Instead of pausing the server (time.sleep), we pre-calculate every single frame 
# and package them together. The user's browser handles the playback smoothly.
frames = []

for i in range(len(t_data)):
    xi = x_data[i] # Current displacement
    ti = t_data[i] # Current time
    
    # Calculate the vertical position of the mass for this specific frame.
    # We use max(0.01, ...) to prevent math domain errors if displacement somehow exceeds string length.
    yi = pivot_y - math.sqrt(max(0.01, L**2 - xi**2)) 
    
    # Create the frame by updating the data of the 4 base traces we made earlier
    frames.append(go.Frame(
        data=[
            go.Scatter(x=[pivot_x, xi], y=[pivot_y, yi]), # Updates Trace 0 (String)
            go.Scatter(x=[xi], y=[yi]),                   # Updates Trace 1 (Mass)
            go.Scatter(x=t_data[:i+1], y=x_data[:i+1]),   # Updates Trace 2 (Line growing over time)
            go.Scatter(x=[ti], y=[xi])                    # Updates Trace 3 (Sync dot jumping to current position)
        ],
        traces=[0, 1, 2, 3] # This array tells Plotly which base traces to map the above data to
    ))

# Attach the compiled frames to the figure
fig.frames = frames

# ---------------------------------------------------------
# 7. LAYOUT & PLAY BUTTON CONFIGURATION
# ---------------------------------------------------------
fig.update_layout(
    height=600,
    showlegend=False,
    # Add Play and Pause buttons to control the animation
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.1, y=1.15,
        buttons=[
            dict(label="▶ Play Animation",
                 method="animate",
                 # duration=40 means 40 milliseconds per frame (roughly 25 FPS)
                 args=[None, dict(frame=dict(duration=40, redraw=False), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
        ]
    )]
)

# --- Synchronize the Axes ---
# This is crucial so that the visual horizontal movement of the pendulum 
# visually matches the vertical movement of the graph perfectly.
disp_padding = max_disp * 1.2 # Add 20% padding so data doesn't touch the edges

# Fix Pendulum limits (Left Chart)
fig.update_xaxes(range=[-disp_padding, disp_padding], title="Displacement [m]", row=1, col=1)
fig.update_yaxes(range=[0, L + (L*0.1)], title="Vertical Height [m]", row=1, col=1)

# Fix Graph limits (Right Chart)
fig.update_xaxes(range=[t_data[0], t_data[-1]], title="Time [s]", row=1, col=2)
fig.update_yaxes(range=[-disp_padding, disp_padding], title="Displacement [m]", row=1, col=2)

# ---------------------------------------------------------
# 8. RENDER APP
# ---------------------------------------------------------
# Finally, push the assembled Plotly figure to the Streamlit app
st.plotly_chart(fig, use_container_width=True)
