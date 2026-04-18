import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. PAGE SETUP & PEDAGOGY HEADER
# ---------------------------------------------------------
st.set_page_config(page_title="Structural Dynamics: Glass Box", layout="wide")

st.title("🎓 Structural Dynamics: The Glass Box")
st.markdown("Explore the **how and why** of single degree of freedom (SDOF) oscillation. Adjust the physical parameters to see how they alter the underlying mathematics and the resulting physical motion.")

# ---------------------------------------------------------
# 2. SIDEBAR: DATA & PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("1. Data Source")
data_mode = st.sidebar.radio("Mode:", ("Simulate Physics", "Upload Custom CSV"))
st.sidebar.markdown("---")

if data_mode == "Simulate Physics":
    st.sidebar.header("Physical Parameters")
    # Educational tooltips added to explain the physics
    m = st.sidebar.slider("Mass (m) [kg]", 1.0, 50.0, 10.0, 1.0, help="Inertia. Higher mass makes the system sluggish and lowers the natural frequency.")
    k = st.sidebar.slider("Stiffness (k) [N/m]", 10.0, 1000.0, 200.0, 10.0, help="Restoring force. Higher stiffness pulls the mass back faster, increasing frequency.")
    c = st.sidebar.slider("Damping (c) [Ns/m]", 0.0, 200.0, 15.0, 1.0, help="Energy dissipation. Removes energy from the system over time (like a shock absorber).")
    x0 = st.sidebar.slider("Initial Displacement [m]", -10.0, 10.0, 5.0, 0.5)

    total_time = st.sidebar.slider("Total Time [s]", 5.0, 50.0, 10.0, 1.0)
    v0 = 0.0  
    num_points = 150  

    # --- PHYSICS CALCULATIONS ---
    omega_n = math.sqrt(k / m)                  
    c_critical = 2 * math.sqrt(k * m)           
    zeta = c / c_critical                       

    # --- PEDAGOGY: THE MATH BEHIND THE MOTION ---
    with st.expander("🔍 THE GLASS BOX: See the Math", expanded=True):
        col_math1, col_math2 = st.columns(2)
        
        with col_math1:
            st.markdown("**1. The Equation of Motion (EOM)**")
            st.markdown("Newton's Second Law applied to our system gives us the governing differential equation:")
            st.latex(r"m\ddot{x} + c\dot{x} + kx = 0")
            st.markdown("With your current parameters, the system must solve:")
            # Live substitution
            st.latex(f"{m}\\ddot{{x}} + {c}\\dot{{x}} + {k}x = 0")
            
        with col_math2:
            st.markdown("**2. System State & Roots**")
            if zeta < 1:
                omega_d = omega_n * math.sqrt(1 - zeta**2)
                st.success(f"**Underdamped (ζ = {zeta:.2f} < 1)**")
                st.markdown("The roots are **Complex Conjugates**, meaning the system will oscillate while decaying.")
                st.latex(f"s_{{1,2}} = {-zeta*omega_n:.2f} \\pm {omega_d:.2f}i")
            elif zeta == 1:
                st.warning(f"**Critically Damped (ζ = {zeta:.2f} = 1)**")
                st.markdown("The roots are **Real and Equal**. The system returns to equilibrium as fast as possible without oscillating.")
                st.latex(f"s_{{1,2}} = {-omega_n:.2f}")
            else:
                st.error(f"**Overdamped (ζ = {zeta:.2f} > 1)**")
                st.markdown("The roots are **Real and Distinct**. The damping is so high the system slowly creeps back to zero without crossing it.")
                root1 = -omega_n * (zeta - math.sqrt(zeta**2 - 1))
                root2 = -omega_n * (zeta + math.sqrt(zeta**2 - 1))
                st.latex(f"s_1 = {root1:.2f}, \\quad s_2 = {root2:.2f}")

    # Generate Analytical Data
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
        
    # Calculate velocity dynamically for the Phase Portrait
    v_data = np.gradient(x_data, time_array)

elif data_mode == "Upload Custom CSV":
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            time_col = next((col for col in df.columns if "time" in col.lower()), None)
            disp_col = next((col for col in df.columns if "disp" in col.lower()), None)

            if not time_col or not disp_col:
                st.error("CSV must contain 'Time' and 'Displacement' columns.")
                st.stop()
                
            time_array = df[time_col].values
            x_data = df[disp_col].values
            total_time = np.max(time_array)
            
            # Calculate velocity from imported data
            v_data = np.gradient(x_data, time_array)
            
            if len(time_array) > 200:
                step = len(time_array) // 200
                time_array = time_array[::step]
                x_data = x_data[::step]
                v_data = v_data[::step]
                
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("👈 Please upload a CSV file.")
        st.stop()

# ---------------------------------------------------------
# 3. ANIMATION SETUP (Including Phase Portrait)
# ---------------------------------------------------------
max_disp = np.max(np.abs(x_data))
if max_disp == 0: max_disp = 1.0 
max_vel = np.max(np.abs(v_data))
if max_vel == 0: max_vel = 1.0

# Pendulum visual constraints
L = max_disp * 1.5 
pivot_x, pivot_y = 0.0, L 
y0 = pivot_y - math.sqrt(L**2 - x_data[0]**2)

# Subplots: Row 1 (Phase Portrait | Time Graph), Row 2 (Pendulum span across bottom)
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy", "colspan": 2}, None]],
    row_heights=[0.5, 0.5],
    subplot_titles=(
        "Phase Portrait (Velocity vs Displacement)", 
        "Time History (Displacement vs Time)", 
        "Physical Oscillating Pendulum"
    )
)

# --- BASE TRACES ---
# 1. Phase Portrait (Row 1, Col 1)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[v_data[0]], mode='lines', line=dict(color='purple', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[v_data[0]], mode='markers', marker=dict(color='orange', size=10)), row=1, col=1)

# 2. Time History (Row 1, Col 2)
fig.add_trace(go.Scatter(x=[time_array[0]], y=[x_data[0]], mode='lines', line=dict(color='blue', width=3)), row=1, col=2)
fig.add_trace(go.Scatter(x=[time_array[0]], y=[x_data[0]], mode='markers', marker=dict(color='red', size=10)), row=1, col=2)

# 3. Pendulum (Row 2, Col 1 spanning to Col 2)
fig.add_trace(go.Scatter(x=[0, x_data[0]], y=[pivot_y, y0], mode='lines', line=dict(color='gray', width=4)), row=2, col=1)
fig.add_trace(go.Scatter(x=[x_data[0]], y=[y0], mode='markers', marker=dict(color='red', size=35)), row=2, col=1)

# --- BUILD ANIMATION FRAMES ---
frames = []
for i in range(len(time_array)):
    xi = x_data[i]
    vi = v_data[i]
    ti = time_array[i]
    yi = pivot_y - math.sqrt(max(0.01, L**2 - xi**2)) 
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=x_data[:i+1], y=v_data[:i+1]), # Phase line
            go.Scatter(x=[xi], y=[vi]),                 # Phase dot
            go.Scatter(x=time_array[:i+1], y=x_data[:i+1]), # Time line
            go.Scatter(x=[ti], y=[xi]),                 # Time dot
            go.Scatter(x=[0, xi], y=[pivot_y, yi]),     # Pendulum string
            go.Scatter(x=[xi], y=[yi])                  # Pendulum mass
        ],
        traces=[0, 1, 2, 3, 4, 5] 
    ))

fig.frames = frames

# ---------------------------------------------------------
# 4. LAYOUT & RENDERING
# ---------------------------------------------------------
fig.update_layout(
    height=800, 
    showlegend=False,
    updatemenus=[dict(
        type="buttons", showactive=False, x=0.05, y=1.08,
        buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=40, redraw=False), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
        ]
    )]
)

disp_padding = max_disp * 1.2 
vel_padding = max_vel * 1.2

# Phase Axes
fig.update_xaxes(range=[-disp_padding, disp_padding], title="Displacement [m]", row=1, col=1)
fig.update_yaxes(range=[-vel_padding, vel_padding], title="Velocity [m/s]", row=1, col=1)

# Time Axes
fig.update_xaxes(range=[0, total_time], title="Time [s]", row=1, col=2)
fig.update_yaxes(range=[-disp_padding, disp_padding], title="Displacement [m]", row=1, col=2)

# Pendulum Axes
fig.update_xaxes(range=[-disp_padding, disp_padding], title="Horizontal Space [m]", row=2, col=1)
fig.update_yaxes(range=[0, L + (L*0.1)], title="Vertical Space", row=2, col=1, showticklabels=False)

st.plotly_chart(fig, use_container_width=True)
