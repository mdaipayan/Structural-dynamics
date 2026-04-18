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
st.markdown(
    "Adjust parameters below. The model represents a **cantilever column** "
    "fixed at the base (zero displacement + zero slope) with a circular mass at the free end."
)

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
    m   = st.sidebar.slider("Mass (m) [kg]",                   1.0,   50.0,  10.0, 1.0)
    k   = st.sidebar.slider("Stiffness (k) [N/m]",            10.0, 1000.0, 200.0, 10.0)
    c   = st.sidebar.slider("Damping Coefficient (c) [Ns/m]",  0.0,  200.0,  15.0, 1.0)
    x0  = st.sidebar.slider("Initial Displacement (x0) [m]", -10.0,   10.0,   5.0, 0.5)

    st.sidebar.header("Simulation Settings")
    total_time = st.sidebar.slider("Total Time [s]", 5.0, 50.0, 10.0, 1.0)

    v0         = 0.0
    num_points = 150

    # --- Physics ---
    omega_n    = math.sqrt(k / m)
    c_critical = 2 * math.sqrt(k * m)
    zeta       = c / c_critical

    col1, col2, col3 = st.columns(3)
    col1.metric("Natural Frequency (ω_n)", f"{omega_n:.2f} rad/s")
    col2.metric("Damping Ratio (ζ)",        f"{zeta:.4f}")
    if   zeta < 1: col3.metric("System State", "Underdamped")
    elif zeta == 1: col3.metric("System State", "Critically Damped")
    else:           col3.metric("System State", "Overdamped")

    time_array = np.linspace(0, total_time, num_points)
    x_data     = []

    for t in time_array:
        if zeta < 1:
            omega_d = omega_n * math.sqrt(1 - zeta**2)
            A = x0
            B = (v0 + zeta * omega_n * x0) / omega_d
            x = math.exp(-zeta * omega_n * t) * (A * math.cos(omega_d * t) + B * math.sin(omega_d * t))
        elif zeta == 1:
            A = x0;  B = v0 + omega_n * x0
            x = (A + B * t) * math.exp(-omega_n * t)
        else:
            s1 = -omega_n * (zeta - math.sqrt(zeta**2 - 1))
            s2 = -omega_n * (zeta + math.sqrt(zeta**2 - 1))
            A  = (v0 - s2 * x0) / (s1 - s2)
            B  = (s1 * x0 - v0) / (s1 - s2)
            x  = A * math.exp(s1 * t) + B * math.exp(s2 * t)
        x_data.append(x)

    # Export
    st.sidebar.markdown("---")
    st.sidebar.header("Export Data")
    results_df = pd.DataFrame({
        "Time [s]":         np.round(time_array, 3),
        "Displacement [m]": np.round(x_data,     4)
    })
    st.sidebar.download_button(
        "📥 Download Oscillation Data (CSV)",
        results_df.to_csv(index=False).encode("utf-8"),
        "single_column_data.csv", "text/csv"
    )

elif data_mode == "Upload Custom CSV":
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df       = pd.read_csv(uploaded_file)
            time_col = next((c for c in df.columns if "time" in c.lower()), None)
            disp_col = next((c for c in df.columns if "disp" in c.lower()), None)

            if time_col is None or disp_col is None:
                st.error("CSV must contain columns with 'Time' and 'Displacement'.")
                st.stop()

            time_array = df[time_col].values
            x_data     = df[disp_col].values
            total_time = float(np.max(time_array))

            if len(time_array) > 200:
                step       = len(time_array) // 150
                time_array = time_array[::step]
                x_data     = x_data[::step]
                st.sidebar.warning(f"Downsampled to {len(time_array)} frames for smooth animation.")

            st.sidebar.success("Custom CSV loaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("👈 Please upload a CSV file in the sidebar to view the animation.")
        st.stop()

x_data = np.array(x_data, dtype=float)

# ---------------------------------------------------------
# 3. CANTILEVER SHAPE FUNCTION
# ---------------------------------------------------------
# For a cantilever beam fixed at base with tip displacement δ:
#   u(s) = δ · (3(s/L)² − (s/L)³) / 2
#
# This satisfies:
#   u(0)   = 0  → zero displacement at fixed base  ✓
#   u'(0)  = 0  → zero slope (no rotation) at base ✓
#   u(L)   = δ  → full tip displacement             ✓
# ---------------------------------------------------------
def cantilever_shape(tip_disp: float, L: float, n_pts: int = 50):
    """Return (x_deflected, y_heights) arrays for a deformed cantilever column."""
    s     = np.linspace(0.0, L, n_pts)
    ratio = s / L
    u     = tip_disp * (3.0 * ratio**2 - ratio**3) / 2.0
    return u, s   # horizontal offset, height along column

# ---------------------------------------------------------
# 4. GEOMETRY CONSTANTS
# ---------------------------------------------------------
max_disp = float(np.max(np.abs(x_data)))
if max_disp == 0:
    max_disp = 1.0

L              = max_disp * 3.5          # Column height (slender cantilever look)
disp_pad       = max_disp * 2.2          # Plot x-padding
support_w      = max_disp * 0.9          # Width of base-plate symbol
support_h      = L * 0.07               # Height of base-plate rectangle
hatch_n        = 10                      # Number of hatching lines
hatch_len      = support_w * 0.18        # Length of each hatch stroke

dt_seconds       = total_time / max(1, len(time_array) - 1)
frame_duration_ms = max(20, int(dt_seconds * 1000))

# ---------------------------------------------------------
# 5. FIGURE SETUP
# ---------------------------------------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.38, 0.62],
    vertical_spacing=0.07,
    subplot_titles=(
        "Displacement Time-History",
        "Fixed Cantilever Column – Real-Time Animation"
    )
)

# ==========================================================
# A) STATIC TRACES  (added first; never updated in frames)
# ==========================================================

# --- Ground line ---
fig.add_trace(go.Scatter(
    x=[-disp_pad * 1.3,  disp_pad * 1.3],
    y=[-support_h,       -support_h],
    mode="lines",
    line=dict(color="#444444", width=2),
    showlegend=False
), row=2, col=1)

# --- Fixed-support plate (filled rectangle) ---
fig.add_trace(go.Scatter(
    x=[-support_w / 2, support_w / 2, support_w / 2, -support_w / 2, -support_w / 2],
    y=[0,              0,             -support_h,      -support_h,     0],
    fill="toself",
    fillcolor="#b0b8c1",
    line=dict(color="#2c3e50", width=2.5),
    showlegend=False
), row=2, col=1)

# --- Diagonal hatching lines (fixed-end symbol) ---
for i in range(hatch_n):
    hx = -support_w / 2 + i * support_w / (hatch_n - 1)
    fig.add_trace(go.Scatter(
        x=[hx, hx - hatch_len],
        y=[-support_h, -support_h - hatch_len],
        mode="lines",
        line=dict(color="#2c3e50", width=1.5),
        showlegend=False
    ), row=2, col=1)

# --- Thick base clamping line (emphasises zero-rotation condition) ---
fig.add_trace(go.Scatter(
    x=[-support_w / 2, support_w / 2],
    y=[0, 0],
    mode="lines",
    line=dict(color="#1a252f", width=5),
    showlegend=False
), row=2, col=1)

static_count = 1 + 1 + hatch_n + 1   # ground + plate + hatches + clamp line

# ==========================================================
# B) ANIMATED BASE TRACES  (indices static_count … +3)
# ==========================================================
col_x0, col_y0 = cantilever_shape(x_data[0], L)

# Trace [static_count + 0] — time-history line
fig.add_trace(go.Scatter(
    x=[x_data[0]], y=[time_array[0]],
    mode="lines",
    line=dict(color="royalblue", width=2.5),
    showlegend=False
), row=1, col=1)

# Trace [static_count + 1] — time-history moving dot
fig.add_trace(go.Scatter(
    x=[x_data[0]], y=[time_array[0]],
    mode="markers",
    marker=dict(color="crimson", size=11),
    showlegend=False
), row=1, col=1)

# Trace [static_count + 2] — deformed column (cubic cantilever curve)
fig.add_trace(go.Scatter(
    x=col_x0, y=col_y0,
    mode="lines",
    line=dict(color="#2980b9", width=6),
    showlegend=False
), row=2, col=1)

# Trace [static_count + 3] — circular mass at column tip
fig.add_trace(go.Scatter(
    x=[col_x0[-1]], y=[col_y0[-1]],
    mode="markers",
    marker=dict(
        color="firebrick", size=40,
        symbol="circle",
        line=dict(color="#7b0000", width=2.5)
    ),
    showlegend=False
), row=2, col=1)

anim_idx = list(range(static_count, static_count + 4))

# ==========================================================
# C) BUILD ANIMATION FRAMES
# ==========================================================
frames = []
for i in range(len(time_array)):
    xi      = float(x_data[i])
    ti      = float(time_array[i])
    cx, cy  = cantilever_shape(xi, L)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=x_data[: i + 1], y=time_array[: i + 1]),   # history line
            go.Scatter(x=[xi],            y=[ti]),                    # moving dot
            go.Scatter(x=cx,              y=cy),                      # column curve
            go.Scatter(x=[cx[-1]],        y=[cy[-1]])                 # mass
        ],
        traces=anim_idx
    ))

fig.frames = frames

# ==========================================================
# D) LAYOUT
# ==========================================================
fig.update_layout(
    height=880,
    showlegend=False,
    plot_bgcolor="#f5f6fa",
    paper_bgcolor="#ffffff",
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05, y=1.06,
        buttons=[
            dict(
                label="▶  Play Real-Time",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=frame_duration_ms, redraw=False),
                    transition=dict(duration=0),
                    fromcurrent=True, mode="immediate"
                )]
            ),
            dict(
                label="⏸  Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                    transition=dict(duration=0)
                )]
            )
        ]
    )]
)

# Time-history axes
fig.update_xaxes(range=[-disp_pad, disp_pad], showgrid=True, gridcolor="#dde", row=1, col=1)
fig.update_yaxes(range=[0, total_time], title_text="Time [s]",
                 showgrid=True, gridcolor="#dde", row=1, col=1)

# Structure axes — locked 1:1 aspect ratio so column proportions are true
fig.update_xaxes(
    range=[-disp_pad, disp_pad],
    title_text="Horizontal Displacement [m]",
    showgrid=True, gridcolor="#dde", row=2, col=1
)
fig.update_yaxes(
    range=[-(support_h + L * 0.18), L * 1.25],
    title_text="Elevation [m]",
    showgrid=True, gridcolor="#dde",
    scaleanchor="x", scaleratio=1,
    row=2, col=1
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True, "displayModeBar": True}
)
