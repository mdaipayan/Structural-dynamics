import numpy as np
import pandas as pd
import math

# --- 1. Structural Dynamics Parameters ---
m = 10.0      # Mass (kg)
k = 200.0     # Stiffness (N/m)
c = 15.0      # Damping coefficient (Ns/m)
x0 = 5.0      # Initial displacement (m)
v0 = 0.0      # Initial velocity (m/s)

total_time = 10.0     # Total simulation time (s)
num_points = 100      # Exactly 100 data points

# --- 2. Physics Calculations ---
omega_n = math.sqrt(k / m)                  # Natural frequency (rad/s)
c_critical = 2 * math.sqrt(k * m)           # Critical damping coefficient
zeta = c / c_critical                       # Damping ratio

print(f"System State: Natural Frequency = {omega_n:.2f} rad/s, Damping Ratio = {zeta:.2f}")

# Create exactly 100 time steps
time_array = np.linspace(0, total_time, num_points)
displacement_array = []

# --- 3. Generate Analytical Data ---
for t in time_array:
    if zeta < 1: # Underdamped (Oscillates and decays)
        omega_d = omega_n * math.sqrt(1 - zeta**2)
        A = x0
        B = (v0 + zeta * omega_n * x0) / omega_d
        x = math.exp(-zeta * omega_n * t) * (A * math.cos(omega_d * t) + B * math.sin(omega_d * t))
        
    elif zeta == 1: # Critically damped (Returns to zero fastest, no oscillation)
        A = x0
        B = v0 + omega_n * x0
        x = (A + B * t) * math.exp(-omega_n * t)
        
    else: # Overdamped (Slowly crawls back to zero, no oscillation)
        s1 = -omega_n * (zeta - math.sqrt(zeta**2 - 1))
        s2 = -omega_n * (zeta + math.sqrt(zeta**2 - 1))
        A = (v0 - s2 * x0) / (s1 - s2)
        B = (s1 * x0 - v0) / (s1 - s2)
        x = A * math.exp(s1 * t) + B * math.exp(s2 * t)
        
    displacement_array.append(x)

# --- 4. Export to CSV ---
df = pd.DataFrame({
    "Time": time_array,
    "Displacement": displacement_array
})

# Round data for clean output
df = df.round({"Time": 3, "Displacement": 4})

file_name = "structural_dynamics_100_points.csv"
df.to_csv(file_name, index=False)

print(f"✅ Successfully generated '{file_name}' based on physical mass, stiffness, and damping.")
