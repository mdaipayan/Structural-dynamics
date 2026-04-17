import numpy as np
import pandas as pd

# --- Parameters ---
num_points = 100      # Exactly 100 data points
total_time = 10.0     # Total simulation time (seconds)
amplitude = 5.0       # Initial displacement (m)
damping = 0.25        # Damping ratio (how fast it decays)
frequency = 3.0       # Oscillation frequency (rad/s)

# --- Generate Data ---
# Create exactly 100 evenly spaced time steps between 0 and 10 seconds
time_array = np.linspace(0, total_time, num_points)

# Calculate displacement using a damped sine wave formula: x(t) = A * e^(-d*t) * cos(w*t)
displacement_array = amplitude * np.exp(-damping * time_array) * np.cos(frequency * time_array)

# --- Create and Save CSV ---
df = pd.DataFrame({
    "Time": time_array,
    "Displacement": displacement_array
})

# Round for cleaner CSV output
df = df.round({"Time": 3, "Displacement": 4})

# Save to file
file_name = "pendulum_100_points.csv"
df.to_csv(file_name, index=False)

print(f"✅ Successfully generated '{file_name}' with exactly {len(df)} rows of data.")
